#!/usr/bin/env python3
"""
refair_eval_commonval.py — FAIR comparison of COMBINED vs SEPARATE,
                           with BOTH resolutions scored on ONE common val set.

WHY THIS FILE EXISTS
--------------------
The 640 and 1280 experiments were built from two different Roboflow exports of the
SAME 1434-image pool. The exports drew different train/val splits:

    640 :  train 1232 | valid 202
    1280:  train 1247 | valid 187        (15 Algal-only images moved valid -> train)

Verified with a cross-export image-identity audit:
    * 1280_valid  is a strict SUBSET of 640_valid          (0 images outside)
    * 1280_valid  INTERSECT  640_train  =  0               <-- no leakage
    * neither dataset has train/valid leakage internally   (0 duplicated source images)

Therefore the 640-trained models can be scored on the 1280 validation set without
ever having seen those images. Doing so puts BOTH resolutions on ONE common
examination (n = 187), which is what the cross-resolution comparison requires.

Residual confound, to be disclosed: the 1280 models trained on 15 extra Algal images
(159 extra Algal instances). This affects the Algal class only; all other 11 classes
have identical training instance counts.

Everything else is unchanged from refair_eval_B1B2.py:
  B1 = merged leaf+pest, NO cross-model fusion (strict lower bound)
  B2 = merged leaf+pest, WITH class-agnostic cross-model NMS (real deployment)
"""
import os, glob
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import xywhn2xyxy

# ============================ EDIT THESE PATHS ============================
# ---------------------------------------------------------------------------
# Set the project root. Either export it:
#     export DURIAN_ROOT=/path/to/project      (Windows: set DURIAN_ROOT=...)
# or edit the fallback below.
# ---------------------------------------------------------------------------
BASE = os.environ.get("DURIAN_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# THE COMMON VALIDATION SET — the 1280 export's valid split (n=187).
# Both resolutions are scored on this.
COMMON_VAL_IMAGES = os.path.join(BASE, r"combine_model_1280\_merged\valid\images")
COMMON_VAL_LABELS = os.path.join(BASE, r"combine_model_1280\_merged\valid\labels")
COMMON_VAL_YAML   = os.path.join(BASE, r"combine_model_1280\_merged\data.yaml")

# Weights stay exactly as before. Only the val set is now shared.
CONFIGS = {
    640: dict(
        combined = os.path.join(BASE, r"six paper\640_combined_run\combined_640\weights\best.pt"),
        leaf     = os.path.join(BASE, r"six paper\Seperate_640_Run\leaf_640\weights\best.pt"),
        pest     = os.path.join(BASE, r"six paper\Seperate_640_Run\pest_640-3\weights\best.pt"),
    ),
    1280: dict(
        combined = os.path.join(BASE, r"six paper\combined_1280_run\weights\best.pt"),
        leaf     = os.path.join(BASE, r"six paper\1280leaf_and_pest_run\leaf_1280\weights\best.pt"),
        pest     = os.path.join(BASE, r"six paper\1280leaf_and_pest_run\pest_1280\weights\best.pt"),
    ),
}

CONF = 0.001
IOU_NMS = 0.7
# =========================================================================

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def read_names_from_yaml(path):
    import re
    txt = open(path, encoding='utf-8', errors='ignore').read()
    m = re.search(r'names:\s*\[([^\]]*)\]', txt)
    if m:
        return [x.strip().strip('\'"') for x in m.group(1).split(',') if x.strip()]
    names, lines = [], txt.splitlines()
    for i, l in enumerate(lines):
        if re.match(r'\s*names\s*:', l) and '[' not in l:
            j = i + 1
            while j < len(lines) and re.match(r'\s*-\s+', lines[j]):
                names.append(re.sub(r'\s*-\s+', '', lines[j]).strip().strip('\'"'))
                j += 1
            break
    return names


def list_images(img_dir):
    return [f for f in sorted(glob.glob(os.path.join(img_dir, '*')))
            if f.lower().endswith(IMG_EXT)]


def load_gt(label_path):
    cls, box = [], []
    if os.path.exists(label_path):
        for line in open(label_path, encoding='utf-8', errors='ignore'):
            p = line.split()
            if len(p) >= 5:
                cls.append(int(float(p[0])))
                box.append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    return np.array(cls, dtype=int), np.array(box, dtype=float).reshape(-1, 4)


def build_remap(model, merged_names):
    name2merged = {n: i for i, n in enumerate(merged_names)}
    mn = model.names
    remap = {}
    for lid in range(len(mn)):
        nm = mn[lid]
        if nm not in name2merged:
            raise ValueError(f"model class '{nm}' not in merged names {merged_names}")
        remap[lid] = name2merged[nm]
    return remap


def predict_boxes(model, img_path, imgsz, id_remap):
    r = model.predict(img_path, imgsz=imgsz, conf=CONF, iou=IOU_NMS, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    cls = np.array([id_remap[c] for c in cls], dtype=int)
    return xyxy, conf, cls


def cross_model_nms(xyxy, conf, cls, iou_thr=0.5):
    if len(conf) == 0:
        return xyxy, conf, cls
    idx = np.argsort(-conf)
    keep, taken = [], np.zeros(len(conf), dtype=bool)

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / u if u > 0 else 0

    for i in idx:
        if taken[i]:
            continue
        keep.append(i); taken[i] = True
        for j in idx:
            if not taken[j] and iou(xyxy[i], xyxy[j]) > iou_thr:
                taken[j] = True
    keep = np.array(keep, dtype=int)
    return xyxy[keep], conf[keep], cls[keep]


def evaluate(pred_per_image, gts, merged_names, whs, tag):
    from ultralytics.utils.metrics import ap_per_class, box_iou
    iouv = torch.linspace(0.5, 0.95, 10)
    niou = iouv.numel()
    stats = []
    for (pxyxy, pconf, pcls), (gcls, gxywhn), (W, H) in zip(pred_per_image, gts, whs):
        nl = len(gcls)
        tcls = gcls.tolist() if nl else []
        if len(pconf) == 0:
            if nl:
                stats.append((np.zeros((0, niou), dtype=bool), np.zeros(0),
                              np.zeros(0), np.array(tcls)))
            continue
        correct = np.zeros((len(pconf), niou), dtype=bool)
        if nl:
            gxyxy = xywhn2xyxy(torch.tensor(gxywhn, dtype=torch.float32), w=W, h=H).numpy()
            ious = box_iou(torch.tensor(gxyxy, dtype=torch.float32),
                           torch.tensor(pxyxy, dtype=torch.float32)).numpy()
            cls_match = (gcls.reshape(-1, 1) == pcls.reshape(1, -1))
            for k in range(niou):
                ok = (ious >= iouv[k].item()) & cls_match
                gi, pi = np.where(ok)
                if len(gi):
                    order = np.argsort(-ious[gi, pi])
                    gi, pi = gi[order], pi[order]
                    ug, up = set(), set()
                    for g, p in zip(gi, pi):
                        if g in ug or p in up:
                            continue
                        ug.add(g); up.add(p); correct[p, k] = True
        stats.append((correct, pconf, pcls, np.array(tcls)))

    correct = np.concatenate([s[0] for s in stats], 0)
    conf = np.concatenate([s[1] for s in stats], 0)
    pred_cls = np.concatenate([s[2] for s in stats], 0)
    target_cls = np.concatenate([s[3] for s in stats], 0)

    tp, fp, p, r, f1, ap, unique_cls, *_ = ap_per_class(
        correct, conf, pred_cls, target_cls, plot=False,
        names={i: n for i, n in enumerate(merged_names)})
    ap50 = ap[:, 0]
    per50 = {int(c): ap50[i] for i, c in enumerate(unique_cls)}

    print(f"\n[{tag}]")
    print(f"  {'class':22s}{'AP50':>9}")
    rows = []
    for i, nm in enumerate(merged_names):
        a = per50.get(i, float('nan'))
        rows.append((nm, a))
        print(f"  {nm:22s}{a:9.3f}")
    m = np.nanmean([x[1] for x in rows])
    print(f"  {'-'*31}")
    print(f"  {'mAP@0.5 (12-class)':22s}{m:9.3f}")
    return dict(per50=per50, mAP50=m)


def preflight():
    print("=" * 72)
    print("PRE-FLIGHT: checking every path exists")
    print("=" * 72)
    ok = True
    for p in (COMMON_VAL_IMAGES, COMMON_VAL_LABELS, COMMON_VAL_YAML):
        e = os.path.exists(p)
        ok &= e
        print(f"  [{'OK ' if e else 'MISS'}] {p}")
    for imgsz, cfg in CONFIGS.items():
        for role in ("combined", "leaf", "pest"):
            p = cfg[role]
            e = os.path.exists(p)
            ok &= e
            print(f"  [{'OK ' if e else 'MISS'}] {imgsz:>4} {role:<8} {p}")
    if not ok:
        raise SystemExit("\n!! Fix the MISS paths above before running.\n")
    print("  all paths present.\n")


def main():
    preflight()
    merged_names = read_names_from_yaml(COMMON_VAL_YAML)
    assert len(merged_names) == 12, f"expected 12 names, got {merged_names}"
    imgs = list_images(COMMON_VAL_IMAGES)
    print("=" * 72)
    print("COMMON VALIDATION SET (both resolutions scored on this)")
    print("=" * 72)
    print(f"  images : {len(imgs)}")
    print(f"  classes: {merged_names}")

    from PIL import Image
    gts, whs = [], []
    for ip in imgs:
        lp = os.path.join(COMMON_VAL_LABELS,
                          os.path.splitext(os.path.basename(ip))[0] + '.txt')
        gts.append(load_gt(lp))
        with Image.open(ip) as im:
            whs.append(im.size)
    inst = sum(len(g[0]) for g in gts)
    print(f"  gt instances: {inst}")

    results = {}
    for imgsz, cfg in CONFIGS.items():
        print("\n" + "=" * 72)
        print(f"imgsz = {imgsz}")
        print("=" * 72)

        cm = YOLO(cfg['combined'])
        comb = [predict_boxes(cm, ip, imgsz, build_remap(cm, merged_names)) for ip in imgs]
        r_comb = evaluate(comb, gts, merged_names, whs, f"UNIFIED @{imgsz}")

        lm, pm = YOLO(cfg['leaf']), YOLO(cfg['pest'])
        lr, pr = build_remap(lm, merged_names), build_remap(pm, merged_names)
        sep = []
        for ip in imgs:
            lx, lc, lcl = predict_boxes(lm, ip, imgsz, lr)
            px, pc, pcl = predict_boxes(pm, ip, imgsz, pr)
            sep.append((np.concatenate([lx, px], 0) if len(lx) + len(px) else np.zeros((0, 4)),
                        np.concatenate([lc, pc], 0) if len(lc) + len(pc) else np.zeros((0,)),
                        np.concatenate([lcl, pcl], 0) if len(lcl) + len(pcl) else np.zeros((0,), int)))
        r_sep = evaluate(sep, gts, merged_names, whs, f"SEPARATE (B1, no fusion) @{imgsz}")

        fused = [cross_model_nms(*s, iou_thr=0.5) for s in sep]
        r_fus = evaluate(fused, gts, merged_names, whs, f"SEPARATE + FUSION (B2) @{imgsz}")

        results[imgsz] = (r_comb, r_sep, r_fus)

    # index into results[imgsz]: 0 = unified, 1 = separate B1 (no fusion), 2 = separate B2 (fusion)
    UNI, B1, B2 = 0, 1, 2

    print("\n\n" + "=" * 88)
    print("PER-CLASS AP@0.5 ON THE COMMON VAL SET  (all three policies, both resolutions)")
    print("=" * 88)
    print("  B1 = leaf+pest merged, NO fusion (strict lower bound)")
    print("  B2 = class-agnostic NMS (IoU 0.5) over the merged set (the fair operating point)")
    print("  -> the paper's Table I prints the Uni and B2 columns; Table S-V prints all six.\n")
    print(f"  {'class':22s}{'Uni@640':>10}{'B1@640':>10}{'B2@640':>10}"
          f"{'Uni@1280':>10}{'B1@1280':>10}{'B2@1280':>10}")
    for i, nm in enumerate(merged_names):
        vals = [results[640][UNI]['per50'].get(i, float('nan')),
                results[640][B1]['per50'].get(i, float('nan')),
                results[640][B2]['per50'].get(i, float('nan')),
                results[1280][UNI]['per50'].get(i, float('nan')),
                results[1280][B1]['per50'].get(i, float('nan')),
                results[1280][B2]['per50'].get(i, float('nan'))]
        print(f"  {nm:22s}" + "".join(f"{v:10.3f}" for v in vals))
    print("  " + "-" * 82)
    print(f"  {'mAP@0.5 (12-class)':22s}" + "".join(
        f"{results[r][k]['mAP50']:10.3f}" for r in (640, 1280) for k in (UNI, B1, B2)))

    print("\n" + "=" * 72)
    print("TABLE II  (fusion)")
    print("=" * 72)
    print(f"  {'configuration':28s}{'mAP@0.5 (640)':>16}{'mAP@0.5 (1280)':>16}")
    print(f"  {'Unified (single model)':28s}{results[640][0]['mAP50']:16.3f}{results[1280][0]['mAP50']:16.3f}")
    print(f"  {'Separate, no fusion':28s}{results[640][1]['mAP50']:16.3f}{results[1280][1]['mAP50']:16.3f}")
    print(f"  {'Separate + fusion (NMS)':28s}{results[640][2]['mAP50']:16.3f}{results[1280][2]['mAP50']:16.3f}")

    print("\nDONE — all four configurations scored on ONE common 12-class val set.")


if __name__ == '__main__':
    main()
