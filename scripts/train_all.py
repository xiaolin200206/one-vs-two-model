#!/usr/bin/env python3
"""
train_all.py — train the six detectors of this study under identical hyperparameters.

Experiment matrix: {unified, separate} x {640, 1280}
    combined_640  / combined_1280   12-class detector          yolo11s
    leaf_640      / leaf_1280        5-class foliar detector   yolo11s
    pest_640      / pest_1280        7-class pest detector     yolo11n

The leaf detector uses yolo11s and the pest detector yolo11n, matching the backbones
of the two-model system this study compares against. The unified detector uses yolo11s,
the same backbone as the leaf model it replaces.

HYPERPARAMETERS
---------------
Every configuration is trained with the settings below. They are not defaults chosen
for convenience -- they are the values recorded in the Ultralytics `args.yaml` of the
actual runs, and they are what the paper's claim of "identical training hyperparameters"
refers to:

    epochs        100
    batch         16
    imgsz         640 or 1280   (the variable under test)
    optimizer     auto
    lr0           0.01
    seed          0
    deterministic True
    pretrained    True   (COCO weights)
    patience      100    (never triggers on a 100-epoch run; no early stopping)

Augmentation is ONLINE ONLY -- Ultralytics applies it in memory, per batch, to the
training partition. No augmented file is ever written to disk. This is deliberate:
it makes the pipeline split-before-augment by construction, and closes the
augment-before-split leakage path documented in the earlier dataset audit.
The active augmentations are the Ultralytics defaults:
    mosaic 1.0, fliplr 0.5, hsv_h 0.015, hsv_s 0.7, hsv_v 0.4,
    translate 0.1, scale 0.5, auto_augment randaugment, erasing 0.4,
    close_mosaic 10   (mosaic is disabled for the final 10 epochs)

USAGE
-----
    python train_all.py --config combined_1280 --data /path/to/_merged/data.yaml
    python train_all.py --all --root /path/to/project

The runs reported in the paper were executed on a cloud GPU. Nothing in this script
depends on that; --data / --root take any path.
"""
import argparse
import os

# The six configurations. Paths are resolved from --root (or overridden with --data).
CONFIGS = {
    # name            (dataset subdir,          weights,       imgsz)
    'combined_640':  ('Combined_model/_merged',       'yolo11s.pt', 640),
    'combined_1280': ('combine_model_1280/_merged',   'yolo11s.pt', 1280),
    'leaf_640':      ('Combined_model/_leaf_merged',  'yolo11s.pt', 640),
    'leaf_1280':     ('combine_model_1280/_leaf_merged', 'yolo11s.pt', 1280),
    'pest_640':      ('Combined_model/_pest_merged',  'yolo11n.pt', 640),
    'pest_1280':     ('combine_model_1280/_pest_merged', 'yolo11n.pt', 1280),
}

# Fixed across all six configurations. Changing any of these breaks the comparison.
HP = dict(
    epochs=100,
    batch=16,
    optimizer='auto',
    lr0=0.01,
    seed=0,
    deterministic=True,
    pretrained=True,
    patience=100,
    plots=True,
    exist_ok=True,
)


def train_one(name, data, weights, imgsz, device=None, overrides=None):
    from ultralytics import YOLO

    if not os.path.isfile(data):
        print(f"[skip] {name}: data.yaml not found at {data}")
        return None

    hp = dict(HP)
    if overrides:
        hp.update(overrides)

    print("\n" + "=" * 70)
    print(f"train {name}   weights={weights}   imgsz={imgsz}")
    print(f"data={data}")
    print(f"hyperparameters: " + "  ".join(f"{k}={v}" for k, v in sorted(hp.items())
                                           if k not in ('plots', 'exist_ok')))
    print("=" * 70)

    m = YOLO(weights)
    m.train(data=data, imgsz=imgsz, name=name, device=device, **hp)

    best = os.path.join('runs', 'detect', name, 'weights', 'best.pt')
    if os.path.isfile(best):
        print(f"\n--- {name}: per-class AP@0.5 ---")
        r = YOLO(best).val(data=data, imgsz=imgsz, split='val', plots=True, verbose=False)
        for i, c in enumerate(r.box.ap_class_index):
            print(f"  {r.names[c]:22s} {float(r.box.ap50[i]):.3f}")
        print(f"  {'mAP@0.5':22s} {r.box.map50:.3f}")
        print(f"\nweights: {best}")
        print(f"export : yolo export model={best} format=onnx imgsz={imgsz}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', choices=list(CONFIGS))
    ap.add_argument('--all', action='store_true', help='train every configuration in turn')
    ap.add_argument('--root', default=os.environ.get('DURIAN_ROOT', '.'),
                    help='project root containing the _merged dataset folders')
    ap.add_argument('--data', help='explicit data.yaml; only valid with --config')
    ap.add_argument('--device', default=None, help="'0', 'cpu', or leave unset for auto")
    ap.add_argument('--batch', type=int, default=None,
                    help='override batch size (a 6 GB GPU needs 2 at imgsz 1280)')
    a = ap.parse_args()

    if a.data and a.all:
        ap.error("--data applies to a single --config; it cannot be combined with --all "
                 "(it would feed the same dataset to all six configurations).")

    overrides = {'batch': a.batch} if a.batch else None

    if a.all:
        for name, (subdir, weights, imgsz) in CONFIGS.items():
            data = os.path.join(a.root, subdir, 'data.yaml')
            train_one(name, data, weights, imgsz, a.device, overrides)
    elif a.config:
        subdir, weights, imgsz = CONFIGS[a.config]
        data = a.data or os.path.join(a.root, subdir, 'data.yaml')
        train_one(a.config, data, weights, imgsz, a.device, overrides)
    else:
        ap.print_help()
        print("\nconfigurations:", ', '.join(CONFIGS))


if __name__ == '__main__':
    main()
