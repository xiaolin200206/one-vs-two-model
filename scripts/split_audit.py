#!/usr/bin/env python3
"""
split_audit.py — cross-export image-identity audit.

WHY THIS EXISTS
---------------
The 640 and 1280 experiments were built from two Roboflow exports of the same
1,434-image pool. The exports drew DIFFERENT train/validation splits:

    640 :  train 1232 | valid 202
    1280:  train 1247 | valid 187      (15 Algal-only images moved valid -> train)

That made any cross-resolution accuracy comparison invalid: the two conditions
were being scored on two different exam papers. The tell was one integer -- the
training-label plot showed Algal = 939 for one export and 1,098 for the other,
while all eleven other classes matched exactly.

WHAT THIS SCRIPT CHECKS
-----------------------
Images cannot be matched by filename, because Roboflow rewrites the name on every
export: it prepends a source+version token and appends a content hash.

    Background_v1i_yolov__IMG_0003_jpg.rf.814fb50a....txt      (640 export)
    Background_v2i_yolov__IMG_0003_jpg.rf.b601f2d3....txt      (1280 export)
    Durian_Root_v5i_yolo__IMG_1775_JPG.rf.7c4a1e09....txt

Two keys are therefore needed, and they answer different questions:

  key_strict  : keep the source AND the version, drop only the hash.
                Two files sharing this key are augmented/re-encoded copies of the
                SAME source image within ONE export.
                -> use for the INTRA-dataset leakage check.

  key_cross   : keep the source, collapse the version token, drop the hash.
                Matches the same image ACROSS the two exports.
                -> use for the CROSS-dataset subset / intersection checks.

A naive key that strips the source prefix as well produces false positives:
different source datasets both contain an "IMG_0003", and they are not the same
image. That naive key reported 13-14 phantom leaks here. Both real keys report zero.

RESULTS ON THIS DATASET
-----------------------
  no source image appears in both partitions of either export     -> no leakage
  no augmented copies on disk (augmentation is online-only)       -> by construction
  1280_valid is a strict SUBSET of 640_valid                      -> 0 images outside
  1280_valid INTERSECT 640_train = 0                              -> re-scoring is safe

The last line is the one that matters: it means the 640-trained models can be
scored on the 1280 validation set without ever having seen those images, which is
what refair_eval_commonval.py does.
"""
import collections
import glob
import os
import re

# ---------------------------------------------------------------- paths
SETS = {
    "640":  r"Combined_model\_merged",
    "1280": r"combine_model_1280\_merged",
}
NAMES = ["Algal", "Leaf_rot", "leafhopper_damage", "Phomopsis", "Pink_disease",
         "Psyllid", "Psyllid_damage", "Root_disease", "Scale_insect",
         "Stem_borer", "weevil", "weevil_damage"]


# ---------------------------------------------------------------- keys
def key_strict(fn):
    """Source + version kept, .rf.<hash> dropped.
    Same key twice inside one export = two copies of the same source image."""
    s = re.sub(r"\.rf\.[0-9a-f]+", "", fn)
    return re.sub(r"\.txt$", "", s)


def key_cross(fn):
    """Source kept, Roboflow version token collapsed, hash dropped.
    Matches the same image across the two exports.
        Background_v1i_yolov__IMG_0003  -> Background__IMG_0003
        Durian_Phomopsis_v4i__IMG_9902  -> Durian_Phomopsis__IMG_9902
        Durian_Root_v5i_yolo__IMG_1775  -> Durian_Root__IMG_1775
    """
    s = re.sub(r"\.rf\.[0-9a-f]+", "", fn)
    s = re.sub(r"\.txt$", "", s)
    s = re.sub(r"_v\d+i.*?__", "__", s, count=1)   # eat from _v<N>i to the first __
    return s


def scan(base, split, keyfn):
    d = os.path.join(base, split, "labels")
    out = collections.defaultdict(list)
    for p in glob.glob(os.path.join(d, "*.txt")):
        out[keyfn(os.path.basename(p))].append(os.path.basename(p))
    return out


def class_counts(base, split):
    c = collections.Counter()
    for p in glob.glob(os.path.join(base, split, "labels", "*.txt")):
        with open(p, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                s = line.split()
                if s:
                    c[int(float(s[0]))] += 1
    return c


# ---------------------------------------------------------------- 1. what tipped us off
print("=" * 74)
print("1. THE TELL — per-class instance counts should not depend on input resolution")
print("=" * 74)
for split in ("train", "valid"):
    a = class_counts(SETS["640"], split)
    b = class_counts(SETS["1280"], split)
    print(f"\n  split = {split}")
    print(f"    {'class':<20}{'640':>8}{'1280':>8}{'diff':>8}")
    for k in range(12):
        d = b[k] - a[k]
        print(f"    {NAMES[k]:<20}{a[k]:>8}{b[k]:>8}{d:>+8}{'   <<<' if d else ''}")
    print(f"    {'TOTAL':<20}{sum(a.values()):>8}{sum(b.values()):>8}"
          f"{sum(b.values()) - sum(a.values()):>+8}")

# ---------------------------------------------------------------- 2. leakage inside each export
print("\n" + "=" * 74)
print("2. INTRA-EXPORT LEAKAGE  (key_strict: same source image in train AND valid?)")
print("=" * 74)
for tag, base in SETS.items():
    tr = scan(base, "train", key_strict)
    va = scan(base, "valid", key_strict)
    ntr = sum(len(v) for v in tr.values())
    nva = sum(len(v) for v in va.values())
    dup_tr = sum(1 for v in tr.values() if len(v) > 1)
    dup_va = sum(1 for v in va.values() if len(v) > 1)
    leak = sorted(set(tr) & set(va))
    print(f"\n  {tag}")
    print(f"    train {ntr:5d} files -> {len(tr):5d} distinct source images "
          f"({dup_tr} with >1 copy)")
    print(f"    valid {nva:5d} files -> {len(va):5d} distinct source images "
          f"({dup_va} with >1 copy)")
    print(f"    LEAKAGE (source image in both partitions): {len(leak)}"
          f"   {'<-- CLEAN' if not leak else '<-- *** LEAK ***'}")
    for k in leak[:5]:
        print(f"        {k}")

# ---------------------------------------------------------------- 3. cross-export
print("\n" + "=" * 74)
print("3. CROSS-EXPORT  (key_cross)")
print("=" * 74)
tr640 = set(scan(SETS["640"], "train", key_cross))
va640 = set(scan(SETS["640"], "valid", key_cross))
tr1280 = set(scan(SETS["1280"], "train", key_cross))
va1280 = set(scan(SETS["1280"], "valid", key_cross))

pool640 = tr640 | va640
pool1280 = tr1280 | va1280
print(f"\n  image pool  640 = {len(pool640)}   1280 = {len(pool1280)}   "
      f"symmetric difference = {len(pool640 ^ pool1280)}   (0 = same pool)")
print(f"\n  Q1  1280_valid NOT in 640_valid      : {len(va1280 - va640)}"
      f"   {'(subset)' if not (va1280 - va640) else '(NOT a subset)'}")
print(f"  Q2  1280_valid INTERSECT 640_train   : {len(va1280 & tr640)}"
      f"   <-- MUST BE 0 to re-score the 640 models on the 1280 val set")
print(f"  Q3  640_valid  INTERSECT 1280_train  : {len(va640 & tr1280)}"
      f"   (the images that moved)")

bad = sorted(va1280 & tr640)
print()
if bad:
    print("  >>> UNSAFE. The 640 models trained on images in the 1280 validation set:")
    for k in bad[:10]:
        print(f"        {k}")
else:
    print("  >>> SAFE. The 640 models never saw any image in the 1280 validation set.")
    print("  >>> Score both resolutions on the 1280 validation split "
          f"(n = {len(va1280)}) — see refair_eval_commonval.py")

print(f"\n  residual training-data difference: "
      f"1280-only {len(tr1280 - tr640)}, 640-only {len(tr640 - tr1280)}")
print("  (the 1280 models trained on these extras; disclose in the paper)")
print("\ndone.")
