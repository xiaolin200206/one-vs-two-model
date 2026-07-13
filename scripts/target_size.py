#!/usr/bin/env python3
"""
target_size.py — compute per-class average bbox PIXEL AREA from YOLO labels,
to quantify which disease/pest classes are 'small' vs 'large' targets (COCO scale).

COCO scale convention:
  small  : area < 32^2  = 1024 px^2
  medium : 1024 <= area < 96^2 = 9216 px^2
  large  : area >= 9216 px^2

Run (edit PATHS below):
  python target_size.py
Then paste the printed table back.
"""
import os, glob
from collections import defaultdict
from PIL import Image

# ============ EDIT THESE (use the 12-class merged valid; do both 640 & 1280 if you like) ============
# ---------------------------------------------------------------------------
# Set the project root. Either export it:
#     export DURIAN_ROOT=/path/to/project      (Windows: set DURIAN_ROOT=...)
# or edit the fallback below.
# ---------------------------------------------------------------------------
BASE = os.environ.get("DURIAN_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# point at the 12-class merged validation set (labels + images)
LABELS_DIR = os.path.join(BASE, r"combine_model_1280\_merged\valid\labels")
IMAGES_DIR = os.path.join(BASE, r"combine_model_1280\_merged\valid\images")
MERGED_NAMES = ['Algal','Leaf_rot','leafhopper_damage','Phomopsis','Pink_disease',
                'Psyllid','Psyllid_damage','Root_disease','Scale_insect','Stem_borer','weevil','weevil_damage']
# ===================================================================================================

IMG_EXT=('.jpg','.jpeg','.png','.bmp','.webp')

def find_image(stem):
    for e in IMG_EXT:
        p=os.path.join(IMAGES_DIR, stem+e)
        if os.path.exists(p): return p
    # case-insensitive fallback
    for f in glob.glob(os.path.join(IMAGES_DIR, stem+'.*')):
        return f
    return None

areas=defaultdict(list)   # class_id -> list of pixel areas
counts=defaultdict(int)

for lbl in glob.glob(os.path.join(LABELS_DIR,'*.txt')):
    stem=os.path.splitext(os.path.basename(lbl))[0]
    img=find_image(stem)
    if img is None:
        continue
    with Image.open(img) as im:
        W,H=im.size
    for line in open(lbl, encoding='utf-8', errors='ignore'):
        p=line.split()
        if len(p)<5: continue
        cid=int(float(p[0])); w=float(p[3]); h=float(p[4])
        area=(w*W)*(h*H)
        areas[cid].append(area)
        counts[cid]+=1

def bucket(a):
    if a<1024: return 'small'
    if a<9216: return 'medium'
    return 'large'

print(f"{'class':22s}{'n':>6}{'mean_area(px^2)':>16}{'sqrt(area)':>12}{'COCO':>8}")
print('-'*66)
# also group leaf vs pest for the paper
leaf_ids={0,1,3,4,7}  # Algal,Leaf_rot,Phomopsis,Pink_disease,Root_disease
pest_ids=set(range(12))-leaf_ids
leaf_areas=[]; pest_areas=[]
for cid in range(len(MERGED_NAMES)):
    if cid not in areas or not areas[cid]:
        print(f"{MERGED_NAMES[cid]:22s}{0:>6}{'-':>16}{'-':>12}{'-':>8}")
        continue
    m=sum(areas[cid])/len(areas[cid])
    import math
    print(f"{MERGED_NAMES[cid]:22s}{counts[cid]:>6}{m:>16.0f}{math.sqrt(m):>12.1f}{bucket(m):>8}")
    if cid in leaf_ids: leaf_areas+=areas[cid]
    else: pest_areas+=areas[cid]

import math
print('-'*66)
if leaf_areas:
    lm=sum(leaf_areas)/len(leaf_areas)
    print(f"{'ALL LEAF classes':22s}{len(leaf_areas):>6}{lm:>16.0f}{math.sqrt(lm):>12.1f}{bucket(lm):>8}")
if pest_areas:
    pm=sum(pest_areas)/len(pest_areas)
    print(f"{'ALL PEST classes':22s}{len(pest_areas):>6}{pm:>16.0f}{math.sqrt(pm):>12.1f}{bucket(pm):>8}")
print("\nPaste this whole table back.")
