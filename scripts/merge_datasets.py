#!/usr/bin/env python3
"""
merge_datasets.py  —  merge the per-class YOLO exports + Background negatives into one training set
                      The class table is discovered from the source data.yaml files, not hard-coded;
                      for this dataset it resolves to 12 classes (5 foliar + 7 pest).

设计原则(全部为了不重蹈 Paper 2 的泄漏覆辙):
  - 保持每个数据集各自的 train/valid 划分,绝不重新洗牌
    (每个数据集导出时已是"只增强 train",拼起来 val 依旧干净)
  - Background(无标注)只进 train 作负样本,绝不进 val
  - 类名清理: Pink_diease->Pink_disease, 'leafhopper damage'->leafhopper_damage,
    Stem-borer->Stem_borer  (只改名,不动数据、不改类数)
  - 参数化: 类表由源数据集自动发现(本数据集为 12 类); 要丢类/合并类改 DROP / MERGE

用法:
  python merge_datasets.py "C:/.../Combined_model" --out "C:/.../Combined_model/_merged"

产物:
  <out>/train/images, <out>/train/labels
  <out>/valid/images, <out>/valid/labels
  <out>/data.yaml            (合并后的统一类表)
  <out>/_merge_report.txt    (合并统计: 每类 train/val instance、来源、负样本数)
"""

import os, sys, glob, shutil, re, argparse
from collections import Counter, defaultdict

IMG_EXT=('.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp')

# ---- 类名清理映射(只改这些,其余原样) ----
RENAME = {
    'Pink_diease':'Pink_disease',
    'leafhopper damage':'leafhopper_damage',
    'Stem-borer':'Stem_borer',
}
# ---- 可选:丢弃的类(这次基线为空;下次要丢 weevil 就填 {'weevil'}) ----
DROP = set()
# ---- 可选:合并映射(这次基线为空;下次要合并 psyllid 就填 {'Psyllid_damage':'Psyllid'}) ----
MERGE = {}

def clean(name):
    name=RENAME.get(name, name)
    name=MERGE.get(name, name)
    return name

def parse_names(yaml_path):
    txt=open(yaml_path,encoding='utf-8',errors='ignore').read()
    m=re.search(r'names:\s*\[([^\]]*)\]', txt)
    if m:
        return {i:x.strip().strip('\'"') for i,x in enumerate(
            [y for y in m.group(1).split(',') if y.strip()])}
    names={}; lines=txt.splitlines()
    for i,l in enumerate(lines):
        if re.match(r'\s*names\s*:',l) and '[' not in l:
            j=i+1; idx=0
            while j<len(lines) and re.match(r'\s*-\s+',lines[j]):
                names[idx]=re.sub(r'\s*-\s+','',lines[j]).strip().strip('\'"'); idx+=1; j+=1
            break
    return names

def find_datasets(root):
    out=[]
    for dp,_,fns in os.walk(root):
        if ('data.yaml' in fns or 'data.yml' in fns) and os.path.abspath(dp)!=os.path.abspath(root):
            # 跳过已经生成的 _merged
            if os.sep+'_merged' in dp: continue
            out.append(dp)
    return sorted(set(out))

def split_dirs(ds, split):
    for s in ([split] if split!='valid' else ['valid','val']):
        img=os.path.join(ds,s,'images'); lbl=os.path.join(ds,s,'labels')
        if os.path.isdir(img): return img, lbl
    return None, None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--out', default=None)
    a=ap.parse_args()
    root=a.root
    out=a.out or os.path.join(root,'_merged')
    if os.path.isdir(out): shutil.rmtree(out)
    for s in ('train','valid'):
        os.makedirs(os.path.join(out,s,'images'),exist_ok=True)
        os.makedirs(os.path.join(out,s,'labels'),exist_ok=True)

    datasets=find_datasets(root)
    # 第一遍: 收集全局类表(清理+丢弃+合并后)
    global_classes=[]
    seen=set()
    ds_names={}
    for ds in datasets:
        yp=os.path.join(ds,'data.yaml')
        if not os.path.exists(yp): yp=os.path.join(ds,'data.yml')
        names=parse_names(yp); ds_names[ds]=names
        for cid,nm in names.items():
            c=clean(nm)
            if c in DROP or not nm: continue
            if c not in seen: seen.add(c); global_classes.append(c)
    global_classes.sort(key=str.lower)
    cls2gid={c:i for i,c in enumerate(global_classes)}

    rep=[]; wtr=lambda s='': (print(s), rep.append(s))
    wtr("="*74); wtr("合并 YOLO 数据集"); wtr("="*74)
    wtr(f"全局类表 ({len(global_classes)} 类): {global_classes}")
    if DROP:  wtr(f"丢弃: {sorted(DROP)}")
    if MERGE: wtr(f"合并: {MERGE}")
    wtr(f"重命名: {RENAME}")

    tr_inst=Counter(); va_inst=Counter(); tr_img=Counter(); va_img=Counter()
    neg_train=0; dup=0; used=set()

    def copy_split(ds, names, split):
        nonlocal neg_train, dup
        img_dir,lbl_dir=split_dirs(ds,split)
        if not img_dir: return
        dst_img=os.path.join(out,split,'images'); dst_lbl=os.path.join(out,split,'labels')
        for f in os.listdir(img_dir):
            if not f.lower().endswith(IMG_EXT): continue
            stem=os.path.splitext(f)[0]
            # 防止跨数据集同名覆盖: 加数据集前缀
            pfx=re.sub(r'[^A-Za-z0-9]+','_',os.path.basename(ds))[:20]
            newname=f"{pfx}__{f}"
            if newname in used: dup+=1; continue
            used.add(newname)
            shutil.copy2(os.path.join(img_dir,f), os.path.join(dst_img,newname))
            # label
            src_lbl=os.path.join(lbl_dir, stem+'.txt') if lbl_dir else None
            out_lines=[]
            if src_lbl and os.path.exists(src_lbl) and os.path.getsize(src_lbl)>0:
                for line in open(src_lbl,encoding='utf-8',errors='ignore'):
                    p=line.split()
                    if not p: continue
                    try: cid=int(float(p[0]))
                    except: continue
                    nm=names.get(cid,'')
                    c=clean(nm)
                    if not nm or c in DROP: continue          # 丢弃该类的行
                    gid=cls2gid[c]
                    out_lines.append(" ".join([str(gid)]+p[1:]))
                    if split=='train': tr_inst[c]+=1
                    else: va_inst[c]+=1
            # 写 label(可能为空 => 负样本)
            with open(os.path.join(dst_lbl, os.path.splitext(newname)[0]+'.txt'),'w') as g:
                g.write("\n".join(out_lines))
            if not out_lines and split=='train': neg_train+=1
            # 记录 image 级
            cls_here=set()
            if out_lines:
                for l in out_lines: cls_here.add(global_classes[int(l.split()[0])])
            for c in cls_here:
                (tr_img if split=='train' else va_img)[c]+=1

    BG_KEYS=('background',)
    for ds in datasets:
        names=ds_names[ds]
        is_bg = (len(names)==0) or any(k in os.path.basename(ds).lower() for k in BG_KEYS)
        copy_split(ds, names, 'train')
        if is_bg:
            wtr(f"[BG ] {os.path.basename(ds):40s} -> train 负样本(不进 val)")
            continue                      # Background 只进 train
        copy_split(ds, names, 'valid')

    # data.yaml
    yaml_txt = ("train: ./train/images\nval: ./valid/images\n"
                f"nc: {len(global_classes)}\nnames:\n" +
                "".join(f"- {c}\n" for c in global_classes))
    open(os.path.join(out,'data.yaml'),'w',encoding='utf-8').write(yaml_txt)

    wtr("\n"+"="*74); wtr("合并结果(每类 instance / image)"); wtr("="*74)
    wtr(f"{'class':22s}{'train_inst':>11}{'train_img':>10}{'val_inst':>10}{'val_img':>9}")
    wtr("-"*62)
    for c in global_classes:
        wtr(f"{c:22s}{tr_inst[c]:>11}{tr_img[c]:>10}{va_inst[c]:>10}{va_img[c]:>9}")
    wtr("-"*62)
    ntr=sum(len(os.listdir(os.path.join(out,'train','images'))) for _ in [0])
    wtr(f"train images: {len(os.listdir(os.path.join(out,'train','images')))} "
        f"(含 {neg_train} 张负样本)")
    wtr(f"valid images: {len(os.listdir(os.path.join(out,'valid','images')))}")
    if dup: wtr(f"⚠️ 跳过重名图: {dup}")
    wtr(f"\n最强势类 train_inst: {tr_inst.most_common(1)[0] if tr_inst else '-'}")
    weak=[f'{c}({tr_inst[c]})' for c in global_classes if tr_inst[c]<50]
    if weak: wtr(f"高危弱势类(train_inst<50): {', '.join(weak)}")
    wtr(f"\ndata.yaml -> {os.path.join(out,'data.yaml')}")

    open(os.path.join(out,'_merge_report.txt'),'w',encoding='utf-8').write("\n".join(rep))
    wtr(f"报告 -> {os.path.join(out,'_merge_report.txt')}  (把这个发我)")

if __name__=='__main__':
    main()
