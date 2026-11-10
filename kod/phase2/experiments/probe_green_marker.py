"""probe_green_marker.py — Yeşil işaretçi kalıntısı probu (TR): val çiftlerinde forged / onarılmış forged / authentic / authentic+işaretçi koşullarında 9 modeli çalıştırır.
Çıktı: results/per_image_green_marker.csv. Kullanım: python probe_green_marker.py (VIZE_ROOT env ile kök).
"""

import os, sys, json, time, numpy as np, pandas as pd, cv2, torch
os.environ.setdefault("USERNAME","user")
EXP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, EXP)
import common.p2data as D, infer_fullimage as IF
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score
import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
VP = str(_ROOT); DATA = os.path.join(VP,"data","AIForge-Doc-v1")
OUT = os.path.join(VP,"phase2","experiments","results"); os.makedirs(OUT, exist_ok=True)
dev = "cuda"
models = [f"unet_{g}_s{s}" for g in ["bbox","pixel_eps8_morph","glyph_v2"] for s in (42,43,44)]
nets = [IF.load_model(m, dev) for m in models]
tau = json.load(open(os.path.join(OUT,"tau.json")))
def get_tau(m, gt): return float(tau[m][gt])
splits = json.load(open(os.path.join(VP,"dataset_audit","splits.json")))
sf = pd.read_csv(os.path.join(VP,"dataset_audit","samples_forged.csv")); sf = sf[(sf.split=="training")&(sf.pair_accepted==True)].set_index("new_id")
val = [i for i in splits["val"] if int(i) in sf.index]
def gmask(rgb):
    r,g,b = rgb[...,0].astype(np.int16), rgb[...,1].astype(np.int16), rgb[...,2].astype(np.int16)
    return (g-np.maximum(r,b)>50)&(g>110)
rng = np.random.default_rng(0)
def ap_img(prob, gt, nneg=300000):
    pos = prob[gt]; neg = prob[~gt]
    if pos.size==0: return np.nan
    if neg.size>nneg: neg = neg[rng.choice(neg.size, nneg, replace=False)]
    y = np.r_[np.ones(pos.size), np.zeros(neg.size)]; s = np.r_[pos, neg]
    return float(average_precision_score(y, s))
def f1(pred, gt):
    tp = (pred&gt).sum(); fp = (pred&~gt).sum(); fn = (~pred&gt).sum()
    return float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn)>0 else np.nan
rows=[]; t0=time.time()
for k,nid in enumerate(val):
    s = f"{int(nid):09d}"
    f = cv2.cvtColor(cv2.imread(os.path.join(DATA,"TrainingSet","images",s+".png")), cv2.COLOR_BGR2RGB)
    a = cv2.cvtColor(cv2.imread(os.path.join(DATA,"TrainingSet","authentic", sf.loc[int(nid),"pair_name"])), cv2.COLOR_BGR2RGB)
    if f.shape!=a.shape: continue
    mb = cv2.imread(os.path.join(DATA,"TrainingSet","masks",s+".png"),0)>0
    mp = cv2.imread(os.path.join(VP,"phase2","gt","pixel_eps8_morph",s+".png"),0)>0
    ng = gmask(f)&~gmask(a); ring = ndi.binary_dilation(mb,iterations=6)&~ndi.binary_erosion(mb,iterations=6)
    res = ng&ring; n_res = int(res.sum()); resd = ndi.binary_dilation(res, iterations=1)
    repaired = f.copy(); repaired[resd] = a[resd]
    auth_marker = a.copy(); auth_marker[resd] = f[resd]
    conds = {"forged": f, "forged_repaired": repaired, "authentic": a, "authentic_plus_marker": auth_marker}
    for cname, img in conds.items():
        probs,_ = IF.predict_full_multi(nets, img, dev)
        for m,p in zip(models, probs):
            tgt = m.split("_s")[0].replace("unet_","")
            tb, tp_ = get_tau(m,"bbox"), get_tau(m,"pixel_eps8_morph")
            pb = p>=tb
            rows.append(dict(new_id=s, cond=cname, model=m, train_gt=tgt, n_residue=n_res, has_residue=int(n_res>=20), tool=sf.loc[int(nid),"assigned_tool"], source=sf.loc[int(nid),"source_dataset"], group=sf.loc[int(nid),"group"],
                ap_bbox=ap_img(p,mb), ap_pixel=ap_img(p,mp), f1_bbox_tau=f1(pb,mb), f1_pixel_tau=f1(p>=tp_,mp),
                pos_frac_05=float((p>=0.5).mean()), pos_frac_tau=float(pb.mean()), top1=float(np.percentile(p,99)), score_max=float(p.max()),
                iou_predbox_tau=float((pb&mb).sum()/max((pb|mb).sum(),1)), bbox_cover_tau=float((pb&mb).sum()/max(mb.sum(),1))))
    if k%50==0: print(k, len(val), f"{time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT,"per_image_green_marker.csv"), index=False)
print("done", len(df), f"{time.time()-t0:.0f}s")
