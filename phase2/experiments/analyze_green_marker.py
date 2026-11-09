"""analyze_green_marker.py — Yeşil işaretçi probunun özeti (TR): koşul × model ailesi tabloları, grup-bootstrap CI'lı farklar ve figür.
Girdi: results/per_image_green_marker.csv ; Çıktı: results/green_marker_summary.csv, results/green_marker_contrasts.csv, figures/fig_green_marker.png
"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE,"results"); F = os.path.join(HERE,"figures")
df = pd.read_csv(os.path.join(R,"per_image_green_marker.csv"))
df["family"] = df.train_gt.map({"bbox":"bbox","pixel_eps8_morph":"pixel","glyph_v2":"glyph"})
# seed-average per image
key = ["new_id","cond","family","has_residue","tool","source","group"]
num = ["ap_bbox","ap_pixel","f1_bbox_tau","f1_pixel_tau","pos_frac_05","pos_frac_tau","top1","score_max","iou_predbox_tau","bbox_cover_tau"]
g = df.groupby(key)[num].mean().reset_index()
def boot_mean(x, groups, B=1000, seed=0):
    rng = np.random.default_rng(seed); x=np.asarray(x,float); groups=np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True); sums = np.bincount(inv, weights=x); cnts = np.bincount(inv)
    out=[]
    for _ in range(B):
        idx = rng.integers(0, len(ug), len(ug)); out.append(sums[idx].sum()/cnts[idx].sum())
    return np.nanmean(x), np.percentile(out,2.5), np.percentile(out,97.5)
rows=[]
for (fam,cond,res), s in g.groupby(["family","cond","has_residue"]):
    r = dict(family=fam, cond=cond, has_residue=res, n=len(s))
    for m in ["ap_bbox","ap_pixel","f1_bbox_tau","pos_frac_tau","iou_predbox_tau","bbox_cover_tau","top1"]:
        ss = s[["group", m]].copy()
        if m.startswith(("pos","iou","bbox","top")): ss[m] = ss[m].fillna(0)
        ss = ss.dropna(subset=[m])
        mm, lo, hi = boot_mean(ss[m].values, ss["group"].values)
        r[m]=mm; r[m+"_lo"]=lo; r[m+"_hi"]=hi
    rows.append(r)
summ = pd.DataFrame(rows); summ.to_csv(os.path.join(R,"green_marker_summary.csv"), index=False)
# paired contrasts within residue images: forged - repaired (AP bbox), authentic_plus_marker vs authentic (pos_frac, iou_predbox)
con=[]
for fam, s in g[g.has_residue==1].groupby("family"):
    w = s.pivot_table(index=["new_id","group"], columns="cond", values=["ap_bbox","ap_pixel","pos_frac_tau","iou_predbox_tau","bbox_cover_tau","top1"]).reset_index()
    def pc(metric, a, b, label):
        d = (w[(metric,a)] - w[(metric,b)]).values; ok = ~np.isnan(d); m_,lo,hi = boot_mean(d[ok], w["group"].values[ok])
        con.append(dict(family=fam, contrast=label, metric=metric, n=len(d), mean=m_, ci_lo=lo, ci_hi=hi))
    pc("ap_bbox","forged_repaired","forged","repaired − forged (ΔAP box GT)")
    pc("ap_pixel","forged_repaired","forged","repaired − forged (ΔAP pixel GT)")
    pc("pos_frac_tau","authentic_plus_marker","authentic","authentic+marker − authentic (Δ positive fraction @τ*)")
    pc("iou_predbox_tau","authentic_plus_marker","authentic","authentic+marker − authentic (Δ IoU(pred, box))")
    pc("bbox_cover_tau","authentic_plus_marker","authentic","authentic+marker − authentic (Δ box coverage)")
    pc("top1","authentic_plus_marker","authentic","authentic+marker − authentic (Δ top-1 % score)")
cont = pd.DataFrame(con); cont.to_csv(os.path.join(R,"green_marker_contrasts.csv"), index=False)
# figure
fig, axes = plt.subplots(1,3, figsize=(13,3.8))
conds = ["forged","forged_repaired","authentic","authentic_plus_marker"]; fams=["bbox","pixel","glyph"]; cols={"bbox":"#1f77b4","pixel":"#ff7f0e","glyph":"#7b5aa6"}
for ax, metric, title in zip(axes, ["ap_bbox","bbox_cover_tau","top1"], ["Per-image AP vs box GT (residue images)","Box coverage of prediction @τ*","Top-1 % score (image level)"]):
    for i,fam in enumerate(fams):
        s = summ[(summ.family==fam)&(summ.has_residue==1)].set_index("cond").reindex(conds)
        x = np.arange(len(conds))+ (i-1)*0.26
        ax.bar(x, s[metric], 0.26, yerr=[s[metric]-s[metric+"_lo"], s[metric+"_hi"]-s[metric]], color=cols[fam], label=f"{fam}-trained", capsize=2)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(["forged","forged\nmarker repaired","authentic","authentic\n+ marker"], fontsize=8); ax.set_title(title, fontsize=10, loc="left"); ax.spines[["top","right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=8); fig.suptitle("Green marker probe — validation pairs with residue (Gemini), 3-seed means, group-bootstrap 95 % CI", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(F,"fig_green_marker.png"), dpi=200); print(summ.round(3).to_string()); print(cont.round(3).to_string())
