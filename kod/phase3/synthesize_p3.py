"""
synthesize_p3.py — Faz 3 sentezi (TR): tablolar (MD + LaTeX), grup-bootstrap CI'lı kontrastlar, şekiller, numbers.json.
Kaynaklar (yalnız okunur): phase2/experiments/results/{summary_*,per_image_*,robustness.csv,per_image_controls.csv,tau.json},
phase3/results/{probes_p3_summary_*.csv,per_image_probes_p3_*.csv,marker_residue.csv,speed_bench.csv,phase1_exact_auprc.csv},
phase3_yolo/results/{detection_val.csv,detection_test.csv,checkpoint_summary_yolo.csv}, phase2/literature_baselines/zero_shot_results.csv.
Eksik kaynaklar atlanır (kısmi çalıştırma güvenli). Bootstrap: kaynak-belge grupları üzerinde B=1000 (Faz 2 ile aynı).
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
ROOT = _ROOT
R2 = ROOT / "phase2" / "experiments" / "results"; R3 = ROOT / "phase3" / "results"; RY = ROOT / "phase3_yolo" / "results"
OUT = R3; TEX = R3 / "tex"; FIG = ROOT / "phase3" / "figures"
for d in (OUT, TEX, FIG): d.mkdir(parents=True, exist_ok=True)
B, SEED = 1000, 20261109
GTS = ["bbox", "pixel_eps8_morph", "glyph_v2"]; GS = {"bbox": "box", "pixel_eps8_morph": "pix", "glyph_v2": "glyph"}
NUM: dict = {}
MD: list = ["# Faz 3 — sentez tabloları (otomatik; synthesize_p3.py)\n"]


def rd(p, **kw):
    p = Path(p)
    return pd.read_csv(p, dtype={"new_id": str}, low_memory=False, **kw) if p.exists() else None


def parse(m: str) -> dict:
    """model adı → aile, eğitim GT, seed, işaretler."""
    if m.startswith("aiforge-"):
        mm = re.match(r"aiforge-(?P<data>[a-z_]+)-(?P<arch>yolo11[nsm])-img(?P<img>\d+)-(?P<aug>[a-z]+)-s(?P<seed>\d+)", m)
        d = mm.groupdict() if mm else {}
        fam = "YOLO11" + d.get("arch", "")[-1:] if d else "YOLO"
        var = {"tiles": "", "full": "full1280", "tiles_authneg": "authneg", "tiles_geminionly": "geminionly"}.get(d.get("data"), d.get("data"))
        if d.get("aug") == "default": var = (var + "+defaug").strip("+")
        return dict(model=m, family=fam, train_gt="bbox", seed=int(d.get("seed", 0)), variant=var or "main", is_yolo=True,
                    gemini_only=d.get("data") == "tiles_geminionly")
    arch = m.split("_")[0]
    seed = int(re.search(r"_s(\d+)", m).group(1))
    gt = next((g for g in ["pixel_eps8_morph", "glyph_v2", "bbox"] if g in m), "?")
    var = []
    if "_mr" in m: var.append("marker-repaired")
    if "_nm" in m: var.append("nm-GT")
    if m.endswith("_e20"): var.append("20ep")
    if "geminionly" in m: var.append("geminionly")
    fam = {"unet": "U-Net", "dlv3p": "DeepLabV3+", "unetjoint": "U-Net+box-aux"}[arch]
    return dict(model=m, family=fam, train_gt=gt, seed=seed, variant="+".join(var) or "main", is_yolo=False,
                gemini_only="geminionly" in m)


def addcols(df):
    if df is None or not len(df): return df
    P = pd.DataFrame([parse(m) for m in df.model.unique()])
    df = df.drop(columns=[c for c in P.columns if c != "model" and c in df.columns])
    return df.merge(P, on="model", how="left")


def gboot_mean(vals: pd.Series, groups: pd.Series, B=B, seed=SEED):
    v = vals.to_numpy(float); g = groups.astype(str).to_numpy(); ok = ~np.isnan(v); v, g = v[ok], g[ok]
    if len(v) == 0: return np.nan, np.nan, np.nan
    ug, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=v, minlength=len(ug)); c = np.bincount(inv, minlength=len(ug)).astype(float)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(ug), size=(B, len(ug)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(m, s=None, nd=3):
    if m is None or (isinstance(m, float) and np.isnan(m)): return "—"
    return f"{m:.{nd}f}" + ("" if s is None or np.isnan(s) else f" ± {s:.{nd}f}")


def fci(t, nd=3, sign=False):
    m, lo, hi = t
    if np.isnan(m): return "—"
    f = f"{{:{'+' if sign else ''}.{nd}f}}"
    return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]"


def md_table(df, cols=None):
    cols = cols or list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()])


# ============================================================ 1. dual-GT (val) — tüm aileler
sv = addcols(rd(R2 / "summary_val.csv"))
if sv is not None:
    a = sv[(sv.subset == "all") & (sv.condition == "clean") & (~sv.gemini_only)]
    g = a.groupby(["family", "variant", "train_gt", "gt"]).agg(n=("seed", "nunique"), auprc=("auprc_pooled", "mean"), auprc_sd=("auprc_pooled", "std"),
                                                             f1t=("f1_tau", "mean"), f1t_sd=("f1_tau", "std"), ap=("ap_img", "mean")).reset_index()
    rows = []
    for (fam, var, tg), d in g.groupby(["family", "variant", "train_gt"]):
        r = dict(family=fam, variant=var, train_gt=GS.get(tg, tg), seeds=int(d.n.max()))
        for _, x in d.iterrows():
            r[f"AUPRC_{GS[x['gt']]}"] = fmt(x.auprc, x.auprc_sd if x.n > 1 else None)
            r[f"F1tau_{GS[x['gt']]}"] = fmt(x.f1t, x.f1t_sd if x.n > 1 else None)
            NUM[f"val_{fam}_{var}_{GS.get(tg, tg)}_on_{GS[x['gt']]}_auprc"] = round(float(x.auprc), 4)
            NUM[f"val_{fam}_{var}_{GS.get(tg, tg)}_on_{GS[x['gt']]}_f1tau"] = round(float(x.f1t), 4)
        rows.append(r)
    T1 = pd.DataFrame(rows); T1.to_csv(OUT / "T1_dual_gt_val.csv", index=False)
    MD.append("## T1. Dual-GT (val 487, temiz) — tüm aileler\n\n" + md_table(T1))
    # sıralama: her eval GT'de en iyi model (yalnız 'main' varyantları)
    mainv = g[g.variant == "main"]
    for gt_ in GTS:
        d = mainv[mainv['gt'] == gt_].sort_values("auprc", ascending=False)
        if len(d): NUM[f"best_on_{GS[gt_]}"] = f"{d.iloc[0].family}/{GS.get(d.iloc[0].train_gt)} {d.iloc[0].auprc:.3f}"

# ============================================================ 2. resmî test
st = addcols(rd(R2 / "summary_test.csv"))
det_t = rd(RY / "detection_test.csv"); det_v = rd(RY / "detection_val.csv")
if st is not None:
    a = st[(st.condition == "clean") & (~st.gemini_only)]
    rows = []
    for (fam, var, tg), d in a.groupby(["family", "variant", "train_gt"]):
        def m(sub, gt_, col):
            x = d[(d.subset == sub) & (d['gt'] == gt_)][col]
            return (x.mean(), x.std() if len(x) > 1 else np.nan)
        r = dict(family=fam, variant=var, train_gt=GS.get(tg, tg), seeds=d.seed.nunique(),
                 AUPRC_box_812=fmt(*m("all", "bbox", "auprc_pooled")), AUPRC_box_740=fmt(*m("clean_test", "bbox", "auprc_pooled")),
                 F1tau_box=fmt(*m("all", "bbox", "f1_tau")), IoUtau_box=fmt(*m("all", "bbox", "iou_tau")),
                 AUPRC_pix_89=fmt(*m("paired_89", "pixel_eps8_morph", "auprc_pooled")))
        if det_t is not None:
            dd = det_t[(det_t.model.isin(d.model.unique())) & (det_t.group == "all")]
            if len(dd):
                r["AP50"] = fmt(dd.AP50.mean(), dd.AP50.std() if len(dd) > 1 else None)
                r["AP50_95"] = fmt(dd.AP50_95.mean(), dd.AP50_95.std() if len(dd) > 1 else None)
                NUM[f"test_{fam}_{var}_{GS.get(tg, tg)}_AP50"] = round(float(dd.AP50.mean()), 4)
        for k, gt_, sub in [("auprc_box", "bbox", "all"), ("auprc_pix89", "pixel_eps8_morph", "paired_89")]:
            NUM[f"test_{fam}_{var}_{GS.get(tg, tg)}_{k}"] = round(float(m(sub, gt_, "auprc_pooled")[0]), 4)
        NUM[f"test_{fam}_{var}_{GS.get(tg, tg)}_f1tau_box"] = round(float(m("all", "bbox", "f1_tau")[0]), 4)
        rows.append(r)
    zs = rd(ROOT / "phase2" / "literature_baselines" / "zero_shot_results.csv")
    if zs is not None:
        for c in ["trufor", "adcd"]:
            z = zs[(zs.cond.str.contains(c)) & (zs.split == "test") & (zs.subset == "all")]
            if len(z):
                rows.append(dict(family=f"{c} (zero-shot)", variant="ref", train_gt="—", seeds="—",
                                 AUPRC_box_812=fmt(z.pixel_auprc_pooled.iloc[0]), F1tau_box=f"F1@.5 {z.pixel_f1_05.iloc[0]:.3f}", IoUtau_box=f"IoU@.5 {z.iou_05.iloc[0]:.3f}"))
    T2 = pd.DataFrame(rows).fillna("—"); T2.to_csv(OUT / "T2_test.csv", index=False)
    MD.append("\n## T2. Resmî test (box GT; piksel GT 89 çift) + tespit AP\n\n" + md_table(T2))

# ============================================================ 3. işaretçi sondası + işaretçisiz yeniden eğitim
pr = pd.concat([x for x in [rd(R3 / f"per_image_probes_p3_{t}.csv") for t in ("unet", "unet_s4344", "unet_mrj_s4344", "unet_q5a", "unet_q5b", "unet_q5c", "unet_q5d", "yolo", "yolo_s4344")] if x is not None], ignore_index=True) \
    if any((R3 / f"per_image_probes_p3_{t}.csv").exists() for t in ("unet", "unet_s4344", "unet_mrj_s4344", "unet_q5a", "unet_q5b", "unet_q5c", "unet_q5d", "yolo", "yolo_s4344")) else None
if pr is not None:
    pr = addcols(pr.drop(columns=[c for c in ("family",) if c in pr]))
    pr["fam_key"] = pr.family + "|" + pr.variant + "|" + pr.train_gt
    conds = ["forged", "forged_repaired", "authentic_plus_marker", "authentic", "render_only", "content_only", "clahe", "gray"]
    rows = []
    for fk, d in pr.groupby("fam_key"):
        # seed ortalaması (görüntü düzeyinde), sonra grup bootstrap
        pv = d.groupby(["cond", "new_id", "group", "has_residue"])[["ap_bbox", "ap_pixel", "bbox_cover_tau", "mean_in_bbox", "score_max", "top1"]].mean().reset_index()
        res = pv[pv.has_residue == 1]
        r = dict(model=fk, seeds=d.seed.nunique(), n_res=res.new_id.nunique())
        for c in ["forged", "forged_repaired", "authentic_plus_marker", "authentic"]:
            x = res[res.cond == c]; r[f"APbox_{c}"] = fci(gboot_mean(x.ap_bbox, x.group))
            r[f"cover_{c}"] = f"{x.bbox_cover_tau.mean():.3f}" if len(x) else "—"
        f_ = res[res.cond == "forged"].set_index("new_id"); rp = res[res.cond == "forged_repaired"].set_index("new_id")
        cm = f_.index.intersection(rp.index)
        if len(cm):
            t = gboot_mean(rp.loc[cm, "ap_bbox"] - f_.loc[cm, "ap_bbox"], f_.loc[cm, "group"]); r["dAP_repair"] = fci(t, sign=True)
            NUM[f"marker_dAPrepair_{fk}"] = [round(v, 4) for v in t]
        # görüntü düzeyi AUROC (max skor) — koşul vs authentic, tüm çiftler
        from sklearn.metrics import roc_auc_score
        au = pv[pv.cond == "authentic"].set_index("new_id").score_max
        for c in ["forged", "forged_repaired", "authentic_plus_marker", "render_only", "content_only"]:
            x = pv[pv.cond == c].set_index("new_id").score_max; cmn = x.index.intersection(au.index)
            if len(cmn) > 10:
                v = roc_auc_score(np.r_[np.ones(len(cmn)), np.zeros(len(cmn))], np.r_[x[cmn], au[cmn]])
                r[f"AUROC_{c}"] = f"{v:.3f}"; NUM[f"auroc_{c}_{fk}"] = round(float(v), 4)
        for c in ["render_only", "content_only", "forged"]:
            x = pv[pv.cond == c]
            r[f"inbox_{c}"] = f"{x.mean_in_bbox.mean():.3f}"; r[f"APbox_all_{c}"] = fci(gboot_mean(x.ap_bbox, x.group))
            NUM[f"apbox_{c}_{fk}"] = round(float(x.ap_bbox.mean()), 4)
        for c in ["clahe", "gray"]:
            x = pv[pv.cond == c].set_index("new_id"); fo = pv[pv.cond == "forged"].set_index("new_id"); cmn = x.index.intersection(fo.index)
            if len(cmn): r[f"dAPbox_{c}"] = fci(gboot_mean(x.loc[cmn, "ap_bbox"] - fo.loc[cmn, "ap_bbox"], fo.loc[cmn, "group"]), sign=True)
        rows.append(r)
    T3 = pd.DataFrame(rows); T3.to_csv(OUT / "T3_probes.csv", index=False)
    MD.append("\n## T3. Sondalar (val çiftleri; işaretçili alt küme için AP; seed-ortalaması + grup bootstrap CI)\n\n" + md_table(T3))

# işaretçili / işaretçisiz test alt kümeleri (model girdisi DEĞİŞMEDİ; yalnız alt küme etiketi)
pt = addcols(rd(R2 / "per_image_test.csv")); mr = rd(R3 / "marker_residue.csv")
if pt is not None and mr is not None:
    mt = mr[mr.split == "test"][["new_id", "ring_green_px"]]
    x = pt[(pt['gt'] == "bbox") & (~pt.gemini_only)].merge(mt, on="new_id", how="left")
    x["marker"] = np.where(x.ring_green_px.fillna(0) > 0, "with_marker", "no_marker")
    rows = []
    for (fam, var, tg), d in x.groupby(["family", "variant", "train_gt"]):
        pv = d.groupby(["new_id", "group", "marker", "tool"])[["ap_img", "f1_tau"]].mean().reset_index()
        r = dict(family=fam, variant=var, train_gt=GS.get(tg, tg), seeds=d.seed.nunique())
        for mk in ["with_marker", "no_marker"]:
            y = pv[pv.marker == mk]; r[f"AP_{mk}"] = fci(gboot_mean(y.ap_img, y.group)); r[f"n_{mk}"] = len(y)
            NUM[f"test_ap_{mk}_{fam}_{var}_{GS.get(tg, tg)}"] = round(float(y.ap_img.mean()), 4)
        y = pv[(pv.marker == "no_marker") & (pv.tool == "gemini")]; r["AP_no_marker_gemini"] = fci(gboot_mean(y.ap_img, y.group)); r["n_nm_gem"] = len(y)
        rows.append(r)
    T3b = pd.DataFrame(rows); T3b.to_csv(OUT / "T3b_test_marker_subsets.csv", index=False)
    MD.append("\n## T3b. Test: işaretçili vs işaretçisiz görüntüler (box GT, görüntü-AP)\n\n" + md_table(T3b))

# ============================================================ 4. held-out üretici (Gemini-only, 3 seed)
pv_ = rd(R2 / "per_image_val.csv"); pt_ = rd(R2 / "per_image_test.csv")
if pv_ is not None and pt_ is not None:
    pv_["split"] = "val"; pt_["split"] = "test"
    pi = addcols(pd.concat([pv_, pt_], ignore_index=True)); pi = pi[(pi.condition == "clean") & (pi.n_pos > 0)]
    pi["uid"] = pi.split + ":" + pi.new_id
    rows = []
    for (fam, var, tg, go), d in pi.groupby(["family", "variant", "train_gt", "gemini_only"]):
        # tüm üreticilerle eğitilen karşılaştırma satırları: ana modeller ve (11.11.2026) işaretçisiz eğitilen U-Net'ler
        if not go and not ((var == "main" and fam in ("U-Net", "YOLO11s")) or (fam == "U-Net" and var in ("marker-repaired", "marker-repaired+nm-GT"))): continue
        for gt_ in (["bbox", "pixel_eps8_morph"] if not d.is_yolo.iloc[0] else ["bbox", "pixel_eps8_morph"]):
            e = d[d['gt'] == gt_]
            if not len(e): continue
            per = e.groupby(["uid", "group", "tool", "source"]).ap_img.mean().reset_index()
            ide = per[per.tool == "ideogram"]; gem = per[(per.tool == "gemini") & per.source.isin(["cord", "sroie", "xfund"])]
            ti, tgm = gboot_mean(ide.ap_img, ide.group), gboot_mean(gem.ap_img, gem.group)
            rows.append(dict(family=fam, variant=var, train_gt=GS.get(tg, tg), gemini_only=go, seeds=e.seed.nunique(), eval_gt=GS[gt_],
                             n_ideogram=len(ide), AP_ideogram=fci(ti), AP_gemini_matched=fci(tgm), gap=f"{tgm[0] - ti[0]:.3f}"))
            NUM[f"heldout_{fam}_{var}_{GS.get(tg, tg)}_go{int(go)}_{GS[gt_]}"] = dict(ideogram=round(ti[0], 4), gemini=round(tgm[0], 4))
    T4 = pd.DataFrame(rows); T4.to_csv(OUT / "T4_heldout.csv", index=False)
    MD.append("\n## T4. Held-out üretici (val+test; Ideogram vs kaynak-eşlenmiş Gemini; görüntü-AP)\n\n" + md_table(T4))

# ============================================================ 5. YOLO ablasyonları + maliyet
ck = rd(RY / "checkpoint_summary_yolo.csv"); sp = rd(R3 / "speed_bench.csv")
if ck is not None:
    ck = ck[~ck.dry_run.astype(str).str.lower().eq("true")].drop_duplicates("run", keep="last")
    rows = []
    for _, c in ck.iterrows():
        m = c.run; r = dict(run=m, epochs=c.epochs, best_epoch=c.best_epoch, train_min=c.time_min, params_M=c.params_M, val_tile_mAP50=c.best_val_tile_map50)
        if det_v is not None:
            for nm, dd in (("val", det_v), ("test", det_t)):
                if dd is None: continue
                z = dd[(dd.model == m) & (dd.group == "all")]
                if len(z): r[f"AP50_{nm}"] = f"{z.AP50.iloc[0]:.3f}"; r[f"AP5095_{nm}"] = f"{z.AP50_95.iloc[0]:.3f}"
                z = dd[(dd.model == m) & (dd.group == "area_<1%")]
                if len(z): r[f"AP50_{nm}_lt1pct"] = f"{z.AP50.iloc[0]:.3f}"
        if sv is not None:
            for gt_ in ["bbox", "pixel_eps8_morph"]:
                z = sv[(sv.model == m) & (sv['gt'] == gt_) & (sv.subset == "all") & (sv.condition == "clean")]
                if len(z): r[f"val_AUPRC_{GS[gt_]}"] = f"{z.auprc_pooled.iloc[0]:.3f}"
        if sp is not None and (sp.model == m).any(): r["ms_per_img"] = f"{sp[sp.model == m].ms_per_img.iloc[0]:.0f}"
        rows.append(r)
    T5 = pd.DataFrame(rows).fillna("—"); T5.to_csv(OUT / "T5_yolo_ablation.csv", index=False)
    MD.append("\n## T5. YOLO ablasyonları ve maliyet\n\n" + md_table(T5))
if sp is not None:
    MD.append("\n## T5b. Hız (GPU boş, 100 test görüntüsü, ms/görüntü)\n\n" + md_table(sp.round(2)))
    for _, x in sp.iterrows(): NUM[f"ms_{x.model}"] = round(float(x.ms_per_img), 1)
if det_v is not None:
    MD.append("\n## T5c. Tespit metriği (val, tüm aileler, alan/üretici kırılımı)\n\n" + md_table(det_v))

# ============================================================ 6. dayanıklılık (relΔ AUPRC, kendi GT'sinde)
rob = []
r_ = addcols(rd(R2 / "robustness.csv"))
if r_ is not None:
    for _, x in r_[r_['gt'] == r_.train_gt.map(lambda t: t.replace("_nm", ""))].iterrows():
        rob.append(dict(model=x.model, family=x.family, variant=x.variant, train_gt=GS.get(x.train_gt.replace("_nm", "")), condition=x.condition,
                        rel=(x.auprc_pooled - x.auprc_pooled_clean) / x.auprc_pooled_clean if x.auprc_pooled_clean else np.nan))
if sv is not None:
    for c in ["jpeg75", "rescale075", "blur1", "clahe", "gray"]:
        s = addcols(rd(R2 / f"summary_val_{c}.csv"))
        if s is None: continue
        for _, x in s[(s.subset == "all") & (s['gt'] == "bbox")].iterrows():
            cl = sv[(sv.model == x.model) & (sv['gt'] == "bbox") & (sv.subset == "all") & (sv.condition == "clean")].auprc_pooled
            if len(cl): rob.append(dict(model=x.model, family=x.family, variant=x.variant, train_gt="box", condition=c, rel=(x.auprc_pooled - cl.iloc[0]) / cl.iloc[0]))
if rob:
    RB = pd.DataFrame(rob); A = RB.groupby(["family", "variant", "train_gt", "condition"]).rel.agg(["mean", "std", "count"]).reset_index()
    A["relΔAUPRC"] = [f"{m * 100:+.1f}%" + ("" if n < 2 else f" ± {s * 100:.1f}") for m, s, n in zip(A["mean"], A["std"], A["count"])]
    T6 = A.pivot_table(index=["family", "variant", "train_gt"], columns="condition", values="relΔAUPRC", aggfunc="first").reset_index().fillna("—")
    T6.to_csv(OUT / "T6_robustness.csv", index=False); MD.append("\n## T6. Dayanıklılık (relΔ pooled AUPRC, kendi GT'sinde)\n\n" + md_table(T6))
    for _, x in A.iterrows(): NUM[f"rob_{x.family}_{x.variant}_{x.train_gt}_{x.condition}"] = round(float(x["mean"]), 4)

# ============================================================ 7. kontroller (görüntü düzeyi; U-Net ailesi)
pc = addcols(rd(R2 / "per_image_controls.csv"))
if pc is not None:
    from sklearn.metrics import roc_auc_score
    rows = []
    _pv = rd(R2 / "per_image_val.csv")
    _pv = _pv[(_pv["gt"] == "bbox") & (_pv.condition == "clean")] if _pv is not None else None
    for m, d in pc.groupby("model"):
        if _pv is None: break
        fo = _pv[_pv.model == m].set_index("new_id")
        for src in ["authentic", "benign_pasteback", "fullimage_resample"]:
            ng = d[d.source == src].set_index("new_id"); cm = fo.index.intersection(ng.index)
            if len(cm) < 10: continue
            y = np.r_[np.ones(len(cm)), np.zeros(len(cm))]
            rows.append(dict(model=m, family=d.family.iloc[0], variant=d.variant.iloc[0], train_gt=GS.get(d.train_gt.iloc[0], d.train_gt.iloc[0]), seed=d.seed.iloc[0], source=src,
                             auroc_top1=roc_auc_score(y, np.r_[fo.loc[cm, "score_top1pct"], ng.loc[cm, "score_top1pct"]]),
                             auroc_max=roc_auc_score(y, np.r_[fo.loc[cm, "score_max"], ng.loc[cm, "score_max"]]),
                             flag05=float((ng.loc[cm, "pos_frac_0p5"] > 0.001).mean()), fp_tau=float(ng.loc[cm, "pos_frac_tau"].mean())))
    C = pd.DataFrame(rows)
    if len(C):
        A = C.groupby(["family", "variant", "train_gt", "source"]).agg(seeds=("seed", "nunique"), auroc_top1=("auroc_top1", "mean"), auroc_top1_sd=("auroc_top1", "std"),
                                                                        auroc_max=("auroc_max", "mean"), flag05=("flag05", "mean"), fp_tau=("fp_tau", "mean")).reset_index().round(4)
        A.to_csv(OUT / "T7_controls.csv", index=False); MD.append("\n## T7. Kontroller (val; image-level AUROC forged vs kontrol; bayrak oranı @0.5 (>0.1 % piksel); FP@τ*)\n\n" + md_table(A))
        for _, x in A.iterrows(): NUM[f"ctrl_{x.family}_{x.variant}_{x.train_gt}_{x.source}_auroc_top1"] = float(x.auroc_top1); NUM[f"ctrl_{x.family}_{x.variant}_{x.train_gt}_{x.source}_flag05"] = float(x.flag05)

# ============================================================ 8. Faz 1 kesin AUPRC
p1 = rd(R3 / "phase1_exact_auprc.csv")
if p1 is not None:
    MD.append("\n## T8. Faz 1 AUPRC düzeltmesi (W6)\n\n" + md_table(p1))

# ============================================================ şekiller
def fig_dual():
    if sv is None: return
    a = sv[(sv.subset == "all") & (sv.condition == "clean") & (~sv.gemini_only)]
    keep = [("U-Net", "main"), ("DeepLabV3+", "main"), ("YOLO11s", "main")]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.1), dpi=300, sharey=True)
    cols = {"bbox": "#4c72b0", "pixel_eps8_morph": "#dd8452", "glyph_v2": "#55a868"}
    for ax, (fam, var) in zip(axs, keep):
        d = a[(a.family == fam) & (a.variant == var)]
        tgs = [t for t in GTS if t in d.train_gt.unique()]
        for j, eg in enumerate(GTS):
            ys = [d[(d.train_gt == t) & (d['gt'] == eg)].auprc_pooled for t in tgs]
            ax.bar(np.arange(len(tgs)) + (j - 1) * 0.26, [y.mean() for y in ys], 0.26, yerr=[y.std() if len(y) > 1 else 0 for y in ys],
                   color=cols[eg], label=f"scored on {GS[eg]}", capsize=1.5, error_kw=dict(lw=0.6))
        ax.set_xticks(np.arange(len(tgs))); ax.set_xticklabels([f"trained on\n{GS[t]}" for t in tgs], fontsize=6.5)
        ax.set_title(fam, fontsize=8); ax.tick_params(labelsize=6.5); ax.set_ylim(0, 1); ax.grid(axis="y", lw=0.3, alpha=0.5)
    axs[0].set_ylabel("pooled pixel AUPRC (val)", fontsize=7); axs[0].legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3); fig.savefig(FIG / "fig_dual_gt_families.png"); plt.close(fig)


def fig_marker():
    p = OUT / "T3_probes.csv"
    if pr is None: return
    sel = [k for k in pr.fam_key.unique() if k.split("|")[2] == "bbox" or "marker-repaired" in k]
    conds = ["forged", "forged_repaired", "authentic_plus_marker", "render_only", "content_only", "authentic"]
    lab = ["forged", "repaired", "auth.+marker", "render-only", "content-only", "authentic"]
    fig, ax = plt.subplots(figsize=(3.5, 2.2), dpi=300)
    w = 0.8 / max(len(sel), 1)
    for i, k in enumerate(sel):
        d = pr[pr.fam_key == k]; pv = d.groupby(["cond", "new_id", "group"]).ap_bbox.mean().reset_index()
        ms = [gboot_mean(pv[pv.cond == c].ap_bbox, pv[pv.cond == c].group) for c in conds]
        ax.bar(np.arange(len(conds)) + (i - (len(sel) - 1) / 2) * w, [m[0] for m in ms], w,
               yerr=np.array([[m[0] - m[1], m[2] - m[0]] for m in ms]).T, capsize=1, error_kw=dict(lw=0.5),
               label=k.replace("|main|bbox", " (box)").replace("|marker-repaired|bbox", " repaired (box)").replace("|", " "))
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(lab, fontsize=6, rotation=20); ax.set_ylabel("per-image AP (box GT)", fontsize=7)
    ax.tick_params(labelsize=6); ax.set_ylim(0, 1.05); ax.legend(fontsize=5, frameon=False, ncol=2, loc="upper right"); ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout(pad=0.3); fig.savefig(FIG / "fig_marker_render.png"); plt.close(fig)


def fig_pareto():
    if sp is None or det_t is None: return
    fig, ax = plt.subplots(figsize=(3.5, 2.1), dpi=300)
    for _, x in sp.iterrows():
        z = det_t[(det_t.model == x.model) & (det_t.group == "all")]
        if not len(z): continue
        pz = parse(x.model); lab = pz["family"] + ("" if pz["variant"] == "main" else f" {pz['variant']}")
        ax.scatter(x.ms_per_img, z.AP50.iloc[0], s=14, marker="s" if pz["is_yolo"] else "o")
        ax.annotate(lab, (x.ms_per_img, z.AP50.iloc[0]), fontsize=5, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("ms / image (RTX 5080, full image)", fontsize=7); ax.set_ylabel("box AP@0.5 (test)", fontsize=7)
    ax.tick_params(labelsize=6); ax.grid(lw=0.3, alpha=0.5); fig.tight_layout(pad=0.3); fig.savefig(FIG / "fig_cost_ap.png"); plt.close(fig)


for f in (fig_dual, fig_marker, fig_pareto):
    try: f()
    except Exception as e: print("figure fail", f.__name__, e)

(OUT / "tables_p3.md").write_text("\n".join(MD) + "\n", encoding="utf-8")
json.dump(NUM, open(OUT / "numbers.json", "w"), indent=1, default=float)
print("\n".join(MD)[:20000])
print("numbers:", len(NUM))
