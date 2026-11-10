"""appcore.batch — toplu test: batch_test, aggregate, batch_figure."""
from __future__ import annotations
from ._common import *  # noqa: F401,F403 — ortak içe aktarmalar
from .config import (  # noqa: F401
    DEVICE, ZERO_GPU
)
from .imaging import (  # noqa: F401
    derive_pixel_gt
)
from .dataset import (  # noqa: F401
    filter_index, load_dataset_sample, sample_exists
)
from .models import (  # noqa: F401
    friendly, gpu_infer
)
from .metrics import (  # noqa: F401
    default_tau, hist_metrics, image_score, image_threshold, postprocess, ref_lookup, resnet_decision
)
from .scenarios import (  # noqa: F401
    degrade, show_df
)
from .simulation import (  # noqa: F401
    _outdir
)

# ================================================================== TOPLU TEST
BATCH_MODES = ["Yalnız sahte görüntüler (lokalizasyon)", "Sahte + authentic eşi (dengeli tespit)"]


def batch_test(split, source, tool, area, marker, n, seed, mode, degradation, unet_sel, resnet_sel, max_mp):
    t0 = time.time()
    if not unet_sel and not resnet_sel:
        raise gr.Error("En az bir model seçin.")
    pair_only = mode == BATCH_MODES[1]
    df = filter_index(split, source, tool, area, pair_only, marker, max_mp)
    keys = list(df.index)
    random.Random(int(seed)).shuffle(keys)
    keys = [k for k in keys if sample_exists(k)][: int(n)] if len(keys) else []
    if not keys:
        raise gr.Error("Filtrelere uyan (ve bucket'ta mevcut) örnek yok. Filtreleri gevşetin.")
    rows, msg = [], []
    cols_ps = ["Örnek", "Kaynak", "Üretici", "Alan", "İşaretçi", "Girdi", "Etiket", "Model", "Görüntü skoru", "Karar", "Doğru",
               "F1 bbox-GT", "IoU bbox-GT", "AP bbox-GT", "F1 piksel-GT", "AP piksel-GT", "Uyg. F1@0.5", "Ref F1@0.5", "Uyg. AP", "Ref AP", "Pozitif piksel %"]
    for i, k in enumerate(keys, 1):
        try:
            forged, mask, auth, meta = load_dataset_sample(k)
        except Exception as e:
            msg.append(f"{k}: {e}"); continue
        inputs = [("sahte", degrade(forged, degradation), 1)]
        if pair_only and auth is not None:
            inputs.append(("authentic", degrade(auth, degradation), 0))
        gtp = derive_pixel_gt(forged, auth)[0] if auth is not None else None
        res = gpu_infer(list(unet_sel), list(resnet_sel), [x[1] for x in inputs], 112)
        clean = degradation.startswith("Yok")
        for (kind, img, label), rr in zip(inputs, res):
            base = {"Örnek": k, "Kaynak": meta.get("source"), "Üretici": meta.get("tool"), "Alan": meta.get("area_bin"),
                    "İşaretçi": "var" if meta.get("has_marker_index") else "yok", "Girdi": kind, "Etiket": label}
            for nme in resnet_sel:
                hm, info = rr[nme]
                smax, rthr, _ = resnet_decision(nme, info["scores"])
                rows.append({**base, "Model": friendly(nme), "Görüntü skoru": smax, "Karar": int(smax >= rthr), "Doğru": int((smax >= rthr) == label)})
            for nme in unet_sel:
                prob, _ = rr[nme]
                stem = nme.split("/", 1)[-1]
                sc = image_score(prob); it, _ = image_threshold(nme); dec = int(sc >= it)
                tb, tp = default_tau(nme, "bbox"), default_tau(nme, "pixel_eps8_morph")
                r = {**base, "Model": friendly(nme), "Görüntü skoru": sc, "Karar": dec, "Doğru": int(dec == label)}
                binm, _ = postprocess(prob, tb, 0, False)
                r["Pozitif piksel %"] = 100 * binm.mean()
                if label == 1 and mask is not None:
                    gb = mask > 127
                    m = hist_metrics(prob, gb, tb); m5 = hist_metrics(prob, gb, 0.5)
                    r.update({"F1 bbox-GT": m["F1"], "IoU bbox-GT": m["IoU"], "AP bbox-GT": m["AP"], "Uyg. F1@0.5": m5["F1"], "Uyg. AP": m["AP"]})
                    ref = ref_lookup(meta["split"], meta["new_id"], stem, "bbox") if clean else None
                    if ref is not None:
                        r.update({"Ref F1@0.5": float(ref["f1_0p5"]), "Ref AP": float(ref["ap_img"])})
                    if gtp is not None:
                        mp_ = hist_metrics(prob, gtp, tp)
                        r.update({"F1 piksel-GT": mp_["F1"], "AP piksel-GT": mp_["AP"]})
                rows.append(r)
        per = pd.DataFrame(rows, columns=cols_ps)
        agg = aggregate(per)
        prog = f"**{i}/{len(keys)}** örnek işlendi · {time.time() - t0:.0f} s · {'ZeroGPU' if ZERO_GPU else DEVICE}" + (f"  \n⚠ {len(msg)} örnek atlandı" if msg else "")
        yield prog, show_df(per), show_df(agg), None, None
    per = pd.DataFrame(rows, columns=cols_ps)
    agg = aggregate(per)
    fig = batch_figure(per)
    d = _outdir("ai515_batch_")
    per.to_csv(d / "toplu_test_ornek_bazinda.csv", index=False); agg.to_csv(d / "toplu_test_ozet.csv", index=False)
    cfg = dict(zaman=datetime.datetime.now().isoformat(timespec="seconds"), filtre=dict(split=split, source=source, tool=tool, area=area, marker=marker, max_mp=max_mp),
               n=len(keys), seed=int(seed), mod=mode, bozulma=degradation, unet=list(unet_sel), resnet=list(resnet_sel), cihaz=str(DEVICE), zero_gpu=ZERO_GPU,
               sure_s=round(time.time() - t0, 1), ornekler=keys, ozet=json.loads(agg.to_json(orient="records", force_ascii=False)))
    (d / "toplu_test.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    fig.savefig(d / "toplu_test_grafik.png", dpi=120)
    prog = (f"### Tamamlandı — {len(keys)} örnek, {time.time() - t0:.0f} s ({'ZeroGPU' if ZERO_GPU else DEVICE})\n"
            "Referans (rapor, resmî test 812 görüntü, 3 tohum): bbox-GT modeli F1@τ* 0,846 · AUPRC 0,950; aynı tahminler piksel-GT ile F1 ≈ 0,45. "
            "Küçük örneklemde değerler örnekleme hatası içerir; 'Ref' sütunları aynı görüntülerde boru hattının kayıtlı sonuçlarıdır.")
    yield prog, show_df(per), show_df(agg), fig, [str(d / f) for f in ("toplu_test_ozet.csv", "toplu_test_ornek_bazinda.csv", "toplu_test.json", "toplu_test_grafik.png")]


def aggregate(per: pd.DataFrame) -> pd.DataFrame:
    out = []
    from sklearn.metrics import roc_auc_score
    for m, d in per.groupby("Model", sort=False):
        f = d[d["Etiket"] == 1]; a = d[d["Etiket"] == 0]
        r = {"Model": m, "n sahte": len(f), "n authentic": len(a)}
        for c in ("F1 bbox-GT", "IoU bbox-GT", "AP bbox-GT", "F1 piksel-GT", "AP piksel-GT"):
            v = pd.to_numeric(f[c], errors="coerce").dropna()
            r[c] = round(float(v.mean()), 4) if len(v) else None
        r["Tespit TPR"] = round(float(f["Karar"].mean()), 3) if len(f) else None
        r["Yanlış alarm FPR"] = round(float(a["Karar"].mean()), 3) if len(a) else None
        r["Doğruluk"] = round(float(d["Doğru"].mean()), 3)
        try:
            r["Görüntü AUROC"] = round(float(roc_auc_score(d["Etiket"], d["Görüntü skoru"])), 3) if (len(f) and len(a)) else None
        except Exception:
            r["Görüntü AUROC"] = None
        both = f.dropna(subset=["Ref F1@0.5", "Uyg. F1@0.5"]) if "Ref F1@0.5" in f else f.iloc[0:0]
        r["Yeniden üretim n"] = len(both)
        r["|ΔF1@0.5| ort."] = round(float((both["Uyg. F1@0.5"] - both["Ref F1@0.5"]).abs().mean()), 4) if len(both) else None
        r["|ΔAP| ort."] = round(float((both["Uyg. AP"] - both["Ref AP"]).abs().mean()), 4) if len(both) else None
        out.append(r)
    return pd.DataFrame(out)


def batch_figure(per):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), dpi=100)
    f = per[per["Etiket"] == 1]
    models = list(dict.fromkeys(per["Model"]))
    x = np.arange(len(models)); w = 0.38
    vb = [pd.to_numeric(f[f["Model"] == m]["F1 bbox-GT"], errors="coerce").mean() for m in models]
    vp = [pd.to_numeric(f[f["Model"] == m]["F1 piksel-GT"], errors="coerce").mean() for m in models]
    ax[0].bar(x - w / 2, np.nan_to_num(vb), w, label="bbox-GT", color="#4C72B0")
    ax[0].bar(x + w / 2, np.nan_to_num(vp), w, label="piksel-değişim GT", color="#DD8452")
    ax[0].set_xticks(x); ax[0].set_xticklabels([m.replace(" · ", "\n", 1) for m in models], fontsize=7)
    ax[0].set_ylim(0, 1); ax[0].set_ylabel("ortalama F1@τ"); ax[0].set_title("Lokalizasyon: aynı tahmin, iki GT", fontsize=9); ax[0].legend(fontsize=7)
    for i, m in enumerate(models):
        d = per[per["Model"] == m]
        for lab, col in ((1, "#C44E52"), (0, "#55A868")):
            s = d[d["Etiket"] == lab]["Görüntü skoru"]
            ax[1].scatter(np.full(len(s), i) + (0.12 if lab else -0.12), s, s=12, color=col, alpha=0.7, label=("sahte" if lab else "authentic") if i == 0 else None)
    ax[1].set_xticks(range(len(models))); ax[1].set_xticklabels([m.replace(" · ", "\n", 1) for m in models], fontsize=7)
    ax[1].set_ylabel("görüntü skoru (top-%1)"); ax[1].set_title("Görüntü-düzeyi skorlar", fontsize=9); ax[1].legend(fontsize=7)
    fig.tight_layout()
    return fig
