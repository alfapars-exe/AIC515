#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_gt.py — AIForge-Doc v1 için piksel-değişim ground-truth (GT) varyantları.

Kullanım (örnek):
  python make_gt.py --root "C:\\...\\Vize Proje" --splits training testing --workers 8
  python make_gt.py --root "C:\\...\\Vize Proje" --summarize-only      # yalnız istatistik/rapor

Girdi:
  <root>/data/AIForge-Doc-v1/{TrainingSet,TestingSet}/{images,masks,authentic}
  <root>/dataset_audit/samples_forged.csv  (pair_accepted==True çiftler kullanılır)
Çıktı:
  <root>/phase2/gt/<variant>/<new_id>.png        (training; 0/255, mod 'L', görüntüyle aynı boyut)
  <root>/phase2/gt/<variant>/test/<new_id>.png   (testing — TestingSet id'leri TrainingSet id'leriyle ÇAKIŞIR,
                                                   bu yüzden ayrı alt klasör; gt_index.csv'de gt_relpath sütunu)
  <root>/phase2/gt/gt_index.csv, gt_hist.npz, gt_stats.json, gt_report.md, fig_*.png

D = max_kanal |forged - authentic| (uint8). Varyantlar:
  pixel_eps4       : D > 4
  pixel_eps8       : D > 8
  pixel_eps16      : D > 16
  pixel_eps8_morph : (D > 8) -> morfolojik açma 3x3 -> <8 px bağlı bileşenleri at -> kapama 3x3
  glyph            : [Sauvola(forged gri; w=25, k=0.2) ∪ Sauvola(authentic gri)] ∩ dilate(D>8, 3 px)
                     (metin = gri < eşik; bbox içi medyan gri < 128 ise polarite ters çevrilir ve bayraklanır)
  glyph_v2         : glyph ile aynı, ancak polarite 'metin = azınlık sınıfı' kuralıyla seçilir:
                     bbox içinde Sauvola-koyu piksel oranı > 0.5 ise ters çevrilir (loş fotoğraflarda
                     medyan-gri kuralı yanlış tetiklenir; bkz. gt_report.md)
Ek sütunlar: <variant>_n_out_far = bbox'ın 3 px genişletilmişinin DIŞINDA kalan pozitifler (halo hariç).
bbox her zaman MASKE DOSYASINDAN alınır (metadata bbox'ı kullanılmaz).
"""
import argparse, os, sys, json, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from skimage.filters import threshold_sauvola

VARIANTS = ["pixel_eps4", "pixel_eps8", "pixel_eps16", "pixel_eps8_morph", "glyph", "glyph_v2", "bbox_and_eps8"]
K3 = np.ones((3, 3), np.uint8)


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def write_mask_png(path, mask_bool):
    ok, buf = cv2.imencode(".png", (mask_bool.astype(np.uint8) * 255), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("PNG encode failed: " + path)
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())


def cc_stats(mask_bool, bbox_bool):
    """Bağlı bileşen sayısı ve bbox'a hiç değmeyen bileşen sayısı (8-komşuluk)."""
    if not mask_bool.any():
        return 0, 0
    n, lab = cv2.connectedComponents(mask_bool.astype(np.uint8), connectivity=8)
    inside_counts = np.bincount(lab[bbox_bool].ravel(), minlength=n)
    n_cc = n - 1
    n_cc_out = int(((inside_counts == 0)[1:]).sum())
    return int(n_cc), n_cc_out


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else float("nan")


def process_pair(rec, data_dir, out_dir, write=True):
    sub = "TrainingSet" if rec["split"] == "training" else "TestingSet"
    nid = "%09d" % int(rec["new_id"])
    fp = os.path.join(data_dir, sub, "images", nid + ".png")
    ap = os.path.join(data_dir, sub, "authentic", rec["pair_name"])
    mp = os.path.join(data_dir, sub, "masks", nid + ".png")
    f = load_rgb(fp)
    a = load_rgb(ap)
    sub_out = "test" if rec["split"] == "testing" else ""
    out = {"new_id": nid, "split": rec["split"], "tool": rec["assigned_tool"], "source": rec["source_dataset"],
           "area_bin": rec["area_bin"], "pair_name": rec["pair_name"], "H": f.shape[0], "W": f.shape[1],
           "gt_relpath": os.path.join("<variant>", sub_out, nid + ".png") if sub_out else os.path.join("<variant>", nid + ".png")}
    if f.shape != a.shape:
        out["skipped"] = "size_mismatch(%s vs %s)" % (f.shape, a.shape)
        return out, None
    m = np.asarray(Image.open(mp).convert("L")) > 127
    if m.shape != f.shape[:2]:
        out["skipped"] = "mask_size_mismatch"
        return out, None
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        out["skipped"] = "empty_bbox_mask"
        return out, None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bbox = np.zeros_like(m)
    bbox[y1:y2 + 1, x1:x2 + 1] = True
    out.update({"bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                "bbox_px": int(bbox.sum()), "img_px": int(m.size),
                "mask_rect_frac": float(m.sum() / bbox.sum())})

    D = np.abs(f.astype(np.int16) - a.astype(np.int16)).max(axis=2).astype(np.uint8)
    hist_in = np.bincount(D[bbox], minlength=256)[:256]
    hist_out = np.bincount(D[~bbox], minlength=256)[:256]
    out["D_max"] = int(D.max())
    out["D_mean"] = float(D.mean())
    out["D_mean_outside"] = float(D[~bbox].mean())

    e4 = D > 4
    e8 = D > 8
    e16 = D > 16

    # (d) morfolojik varyant
    o = cv2.morphologyEx(e8.astype(np.uint8), cv2.MORPH_OPEN, K3)
    n, lab, st, _ = cv2.connectedComponentsWithStats(o, connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= 8
    keep[0] = False
    o = keep[lab].astype(np.uint8)
    morph = cv2.morphologyEx(o, cv2.MORPH_CLOSE, K3).astype(bool)

    # (e) glif varyantı
    gf = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    bbox_med = float(np.median(gf[bbox]))
    inverted = bbox_med < 128
    thf = threshold_sauvola(gf, window_size=25, k=0.2)
    tha = threshold_sauvola(ga, window_size=25, k=0.2)
    if inverted:
        tf, ta = gf > thf, ga > tha
    else:
        tf, ta = gf < thf, ga < tha
    dil = cv2.dilate(e8.astype(np.uint8), K3, iterations=3).astype(bool)  # 3 px yarıçap (7x7 kare)
    glyph = (tf | ta) & dil
    # glyph_v2: polarite = azınlık sınıfı (bbox içinde Sauvola-koyu oranı > 0.5 ise ters)
    dark_frac = float((gf[bbox] < thf[bbox]).mean())
    inverted_v2 = dark_frac > 0.5
    if inverted_v2:
        tf2, ta2 = gf > thf, ga > tha
    else:
        tf2, ta2 = gf < thf, ga < tha
    glyph_v2 = (tf2 | ta2) & dil
    bbox_d = cv2.dilate(bbox.astype(np.uint8), K3, iterations=3).astype(bool)  # 3 px halo
    out["bbox_median_gray"] = bbox_med
    out["polarity_inverted"] = bool(inverted)
    out["bbox_dark_frac"] = dark_frac
    out["polarity_inverted_v2"] = bool(inverted_v2)
    out["glyph_text_frac_forged"] = float(tf.mean())

    bbe8 = bbox & e8
    variants = {"pixel_eps4": e4, "pixel_eps8": e8, "pixel_eps16": e16,
                "pixel_eps8_morph": morph, "glyph": glyph, "glyph_v2": glyph_v2, "bbox_and_eps8": bbe8}

    for name in VARIANTS:
        v = variants[name]
        n_pos = int(v.sum())
        n_in = int(np.logical_and(v, bbox).sum())
        n_cc, n_cc_out = cc_stats(v, bbox)
        out[name + "_n_pos"] = n_pos
        out[name + "_n_in"] = n_in
        out[name + "_n_out"] = n_pos - n_in
        out[name + "_n_out_far"] = int(np.logical_and(v, ~bbox_d).sum())
        out[name + "_iou_bbox"] = iou(v, bbox)
        out[name + "_n_cc"] = n_cc
        out[name + "_n_cc_out"] = n_cc_out
        if write:
            write_mask_png(os.path.join(out_dir, name, sub_out, nid + ".png"), v)
    for i in range(len(VARIANTS)):
        for j in range(i + 1, len(VARIANTS)):
            out["iou_%s__%s" % (VARIANTS[i], VARIANTS[j])] = iou(variants[VARIANTS[i]], variants[VARIANTS[j]])
    return out, (hist_in, hist_out)


def q(x, p):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.percentile(x, p)) if len(x) else float("nan")


def med_iqr(x):
    return {"median": q(x, 50), "q25": q(x, 25), "q75": q(x, 75), "mean": float(np.nanmean(x)) if len(x) else float("nan"), "n": int(np.sum(~np.isnan(np.asarray(x, float))))}


def summarize(gt_dir, eps_max=32):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = pd.read_csv(os.path.join(gt_dir, "gt_index.csv"), dtype={"new_id": str})
    hz = np.load(os.path.join(gt_dir, "gt_hist.npz"), allow_pickle=True)
    h_ids = [str(s) for s in hz["new_id"]]
    H_in, H_out = hz["hist_in"].astype(np.int64), hz["hist_out"].astype(np.int64)
    ok = idx[idx["skipped"].isna()].copy() if "skipped" in idx.columns else idx.copy()
    stats = {"n_pairs_total": int(len(idx)), "n_pairs_ok": int(len(ok)),
             "n_skipped": int(len(idx) - len(ok)),
             "skipped": idx[~idx.index.isin(ok.index)][["new_id", "split", "skipped"]].to_dict("records") if len(idx) != len(ok) else [],
             "by_split": ok["split"].value_counts().to_dict(),
             "n_polarity_inverted": int(ok["polarity_inverted"].sum()),
             "polarity_inverted_ids": ok.loc[ok["polarity_inverted"], "new_id"].tolist(),
             "n_polarity_inverted_v2": int(ok["polarity_inverted_v2"].sum()),
             "polarity_inverted_v2_ids": ok.loc[ok["polarity_inverted_v2"], "new_id"].tolist(),
             "polarity_inverted_by_source": ok.loc[ok["polarity_inverted"]].groupby("source").size().to_dict(),
             "degenerate_pairs_Dmax_le4": ok.loc[ok["D_max"] <= 4, ["new_id", "split", "tool", "source", "D_max"]].to_dict("records"),
             "pairs_all_eps8_change_outside_bbox": ok.loc[(ok["bbox_and_eps8_n_pos"] == 0) & (ok["pixel_eps8_n_pos"] > 0), ["new_id", "split", "tool", "source", "D_max", "pixel_eps8_n_pos"]].to_dict("records"),
             "variants": {}, "by_tool": {}, "by_source": {}, "by_split_variant": {}}
    for v in VARIANTS:
        pos_frac_img = ok[v + "_n_pos"] / ok["img_px"]
        pos_frac_bbox = ok[v + "_n_in"] / ok["bbox_px"]
        out_share = ok[v + "_n_out"] / ok[v + "_n_pos"].replace(0, np.nan)
        out_share_far = ok[v + "_n_out_far"] / ok[v + "_n_pos"].replace(0, np.nan)
        stats["variants"][v] = {
            "outside_share_far(3px halo hariç)": med_iqr(out_share_far),
            "n_pairs_any_outside_far": int((ok[v + "_n_out_far"] > 0).sum()),
            "n_pairs_outside_far_share_gt_1pct": int((out_share_far > 0.01).sum()),
            "n_pos": med_iqr(ok[v + "_n_pos"]),
            "pos_frac_img": med_iqr(pos_frac_img),
            "bbox_coverage(n_in/bbox_px)": med_iqr(pos_frac_bbox),
            "iou_bbox": med_iqr(ok[v + "_iou_bbox"]),
            "outside_share(n_out/n_pos)": med_iqr(out_share),
            "n_pairs_any_outside": int((ok[v + "_n_out"] > 0).sum()),
            "n_pairs_outside_share_gt_1pct": int((out_share > 0.01).sum()),
            "n_pairs_outside_share_gt_10pct": int((out_share > 0.10).sum()),
            "n_cc": med_iqr(ok[v + "_n_cc"]),
            "n_cc_outside_bbox": med_iqr(ok[v + "_n_cc_out"]),
            "n_pairs_with_cc_outside_bbox": int((ok[v + "_n_cc_out"] > 0).sum()),
            "n_empty": int((ok[v + "_n_pos"] == 0).sum()),
            "n_bbox_coverage_lt_20pct": int((pos_frac_bbox < 0.20).sum()),
        }
        for key, col in [("by_tool", "tool"), ("by_source", "source"), ("by_split_variant", "split")]:
            stats[key].setdefault(v, {})
            for g, sub in ok.groupby(col):
                pf = sub[v + "_n_in"] / sub["bbox_px"]
                osh = sub[v + "_n_out"] / sub[v + "_n_pos"].replace(0, np.nan)
                stats[key][v][g] = {"n": int(len(sub)),
                                    "bbox_coverage_median": q(pf, 50),
                                    "iou_bbox_median": q(sub[v + "_iou_bbox"], 50),
                                    "outside_share_median": q(osh, 50),
                                    "frac_pairs_any_outside": float((sub[v + "_n_out"] > 0).mean()),
                                    "n_empty": int((sub[v + "_n_pos"] == 0).sum()),
                                    "pos_frac_img_median": q(sub[v + "_n_pos"] / sub["img_px"], 50)}
    # ikili IoU matrisi
    M = np.full((len(VARIANTS), len(VARIANTS)), np.nan)
    for i in range(len(VARIANTS)):
        M[i, i] = 1.0
        for j in range(i + 1, len(VARIANTS)):
            c = "iou_%s__%s" % (VARIANTS[i], VARIANTS[j])
            M[i, j] = M[j, i] = q(ok[c], 50)
    stats["pairwise_iou_median"] = {VARIANTS[i]: {VARIANTS[j]: (None if np.isnan(M[i, j]) else float(M[i, j])) for j in range(len(VARIANTS))} for i in range(len(VARIANTS))}

    # ε-duyarlılık eğrisi (histogramlardan, tüm çiftler)
    hid = {h: k for k, h in enumerate(h_ids)}
    sel = [hid[n] for n in ok["new_id"] if n in hid]
    Hi, Ho = H_in[sel], H_out[sel]
    img_px = (Hi.sum(1) + Ho.sum(1)).astype(float)
    out_px = Ho.sum(1).astype(float)
    in_px = Hi.sum(1).astype(float)
    cum_in = Hi[:, ::-1].cumsum(1)[:, ::-1]   # cum_in[:,e] = #D>=e
    cum_out = Ho[:, ::-1].cumsum(1)[:, ::-1]
    curve = []
    eps_list = list(range(0, eps_max + 1))
    for e in eps_list:
        gi = cum_in[:, e + 1] if e + 1 < 256 else np.zeros(len(sel))   # D > e
        go = cum_out[:, e + 1] if e + 1 < 256 else np.zeros(len(sel))
        tot = gi + go
        with np.errstate(invalid="ignore", divide="ignore"):
            out_share = np.where(tot > 0, go / tot, np.nan)
            out_frac = go / out_px
            cov = gi / in_px
            pf = tot / img_px
            iou_b = gi / (in_px + go)
        curve.append({"eps": e,
                      "pos_frac_img_median": q(pf, 50), "pos_frac_img_q25": q(pf, 25), "pos_frac_img_q75": q(pf, 75),
                      "bbox_coverage_median": q(cov, 50),
                      "outside_share_median": q(out_share, 50), "outside_share_mean": float(np.nanmean(out_share)),
                      "outside_frac_median": q(out_frac, 50), "outside_frac_q75": q(out_frac, 75),
                      "iou_bbox_median": q(iou_b, 50),
                      "frac_pairs_any_outside": float(np.mean(go > 0)),
                      "frac_pairs_outside_frac_gt_1pct": float(np.mean(out_frac > 0.01)),
                      "frac_pairs_outside_frac_gt_0p1pct": float(np.mean(out_frac > 0.001)),
                      "frac_pairs_outside_share_gt_10pct": float(np.nanmean(out_share > 0.10)),
                      "n": int(len(sel))})
    stats["eps_curve"] = curve
    # tool'a göre eps eğrisi (bbox-dışı pay)
    stats["eps_curve_by_tool"] = {}
    tools = ok["tool"].values
    for t in np.unique(tools):
        mk = tools == t
        rows = []
        for e in eps_list:
            gi = cum_in[mk, e + 1]; go = cum_out[mk, e + 1]
            tot = gi + go
            with np.errstate(invalid="ignore", divide="ignore"):
                osh = np.where(tot > 0, go / tot, np.nan); of = go / out_px[mk]
            rows.append({"eps": e, "outside_share_median": q(osh, 50), "outside_frac_median": q(of, 50),
                         "frac_pairs_any_outside": float(np.mean(go > 0)),
                         "frac_pairs_outside_frac_gt_1pct": float(np.mean(of > 0.01))})
        stats["eps_curve_by_tool"][t] = rows

    with open(os.path.join(gt_dir, "gt_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1, ensure_ascii=False, default=float)

    # ---- figürler ----
    ce = pd.DataFrame(curve)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(ce.eps, ce.pos_frac_img_median * 100, "o-", ms=3, label="medyan")
    ax.fill_between(ce.eps, ce.pos_frac_img_q25 * 100, ce.pos_frac_img_q75 * 100, alpha=0.25, label="IQR")
    ax.set_yscale("log"); ax.set_xlabel("ε"); ax.set_ylabel("pozitif piksel oranı (% görüntü)"); ax.set_title("D>ε pozitif oranı"); ax.legend()
    ax = axes[1]
    ax.plot(ce.eps, ce.outside_share_median, "o-", ms=3, label="bbox-dışı pay (medyan)")
    ax.plot(ce.eps, ce.outside_share_mean, "s--", ms=3, label="bbox-dışı pay (ortalama)")
    ax.plot(ce.eps, ce.frac_pairs_any_outside, "^-", ms=3, label="çift oranı: ≥1 bbox-dışı px")
    ax.plot(ce.eps, ce.frac_pairs_outside_frac_gt_1pct, "v-", ms=3, label="çift oranı: bbox-dışı px >%1")
    ax.set_xlabel("ε"); ax.set_ylabel("oran"); ax.set_title("bbox-dışı değişim vs ε"); ax.legend(fontsize=8)
    ax = axes[2]
    for t, rows in stats["eps_curve_by_tool"].items():
        r = pd.DataFrame(rows)
        ax.plot(r.eps, r.frac_pairs_any_outside, "o-", ms=3, label="%s: ≥1 bbox-dışı px" % t)
        ax.plot(r.eps, r.outside_share_median, "--", label="%s: bbox-dışı pay medyan" % t)
    ax.set_xlabel("ε"); ax.set_ylabel("oran"); ax.set_title("tool kırılımı"); ax.legend(fontsize=8)
    for a_ in axes:
        a_.axvline(4, color="gray", lw=0.6, ls=":"); a_.axvline(8, color="gray", lw=0.6, ls=":"); a_.axvline(16, color="gray", lw=0.6, ls=":")
    fig.tight_layout(); fig.savefig(os.path.join(gt_dir, "fig_eps_sensitivity.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(VARIANTS))); ax.set_yticks(range(len(VARIANTS)))
    ax.set_xticklabels(VARIANTS, rotation=45, ha="right"); ax.set_yticklabels(VARIANTS)
    for i in range(len(VARIANTS)):
        for j in range(len(VARIANTS)):
            ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", color="w" if M[i, j] < 0.6 else "k", fontsize=8)
    fig.colorbar(im, ax=ax, label="medyan IoU"); ax.set_title("Varyantlar arası ikili IoU (çift başına medyan)")
    fig.tight_layout(); fig.savefig(os.path.join(gt_dir, "fig_variant_iou_matrix.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    data = [ok[v + "_iou_bbox"].dropna().values for v in VARIANTS]
    axes[0].boxplot(data, showfliers=False); axes[0].set_xticks(range(1, len(VARIANTS) + 1)); axes[0].set_xticklabels(VARIANTS, rotation=30, ha="right"); axes[0].set_ylabel("IoU(variant, bbox)"); axes[0].set_title("IoU ile bbox GT")
    data = [(ok[v + "_n_in"] / ok["bbox_px"]).values for v in VARIANTS]
    axes[1].boxplot(data, showfliers=False); axes[1].set_xticks(range(1, len(VARIANTS) + 1)); axes[1].set_xticklabels(VARIANTS, rotation=30, ha="right"); axes[1].set_ylabel("bbox kapsama (n_in / bbox_px)"); axes[1].set_title("bbox içinde değişen pay")
    fig.tight_layout(); fig.savefig(os.path.join(gt_dir, "fig_variant_boxplots.png"), dpi=150); plt.close(fig)

    # ---- rapor ----
    L = []
    L.append("# Piksel-değişim GT varyantları — rapor (AIForge-Doc v1)\n")
    L.append("Üretim: `make_gt.py`. Çift sayısı: %d (işlenen %d, atlanan %d). Split: %s.\n" % (stats["n_pairs_total"], stats["n_pairs_ok"], stats["n_skipped"], stats["by_split"]))
    L.append("D = max_kanal |forged − authentic|; bbox = maske dosyasından (rect). Pozitif oranlar görüntü pikseline göre; 'bbox kapsama' = bbox içindeki pozitif / bbox alanı; 'bbox-dışı pay' = bbox dışındaki pozitif / tüm pozitif.\n")
    L.append("Glif varyantında (spec kuralı: bbox içi medyan gri < 128) polarite ters çevrilen çift sayısı: **%d** (kaynak kırılımı: %s)%s\n" % (stats["n_polarity_inverted"], stats["polarity_inverted_by_source"], (" — ilk id'ler: " + ", ".join(stats["polarity_inverted_ids"][:12])) if stats["n_polarity_inverted"] else ""))
    L.append("**Uyarı:** medyan-gri kuralı loş/gölgeli fotoğraflarda (gri zemin, koyu metin) yanlış tetiklenir ve `glyph` maskesi metnin TÜMLEYENİNİ (zemini) seçer. Bu nedenle `glyph_v2` eklendi: polarite = 'metin azınlık sınıfı' (bbox içinde Sauvola-koyu oranı > 0.5 ise ters). glyph_v2'de ters çevrilen çift sayısı: **%d**%s. Model eğitimi için `glyph_v2` önerilir; `glyph` spec uyumluluğu için saklanmıştır.\n" % (stats["n_polarity_inverted_v2"], (" — id: " + ", ".join(stats["polarity_inverted_v2_ids"][:20])) if stats["n_polarity_inverted_v2"] else ""))
    L.append("**Dejenere çiftler (D_max ≤ 4, forged ≈ authentic; sahtelik piksel düzeyinde görünmez):** %d — %s\n" % (len(stats["degenerate_pairs_Dmax_le4"]), ", ".join("%s(%s,%s,D=%d)" % (r["new_id"], r["split"][:5], r["source"], r["D_max"]) for r in stats["degenerate_pairs_Dmax_le4"])))
    L.append("**Tüm D>8 değişimi bbox DIŞINDA olan çiftler (bbox∩eps8 boş, eps8 dolu; anotasyon-değişim uyumsuzluğu):** %d — %s\n" % (len(stats["pairs_all_eps8_change_outside_bbox"]), ", ".join("%s(%s,%s,D=%d,n=%d)" % (r["new_id"], r["split"][:5], r["source"], r["D_max"], r["pixel_eps8_n_pos"]) for r in stats["pairs_all_eps8_change_outside_bbox"])))
    L.append("\n## 1. Varyant başına özet (tüm işlenen çiftler)\n")
    L.append("| varyant | pozitif px medyan [IQR] | pozitif oran (görüntü) medyan | bbox kapsama medyan [IQR] | IoU(bbox) medyan [IQR] | bbox-dışı pay medyan / ort. | çift: ≥1 bbox-dışı px | bbox-dışı pay (3 px halo hariç) ort. | çift: ≥1 bbox-dışı px (halo hariç) | çift: bbox-dışı bileşen | bileşen sayısı medyan | boş maske | kapsama <%20 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        s = stats["variants"][v]
        L.append("| %s | %d [%d–%d] | %.4f%% | %.3f [%.3f–%.3f] | %.3f [%.3f–%.3f] | %.3f / %.3f | %d (%.1f%%) | %.3f | %d (%.1f%%) | %d | %d [%d–%d] | %d | %d |" % (
            v, s["n_pos"]["median"], s["n_pos"]["q25"], s["n_pos"]["q75"], s["pos_frac_img"]["median"] * 100,
            s["bbox_coverage(n_in/bbox_px)"]["median"], s["bbox_coverage(n_in/bbox_px)"]["q25"], s["bbox_coverage(n_in/bbox_px)"]["q75"],
            s["iou_bbox"]["median"], s["iou_bbox"]["q25"], s["iou_bbox"]["q75"],
            s["outside_share(n_out/n_pos)"]["median"], s["outside_share(n_out/n_pos)"]["mean"],
            s["n_pairs_any_outside"], 100.0 * s["n_pairs_any_outside"] / max(1, stats["n_pairs_ok"]),
            s["outside_share_far(3px halo hariç)"]["mean"], s["n_pairs_any_outside_far"], 100.0 * s["n_pairs_any_outside_far"] / max(1, stats["n_pairs_ok"]),
            s["n_pairs_with_cc_outside_bbox"],
            s["n_cc"]["median"], s["n_cc"]["q25"], s["n_cc"]["q75"], s["n_empty"], s["n_bbox_coverage_lt_20pct"]))
    for key, title in [("by_tool", "tool"), ("by_source", "kaynak"), ("by_split_variant", "split")]:
        L.append("\n## Kırılım: %s\n" % title)
        L.append("| varyant | grup | n | bbox kapsama medyan | IoU(bbox) medyan | bbox-dışı pay medyan | çift oranı: ≥1 bbox-dışı px | boş |")
        L.append("|---|---|---|---|---|---|---|---|")
        for v in VARIANTS:
            for g, s in stats[key][v].items():
                L.append("| %s | %s | %d | %.3f | %.3f | %.3f | %.3f | %d |" % (v, g, s["n"], s["bbox_coverage_median"], s["iou_bbox_median"], s["outside_share_median"], s["frac_pairs_any_outside"], s["n_empty"]))
    L.append("\n## 2. ε-duyarlılık (D>ε; histogram tabanlı, TÜM işlenen çiftler, n=%d)\n" % curve[0]["n"])
    L.append("![eps](fig_eps_sensitivity.png)\n")
    L.append("| ε | pozitif oran (görüntü) medyan | bbox kapsama medyan | IoU(bbox) medyan | bbox-dışı pay medyan | bbox-dışı pay ort. | çift oranı ≥1 bbox-dışı px | çift oranı bbox-dışı px >%0.1 | çift oranı bbox-dışı px >%1 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in curve:
        if r["eps"] in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
            L.append("| %d | %.4f%% | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (r["eps"], r["pos_frac_img_median"] * 100, r["bbox_coverage_median"], r["iou_bbox_median"], r["outside_share_median"], r["outside_share_mean"], r["frac_pairs_any_outside"], r["frac_pairs_outside_frac_gt_0p1pct"], r["frac_pairs_outside_frac_gt_1pct"]))
    c0, c4, c8, c16 = curve[0], curve[4], curve[8], curve[16]
    L.append("\n**Yeniden-kodlama izi (ε=0'daki global ±1–3 fark) ε≥4'te kayboluyor mu?** ε=0'da çiftlerin %.1f%%'inde bbox-dışı piksellerin >%%1'i değişmiş (medyan bbox-dışı pay %.3f); ε=4'te bu oran %.1f%% (medyan pay %.3f), ε=8'de %.1f%% (%.3f), ε=16'da %.1f%% (%.3f). ≥1 bbox-dışı piksel içeren çift oranı: ε=0 %.3f → ε=4 %.3f → ε=8 %.3f → ε=16 %.3f.\n" % (
        100 * c0["frac_pairs_outside_frac_gt_1pct"], c0["outside_share_median"], 100 * c4["frac_pairs_outside_frac_gt_1pct"], c4["outside_share_median"],
        100 * c8["frac_pairs_outside_frac_gt_1pct"], c8["outside_share_median"], 100 * c16["frac_pairs_outside_frac_gt_1pct"], c16["outside_share_median"],
        c0["frac_pairs_any_outside"], c4["frac_pairs_any_outside"], c8["frac_pairs_any_outside"], c16["frac_pairs_any_outside"]))
    L.append("\nNot: ε=0–1'de IoU(bbox) ve bbox-dışı pay medyanları bimodal dağılım (çiftlerin ~%49'unda global yeniden-kodlama izi var, ~%51'inde hiç yok) nedeniyle kararsızdır; ortalama ve 'çift oranı' sütunları daha bilgilendiricidir.\n")
    L.append("\nTool kırılımı (çift oranı: ≥1 bbox-dışı px / bbox-dışı pay medyan):\n")
    L.append("| ε | " + " | ".join(stats["eps_curve_by_tool"].keys()) + " |")
    L.append("|---|" + "---|" * len(stats["eps_curve_by_tool"]))
    for e in (0, 2, 4, 8, 16, 32):
        L.append("| %d | " % e + " | ".join("%.3f / %.3f" % (rows[e]["frac_pairs_any_outside"], rows[e]["outside_share_median"]) for rows in stats["eps_curve_by_tool"].values()) + " |")
    L.append("\n## 3. Varyantlar arası ikili IoU (çift başına medyan)\n")
    L.append("![iou](fig_variant_iou_matrix.png)\n")
    L.append("| | " + " | ".join(VARIANTS) + " |")
    L.append("|---|" + "---|" * len(VARIANTS))
    for i, v in enumerate(VARIANTS):
        L.append("| %s | " % v + " | ".join("%.3f" % M[i, j] for j in range(len(VARIANTS))) + " |")
    L.append("\n![box](fig_variant_boxplots.png)\n")
    L.append("\n## 4. Notlar\n")
    L.append("- `glyph`: Sauvola (skimage.filters.threshold_sauvola, window_size=25, k=0.2) ile forged ve authentic gri görüntülerde metin pikselleri (gri < eşik) birleştirilip dilate(D>8, 3×3 kare, 3 iterasyon = 3 px yarıçap) ile kesiştirildi.")
    L.append("- `pixel_eps8_morph`: cv2 MORPH_OPEN 3×3 → 8-komşuluk bağlı bileşen alanı <8 px olanlar atıldı → MORPH_CLOSE 3×3.")
    L.append("- 'bbox-dışı bileşen' = varyantın bbox ile hiç kesişmeyen 8-komşuluk bağlı bileşeni. '3 px halo hariç' = bbox'ın 3 px genişletilmişinin dışında kalan pozitifler (glif varyantlarının bbox kenarına taşan dilate halosu bu sayımda yok sayılır).")
    L.append("- `glyph_v2`: `glyph` ile aynı işlem; polarite kuralı 'bbox içinde Sauvola-koyu piksel oranı > 0.5 ise ters' (metin azınlık sınıfı).")
    L.append("- Dosya düzeni: training maskeleri `<variant>/<new_id>.png`, testing maskeleri `<variant>/test/<new_id>.png` (TestingSet new_id'leri 1–812 TrainingSet id'leriyle çakıştığı için ayrı alt klasör; `gt_index.csv` → `gt_relpath`).")
    L.append("- Atlanan çiftler: %s" % (stats["skipped"] if stats["skipped"] else "yok"))
    with open(os.path.join(gt_dir, "gt_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Vize Proje kök klasörü")
    ap.add_argument("--data", default=None, help="AIForge-Doc-v1 klasörü (vars.: <root>/data/AIForge-Doc-v1)")
    ap.add_argument("--out", default=None, help="çıktı klasörü (vars.: <root>/phase2/gt)")
    ap.add_argument("--splits", nargs="+", default=["training", "testing"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="hızlı test için ilk N çift")
    ap.add_argument("--no-write", action="store_true", help="PNG yazma, yalnız istatistik")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    data_dir = args.data or os.path.join(args.root, "data", "AIForge-Doc-v1")
    gt_dir = args.out or os.path.join(args.root, "phase2", "gt")
    os.makedirs(gt_dir, exist_ok=True)
    log_path = args.log or os.path.join(gt_dir, "make_gt.log")
    logf = open(log_path, "a", encoding="utf-8")

    def log(msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    if args.summarize_only:
        st = summarize(gt_dir)
        log("summary done: %d pairs" % st["n_pairs_ok"]); return

    df = pd.read_csv(os.path.join(args.root, "dataset_audit", "samples_forged.csv"))
    df = df[(df["pair_accepted"] == True) & (df["split"].isin(args.splits))].copy()
    df = df.sort_values(["split", "new_id"], ascending=[False, True]).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    for v in VARIANTS:
        os.makedirs(os.path.join(gt_dir, v, "test"), exist_ok=True)
    log("start: %d pairs, splits=%s, workers=%d, write=%s" % (len(df), args.splits, args.workers, not args.no_write))
    recs = df.to_dict("records")
    rows, hists = [], {}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_pair, r, data_dir, gt_dir, not args.no_write): r for r in recs}
        for fu in as_completed(futs):
            r = futs[fu]
            try:
                out, h = fu.result()
            except Exception as e:
                out = {"new_id": "%09d" % int(r["new_id"]), "split": r["split"], "skipped": "error:%s" % repr(e)}
                h = None
                log("ERROR %s: %s" % (out["new_id"], traceback.format_exc().splitlines()[-1]))
            rows.append(out)
            if h is not None:
                hists[out["new_id"]] = h
            done += 1
            if done % 100 == 0 or done == len(recs):
                el = time.time() - t0
                log("%d/%d (%.1f s, %.2f s/pair, eta %.0f s)" % (done, len(recs), el, el / done, el / done * (len(recs) - done)))
    idx = pd.DataFrame(rows)
    if "skipped" not in idx.columns:
        idx["skipped"] = np.nan
    idx = idx.sort_values(["split", "new_id"], ascending=[False, True])
    idx.to_csv(os.path.join(gt_dir, "gt_index.csv"), index=False)
    ids = list(hists.keys())
    np.savez_compressed(os.path.join(gt_dir, "gt_hist.npz"), new_id=np.array(ids),
                        hist_in=np.stack([hists[i][0] for i in ids]) if ids else np.zeros((0, 256)),
                        hist_out=np.stack([hists[i][1] for i in ids]) if ids else np.zeros((0, 256)))
    log("index written: %d rows, %d skipped" % (len(idx), idx["skipped"].notna().sum()))
    st = summarize(gt_dir)
    log("summary done; total %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
