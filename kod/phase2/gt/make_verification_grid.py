#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_verification_grid.py — 100 çiftlik insan doğrulama grid'i (AIForge-Doc v1, phase2/gt).

Kullanım:
  python make_verification_grid.py --root "C:\\...\\Vize Proje" [--n 100] [--seed 42] [--per-page 5]

Girdi : <root>/phase2/gt/gt_index.csv + <root>/phase2/gt/{pixel_eps8,glyph}/<new_id>.png + veri seti
Çıktı : <root>/phase2/gt/verification_grid.pdf, verification_sheet.csv, verification_summary.md

Örnekleme: training çiftleri kaynak×tool×area_bin tabakalarına ayrılır; tabaka başına en az 1,
geri kalan orantılı (en büyük kalan yöntemi), tabaka içi rastgele (seed).
Satır: forged crop | authentic crop | bbox GT | pixel_eps8 | glyph (spec) | glyph_v2 (azınlık-polarite). Crop = bbox'ın her yönde kendi
genişliği/yüksekliği kadar (2×) genişletilmiş penceresi, kenar en az 256 px, görüntüye kırpılır.
"""
import argparse, os, json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


def stratified_sample(df, n, seed, strata_cols):
    rng = np.random.RandomState(seed)
    groups = {k: g.sort_values("new_id") for k, g in df.groupby(strata_cols, observed=True)}
    keys = sorted(groups.keys(), key=lambda k: str(k))
    sizes = np.array([len(groups[k]) for k in keys], dtype=float)
    alloc = np.ones(len(keys), dtype=int)                      # tabaka başına en az 1
    alloc = np.minimum(alloc, sizes.astype(int))
    rem = n - alloc.sum()
    if rem > 0:
        prop = sizes / sizes.sum() * rem
        base = np.floor(prop).astype(int)
        base = np.minimum(base, sizes.astype(int) - alloc)
        alloc += base
        rem = n - alloc.sum()
        order = np.argsort(-(prop - np.floor(prop)))
        for i in order:
            if rem <= 0:
                break
            if alloc[i] < sizes[i]:
                alloc[i] += 1; rem -= 1
    picks = []
    for k, a in zip(keys, alloc):
        g = groups[k]
        idx = rng.choice(len(g), size=int(a), replace=False)
        picks.append(g.iloc[np.sort(idx)])
    out = pd.concat(picks).reset_index(drop=True)
    alloc_tbl = pd.DataFrame({"stratum": [str(k) for k in keys], "n_available": sizes.astype(int), "n_sampled": alloc})
    return out, alloc_tbl


def crop_window(x1, y1, x2, y2, W, H, min_side=256):
    w, h = x2 - x1 + 1, y2 - y1 + 1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cw, ch = max(3 * w, min_side), max(3 * h, min_side)
    X1, X2 = int(round(cx - cw / 2)), int(round(cx + cw / 2))
    Y1, Y2 = int(round(cy - ch / 2)), int(round(cy + ch / 2))
    X1, Y1 = max(0, X1), max(0, Y1)
    X2, Y2 = min(W, X2), min(H, Y2)
    return X1, Y1, X2, Y2


def flags_for(r):
    fl = []
    if r["pixel_eps8_morph_n_cc_out"] > 0:
        fl.append("outside_bbox_component")
    if r["pixel_eps8_n_pos"] > 0 and r["pixel_eps8_n_out"] / r["pixel_eps8_n_pos"] > 0.01:
        fl.append("outside_px_gt_1pct")
    if r["glyph_n_pos"] == 0:
        fl.append("empty_glyph")
    if r["pixel_eps8_n_pos"] == 0:
        fl.append("empty_eps8")
    if r["pixel_eps8_n_in"] / r["bbox_px"] < 0.20:
        fl.append("low_bbox_coverage_eps8")
    if bool(r["polarity_inverted"]):
        fl.append("polarity_inverted")
    if bool(r["polarity_inverted_v2"]):
        fl.append("polarity_inverted_v2")
    if r["glyph_v2_n_pos"] == 0:
        fl.append("empty_glyph_v2")
    return ";".join(fl)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-page", type=int, default=5)
    ap.add_argument("--preview-pages", type=int, default=1, help="ilk N sayfayı PNG olarak da kaydet (verification_grid_pageK.png)")
    args = ap.parse_args()
    data_dir = args.data or os.path.join(args.root, "data", "AIForge-Doc-v1")
    gt_dir = args.gt or os.path.join(args.root, "phase2", "gt")
    idx = pd.read_csv(os.path.join(gt_dir, "gt_index.csv"), dtype={"new_id": str})
    ok = idx[idx["skipped"].isna()] if "skipped" in idx.columns else idx
    tr = ok[ok["split"] == "training"].copy()
    tr["auto_flags"] = tr.apply(flags_for, axis=1)

    # tüm train çiftleri için tutarlılık özeti
    def consistency(d):
        return {"n": int(len(d)),
                "n_outside_bbox_component(eps8_morph)": int((d["pixel_eps8_morph_n_cc_out"] > 0).sum()),
                "n_outside_px_gt_1pct(eps8)": int(((d["pixel_eps8_n_out"] / d["pixel_eps8_n_pos"].replace(0, np.nan)) > 0.01).sum()),
                "n_any_outside_px(eps8)": int((d["pixel_eps8_n_out"] > 0).sum()),
                "n_empty_glyph": int((d["glyph_n_pos"] == 0).sum()),
                "n_empty_eps8": int((d["pixel_eps8_n_pos"] == 0).sum()),
                "n_eps8_bbox_coverage_lt_20pct": int(((d["pixel_eps8_n_in"] / d["bbox_px"]) < 0.20).sum()),
                "n_polarity_inverted": int(d["polarity_inverted"].astype(bool).sum()),
                "n_polarity_inverted_v2": int(d["polarity_inverted_v2"].astype(bool).sum()),
                "n_empty_glyph_v2": int((d["glyph_v2_n_pos"] == 0).sum())}

    sample, alloc_tbl = stratified_sample(tr, args.n, args.seed, ["source", "tool", "area_bin"])
    sample = sample.sort_values(["source", "tool", "area_bin", "new_id"]).reset_index(drop=True)
    summ = {"all_train": consistency(tr), "sample": consistency(sample), "seed": args.seed,
            "strata_allocation": alloc_tbl.to_dict("records")}

    sheet = sample[["new_id", "tool", "source", "area_bin"]].copy()
    sheet["n_outside_bbox_eps8"] = sample["pixel_eps8_n_out"].values
    sheet["n_outside_cc_eps8_morph"] = sample["pixel_eps8_morph_n_cc_out"].values
    sheet["iou_eps8_bbox"] = sample["pixel_eps8_iou_bbox"].round(4).values
    sheet["bbox_coverage_eps8"] = (sample["pixel_eps8_n_in"] / sample["bbox_px"]).round(4).values
    sheet["glyph_n_pos"] = sample["glyph_n_pos"].values
    sheet["iou_glyph_bbox"] = sample["glyph_iou_bbox"].round(4).values
    sheet["glyph_v2_n_pos"] = sample["glyph_v2_n_pos"].values
    sheet["iou_glyph_v2_bbox"] = sample["glyph_v2_iou_bbox"].round(4).values
    sheet["auto_flags"] = sample["auto_flags"].values
    sheet["page"] = (np.arange(len(sheet)) // args.per_page) + 1
    sheet["human_verdict"] = ""   # OK / PARTIAL / WRONG
    sheet["note"] = ""
    sheet.to_csv(os.path.join(gt_dir, "verification_sheet.csv"), index=False)

    pdf_path = os.path.join(gt_dir, "verification_grid.pdf")
    cols = ["forged crop", "authentic crop", "bbox GT (maske)", "pixel_eps8 (D>8)", "glyph (spec)", "glyph_v2"]
    with PdfPages(pdf_path) as pdf:
        for p0 in range(0, len(sample), args.per_page):
            chunk = sample.iloc[p0:p0 + args.per_page]
            # önce crop pencereleri ve satır yükseklikleri (en-boy oranına göre) hesaplanır
            items = []
            for _, r in chunk.iterrows():
                nid = r["new_id"]
                sub = "TrainingSet"
                f = Image.open(os.path.join(data_dir, sub, "images", nid + ".png")).convert("RGB")
                W, H = f.size
                x1, y1, x2, y2 = int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"])
                X1, Y1, X2, Y2 = crop_window(x1, y1, x2, y2, W, H)
                items.append((r, f, (X1, Y1, X2, Y2), (x1, y1, x2, y2)))
            panel_w = 2.9  # inç
            ratios = [min(max((b[3] - b[1]) / max(1, b[2] - b[0]), 0.22), 1.3) for _, _, b, _ in items]
            heights = [panel_w * rt for rt in ratios]
            total_h = sum(heights) + 0.45 * len(items) + 1.0
            fig = plt.figure(figsize=(6 * panel_w + 0.3, total_h))
            gs = fig.add_gridspec(len(items), 6, height_ratios=heights, left=0.01, right=0.99, top=1 - 1.0 / total_h, bottom=0.01,
                                  hspace=0.45 * len(items) / max(0.1, np.mean(heights)) / len(items), wspace=0.04)
            axes = [[fig.add_subplot(gs[i, j]) for j in range(6)] for i in range(len(items))]
            for ri, (r, f, box, bb) in enumerate(items):
                nid = r["new_id"]
                sub = "TrainingSet"
                a = Image.open(os.path.join(data_dir, sub, "authentic", r["pair_name"])).convert("RGB")
                bm = Image.open(os.path.join(data_dir, sub, "masks", nid + ".png")).convert("L")
                e8 = Image.open(os.path.join(gt_dir, "pixel_eps8", nid + ".png")).convert("L")
                gl = Image.open(os.path.join(gt_dir, "glyph", nid + ".png")).convert("L")
                gl2 = Image.open(os.path.join(gt_dir, "glyph_v2", nid + ".png")).convert("L")
                X1, Y1, X2, Y2 = box
                x1, y1, x2, y2 = bb
                panels = [np.asarray(f.crop(box)), np.asarray(a.crop(box)), np.asarray(bm.crop(box)), np.asarray(e8.crop(box)), np.asarray(gl.crop(box)), np.asarray(gl2.crop(box))]
                for ci, (ax, img) in enumerate(zip(axes[ri], panels)):
                    if img.ndim == 2:
                        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect="auto")
                    else:
                        ax.imshow(img, interpolation="nearest", aspect="auto")
                    ax.add_patch(Rectangle((x1 - X1 - 0.5, y1 - Y1 - 0.5), x2 - x1 + 1, y2 - y1 + 1, fill=False, ec="#d62728" if ci < 2 else "#1f77b4", lw=0.9, ls="--"))
                    ax.set_xticks([]); ax.set_yticks([])
                    if ri == 0:
                        ax.set_title(cols[ci], fontsize=9, pad=18)
                title = "%s | %s | %s | %s | bbox-dışı değişen px (eps8): %d | IoU(eps8,bbox)=%.2f | bbox kapsama=%.2f%s" % (
                    nid, r["tool"], r["source"], r["area_bin"], int(r["pixel_eps8_n_out"]), r["pixel_eps8_iou_bbox"],
                    r["pixel_eps8_n_in"] / r["bbox_px"], (" | flags: " + r["auto_flags"]) if r["auto_flags"] else "")
                axes[ri][0].text(0.0, 1.02, title, transform=axes[ri][0].transAxes, fontsize=8, ha="left", va="bottom", family="monospace")
            fig.suptitle("İnsan doğrulama grid'i — sayfa %d/%d (kesikli kutu = bbox GT; crop = 3×bbox, min 256 px; paneller en-boy korunmadan pencereye oturtulur)" % (p0 // args.per_page + 1, int(np.ceil(len(sample) / args.per_page))), fontsize=10, y=0.998)
            pdf.savefig(fig)
            page_no = p0 // args.per_page + 1
            if page_no <= args.preview_pages:
                fig.savefig(os.path.join(gt_dir, "verification_grid_page%d.png" % page_no), dpi=110)
            plt.close(fig)

    L = ["# İnsan doğrulama grid'i — otomatik tutarlılık özeti\n",
         "Örneklem: %d training çifti, tabaka = kaynak×tool×area_bin, seed=%d. PDF: `verification_grid.pdf` (%d çift/sayfa). Form: `verification_sheet.csv` (human_verdict: OK/PARTIAL/WRONG, note).\n" % (len(sample), args.seed, args.per_page),
         "| ölçüt | örneklem (n=%d) | tüm train (n=%d) |" % (summ["sample"]["n"], summ["all_train"]["n"]), "|---|---|---|"]
    for k in summ["sample"]:
        if k == "n":
            continue
        L.append("| %s | %d | %d (%.1f%%) |" % (k, summ["sample"][k], summ["all_train"][k], 100.0 * summ["all_train"][k] / summ["all_train"]["n"]))
    L.append("\nTabaka dağılımı:\n")
    L.append("| tabaka (kaynak, tool, area_bin) | mevcut | örneklenen |")
    L.append("|---|---|---|")
    for rec in summ["strata_allocation"]:
        L.append("| %s | %d | %d |" % (rec["stratum"], rec["n_available"], rec["n_sampled"]))
    L.append("\nBayrak tanımları: `outside_bbox_component` = pixel_eps8_morph'ta bbox ile kesişmeyen ≥1 bağlı bileşen; `outside_px_gt_1pct` = eps8 pozitiflerinin >%1'i bbox dışında; `empty_glyph` = glif maskesi boş; `empty_eps8` = D>8 boş; `low_bbox_coverage_eps8` = D>8 bbox'ın <%20'sini kaplıyor; `polarity_inverted` = spec kuralı (bbox içi medyan gri <128) tetiklendi; `polarity_inverted_v2` = azınlık-sınıf kuralı tetiklendi; `empty_glyph_v2` = glyph_v2 boş.")
    with open(os.path.join(gt_dir, "verification_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    with open(os.path.join(gt_dir, "verification_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, ensure_ascii=False)
    print(json.dumps({"sample": summ["sample"], "all_train": summ["all_train"]}, indent=1))
    print("pdf:", pdf_path)


if __name__ == "__main__":
    main()
