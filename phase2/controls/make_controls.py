#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_controls.py — Benign kontrol setleri (AIForge-Doc v1, phase2/controls).

Kullanım:
  python make_controls.py --root "C:\\...\\Vize Proje" [--splits val smoke_val] [--workers 8] [--limit N]

Her id (splits.json → val / smoke_val; TrainingSet) için ilgili forged'ın AUTHENTIC eşi alınır ve
forged'ın MASKE dosyasından bbox okunur. Üç kontrol üretilir (içerik değişmez, sahte yoktur):
  benign_pasteback   : crop = bbox her yönde %50 genişletilmiş (kenar ≥100 px, görüntüye kırpılır)
                       → 2× LANCZOS büyüt → aynı boyuta LANCZOS küçült → YALNIZ bbox bölgesi authentic
                       kopyasına geri yapıştırılır → PNG
  reencode_png       : authentic → PIL open → RGB → PNG (compress_level=6); piksel özdeşlik raporlanır
  fullimage_resample : tüm görüntü 2× LANCZOS büyüt → küçült → PNG
Çıktı: controls/<type>/images/<new_id>.png, controls/<type>/masks/<new_id>.png (tamamen 0),
       controls_index.csv, controls_asymmetry.csv, controls_report.md, examples/*.png
"""
import argparse, os, json, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from PIL import Image

TYPES = ["benign_pasteback", "reencode_png", "fullimage_resample"]


def bbox_from_mask(path):
    m = np.asarray(Image.open(path).convert("L")) > 127
    ys, xs = np.nonzero(m)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def expand_bbox(x1, y1, x2, y2, W, H, frac=0.5, min_side=100):
    w, h = x2 - x1 + 1, y2 - y1 + 1
    cw, ch = max(w * (1 + 2 * frac), min_side), max(h * (1 + 2 * frac), min_side)
    cx, cy = (x1 + x2 + 1) / 2.0, (y1 + y2 + 1) / 2.0
    X1, X2 = int(np.floor(cx - cw / 2)), int(np.ceil(cx + cw / 2))
    Y1, Y2 = int(np.floor(cy - ch / 2)), int(np.ceil(cy + ch / 2))
    return max(0, X1), max(0, Y1), min(W, X2), min(H, Y2)   # PIL crop box (exclusive sağ/alt)


def up_down(img):
    w, h = img.size
    return img.resize((2 * w, 2 * h), Image.LANCZOS).resize((w, h), Image.LANCZOS)


def diff_stats(ctrl, auth, bbox):
    x1, y1, x2, y2 = bbox
    D = np.abs(ctrl.astype(np.int16) - auth.astype(np.int16)).max(axis=2)
    inb = np.zeros(D.shape, bool); inb[y1:y2 + 1, x1:x2 + 1] = True
    g8 = D > 8; g0 = D > 0
    return {"D_max": int(D.max()), "D_mean": float(D.mean()), "D_mean_in": float(D[inb].mean()), "D_mean_out": float(D[~inb].mean()),
            "n_gt8": int(g8.sum()), "n_gt8_in": int((g8 & inb).sum()), "n_gt8_out": int((g8 & ~inb).sum()),
            "n_gt0": int(g0.sum()), "n_gt0_in": int((g0 & inb).sum()), "n_gt0_out": int((g0 & ~inb).sum()),
            "bbox_px": int(inb.sum()), "img_px": int(D.size)}


def process(rec, data_dir, out_dir):
    nid = rec["new_id"]
    sub = "TrainingSet"
    ap = os.path.join(data_dir, sub, "authentic", rec["pair_name"])
    fp = os.path.join(data_dir, sub, "images", nid + ".png")
    mp = os.path.join(data_dir, sub, "masks", nid + ".png")
    auth_img = Image.open(ap).convert("RGB")
    auth = np.asarray(auth_img)
    W, H = auth_img.size
    bbox = bbox_from_mask(mp)
    x1, y1, x2, y2 = bbox
    rows, asym = [], []
    # gerçek forged referansı
    forged = np.asarray(Image.open(fp).convert("RGB"))
    if forged.shape == auth.shape:
        s = diff_stats(forged, auth, bbox); s.update({"new_id": nid, "type": "REAL_FORGED"}); asym.append(s)
    # (a) benign paste-back
    X1, Y1, X2, Y2 = expand_bbox(x1, y1, x2, y2, W, H)
    crop = auth_img.crop((X1, Y1, X2, Y2))
    res = np.asarray(up_down(crop))
    pb = auth.copy()
    pb[y1:y2 + 1, x1:x2 + 1] = res[y1 - Y1:y2 - Y1 + 1, x1 - X1:x2 - X1 + 1]
    # (b) re-encode
    re_img = auth_img  # PIL open → RGB; yeniden PNG yazımı
    # (c) full-image resample
    fr = np.asarray(up_down(auth_img))
    outs = {"benign_pasteback": pb, "reencode_png": np.asarray(re_img), "fullimage_resample": fr}
    empty = Image.fromarray(np.zeros((H, W), np.uint8), "L")
    for t in TYPES:
        ip = os.path.join(out_dir, t, "images", nid + ".png")
        mpth = os.path.join(out_dir, t, "masks", nid + ".png")
        Image.fromarray(outs[t], "RGB").save(ip, format="PNG", compress_level=6)
        empty.save(mpth, format="PNG", compress_level=6)
        # kaydedilen dosyayı geri okuyup authentic ile karşılaştır (asimetri ölçümü)
        back = np.asarray(Image.open(ip).convert("RGB"))
        s = diff_stats(back, auth, bbox)
        s.update({"new_id": nid, "type": t, "pixel_identical": bool(np.array_equal(back, auth))})
        asym.append(s)
        rows.append({"control_id": "%s_%s" % (t, nid), "type": t, "new_id": nid, "authentic_name": rec["pair_name"],
                     "bbox": "[%d, %d, %d, %d]" % bbox, "crop_window": "[%d, %d, %d, %d]" % (X1, Y1, X2, Y2) if t == "benign_pasteback" else "",
                     "path": os.path.relpath(ip, out_dir), "mask_path": os.path.relpath(mpth, out_dir),
                     "split_membership": rec["split_membership"], "W": W, "H": H, "pixel_identical": s["pixel_identical"],
                     "n_gt8": s["n_gt8"], "n_gt0": s["n_gt0"], "D_max": s["D_max"]})
    return rows, asym


def q(x, p):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(np.percentile(x, p)) if len(x) else float("nan")


def summarize(out_dir, idx, asym):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = pd.DataFrame(asym)
    A.to_csv(os.path.join(out_dir, "controls_asymmetry.csv"), index=False)
    L = ["# Benign kontrol setleri — rapor\n",
         "Üretim: `make_controls.py`. Kontrol başına görüntü sayısı: %s. Split üyeliği: %s.\n" % (idx.groupby("type").size().to_dict(), idx[idx.type == TYPES[0]]["split_membership"].value_counts().to_dict()),
         "Boş maskeler `controls/<type>/masks/` altında (tamamen 0). Tüm kontroller içerik değiştirmez; yalnız yeniden-örnekleme/yeniden-kodlama izi taşır.\n"]
    ident = idx[idx.type == "reencode_png"]["pixel_identical"]
    L.append("**reencode_png piksel özdeşlik:** %d/%d görüntü authentic ile birebir özdeş%s\n" % (
        int(ident.sum()), len(ident), " → bu kontrol piksel düzeyinde ANLAMSIZ (yalnız dosya/kodlama düzeyi kontrol; modele girdi olarak authentic ile aynıdır)." if ident.all() else " (özdeş olmayanlar için fark istatistikleri aşağıda)."))
    L.append("\n## Asimetri ölçümü: D = max_kanal |kontrol − authentic| vs gerçek forged çiftleri (aynı id'ler)\n")
    L.append("| tip | n | D_max medyan [IQR] | D_mean medyan | D>8 px medyan [IQR] | D>8 bbox-içi medyan | D>8 bbox-dışı medyan | D>0 px medyan | D>0 bbox-dışı medyan | görüntü oranı: ≥1 px D>8 | görüntü oranı: ≥1 px D>8 bbox-dışı |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    summ = {}
    for t in ["REAL_FORGED"] + TYPES:
        d = A[A.type == t]
        if not len(d):
            continue
        s = {"n": int(len(d)), "D_max_median": q(d.D_max, 50), "D_max_q25": q(d.D_max, 25), "D_max_q75": q(d.D_max, 75), "D_mean_median": q(d.D_mean, 50),
             "n_gt8_median": q(d.n_gt8, 50), "n_gt8_q25": q(d.n_gt8, 25), "n_gt8_q75": q(d.n_gt8, 75), "n_gt8_in_median": q(d.n_gt8_in, 50), "n_gt8_out_median": q(d.n_gt8_out, 50),
             "n_gt0_median": q(d.n_gt0, 50), "n_gt0_out_median": q(d.n_gt0_out, 50),
             "frac_any_gt8": float((d.n_gt8 > 0).mean()), "frac_any_gt8_out": float((d.n_gt8_out > 0).mean()),
             "bbox_coverage_gt8_median": q(d.n_gt8_in / d.bbox_px, 50), "D_mean_in_median": q(d.D_mean_in, 50), "D_mean_out_median": q(d.D_mean_out, 50)}
        summ[t] = s
        L.append("| %s | %d | %.0f [%.0f–%.0f] | %.3f | %.0f [%.0f–%.0f] | %.0f | %.0f | %.0f | %.0f | %.3f | %.3f |" % (
            t, s["n"], s["D_max_median"], s["D_max_q25"], s["D_max_q75"], s["D_mean_median"], s["n_gt8_median"], s["n_gt8_q25"], s["n_gt8_q75"],
            s["n_gt8_in_median"], s["n_gt8_out_median"], s["n_gt0_median"], s["n_gt0_out_median"], s["frac_any_gt8"], s["frac_any_gt8_out"]))
    L.append("\nbbox içinde D>8 kapsama medyanı (n_gt8_in / bbox_px): " + "; ".join("%s %.3f" % (t, s["bbox_coverage_gt8_median"]) for t, s in summ.items()))
    L.append("\nbbox içi / dışı ortalama D medyanı: " + "; ".join("%s %.2f / %.2f" % (t, s["D_mean_in_median"], s["D_mean_out_median"]) for t, s in summ.items()))
    rf, pb, fr = summ.get("REAL_FORGED"), summ.get("benign_pasteback"), summ.get("fullimage_resample")
    if rf and pb and fr:
        L.append("\n**Asimetri bulgusu (hesaplanan):** gerçek forged çiftlerinde D_max medyanı %.0f ve D>8 piksel medyanı %.0f (bbox kapsama %.3f). benign_pasteback: D_max medyanı %.0f, D>8 piksel medyanı %.0f (gerçek sahtenin ~1/%.0f'i; görüntülerin %%%.1f'inde ≥1 px D>8; tamamı bbox içinde, bbox-dışı D>8 medyanı %.0f). fullimage_resample: D_max medyanı %.0f, ancak D>8 piksel medyanı %.0f — gerçek sahtenin D>8 piksel sayısına YAKIN (medyan oranı %.2f) ve görüntülerin %%%.1f'inde mevcut; bu pikseller tüm görüntüdeki metin kenarlarına dağılır (bbox-dışı D>8 medyanı %.0f, bbox içi kapsama %.3f). Sonuç: (i) genlik (D_max) açısından kontroller gerçek sahteden net ayrılır; (ii) D>ε tabanlı bir piksel-değişim GT'si benign_pasteback'te az sayıda, fullimage_resample'da ise gerçek sahteyle karşılaştırılabilir SAYIDA yanlış pozitif piksel üretir — fark sayıda değil, genlikte ve uzamsal dağılımdadır (yoğun blok yerine seyrek kenar pikselleri). Bu nedenle fullimage_resample 'yeniden-örnekleme izi = sahtelik' kısayolunu test eden en zorlu kontroldür; bbox-GT ile eğitilen bir modelin benign_pasteback'in bbox sınırında veya fullimage_resample'ın metin kenarlarında pozitif üretmesi kısayol kanıtı olarak değerlendirilecektir.\n" % (
            rf["D_max_median"], rf["n_gt8_median"], rf["bbox_coverage_gt8_median"],
            pb["D_max_median"], pb["n_gt8_median"], rf["n_gt8_median"] / max(1, pb["n_gt8_median"]), 100 * pb["frac_any_gt8"], pb["n_gt8_out_median"],
            fr["D_max_median"], fr["n_gt8_median"], fr["n_gt8_median"] / max(1, rf["n_gt8_median"]), 100 * fr["frac_any_gt8"], fr["n_gt8_out_median"], fr["bbox_coverage_gt8_median"]))
    with open(os.path.join(out_dir, "controls_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    # örnek görseller (2 adet)
    os.makedirs(os.path.join(out_dir, "examples"), exist_ok=True)
    ex_ids = []
    cand = A[A.type == "REAL_FORGED"].sort_values("n_gt8")
    if len(cand):
        ex_ids = [cand.iloc[len(cand) // 2]["new_id"], cand.iloc[int(len(cand) * 0.9)]["new_id"]]
    return L, ex_ids


def make_examples(out_dir, data_dir, idx, ex_ids, L):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    for k, nid in enumerate(ex_ids):
        r = idx[(idx.new_id == nid) & (idx.type == "benign_pasteback")].iloc[0]
        bbox = json.loads(r["bbox"]); x1, y1, x2, y2 = bbox
        auth = np.asarray(Image.open(os.path.join(data_dir, "TrainingSet", "authentic", r["authentic_name"])).convert("RGB"))
        forged = np.asarray(Image.open(os.path.join(data_dir, "TrainingSet", "images", nid + ".png")).convert("RGB"))
        H, W = auth.shape[:2]
        w, h = x2 - x1 + 1, y2 - y1 + 1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cw, ch = max(3 * w, 256), max(3 * h, 256)
        X1, X2 = max(0, int(cx - cw / 2)), min(W, int(cx + cw / 2)); Y1, Y2 = max(0, int(cy - ch / 2)), min(H, int(cy + ch / 2))
        panels = [("authentic", auth[Y1:Y2, X1:X2], None), ("forged", forged[Y1:Y2, X1:X2], None)]
        for t in ["REAL_FORGED"] + TYPES:
            if t == "REAL_FORGED":
                c = forged
            else:
                c = np.asarray(Image.open(os.path.join(out_dir, t, "images", nid + ".png")).convert("RGB"))
            D = np.abs(c.astype(np.int16) - auth.astype(np.int16)).max(axis=2)
            panels.append(("D: %s (max %d)" % (t if t != "REAL_FORGED" else "forged", D.max()), D[Y1:Y2, X1:X2], "D"))
        fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
        for ax, (title, img, kind) in zip(axes, panels):
            if kind == "D":
                im = ax.imshow(img, cmap="magma", vmin=0, vmax=32, interpolation="nearest")
            else:
                ax.imshow(img, interpolation="nearest")
            ax.add_patch(Rectangle((x1 - X1 - 0.5, y1 - Y1 - 0.5), w, h, fill=False, ec="#2ca02c", lw=0.9, ls="--"))
            ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes[-1], fraction=0.046, label="D (kırpılmış 0–32)")
        fig.suptitle("Örnek %d — new_id %s (kesikli kutu = forged bbox; D = max_kanal |görüntü − authentic|)" % (k + 1, nid), fontsize=9)
        fig.tight_layout()
        p = os.path.join(out_dir, "examples", "example_%d_%s.png" % (k + 1, nid))
        fig.savefig(p, dpi=130); plt.close(fig)
        L.append("\n![örnek %d](examples/%s)\n" % (k + 1, os.path.basename(p)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--splits", nargs="+", default=["val", "smoke_val"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    data_dir = args.data or os.path.join(args.root, "data", "AIForge-Doc-v1")
    out_dir = args.out or os.path.join(args.root, "phase2", "controls")
    os.makedirs(out_dir, exist_ok=True)
    for t in TYPES:
        os.makedirs(os.path.join(out_dir, t, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, t, "masks"), exist_ok=True)
    logf = open(args.log or os.path.join(out_dir, "make_controls.log"), "a", encoding="utf-8")

    def log(msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg); print(line, flush=True); logf.write(line + "\n"); logf.flush()

    sp = json.load(open(os.path.join(args.root, "dataset_audit", "splits.json"), encoding="utf-8"))
    member = {}
    for s in args.splits:
        for i in sp[s]:
            member.setdefault(str(i).zfill(9), []).append(s)
    df = pd.read_csv(os.path.join(args.root, "dataset_audit", "samples_forged.csv"))
    df = df[df["split"] == "training"].copy()
    df["new_id"] = df["new_id"].astype(int).map(lambda v: "%09d" % v)
    df = df[df["new_id"].isin(member)].copy()
    df["split_membership"] = df["new_id"].map(lambda n: ";".join(member[n]))
    missing = sorted(set(member) - set(df["new_id"]))
    not_acc = df[df["pair_accepted"] != True]
    df = df[df["pair_accepted"] == True].sort_values("new_id").reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    log("ids: %d (splits=%s); pair_accepted False → atlanan: %d; samples_forged'da bulunamayan: %d" % (len(df), args.splits, len(not_acc), len(missing)))
    recs = df.to_dict("records")
    rows, asym = [], []
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, r, data_dir, out_dir): r for r in recs}
        for fu in as_completed(futs):
            r = futs[fu]
            try:
                rr, aa = fu.result(); rows += rr; asym += aa
            except Exception as e:
                log("ERROR %s: %s" % (r["new_id"], traceback.format_exc().splitlines()[-1]))
            done += 1
            if done % 50 == 0 or done == len(recs):
                el = time.time() - t0; log("%d/%d (%.1f s, eta %.0f s)" % (done, len(recs), el, el / done * (len(recs) - done)))
    idx = pd.DataFrame(rows).sort_values(["type", "new_id"]).reset_index(drop=True)
    idx.to_csv(os.path.join(out_dir, "controls_index.csv"), index=False)
    for t in TYPES:
        ids = idx[idx.type == t]["new_id"].tolist()
        with open(os.path.join(out_dir, t, "ids.txt"), "w") as fh:
            fh.write("\n".join(ids))
    L, ex_ids = summarize(out_dir, idx, asym)
    make_examples(out_dir, data_dir, idx, ex_ids, L)
    L.append("\n## Atlananlar\n- pair_accepted==False (authentic eşi doğrulanamamış): %s\n- samples_forged.csv'de bulunamayan id: %s\n" % (not_acc["new_id"].tolist() if len(not_acc) else "yok", missing if missing else "yok"))
    with open(os.path.join(out_dir, "controls_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    log("done: %d control rows, %.1f s" % (len(idx), time.time() - t0))


if __name__ == "__main__":
    main()
