r"""
xai_benchmark.py — SHAP/XAI modülünün performans ve doğruluk ölçümü → docs/xai_test_report.md + CSV.

Ölçülenler
  A) Durum      : her checkpoint için karar hedefi + Partition → OK / FAIL; SVM/RF UNSUPPORTED, YOLO/ViT N/A satırları.
  B) Çekirdek   : (varsayılan ResNet ft + U-Net bbox/piksel/glif s42) test örnekleri üzerinde, TP / FN / TN / FP grupları:
                  süre, değerlendirme sayısı, GPU tepe belleği, toplamsallık hatası,
                  SADAKAT (silme eğrisi: en yüksek SHAP blokları referansla değiştirilince skor düşüşü; AOPC, rastgele sıraya karşı),
                  POINTING GAME (maks SHAP bloğu GT kutusuna değiyor mu), İŞARETÇİ BAĞIMLILIĞI (pozitif kütlenin işaretçi
                  pikselleri ve kutu kenar halkasındaki payı, alan payına oranla "lift").
  C) Tohum kararlılığı: U-Net bbox ve glif s42/43/44 aynı ortak pencerede → top-%5 pozitif bölge IoU.

Kullanım (proje kökü):
    python scripts/xai_benchmark.py --n 24                       # GPU önerilir (varsayılan max_evals 1000)
    python scripts/xai_benchmark.py --n 8 --max-evals 200 --status-evals 100 --seed-images 3   # CPU, hızlı
"""
from __future__ import annotations
import argparse, os, sys, time, datetime, json
from pathlib import Path
import numpy as np
import pandas as pd

import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
ROOT = _ROOT
sys.path.insert(0, str(Path(os.environ.get("AI515_APP_DIR", _KOD / "app"))))
os.environ.setdefault("APP_NO_BROWSER", "1")

CORE = ["PHASE1/resnet18_ft_best", "PHASE2/unet_bbox_s42", "PHASE2/unet_pixel_eps8_morph_s42", "PHASE2/unet_glyph_v2_s42"]
SEED_FAMILIES = {"bbox": [f"PHASE2/unet_bbox_s{s}" for s in (42, 43, 44)],
                 "glif": [f"PHASE2/unet_glyph_v2_s{s}" for s in (42, 43, 44)]}


def log(m):
    print(f"[xai-bench {time.strftime('%H:%M:%S')}] {m}", flush=True)


def block_grid(h, w, g=16):
    ys = np.linspace(0, h, g + 1).astype(int); xs = np.linspace(0, w, g + 1).astype(int)
    return [(ys[i], ys[i + 1], xs[j], xs[j + 1]) for i in range(g) for j in range(g)]


def deletion_aopc(ad, t, win, ref, smap, ctx, steps=10, rng=None, n_rand=2):
    """En yüksek SHAP blokları referansla değiştirildikçe hedef skorun düşüşü (AOPC; yüksek = sadık)."""
    blocks = block_grid(*smap.shape)
    sums = np.array([smap[a:b, c:d].sum() for a, b, c, d in blocks])
    orders = {"shap": np.argsort(-sums)}
    for k in range(n_rand):
        orders[f"rand{k}"] = (rng or np.random.default_rng(k)).permutation(len(blocks))
    f0 = float(ad.predict_batch(win[None], t, ctx)[0, 0])
    out = {}
    for name, order in orders.items():
        imgs, cur = [], win.copy()
        per = int(np.ceil(len(blocks) / steps))
        for s in range(steps):
            for bi in order[s * per:(s + 1) * per]:
                a, b, c, d = blocks[bi]; cur[a:b, c:d] = ref[a:b, c:d]
            imgs.append(cur.copy())
        f = ad.predict_batch(np.stack(imgs), t, ctx)[:, 0]
        out[name] = float(np.mean(f0 - f))
    rand = np.mean([v for k, v in out.items() if k.startswith("rand")])
    return f0, out["shap"], float(rand)


def pointing_hit(smap, gt_win):
    if gt_win is None or not gt_win.any():
        return np.nan
    blocks = block_grid(*smap.shape)
    sums = np.array([smap[a:b, c:d].sum() for a, b, c, d in blocks])
    a, b, c, d = blocks[int(np.argmax(sums))]
    return float(gt_win[a:b, c:d].any())


def marker_stats(A, smap, gt_win, mk_win):
    pos = np.clip(smap, 0, None); tot = pos.sum()
    out = dict(marker_mass=np.nan, marker_lift=np.nan, ring_mass=np.nan, ring_lift=np.nan, gt_mass=np.nan)
    if tot <= 0:
        return out
    import cv2
    if mk_win is not None and mk_win.any():
        mk = cv2.dilate(mk_win.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        out["marker_mass"] = float(pos[mk].sum() / tot); out["marker_lift"] = out["marker_mass"] / max(1e-9, mk.mean())
    if gt_win is not None and gt_win.any():
        g = gt_win.astype(np.uint8); k = np.ones((13, 13), np.uint8)
        ring = (cv2.dilate(g, k) > 0) & ~(cv2.erode(g, k) > 0)
        out["ring_mass"] = float(pos[ring].sum() / tot); out["ring_lift"] = out["ring_mass"] / max(1e-9, ring.mean())
        out["gt_mass"] = float(pos[gt_win].sum() / tot)
    return out


def select_inputs(A, n, max_mp, split="testing"):
    """TP / FN (sahte) ve TN / FP (authentic eş) gruplarından dengeli seçim — grup unet_bbox_s42 kararına göre."""
    idx = A.dataset_index()
    cand = idx[(idx.split == split) & (idx.megapixels.astype(float) <= max_mp)]
    cand = cand[[A.sample_exists(k) for k in cand.index]]
    ref_model = "PHASE2/unet_bbox_s42" if "PHASE2/unet_bbox_s42" in A._REG else next(n_ for n_ in A._REG if "unet" in n_)
    thr, _ = A.image_threshold(ref_model)
    items = []
    for k in cand.index[: max(4 * n, 24)]:
        f, m, a, meta = A.load_dataset_sample(k)
        sc = A.image_score(A.gpu_infer([ref_model], [], [f], 112)[0][ref_model][0])
        items.append(dict(key=k, kind="forged", rgb=f, mask=m, auth=a, group="TP" if sc >= thr else "FN", score=sc,
                          tool=meta.get("tool"), area=meta.get("area_bin"), marker=bool(meta.get("has_marker_index"))))
        if a is not None:
            sa = A.image_score(A.gpu_infer([ref_model], [], [a], 112)[0][ref_model][0])
            items.append(dict(key=k + ":auth", kind="authentic", rgb=a, mask=None, auth=None, group="FP" if sa >= thr else "TN",
                              score=sa, tool=meta.get("tool"), area="—", marker=False))
    by = {g: [it for it in items if it["group"] == g] for g in ("TP", "FN", "TN", "FP")}
    out, i = [], 0
    while len(out) < n and any(by.values()):
        g = ("TP", "FN", "TN", "FP")[i % 4]; i += 1
        if by[g]:
            out.append(by[g].pop(0))
    return out, ref_model, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-evals", type=int, default=1000)
    ap.add_argument("--status-evals", type=int, default=300)
    ap.add_argument("--seed-images", type=int, default=6)
    ap.add_argument("--max-mp", type=float, default=2.0)
    ap.add_argument("--models", default=",".join(CORE))
    ap.add_argument("--out", default=str(ROOT / "docs" / "xai_test_report.md"))
    ap.add_argument("--csv", default=str(ROOT / "docs" / "xai_benchmark.csv"))
    args = ap.parse_args()

    import app as A
    X = A.XAI
    t_all = time.time()
    cfg = X.ExplainConfig(method="partition", reference="auto", max_evals=args.max_evals, batch_size=16)
    rng = np.random.default_rng(0)
    log(f"cihaz {A.DEVICE} · {len(A.XAI_SPECS)} checkpoint · max_evals {args.max_evals}")

    inputs, ref_model, thr = select_inputs(A, args.n, args.max_mp)
    log(f"{len(inputs)} girdi seçildi: " + ", ".join(f"{it['key']}[{it['group']}]" for it in inputs))
    if not inputs:
        raise SystemExit("veri yok")

    # ---------------- A) durum
    status_rows = []
    probe = next((it for it in inputs if it["kind"] == "forged"), inputs[0])
    for name, sp in A.XAI_SPECS.items():
        t0 = time.time()
        try:
            res = A.gpu_infer([name] if sp.family == "unet" else [], [name] if sp.family == "resnet" else [], [probe["rgb"]], 112)[0]
            tg = X.decision_targets(res, A.XAI_SPECS, probe["rgb"].shape, A.XAI_DEPS)
            r = X.run_jobs(tg, probe["rgb"], probe["auth"], A.XAI_DEPS, A.XAI_SPECS,
                           X.ExplainConfig(max_evals=args.status_evals, batch_size=16))[0]
            status_rows.append(dict(model=sp.label, name=name, family=sp.family, in_ch=sp.in_ch, status=r.status,
                                    add_err=r.add_err, n_evals=r.n_evals, seconds=r.seconds, note=r.message))
        except Exception as e:
            status_rows.append(dict(model=sp.label, name=name, family=sp.family, in_ch=sp.in_ch, status="FAIL",
                                    note=f"{type(e).__name__}: {e}", seconds=time.time() - t0))
        log(f"durum {sp.label}: {status_rows[-1]['status']} ({status_rows[-1].get('seconds', 0):.1f}s)")
    for sp in X.UNSUPPORTED_MODELS:
        status_rows.append(dict(model=sp.label, name=sp.name, family=sp.family, in_ch=None, status=sp.status, note=sp.reason))

    # ---------------- B) çekirdek modeller
    core = [m for m in args.models.split(",") if m in A.XAI_SPECS]
    rows = []
    for it in inputs:
        res = A.gpu_infer([m for m in core if "unet" in m], [m for m in core if "resnet" in m], [it["rgb"]], 112)[0]
        tg = X.decision_targets(res, A.XAI_SPECS, it["rgb"].shape, A.XAI_DEPS)
        rs = X.run_jobs(tg, it["rgb"], it["auth"], A.XAI_DEPS, A.XAI_SPECS, cfg)
        for t, r in zip(tg, rs):
            row = dict(input=it["key"], kind=it["kind"], group=it["group"], tool=it["tool"], area=it["area"], marker=it["marker"],
                       model=r.label, status=r.status, reference=r.reference, fx=r.fx, base=r.base, add_err=r.add_err,
                       n_evals=r.n_evals, seconds=r.seconds, peak_mem_mb=r.peak_mem_mb)
            if r.ok:
                ad = X.adapter_for(A.XAI_SPECS[t.model], A.XAI_DEPS)
                img_p = X.prepare_for(t.prep, it["rgb"], A.XAI_DEPS)[0]
                win = img_p[t.y:t.y + t.h, t.x:t.x + t.w].copy()
                if it["auth"] is not None and r.reference == "authentic eş":
                    ref = X.prepare_for(t.prep, it["auth"], A.XAI_DEPS)[0][t.y:t.y + t.h, t.x:t.x + t.w]
                else:
                    import cv2
                    ref = cv2.blur(win, (32, 32))
                smap = r.shap_map()
                f0, aopc, aopc_r = deletion_aopc(ad, t, win, ref, smap, {"prepared": img_p}, rng=rng)
                gt_w = X.window_mask(it["mask"], t, A.XAI_DEPS) if it["mask"] is not None else None
                gm = A.green_marker_mask(it["rgb"])
                if it["auth"] is not None:
                    gm &= ~A.green_marker_mask(it["auth"])
                mk_w = X.window_mask(gm, t, A.XAI_DEPS)
                row.update(aopc_shap=aopc, aopc_random=aopc_r, pointing=pointing_hit(smap, gt_w),
                           **marker_stats(A, smap, gt_w, mk_w))
            rows.append(row)
            log(f"{it['key']}[{it['group']}] {r.label}: {r.status} {r.seconds:.1f}s add_err={r.add_err:.1e}")

    # ---------------- C) tohum kararlılığı
    seed_rows = []
    forged = [it for it in inputs if it["kind"] == "forged"][: args.seed_images]
    for fam, models in SEED_FAMILIES.items():
        models = [m for m in models if m in A.XAI_SPECS]
        if len(models) < 2:
            continue
        for it in forged:
            res = A.gpu_infer(models, [], [it["rgb"]], 112)[0]
            tg = X.common_window_targets(models, res, A.XAI_SPECS, it["rgb"].shape, it["mask"], A.XAI_DEPS)
            rs = X.run_jobs(tg, it["rgb"], it["auth"], A.XAI_DEPS, A.XAI_SPECS, cfg)
            cmp = X.compare.compare_results(rs)
            if "iou" in cmp:
                n = len(cmp["names"]); off = [(i, j) for i in range(n) for j in range(i + 1, n)]
                seed_rows.append(dict(family=fam, input=it["key"], iou_mean=float(np.nanmean([cmp["iou"][i, j] for i, j in off])),
                                      spearman_mean=float(np.nanmean([cmp["spearman"][i, j] for i, j in off]))))
                log(f"tohum {fam} {it['key']}: IoU {seed_rows[-1]['iou_mean']:.3f}")

    # ---------------- rapor
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df, st, sd = pd.DataFrame(rows), pd.DataFrame(status_rows), pd.DataFrame(seed_rows)
    df.to_csv(args.csv, index=False)
    st.to_csv(Path(args.csv).with_name("xai_status.csv"), index=False)
    if len(sd):
        sd.to_csv(Path(args.csv).with_name("xai_seed_stability.csv"), index=False)
    write_report(out, A, args, df, st, sd, inputs, ref_model, thr, time.time() - t_all)
    log(f"rapor → {out} ({time.time() - t_all:.0f} s)")


def _fmt(v, nd=3):
    try:
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def write_report(out, A, args, df, st, sd, inputs, ref_model, thr, total_s):
    L = ["# XAI (SHAP) test raporu", "",
         f"Üretildi: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} · `scripts/xai_benchmark.py` · cihaz **{A.DEVICE}** "
         f"(ZeroGPU={A.ZERO_GPU}) · toplam {total_s / 60:.1f} dk",
         f"Ayarlar: Partition, referans = authentic eş (varsa) yoksa blur(32,32), max_evals = {args.max_evals} "
         f"(durum taraması {args.status_evals}), {len(inputs)} girdi, grup modeli `{ref_model}` (eşik {thr:.3f}).", ""]
    if str(A.DEVICE) == "cpu":
        L += ["> **Not:** Bu çalıştırma CPU'da ve düşük bütçeyle yapıldı; süreler GPU'yu temsil etmez ve Partition haritaları "
              "daha kaba bölümlemelidir. GPU'da `python scripts/xai_benchmark.py --n 24` ile yeniden üretin.", ""]
    L += ["## A) Model durumu", "", "| Model | Ad | Aile | Kanal | Durum | Toplamsallık hatası | Değerlendirme | Süre s | Not |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in st.iterrows():
        L.append(f"| {r.model} | `{r['name']}` | {r.family} | {_fmt(r.get('in_ch'), 0)} | **{r.status}** | {_fmt(r.get('add_err'), 6)} | "
                 f"{_fmt(r.get('n_evals'), 0)} | {_fmt(r.get('seconds'), 1)} | {r.get('note') or ''} |")
    vc = st.status.value_counts().to_dict()
    L += ["", "Özet: " + " · ".join(f"{k} {v}" for k, v in vc.items()), ""]
    if len(df):
        ok = df[df.status == "OK"]
        L += ["## B) Çekirdek modeller — performans ve doğruluk", "",
              f"Girdiler: " + ", ".join(f"`{it['key']}` ({it['group']}, {it.get('tool')}, alan {it.get('area')})" for it in inputs), "",
              "| Model | n | Süre s (ort / medyan) | Değerl. | Tepe MB | Toplamsallık hatası (maks) | AOPC SHAP | AOPC rastgele | "
              "SHAP > rastgele | Pointing game | İşaretçi kütlesi (lift) | Kenar halkası kütlesi (lift) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, g in ok.groupby("model", sort=False):
            win = (g.aopc_shap > g.aopc_random).mean()
            L.append(f"| {m} | {len(g)} | {_fmt(g.seconds.mean(), 1)} / {_fmt(g.seconds.median(), 1)} | {_fmt(g.n_evals.mean(), 0)} | "
                     f"{_fmt(g.peak_mem_mb.mean(), 0)} | {_fmt(g.add_err.max(), 6)} | {_fmt(g.aopc_shap.mean())} | "
                     f"{_fmt(g.aopc_random.mean())} | %{100 * win:.0f} | {_fmt(g.pointing.mean(), 2)} (n={int(g.pointing.notna().sum())}) | "
                     f"{_fmt(g.marker_mass.mean())} ({_fmt(g.marker_lift.mean(), 1)}×) | {_fmt(g.ring_mass.mean())} ({_fmt(g.ring_lift.mean(), 1)}×) |")
        L += ["", "### Gruplara göre (TP / FN / TN / FP)", "", "| Model | Grup | n | f(x) ort. | AOPC SHAP | AOPC rastgele | Pointing |",
              "|---|---|---|---|---|---|---|"]
        for (m, gname), g in ok.groupby(["model", "group"], sort=False):
            L.append(f"| {m} | {gname} | {len(g)} | {_fmt(g.fx.mean())} | {_fmt(g.aopc_shap.mean())} | {_fmt(g.aopc_random.mean())} | "
                     f"{_fmt(g.pointing.mean(), 2)} |")
        fails = df[df.status != "OK"]
        if len(fails):
            L += ["", "Başarısız çalıştırmalar:"] + [f"- {r.model} · {r.input}: {r.status}" for _, r in fails.iterrows()]
    if len(sd):
        L += ["", "## C) Tohum kararlılığı (aynı pencere, top-%5 pozitif bölge)", "", "| Aile | n görüntü | IoU ort. | Spearman ort. |",
              "|---|---|---|---|"]
        for fam, g in sd.groupby("family"):
            L.append(f"| U-Net {fam} s42/43/44 | {len(g)} | {_fmt(g.iou_mean.mean())} | {_fmt(g.spearman_mean.mean())} |")
    L += ["", "## Tanımlar", "",
          "- **Toplamsallık hatası** = |E[f] + ΣSHAP − f(x)| (Partition'da kesin olmalı; ≈ 0).",
          "- **AOPC** (silme): pencere 16×16 bloğa bölünür; bloklar SHAP toplamına göre azalan sırada %10'luk adımlarla referansla "
          "değiştirilir; f(x) − f(adım) ortalaması. Rastgele sıra (2 permütasyon) ile karşılaştırılır; SHAP > rastgele = sadakat.",
          "- **Pointing game**: en yüksek SHAP bloğu GT kutusuna değiyorsa isabet (yalnız GT'li sahte girdiler).",
          "- **İşaretçi kütlesi**: pozitif SHAP kütlesinin yeşil işaretçi piksellerindeki (±1 px) payı; *lift* = bu pay / alan payı. "
          "**Kenar halkası**: GT kutusu kenarı ±6 px. Lift ≫ 1 → modelin kararı işaretçi kısayoluna dayanıyor (PHASE 2 Ek-C).",
          "- Gruplar `unet_bbox_s42` görüntü-düzeyi kararına göre: TP/FN sahte girdiler, TN/FP authentic eşler.",
          "- Ham sonuçlar: `docs/xai_benchmark.csv`, `docs/xai_status.csv`, `docs/xai_seed_stability.csv`."]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
