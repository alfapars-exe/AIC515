"""
tools/xai_fullimage_bench.py — Tam görüntü SHAP + döşemeli Kernel aktivasyonu: kapsama, koordinat, toplamsallık, dikiş,
süre ve GPU belleği ölçümü (farklı çözünürlüklerde gerçek test görüntüleri).

  python tools\\xai_fullimage_bench.py [--ev 500] [--n 3]
Çıktı: reports/xai_fullimage/{bench.json, RAPOR_OLCUM.md}  (veri setinden türetilmiş görüntü yazılmaz — lisans)
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from pathlib import Path

os.environ["APP_NO_BROWSER"] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cv2
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--ev", type=int, default=500)
ap.add_argument("--n", type=int, default=3)
args = ap.parse_args()

import app as A
import xai_filters as XF
X, F = A.XAI, A.XAI.fullimage
OUT = ROOT / "reports" / "xai_fullimage"; OUT.mkdir(parents=True, exist_ok=True)

# ---- farklı çözünürlükte test görüntüleri (küçük / orta / büyük)
ims = sorted(glob.glob(str(ROOT / "data" / "AIForge-Doc-v1" / "TestingSet" / "images" / "*.png")))[:400]
sz = []
for p in ims:
    with Image.open(p) as im:
        sz.append((im.size[0] * im.size[1], Path(p).stem))
sz = sorted(s for s in sz if s[0] <= 6e6)   # ≤ 6 MP: 5 kanallı U-Net'in tam görüntü bütçesi makul sürede kalsın
picks = [sz[int(q * (len(sz) - 1))][1] for q in np.linspace(0.05, 0.95, args.n)]
yolo = [n for n in A.list_unet_ckpts() if n.startswith("YOLO/")]
yolo = [n for n in yolo if "-full-" in n][:1] + [n for n in yolo if "-tiles-" in n][:1]
MODELS = ["PHASE2/unet_bbox_s42", "PHASE1/unet_rgbfreq_best", "PHASE1/resnet18_ft_best", "PHASE2/dlv3p_bbox_s42"] + yolo
cfg = X.ExplainConfig(method="partition", reference="auto", max_evals=args.ev, batch_size=32)
rows, krows = [], []


def seam_bound(m, ramp=64, win=512):
    """Döşemeli mozaikte birleştirme ağırlıkları süreklidir: bir pencere kendi kenarında en fazla w_e ağırlıkla katkı verir
    (kosinüs rampası, w_e ≈ 1.5e-4). Kenar çizgisinde birleştirmeden doğabilecek sıçrama ≤ w_e · 2·max|SHAP|. (Haritadaki
    diğer sıçramalar Partition'ın blok sınırlarıdır ve her pencerenin kendi haritasında da bulunur.)"""
    w = F._cosine_weight(win, win, ramp)
    we = float(w[0, win // 2] / w[win // 2, win // 2])
    return we, float(2 * we * np.abs(m).max())


CASES = [(f"testing:{s}", 1.0) for s in picks] + [(f"testing:{picks[len(picks) // 2]}", 2.0)]
for key, up in CASES:
    img, mask, auth, meta = A.load_dataset_sample(key)
    if up != 1.0:  # çözünürlük ölçeklenebilirliği: aynı görüntü 2× (bicubic) — yalnız süre/bellek/kapsama için
        img = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        mask = None if mask is None else cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        auth = None if auth is None else cv2.resize(auth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        key = key + " (2× büyütülmüş)"
    H0, W0 = img.shape[:2]
    print(f"\n### {key} {W0}×{H0} ({W0 * H0 / 1e6:.2f} MP) auth={'var' if auth is not None else 'yok'}", flush=True)
    names = [m for m in MODELS if m in A.XAI_SPECS or m.startswith("YOLO/")]
    if up != 1.0:  # büyütülmüş görüntüde süre ölçeklenmesi için 3 kanallı U-Net, ResNet ve tam görüntü YOLO yeterli
        names = [m for m in names if m in ("PHASE2/unet_bbox_s42", "PHASE1/resnet18_ft_best") or "-full-" in m]
    res = A.gpu_infer([m for m in names if "resnet" not in m], [m for m in names if "resnet" in m], [img], 112)[0]
    preds = {n: res[n] for n in names}
    jobs = F.full_decision_targets(preds, A.XAI_SPECS, img.shape, A.XAI_DEPS)
    jobs += F.tile_targets(["PHASE2/unet_bbox_s42"], A.XAI_SPECS, img.shape, A.XAI_DEPS)
    for t in jobs:
        if A.DEVICE is not None and A.DEVICE.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        r = A._gpu_explain(img, auth, [t], cfg)[0]
        dt = time.time() - t0
        if r.status != "OK":
            print("  !!", t.model, r.status, r.message); rows.append(dict(image=key, model=t.model, status=r.status, msg=r.message)); continue
        m = r.shap_map()
        app_sc = A._xai_app_score(dict(preds=preds), t.model)
        if t.model.startswith("YOLO/"):  # hedef = kutu güveni → uygulamanın tespit listesindeki en yüksek kutu güveni
            bx = preds[t.model][1].get("boxes") or []
            app_sc = bx[0][0] if bx else None
        g = r.target.get("grid") or [0, 0]
        cell = r.target.get("cell_px") or 0
        row = dict(image=key, size=f"{W0}x{H0}", mp=round(W0 * H0 / 1e6, 2), model=t.model, kind=t.kind,
                   map_shape_eq_image=bool(m.shape == (H0, W0)), finite=bool(np.isfinite(m).all()),
                   grid=f"{g[1]}x{g[0]}", cell_px=cell,
                   grid_covers=bool(t.kind == F.TILE_KIND or (g[0] * cell * (1 / max(t.scale, 1e-9)) >= H0 - cell / t.scale - 1 and g[1] * cell / t.scale >= W0 - cell / t.scale - 1)),
                   fx=round(r.fx, 4), app_score=None if app_sc is None else round(float(app_sc), 4), base=round(r.base, 4),
                   sum_shap=round(r.sum_shap, 4), add_err=float(r.add_err), evals=r.n_evals, sec=round(dt, 1),
                   peak_gpu_mb=None if r.peak_mem_mb is None else round(r.peak_mem_mb), reference=r.reference,
                   pos_mass_in_gt=None)
        if mask is not None and (mask > 127).any():
            pos = np.clip(m, 0, None); g_ = mask > 127
            row["pos_mass_in_gt"] = round(float(pos[g_].sum() / (pos.sum() + 1e-12)), 3)
            row["gt_area_frac"] = round(float(g_.mean()), 4)
        if t.kind == F.TILE_KIND:
            row["tiles"] = len(r.target["tiles"]); row["covered"] = r.target["covered"]
            row["base_spread"] = round(r.target["base_spread"], 4)
            row["edge_weight"], row["seam_bound"] = seam_bound(m)
            row["max_abs_shap"] = float(np.abs(m).max())
        rows.append(row)
        print(f"  {t.model:46s} {t.kind:16s} map==img {row['map_shape_eq_image']} f(x)={r.fx:.4f} app={row['app_score']} "
              f"E[f]={r.base:.4f} ΣSHAP={r.sum_shap:+.4f} err={r.add_err:.1e} {dt:.1f}s peak={row['peak_gpu_mb']} MB "
              f"GT-kütle={row['pos_mass_in_gt']}" + (f" dikiş üst sınırı={row.get('seam_bound'):.1e}" if t.kind == F.TILE_KIND else ""), flush=True)
    # ---- Kernel: tam görüntü natif döşemeli aktivasyon (GPU) + tek geçiş eşitliği
    for k in ("PHASE2/unet_bbox_s42", "PHASE1/resnet18_ft_best") + tuple(yolo[:1]):
        specs = {s.key: s for s in XF.discover_models(ROOT / "models", ROOT / "phase3_yolo" / "runs")}
        if k not in specs:
            continue
        an = XF.run_analysis(specs[k], img, mask, XF.MODE_FULL, 512, "otomatik", 6, A.project_freq_channels, None)
        lm = XF.load_model(specs[k])
        x, _ = XF.preprocess(img, lm.preprocess, A.project_freq_channels)
        # döşeme etkisini ayırmak için aynı cihazda tek geçiş (TF32/cuDNN farkı ayrıca: GPU ↔ CPU)
        ref, _, _, _ = XF.analyze(lm, x, (H0, W0), an.device, 6, 4.0, np.float32)
        refc, _, _, _ = XF.analyze(lm, x, (H0, W0), "cpu", 6, 4.0, np.float32)
        rel = lambda a, b: float(np.abs(a.astype(np.float32) - b).max() / (np.abs(b).max() + 1e-12))
        d = dc = float("nan")
        if an.store_factor == 1:
            d = max(rel(L.stages[s], R.stages[s]) for L, R in zip(an.layers, ref) for s in R.stages)
            dc = max(rel(L.stages[s], R.stages[s]) for L, R in zip(an.layers, refc) for s in R.stages)
        cov = all(L.stages[s].shape[1:] == (-(-(-(-H0 // int(L.out_stride))) // an.store_factor),
                                             -(-(-(-W0 // int(L.out_stride))) // an.store_factor)) for L in an.layers for s in L.stages)
        krows.append(dict(image=key, size=f"{W0}x{H0}", model=k, device=an.device, tiles=an.n_tiles, store_factor=an.store_factor,
                          coverage=bool(cov), rel_vs_single_pass_same_device=d, rel_vs_cpu=dc, sec=round(an.sec, 2), peak_gpu_mb=round(an.peak_gpu_mb),
                          stored_mb=round(an.stored_mb), layers=len(an.layers), dtype=an.store_dtype))
        print(f"  KERNEL {k:40s} {an.device} {an.n_tiles} döşeme f={an.store_factor} kapsama {cov} Δ(tek geçiş, aynı cihaz)={d:.1e} Δ(CPU)={dc:.1e} "
              f"{an.sec:.2f}s tepe {an.peak_gpu_mb:.0f} MB saklanan {an.stored_mb:.0f} MB", flush=True)
        XF._RESULTS.clear()

js = dict(ev=args.ev, device=str(A.DEVICE), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          shap=rows, kernel=krows, date=time.strftime("%Y-%m-%d %H:%M"))
(OUT / "bench.json").write_text(json.dumps(js, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

L = [f"# Tam görüntü XAI ölçümleri ({js['date']}, {js['gpu'] or 'CPU'}, max_evals={args.ev})", "",
     "## SHAP (tam görüntü karar hedefi + döşemeli mozaik)", "",
     "| görüntü | boyut | model | hedef | harita = görüntü | f(x) | uygulama skoru | E[f] | ΣSHAP | toplamsallık hatası | süre (s) | tepe GPU (MB) | GT içi + kütle | not |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if "kind" not in r:
        L.append(f"| {r['image']} | | {r['model']} | | | | | | | | | | | {r['status']}: {r['msg']} |"); continue
    note = (f"{r['tiles']} pencere · kapsama {'tam' if r['covered'] else 'eksik'} · taban farkı {r['base_spread']}"
            + (f" · birleştirme dikişi ≤ {r['seam_bound']:.1e} (kenar ağırlığı {r['edge_weight']:.1e})" if r.get("seam_bound") is not None else "")
            if r["kind"] == F.TILE_KIND else
            (f"ızgara {r['grid']} ({r['cell_px']} px)" if r["kind"] != F.TILE_KIND else ""))
    L.append(f"| {r['image']} | {r['size']} | {r['model']} | {r['kind']} | {'✓' if r['map_shape_eq_image'] else '✗'} | {r['fx']} | "
             f"{r['app_score']} | {r['base']} | {r['sum_shap']} | {r['add_err']:.1e} | {r['sec']} | {r['peak_gpu_mb']} | "
             f"{r.get('pos_mass_in_gt')} | {note} |")
L += ["", "Not: tepe GPU belleği uygulamanın önceden GPU'ya yüklediği tüm modelleri (34 checkpoint) içerir; açıklamanın kendi payı "
      "bunun küçük bir kısmıdır. 'uygulama skoru' tahmin tablosundaki değerdir; f(x) aynı hedefin SHAP değerlendiricisindeki değeridir "
      "(fark yalnız toplu/karışık duyarlıklı çıkarımdan kaynaklanan sayısal gürültü). Döşemeli mozaikte f(x)/E[f] pencere ortalamasıdır.",
      "", "## Kernel — tam görüntü natif döşemeli aktivasyon", "",
      "| görüntü | boyut | model | cihaz | döşeme | saklama ölçeği | kapsama | göreli Δ döşemeli ↔ tek geçiş (aynı cihaz) | göreli Δ GPU ↔ CPU (TF32) | süre (s) | tepe GPU (MB) | saklanan (MB) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in krows:
    L.append(f"| {r['image']} | {r['size']} | {r['model']} | {r['device']} | {r['tiles']} | {r['store_factor']} | {'✓' if r['coverage'] else '✗'} | "
             f"{r['rel_vs_single_pass_same_device']:.1e} | {r['rel_vs_cpu']:.1e} | {r['sec']} | {r['peak_gpu_mb']} | {r['stored_mb']} |")
(OUT / "RAPOR_OLCUM.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n→", OUT / "RAPOR_OLCUM.md")
