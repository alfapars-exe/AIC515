"""Faz 3 entegrasyon testi (YOLO11, DeepLabV3+, birleşik/işaretçisiz U-Net, yeni senaryolar, bozulmalar, sonuçlar).
Kullanım (proje kökünde, GPU'lu ortam): python tests/test_app_p3.py
Kontrol: uygulama içi çıkarım, boru hattının kaydettiği val haritalarıyla (phase2/experiments/preds/<model>/val/<id>.png) aynı mı?"""
import os, sys, time, json
os.environ.setdefault("APP_NO_BROWSER", "1")
import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
sys.path.insert(0, os.environ.get("AI515_APP_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))  # Space paketi: AI515_APP_DIR=hf_space
import numpy as np, cv2
t0 = time.time()
import app as A
print(f"import {time.time()-t0:.1f}s · modeller {len(A._REG)} · yükleme hatası {A._LOAD_ERR} · cihaz {A.DEVICE}")
ERR = []
def check(c, m):
    print(("  ✓ " if c else "  ✗ ") + m)
    if not c: ERR.append(m)

names = A.list_unet_ckpts()
for n in names: print("   ", n, "→", A.friendly(n))
P3 = ["PHASE2/dlv3p_bbox_s42", "PHASE2/dlv3p_pixel_eps8_morph_s42", "PHASE2/unetjoint_pixel_eps8_morph_s42",
      "PHASE2/unet_bbox_mr_s42", "YOLO/aiforge-tiles-yolo11s-img1024-conserv-s42", "YOLO/aiforge-full-yolo11s-img1280-conserv-s42",
      "YOLO/aiforge-tiles-yolo11n-img1024-conserv-s42", "YOLO/aiforge-tiles-yolo11m-img1024-conserv-s42",
      "YOLO/aiforge-tiles_geminionly-yolo11s-img1024-conserv-s42"]
for n in P3: check(n in names and n in A._REG, f"listede ve yüklü: {n}")
check(not any(n.endswith(("_e20", "_stem", "-s43", "-s44")) for n in names), "gizli modeller listede yok")
check(all(not n.startswith("YOLO/") for n in A.list_seg_ckpts()), "XAI listesi YOLO içermiyor")

idx = A.dataset_index()
ids = ["000000012", "000000019", "000000021"]
keys = [k for k in idx.index if k.startswith("training:") and k.split(":")[1] in ids and A.sample_exists(k)]
check(len(keys) == 3, f"örnekler bulundu: {keys}")
from pathlib import Path
PRED = Path(os.environ.get("AI515_PRED_ROOT") or (A.ROOT / "phase2" / "experiments" / "preds"))
rows = []
for k in keys:
    forged, mask, auth, meta = A.load_dataset_sample(k)
    nid = k.split(":")[1]
    for n in P3:
        prob, info = A._predict_unet(n, forged)
        stem = n.split("/", 1)[1]
        ref = cv2.imread(str(PRED / stem / "val" / f"{nid}.png"), cv2.IMREAD_GRAYSCALE)
        s_app = A.image_score(prob)
        if ref is not None:
            d = np.abs(A.q_u8(prob).astype(int) - ref.astype(int))
            s_ref = float(np.sort(ref.ravel())[-max(1, int(0.01 * ref.size)):].mean() / 255)
            rows.append((k, stem, round(s_app, 4), round(s_ref, 4), int(d.max()), round(float((d > 2).mean()), 5), round(info["sec"], 2)))
            # YOLO: aynı kutular → aynı harita (fp16/NMS sınır durumları için küçük tolerans); segmenter: bf16 vs kayıt hassasiyeti
            check(abs(s_app - s_ref) < 0.05 and (d > 2).mean() < 0.01, f"{k} {stem}: skor {s_app:.3f} vs kayıt {s_ref:.3f}, |Δ|>2 oranı {(d>2).mean():.4f}")
        else:
            print("    (kayıtlı val haritası yok)", stem, nid)
        if n.startswith("YOLO/"):
            check(info["n_boxes"] >= 1 and len(info["boxes"]) >= 1, f"{stem}: kutu sayısı {info['n_boxes']}, ilk {info['boxes'][:2]}")

forged, mask, auth, meta = A.load_dataset_sample(keys[0])
for it in A.INPUT_TYPES:
    img, lab, note = A.build_input(forged, auth, it)
    check(img.shape == forged.shape and img.dtype == np.uint8, f"senaryo: {it} → etiket {lab} · {note[:70]}")
r = A.build_input(forged, auth, A.INPUT_TYPES[4])[0]; c = A.build_input(forged, auth, A.INPUT_TYPES[5])[0]
check(int((r != forged).any(2).sum()) > 0 and int((c != auth).any(2).sum()) > 0, "render-only/content-only girdileri kaynaktan farklı")
for dg in A.DEGRADATIONS:
    o = A.degrade(forged, dg)
    check(o.shape == forged.shape and o.dtype == np.uint8, f"bozulma: {dg}")
g = A.degrade(forged, A.DEGRADATIONS[-1]); check(bool((g[..., 0] == g[..., 1]).all()), "gri seviye kanalları eşit")
md = A.results_markdown()
check("Faz 3" in md or "Phase 3" in md or "YOLO" in md, f"sonuçlar sekmesi Faz 3 tablolarını içeriyor ({len(md)} karakter)")
for n in ("YOLO/aiforge-tiles-yolo11s-img1024-conserv-s42", "PHASE2/dlv3p_bbox_s42"):
    thr, ok = A.image_threshold(n); check(ok, f"görüntü eşiği {n}: {thr}")
    check(A.default_tau(n, "bbox") != 0.5 or n.startswith("PHASE2/dlv3p"), f"τ* {n}: {A.default_tau(n, 'bbox')}")

# uçtan uca simülasyon (generator) — YOLO + DLV3+ ile, yeni senaryo ve bozulmayla
last, nstep = None, 0
for out in A.simulate(keys[0], A.INPUT_TYPES[4], A.DEGRADATIONS[-2], False, [], ["YOLO/aiforge-tiles-yolo11s-img1024-conserv-s42", "PHASE2/dlv3p_bbox_s42"], A._DEF_RESNET, 112):
    last = out; nstep += 1
check(nstep > 1 and last is not None, f"simulate() {nstep} adım tamamlandı")
import pandas as pd
print(pd.DataFrame(rows, columns=["key", "model", "skor_app", "skor_kayıt", "maxΔ", "|Δ|>2", "sn"]).to_string(index=False))
json.dump(dict(errors=ERR, rows=rows, models=names), open(os.path.join(str(_out_dir(__file__)), os.environ.get("AI515_TEST_TAG", "test_p3_result") + ".json"), "w"), ensure_ascii=False, indent=1)
print(f"\nSONUÇ: {len(ERR)} hata · {time.time()-t0:.0f}s"); sys.exit(1 if ERR else 0)
