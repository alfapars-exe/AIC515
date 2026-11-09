"""check_repro.py — Ortam değişikliği (smp 0.5→0.6, yeniden kurulan torch) sonrası yeniden üretim kontrolü (TR).
Faz 2 checkpoint'leriyle val'in ilk N görüntüsünde tam-görüntü çıkarım yapar ve kayıtlı preds/<m>/val/<id>.png ile karşılaştırır.
Kabul: uint8 haritalarda ortalama |Δ| ≤ 1 ve piksel-F1@0.5 farkı ≤ 0.005."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3ext as X  # noqa: E402
import numpy as np, cv2, torch
D, I = X.D, X.I
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
names = ["unet_bbox_s42", "unet_pixel_eps8_morph_s42", "unet_glyph_v2_s42", "unet_bbox_geminionly_s42"]
dev = torch.device("cuda")
nets = [I.load_model(n, dev) for n in names]
ids = list(D.split_table("val").index)[:N]
out = {}
for n in names:
    out[n] = []
for i in ids:
    img = D.read_rgb(D.img_path(i, "val"))
    probs, _ = I.predict_full_multi(nets, img, dev)
    gt = D.read_mask(D.bbox_mask_path(i, "val")) > 0
    for n, p in zip(names, probs):
        new = I.to_u8(p); old = cv2.imread(str(D.PRED_DIR / n / "val" / f"{i}.png"), 0)
        f = lambda m: 2 * ((m >= 128) & gt).sum() / max((m >= 128).sum() + gt.sum(), 1)
        out[n].append(dict(mad=float(np.abs(new.astype(int) - old.astype(int)).mean()), dF1=float(f(new) - f(old))))
rep = {n: dict(mean_abs_diff=round(np.mean([r["mad"] for r in v]), 4), max_abs_dF1=round(max(abs(r["dF1"]) for r in v), 5)) for n, v in out.items()}
ok = all(r["mean_abs_diff"] <= 1.0 and r["max_abs_dF1"] <= 0.005 for r in rep.values())
rep["PASS"] = ok
print(json.dumps(rep, indent=1))
(X.HERE / "results").mkdir(exist_ok=True)
json.dump(rep, open(X.HERE / "results" / "repro_check_env.json", "w"), indent=1)
