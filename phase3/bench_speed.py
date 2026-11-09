"""bench_speed.py — Maliyet tablosu için GPU boşken tam-görüntü çıkarım süresi ölçümü (TR).
Aynı 100 test görüntüsü (sabit tohum), görüntü başına decode HARİÇ, torch.cuda.synchronize ile; ısınma 5 görüntü.
U-Net/DeepLabV3+: 512/448 kayan pencere (Faz 2 protokolü). YOLO: 1024/768 kare + NMS (phase3_yolo/infer_to_maps).
Çıktı: phase3/results/speed_bench.csv"""
import sys, time, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3ext as X  # noqa: E402
import numpy as np, pandas as pd, torch
D, I = X.D, X.I
unets = [m for m in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if m]
yolos = [m for m in (sys.argv[2] if len(sys.argv) > 2 else "").split(",") if m]
dev = torch.device("cuda")
ids = list(D.split_table("test").index)
rng = np.random.default_rng(0); ids = list(rng.choice(ids, 105, replace=False))
imgs = [D.read_rgb(D.img_path(i, "test")) for i in ids]
mp = np.array([im.shape[0] * im.shape[1] / 1e6 for im in imgs[5:]])
rows = []
for m in unets:
    net = I.load_model(m, dev)
    for k, im in enumerate(imgs):
        if k == 5: torch.cuda.synchronize(); t0 = time.perf_counter()
        I.predict_full_multi([net], im, dev)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / (len(imgs) - 5)
    rows.append(dict(model=m, family="seg", ms_per_img=dt * 1000, params_M=sum(p.numel() for p in net.parameters()) / 1e6))
    del net; torch.cuda.empty_cache()
if yolos:
    sys.path.insert(0, str(X.HERE.parent / "phase3_yolo"))
    import infer_to_maps as Y
    from ultralytics import YOLO
    a = types.SimpleNamespace(tile=1024, stride=768, full=0, conf=0.001, iou=0.6, batch=8)
    for r in yolos:
        ym = YOLO(str(X.HERE.parent / "phase3_yolo" / "runs" / r / "weights" / "best.pt"))
        a.full = 1280 if "-full-" in r else 0
        for k, im in enumerate(imgs):
            if k == 5: torch.cuda.synchronize(); t0 = time.perf_counter()
            Y.detect_full(ym, im, a)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / (len(imgs) - 5)
        rows.append(dict(model=r, family="yolo", ms_per_img=dt * 1000, params_M=sum(p.numel() for p in ym.model.parameters()) / 1e6))
df = pd.DataFrame(rows); df["mean_MP"] = mp.mean(); df["gpu"] = torch.cuda.get_device_name(0)
df.to_csv(X.HERE / "results" / "speed_bench.csv", index=False)
print(df.round(2).to_string(index=False))
