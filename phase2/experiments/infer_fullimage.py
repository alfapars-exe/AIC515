"""
infer_fullimage.py — Tam-görüntü sliding-window çıkarım (TR).

  python infer_fullimage.py --models unet_bbox_s42,unet_pixel_eps8_morph_s42 --split val|test [--condition clean|jpeg75|...]
                            [--source forged|benign_pasteback|fullimage_resample|authentic] [--save-preds]

Pencere 512, stride 448, örtüşmede ortalama, bf16. En uzun kenar > 4096 ise 4096'ya küçültülüp çıkarım yapılır,
olasılık haritası orijinal boyuta geri büyütülür (bilinear) ve resized=1 olarak raporlanır.
Görüntü bir kez decode edilir; tüm modeller (ve koşul) aynı decode üzerinde çalışır.
Çıktı: preds/<model>/<split>[_<condition>|_<source>]/<id>.png (uint8 0-255) ve infer_log CSV (süre, pencere sayısı, resized).
Kütüphane olarak da kullanılır: predict_full(model, img_u8) → prob float32 HxW.
"""
from __future__ import annotations
import os, sys, time, argparse, math
os.environ.setdefault("USERNAME", "user")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2
import torch
import segmentation_models_pytorch as smp

import common.p2data as D

WIN, STRIDE, MAX_SIDE = 512, 448, 4096


def load_model(name: str, device) -> torch.nn.Module:
    m = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
    sd = torch.load(D.MODELS_DIR / f"{name}.pt", map_location="cpu")
    m.load_state_dict(sd)
    return m.to(device).eval().to(memory_format=torch.channels_last)


def _windows(H: int, W: int, win: int = WIN, stride: int = STRIDE):
    ys = list(range(0, max(H - win, 0) + 1, stride))
    xs = list(range(0, max(W - win, 0) + 1, stride))
    if ys[-1] + win < H: ys.append(H - win)
    if xs[-1] + win < W: xs.append(W - win)
    return [(y, x) for y in ys for x in xs]


@torch.no_grad()
def predict_full_multi(models: list, img_u8: np.ndarray, device, batch: int = 32, win: int = WIN, stride: int = STRIDE,
                       max_side: int = MAX_SIDE):
    """Birden çok model için aynı pencerelerle tam-görüntü olasılık haritaları. Dönüş: (list[prob float32 HxW], info)."""
    H0, W0 = img_u8.shape[:2]
    resized = max(H0, W0) > max_side
    if resized:
        s = max_side / max(H0, W0)
        img = cv2.resize(img_u8, (max(1, int(round(W0 * s))), max(1, int(round(H0 * s)))), interpolation=cv2.INTER_AREA)
    else:
        img = img_u8
    img = D.pad_to_min(img, win)
    H, W = img.shape[:2]
    x = torch.from_numpy(D.norm_chw(img.astype(np.float32) / 255.0)).to(device)  # (3,H,W)
    wins = _windows(H, W, win, stride)
    acc = [torch.zeros((H, W), dtype=torch.float32, device=device) for _ in models]
    cnt = torch.zeros((H, W), dtype=torch.float32, device=device)
    for i in range(0, len(wins), batch):
        chunk = wins[i:i + batch]
        xb = torch.stack([x[:, y:y + win, xx:xx + win] for (y, xx) in chunk]).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = [torch.sigmoid(m(xb).float())[:, 0] for m in models]
        for j, (y, xx) in enumerate(chunk):
            cnt[y:y + win, xx:xx + win] += 1
            for k in range(len(models)):
                acc[k][y:y + win, xx:xx + win] += outs[k][j]
    probs = []
    for k in range(len(models)):
        p = (acc[k] / cnt).cpu().numpy()
        p = p[:img.shape[0], :img.shape[1]]
        if resized:
            p = cv2.resize(p, (W0, H0), interpolation=cv2.INTER_LINEAR)
        else:
            p = p[:H0, :W0]
        probs.append(p.astype(np.float32))
    return probs, dict(n_windows=len(wins), resized=int(resized), H=H0, W=W0)


def to_u8(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.round(prob * 255), 0, 255).astype(np.uint8)


def write_png(path, arr_u8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(str(path), arr_u8, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def source_image_path(new_id: str, split: str, source: str, tab: pd.DataFrame):
    if source == "forged":
        return D.img_path(new_id, split)
    if source == "authentic":
        return D.auth_path(tab.loc[new_id, "pair_name"], split)
    return D.control_img_path(source, new_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="virgülle ayrılmış checkpoint adları (models/phase2/<name>.pt)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--condition", default="clean", choices=["clean"] + list(D.DEGRADATIONS))
    ap.add_argument("--source", default="forged", choices=["forged", "authentic", "benign_pasteback", "fullimage_resample", "reencode_png"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    device = torch.device("cuda")
    names = args.models.split(",")
    models = [load_model(n, device) for n in names]
    tab = D.split_table(args.split)
    ids = list(tab.index)[: args.limit or None]
    suffix = args.split + ("" if args.condition == "clean" else f"_{args.condition}") + ("" if args.source == "forged" else f"_{args.source}")
    out_dirs = {n: D.PRED_DIR / n / suffix for n in names}
    deg = D.DEGRADATIONS.get(args.condition)
    pool = ThreadPoolExecutor(args.threads)
    t0 = time.time(); rows = []
    # prefetch decode
    def _load(i):
        p = source_image_path(i, args.split, args.source, tab)
        img = D.read_rgb(p)
        if deg is not None:
            img = deg(img)
        return img
    futs = {}
    look = 6
    for k in range(min(look, len(ids))):
        futs[k] = pool.submit(_load, k)
    for k, new_id in enumerate(ids):
        img = futs.pop(k).result()
        if k + look < len(ids):
            futs[k + look] = pool.submit(_load, k + look)
        t = time.time()
        probs, info = predict_full_multi(models, img, device)
        tg = time.time() - t
        for n, p in zip(names, probs):
            pool.submit(write_png, out_dirs[n] / f"{new_id}.png", to_u8(p))
        rows.append(dict(new_id=new_id, split=args.split, condition=args.condition, source=args.source, gpu_s=tg, **info))
        if (k + 1) % 50 == 0:
            print(f"[{suffix}] {k + 1}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    pool.shutdown(wait=True)
    log = pd.DataFrame(rows)
    D.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.to_csv(D.LOG_DIR / f"infer_{suffix}_{'+'.join(names)[:80]}.csv", index=False)
    print(f"done {len(ids)} images, {len(names)} models, {time.time() - t0:.0f}s, resized={int(log.resized.sum())}, "
          f"windows/img mean {log.n_windows.mean():.1f}", flush=True)


if __name__ == "__main__":
    main()
