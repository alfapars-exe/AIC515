"""appcore.metrics — piksel ve görüntü düzeyi ölçütler, son işleme, bindirme, eşikler (tau.json, image_tau.json)."""
from __future__ import annotations
from ._common import *  # noqa: F401,F403 — ortak içe aktarmalar
from .config import (  # noqa: F401
    META, TAU_JSON
)
from .imaging import (  # noqa: F401
    odd, q_u8, to_rgb
)

def hist_metrics(prob: np.ndarray, gt: np.ndarray, thr: float):
    """Boru hattıyla aynı: olasılıklar 8-bit'e nicemlenir (PNG tahmin), 256 kutulu histogramla TÜM pikseller üzerinde
    kesin AP; F1/IoU eşik τ'da (u8/255 ≥ τ); Oracle-F1 = görüntü başına en iyi eşik."""
    u = q_u8(prob); g = gt.astype(bool)
    hp = np.bincount(u[g], minlength=256)[:256].astype(np.float64)
    hn = np.bincount(u[~g], minlength=256)[:256].astype(np.float64)
    P = hp.sum()
    tp = np.cumsum(hp[::-1])[::-1]; fp = np.cumsum(hn[::-1])[::-1]  # tp[t] = #pos(u ≥ t)
    TPd, FPd = tp[::-1], fp[::-1]                                   # eşik azalan sırada
    prec = TPd / np.maximum(TPd + FPd, 1); rec = TPd / max(P, 1)
    ap = float(np.sum((rec - np.r_[0, rec[:-1]]) * prec)) if P > 0 else float("nan")
    k = int(min(255, max(0, math.ceil(thr * 255 - 1e-9))))
    TP, FP = tp[k], fp[k]; FN = P - TP
    f1 = 2 * TP / max(2 * TP + FP + FN, 1) if P > 0 else float("nan")
    iou = TP / max(TP + FP + FN, 1) if P > 0 else float("nan")
    f1all = 2 * tp / np.maximum(2 * tp + fp + (P - tp), 1)
    return dict(F1=float(f1), IoU=float(iou), AP=ap, OracleF1=float(f1all.max()) if P > 0 else float("nan"),
                TP=int(TP), FP=int(FP), FN=int(FN), P=int(P))


def image_score(prob):
    u = q_u8(prob).ravel()
    k = max(1, int(0.01 * u.size))
    return float(np.partition(u, -k)[-k:].mean() / 255.0)


def rectangularity(binm: np.ndarray):
    ys, xs = np.where(binm)
    if len(xs) == 0:
        return float("nan"), None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return float(binm.sum() / ((x1 - x0) * (y1 - y0))), (int(x0), int(y0), int(x1), int(y1))


def postprocess(prob: np.ndarray, thr: float, morph_k: int, use_otsu: bool):
    if use_otsu:
        T, b = cv2.threshold(q_u8(prob), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binm = b > 0; thr_used = T / 255.0
    else:
        binm = q_u8(prob) >= int(min(255, max(0, math.ceil(thr * 255 - 1e-9)))); thr_used = thr
    if morph_k and morph_k >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (odd(morph_k), odd(morph_k)))
        b = cv2.morphologyEx(binm.astype(np.uint8), cv2.MORPH_OPEN, k)
        binm = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k) > 0
    return binm, thr_used


def heatmap_rgb(prob: np.ndarray):
    return cv2.cvtColor(cv2.applyColorMap(q_u8(prob), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def overlay(rgb_u8: np.ndarray, binm, gt=None, bbox=None, alpha=0.45, gt_color=(0, 160, 255)):
    out = to_rgb(rgb_u8).copy()
    if gt is not None:
        g = gt.astype(bool)
        out[g] = (out[g] * 0.65 + np.array(gt_color) * 0.35).astype(np.uint8)
        ys, xs = np.where(g)
        if len(xs):
            cv2.rectangle(out, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), gt_color, max(1, out.shape[1] // 400))
    if binm is not None:
        b = binm.astype(bool)
        out[b] = (out[b] * (1 - alpha) + np.array([255, 40, 40]) * alpha).astype(np.uint8)
    if bbox is not None:
        cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2] - 1, bbox[3] - 1), (40, 220, 40), max(1, out.shape[1] // 400))
    return out


def disp(img, maxside=1280):
    """Galeri için küçültme (yalnız görüntüleme)."""
    if img is None:
        return None
    h, w = img.shape[:2]
    s = maxside / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def _load_json(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return {}


TAU = _load_json(TAU_JSON)


IMG_TAU = _load_json(META / "image_tau.json")


_REF = None


def reference_table():
    global _REF
    if _REF is None:
        p = META / "reference_per_image_s42.csv"
        _REF = pd.read_csv(p, dtype={"new_id": str}) if p.exists() else pd.DataFrame()
        if len(_REF):
            _REF["new_id"] = _REF["new_id"].astype(str).str.zfill(9)
            _REF = _REF.set_index(["split", "new_id", "model", "gt"]).sort_index()
    return _REF


def ref_lookup(split, nid, stem, gt):
    r = reference_table()
    try:
        row = r.loc[(split, nid, stem, gt)]
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row
    except Exception:
        return None


def default_tau(model_name: str, gt_kind: str):
    stem = model_name.split("/", 1)[-1]
    try:
        return float(TAU[stem][gt_kind])
    except Exception:
        return 0.5


def image_threshold(model_name: str):
    stem = model_name.split("/", 1)[-1]
    if stem in IMG_TAU and isinstance(IMG_TAU[stem], dict):
        return float(IMG_TAU[stem]["thr"]), True
    return 0.5, False


# [FAZ4] ResNet-18 görüntü kararı: skor = en yüksek k patch skorunun ortalaması (seçilen k = 1 → maks patch);
# eşik val 487 sahte + 487 eşleşen authentic üzerinde Youden-J (kod/phase4_resnet/calibrate_p4.py).
RESNET_TAU = _load_json(META / "resnet_image_tau.json")


def resnet_decision(model_name: str, scores):
    """(görüntü skoru, eşik, kalibre mi). scores: [(patch skoru, y, x)]. Kalibrasyon girdisi yoksa maks patch >= 0.5."""
    e = RESNET_TAU.get(model_name.split("/", 1)[-1])
    k, thr, cal = (int(e.get("k", 1)), float(e["thr"]), True) if isinstance(e, dict) else (1, 0.5, False)
    top = sorted((float(s) for s, _, _ in scores), reverse=True)[:max(1, k)]
    return float(np.mean(top)), thr, cal
