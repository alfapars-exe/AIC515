"""Tam görüntü SHAP: orijinal görüntünün TAMAMINI kapsayan açıklamalar (kırpılmış pencere / patch yerine).

Neden ayrı bir yol?
  * Mevcut hedefler (targets.py) tek bir 512×512 pencereye / 224 patch'e bağlıdır; açıklama yalnız o bölgeyi kapsar.
  * shap'ın piksel düzeyi `maskers.Image`'ı tam çözünürlükte çok yavaştır (ölçüm: 1296×750'de modelsiz 27 s, 3 MP'de 99 s
    yalnız maske/ağaç maliyeti). Bu yüzden maske bir IZGARA üzerinde tanımlanır: shap PartitionExplainer'a (gh, gw, 1)
    boyutlu bir gösterge görüntüsü verilir; her değerlendirmede gösterge en yakın komşu ile tam çözünürlüğe taşınır ve
    görüntü = gösterge ? x : referans olarak kurulur. Böylece açıklama TEK hedef ve TEK referansla tüm görüntüyü kapsar;
    toplamsallık kesindir: E[f] + ΣSHAP = f(x), E[f] = f(referans görüntüsü).

Hedefler (modelin gerçek tam görüntü çıktısından türetilen skalerler)
  U-Net / DeepLabV3+ (prep="unet"; uygulamanın _predict_unet protokolü: 512 pencere, 448 adım, örtüşme ortalaması,
  uzun kenar > 4096 ise küçültme):
    full_image_score   : tam olasılık haritasının top-%1 ortalaması = uygulamanın görüntü kararı skoru
    full_region_mean   : tespit bölgesinin (bağlı bileşen, τ*) ortalama olasılığı
    full_gt_bbox_mean  : GT kutusu içi ortalama olasılık
  ResNet-18 (prep="resnet"; 224 patch, adım 112):
    full_max_patch     : tüm ızgaranın maks patch skoru = uygulamanın görüntü kararı skoru
    full_patch         : tek tespit patch'inin skoru (tam görüntüde; patch dışı pikseller etkisizdir → SHAP 0)
    full_gt_max_patch  : GT kutusuyla kesişen patch'lerin maks skoru
  YOLO11 (prep="yolo"; phase3 protokolü: 1024 kare / 768 adım, conf 0.001, NMS iou 0.6):
    yolo_box_conf      : seçilen kutuyla IoU ≥ 0.5 olan tahminlerin en yüksek güveni (tek sınıf: sahte bölge)

Döşemeli yerel SHAP (tile_mosaic): görüntü örtüşen (128 px) pencerelere bölünür, HER pencere kendi pencere skorunu
aynı referans türüyle açıklar ve haritalar kosinüs ağırlıklarıyla orijinal koordinatlarda birleştirilir. Taban değerleri
(E[f]) pencereden pencereye farklıdır; bu harita yüksek çözünürlüklü bir YEREL açıklama mozaiğidir ve tam görüntü SHAP
olarak SUNULMAZ (sonuç notunda ve arayüzde belirtilir; pencere başına toplamsallık ayrıca raporlanır).

Hız / bellek: tam görüntü U-Net'te pencere olasılıkları içerik özetiyle (blake2b) önbelleğe alınır — maskelenmemiş ya da
tamamen referansa dönmüş pencereler yeniden hesaplanmaz (5 kanallı modelde frekans kanalları küresel olduğundan önbellek
kapalı). ResNet'te patch skorları, YOLO'da kare tespitleri aynı şekilde önbelleğe alınır.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import math
import time
import numpy as np
import cv2

from .targets import (TargetSpec, unet_geometry, resnet_geometry, _to_prep_mask, top1_centroid, unet_regions, _iou, _bbox)
from .service import ExplanationResult
from .adapters import XAIUnsupported, REF_LABEL, to_u8_batch, _torch, _peak_reset, _peak_mb, resolve_reference

FULL_KINDS = ("full_image_score", "full_region_mean", "full_gt_bbox_mean", "full_max_patch", "full_patch",
              "full_gt_max_patch", "yolo_box_conf")
TILE_KIND = "tile_mosaic"
GRID_LONG = 192          # gösterge ızgarasının uzun kenarı (hücre boyutu = ceil(uzun kenar / 192) px)
YOLO_TILE, YOLO_STRIDE, YOLO_CONF, YOLO_IOU = 1024, 768, 0.001, 0.6


def is_full(kind):
    return kind in FULL_KINDS


def yolo_boxes(info):
    """Uygulamanın YOLO tahmin bilgisinden (info['boxes'] = [(conf, [x0,y0,x1,y1]), …], güvene göre sıralı) kutular."""
    bl = info.get("boxes") or []
    if not bl:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    return (np.array([b for _, b in bl], np.float32).reshape(-1, 4), np.array([c for c, _ in bl], np.float32))


def yolo_specs(names, deps):
    """YOLO11 kutu dedektörleri için XAI kaydı (model yüklenebiliyorsa OK; ultralytics yoksa gerekçeli UNSUPPORTED)."""
    from .registry import ModelSpec
    out = {}
    try:
        import importlib.util
        ok = importlib.util.find_spec("ultralytics") is not None
    except Exception:
        ok = False
    for n in names:
        out[n] = ModelSpec(
            n, deps.friendly(n), "yolo", "object-detection", 3, YOLO_TILE,
            f"natif {YOLO_TILE} kare / {YOLO_STRIDE} adım · RGB/255 · conf {YOLO_CONF} · kareler arası NMS iou {YOLO_IOU}",
            "kutu + güven (tek sınıf: sahte bölge)", "görüntü skoru = en yüksek kutu güveni",
            "PartitionExplainer · ızgara maskesi · tam görüntü (kutu güveni hedefi)", "",
            "OK" if ok else "UNSUPPORTED", "" if ok else "ultralytics paketi bu ortamda yok (tam YOLO çıkarımı yapılamaz)")
    return out


# ------------------------------------------------------------------ yardımcılar
class LRU:
    def __init__(self, n):
        self.n, self.d = n, OrderedDict()

    def get(self, k):
        v = self.d.get(k)
        if v is not None:
            self.d.move_to_end(k)
        return v

    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        while len(self.d) > self.n:
            self.d.popitem(last=False)


def _digest(a):
    return hashlib.blake2b(np.ascontiguousarray(a).data, digest_size=16).digest()


def _full_window(name, kind, prep, h, w, s, label, source, conf=None):
    return TargetSpec(name, kind, prep, 0, 0, h, w, label=label, source=source, confidence=conf, scale=s)


def _scale_note(t, deps):
    if t.scale != 1.0:
        t.notes.append(f"görüntü U-Net için {t.scale:.3f}× küçültüldü (uzun kenar > {deps.unet_max_side}); SHAP orijinal "
                       "çözünürlüğe alan korunarak taşındı")


def _resnet_grid_all(h, w, deps):
    _, _, (Hr, Wr) = resnet_geometry(h, w, deps.resnet_patch)
    return deps.windows(Hr, Wr, deps.resnet_patch, deps.resnet_stride)


# ------------------------------------------------------------------ hedef üreticiler (tam görüntü)
def full_decision_targets(preds, specs, shape, deps):
    H0, W0 = shape[:2]
    out = []
    for name, (m, info) in preds.items():
        sp = specs.get(name)
        if sp is None or sp.status != "OK":
            continue
        if sp.family == "resnet":
            smax = max(s for s, _, _ in info["scores"])
            t = _full_window(name, "full_max_patch", "resnet", H0, W0, 1.0,
                             f"tam görüntü · görüntü skoru (maks patch) {smax:.3f}", "karar", smax)
            t.patches = _resnet_grid_all(H0, W0, deps)
        elif sp.family == "yolo":
            bx, cf = yolo_boxes(info)
            if not len(cf):
                continue
            i = int(np.argmax(cf))
            b = [float(v) for v in bx[i]]
            t = _full_window(name, "yolo_box_conf", "yolo", H0, W0, 1.0,
                             f"tam görüntü · en güvenli kutu conf {cf[i]:.3f} @ {[int(v) for v in b]}", "karar", float(cf[i]))
            t.patches = [b]
        else:
            s, (h, w), _ = unet_geometry(H0, W0, deps.unet_win, deps.unet_max_side)
            sc = deps.image_score(m)
            t = _full_window(name, "full_image_score", "unet", h, w, s,
                             f"tam görüntü · görüntü skoru (top-%1) {sc:.3f}", "karar", sc)
            _scale_note(t, deps)
        out.append(t)
    return out


def full_detection_targets(preds, specs, shape, deps, k=3):
    H0, W0 = shape[:2]
    out = []
    for name, (m, info) in preds.items():
        sp = specs.get(name)
        if sp is None or sp.status != "OK":
            continue
        if sp.family == "resnet":
            P, picked = deps.resnet_patch, []
            for s_, y, x in sorted(info["scores"], key=lambda t: -t[0]):
                b = (x, y, x + P, y + P)
                if all(_iou(b, q) < 0.25 for q in picked):
                    picked.append(b)
                    t = _full_window(name, "full_patch", "resnet", H0, W0, 1.0,
                                     f"tam görüntü · tespit {len(picked)} · patch p={s_:.3f} @ ({x},{y})", "tespit", float(s_))
                    t.patches = [(int(y), int(x))]
                    out.append(t)
                if len(picked) >= k:
                    break
        elif sp.family == "yolo":
            bx, cf = yolo_boxes(info)
            order = np.argsort(-cf)
            picked = []
            for i in order:
                b = [float(v) for v in bx[i]]
                if all(_iou(b, q) < 0.5 for q in picked):
                    picked.append(b)
                    t = _full_window(name, "yolo_box_conf", "yolo", H0, W0, 1.0,
                                     f"tam görüntü · kutu {len(picked)} conf {cf[i]:.3f} @ {[int(v) for v in b]}",
                                     "tespit", float(cf[i]))
                    t.patches = [b]
                    out.append(t)
                if len(picked) >= k:
                    break
        else:
            s, (h, w), (Hp, Wp) = unet_geometry(H0, W0, deps.unet_win, deps.unet_max_side)
            tau = deps.default_tau(name, sp.train_gt or "bbox")
            for j, r in enumerate(unet_regions(m, tau, deps)[:k], 1):
                x0, y0, x1, y1 = r["bbox"]
                t = _full_window(name, "full_region_mean", "unet", h, w, s,
                                 f"tam görüntü · tespit {j} · bölge güveni {r['conf']:.3f} · kutu ({x0},{y0},{x1},{y1}) · τ={tau:.2f}",
                                 "tespit", r["conf"])
                t.region = _to_prep_mask(r["mask"], s, (h, w), (Hp, Wp))[:h, :w]
                _scale_note(t, deps)
                out.append(t)
    return out


def full_gt_targets(models, specs, shape, gt_mask, deps, preds=None):
    H0, W0 = shape[:2]
    if gt_mask is None:
        return []
    g = gt_mask > 127 if gt_mask.dtype == np.uint8 else gt_mask.astype(bool)
    bb = _bbox(g)
    if bb is None:
        return []
    out = []
    for name in models:
        sp = specs.get(name)
        if sp is None or sp.status != "OK":
            continue
        if sp.family == "resnet":
            P = deps.resnet_patch
            grid = _resnet_grid_all(H0, W0, deps)
            hit = [(y, x) for (y, x) in grid if not (x + P <= bb[0] or x >= bb[2] or y + P <= bb[1] or y >= bb[3])]
            t = _full_window(name, "full_gt_max_patch", "resnet", H0, W0, 1.0,
                             f"tam görüntü · GT kutusuyla kesişen {len(hit)} patch'in maks skoru · {bb}", "gt")
            t.patches = hit
        elif sp.family == "yolo":
            t = _full_window(name, "yolo_box_conf", "yolo", H0, W0, 1.0,
                             f"tam görüntü · GT kutusuyla IoU ≥ 0.5 kutuların maks güveni · {bb}", "gt")
            t.patches = [[float(v) for v in bb]]
        else:
            s, (h, w), (Hp, Wp) = unet_geometry(H0, W0, deps.unet_win, deps.unet_max_side)
            box = np.zeros((H0, W0), bool); box[bb[1]:bb[3], bb[0]:bb[2]] = True
            t = _full_window(name, "full_gt_bbox_mean", "unet", h, w, s, f"tam görüntü · GT kutusu içi ortalama · {bb}", "gt")
            t.region = _to_prep_mask(box, s, (h, w), (Hp, Wp))[:h, :w]
            _scale_note(t, deps)
        out.append(t)
    return out


def tile_grid(h, w, win=512, overlap=128):
    """Örtüşen döşeme ızgarası (hazırlanmış koordinatlar). Kenarda son döşeme sona hizalanır → tüm görüntü kapsanır."""
    stride = win - overlap
    ys = list(range(0, max(h - win, 0) + 1, stride)); xs = list(range(0, max(w - win, 0) + 1, stride))
    if ys[-1] + win < h: ys.append(h - win)
    if xs[-1] + win < w: xs.append(w - win)
    return [(y, x) for y in ys for x in xs]


def tile_targets(models, specs, shape, deps, overlap=128):
    """Döşemeli yerel SHAP için model başına TEK iş (TILE_KIND); döşemeler yürütmede üretilir."""
    H0, W0 = shape[:2]
    out = []
    for name in models:
        sp = specs.get(name)
        if sp is None or sp.status != "OK" or sp.family not in ("unet", "resnet"):
            continue
        if sp.family == "unet":
            s, (h, w), _ = unet_geometry(H0, W0, deps.unet_win, deps.unet_max_side)
            prep = "unet"
        else:
            s, (h, w) = 1.0, (H0, W0)
            prep = "resnet"
        n = len(tile_grid(max(h, deps.unet_win), max(w, deps.unet_win), deps.unet_win, overlap))
        t = _full_window(name, TILE_KIND, prep, h, w, s,
                         f"döşemeli yerel SHAP · {n} pencere (512 px, {overlap} px örtüşme) · her pencere kendi skorunu açıklar",
                         "döşeme")
        t.notes.append("YEREL açıklama mozaiği: pencere tabanları (E[f]) farklıdır → tam görüntü SHAP DEĞİLDİR")
        if prep == "unet":
            _scale_note(t, deps)
        out.append(t)
    return out


# ------------------------------------------------------------------ değerlendiriciler (modelin gerçek çıktısı)
class UNetFull:
    """Uygulamanın _predict_unet protokolüyle tam olasılık haritası. values(): bir grup maskeli görüntünün TÜM
    pencereleri tek kuyrukta toplanır, aynı içerikli pencereler bir kez hesaplanır (önbellek + toplu içi tekilleştirme)."""

    def __init__(self, deps, model, in_ch, cache_windows=96, batch=16):
        self.deps, self.model, self.in_ch = deps, model, in_ch
        self.cache = LRU(cache_windows) if in_ch == 3 else None
        self.batch = batch
        self.forwards = 0

    def _pad(self, im):
        win = self.deps.unet_win
        h, w = im.shape[:2]
        ph, pw = max(0, win - h), max(0, win - w)
        return cv2.copyMakeBorder(im, 0, ph, 0, pw, cv2.BORDER_REPLICATE) if (ph or pw) else im

    def probs(self, ims):
        torch = _torch()
        d = self.deps
        win, stride = d.unet_win, d.unet_stride
        plans, todo, pending = [], [], {}
        for ii, im in enumerate(ims):
            img = self._pad(im)
            H, W = img.shape[:2]
            fc = d.project_freq_channels(img) if self.in_ch == 5 else None
            wins = d.windows(H, W, win, stride)
            got = []
            for (y, x) in wins:
                tile = img[y:y + win, x:x + win]
                if self.cache is not None:
                    key = _digest(tile)
                    p = self.cache.get(key)
                    if p is None:
                        if key not in pending:
                            pending[key] = len(todo); todo.append((tile, None))
                        got.append((y, x, ("q", pending[key])))
                    else:
                        got.append((y, x, ("c", p)))
                else:
                    todo.append((tile, fc[:, y:y + win, x:x + win]))
                    got.append((y, x, ("q", len(todo) - 1)))
            plans.append((im.shape[:2], (H, W), got))
        outs = [None] * len(todo)
        for i in range(0, len(todo), self.batch):
            chunk = todo[i:i + self.batch]
            xs = []
            for tile, fcw in chunk:
                xc = d.norm_chw(tile)
                if fcw is not None:
                    xc = np.concatenate([xc, fcw], 0)
                xs.append(xc)
            xb = torch.from_numpy(np.stack(xs)).to(d.device)
            with torch.no_grad():
                pb = d.unet_window_forward(self.model, self.in_ch, xb).float().cpu().numpy()
            self.forwards += len(chunk)
            for j, pr in enumerate(pb):
                outs[i + j] = pr
        if self.cache is not None:
            for key, idx in pending.items():
                self.cache.put(key, outs[idx])
        res = []
        for (h, w), (H, W), got in plans:
            acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
            for (y, x, (kind, v)) in got:
                pr = outs[v] if kind == "q" else v
                acc[y:y + win, x:x + win] += pr; cnt[y:y + win, x:x + win] += 1
            res.append((acc / cnt)[:h, :w])
        return res

    def prob(self, im):
        return self.probs([im])[0]

    def _reduce(self, p, target):
        if target.kind == "full_image_score":
            return float(self.deps.image_score(p))
        r = target.region
        return float(p[r].mean()) if r is not None and r.any() else float("nan")

    def values(self, ims, target):
        return [self._reduce(p, target) for p in self.probs(ims)]

    def value(self, im, target):
        return self.values([im], target)[0]


class UNetWindow:
    """Döşeme hedefi: tek 512 pencerenin geçerli piksellerindeki top-%1 skoru (targets.window_image_score ile aynı)."""

    def __init__(self, deps, model, in_ch, context=None, origin=(0, 0), batch=16):
        self.deps, self.model, self.in_ch = deps, model, in_ch
        self.context, self.origin = context, origin       # 5 kanal: frekans kanalları bağlam üzerinde
        self.batch = batch
        self.forwards = 0

    def _x(self, im):
        d = self.deps
        x = d.norm_chw(im)
        if self.in_ch == 5:
            c = self.context.copy(); oy, ox = self.origin
            c[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
            x = np.concatenate([x, d.project_freq_channels(c)[:, oy:oy + im.shape[0], ox:ox + im.shape[1]]], 0)
        return x

    def values(self, ims, target):
        torch = _torch()
        d = self.deps
        out = []
        for i in range(0, len(ims), self.batch):
            xb = torch.from_numpy(np.stack([self._x(im) for im in ims[i:i + self.batch]])).to(d.device)
            with torch.no_grad():
                pb = d.unet_window_forward(self.model, self.in_ch, xb).float().cpu().numpy()
            self.forwards += len(pb)
            for p in pb:
                v = p[target.valid] if target.valid is not None else p
                out.append(float(d.image_score(v)))
        return out

    def value(self, im, target):
        return self.values([im], target)[0]


class ResNetFull:
    """224 patch skorları (uygulamanın _predict_resnet ızgarası); patch içeriğiyle önbellekli, görüntüler arası toplu."""

    def __init__(self, deps, model, cache=400_000, batch=128):
        self.deps, self.model = deps, model
        self.cache = LRU(cache)
        self.batch = batch
        self.forwards = 0

    def _pad(self, im):
        P = self.deps.resnet_patch
        h, w = im.shape[:2]
        ph, pw = max(0, P - h), max(0, P - w)
        return cv2.copyMakeBorder(im, 0, ph, 0, pw, cv2.BORDER_REPLICATE) if (ph or pw) else im

    def scores_many(self, ims, patches):
        torch = _torch()
        d = self.deps
        P = d.resnet_patch
        res = [np.zeros(len(patches), np.float64) for _ in ims]
        todo, where = [], {}
        for ii, im in enumerate(ims):
            img = self._pad(im)
            for pi, (y, x) in enumerate(patches):
                pt = img[y:y + P, x:x + P]
                k = _digest(pt)
                v = self.cache.get(k)
                if v is not None:
                    res[ii][pi] = v
                elif k in where:
                    where[k][1].append((ii, pi))
                else:
                    where[k] = (len(todo), [(ii, pi)]); todo.append(pt)
        keys = list(where.keys())
        vals = np.zeros(len(todo), np.float64)
        for j in range(0, len(todo), self.batch):
            xb = torch.from_numpy(np.stack([d.norm_chw(pt) for pt in todo[j:j + self.batch]])).to(d.device)
            with torch.no_grad():
                with d.autocast():
                    s = torch.sigmoid(self.model(xb).float()).squeeze(1).cpu().numpy()
            vals[j:j + len(s)] = s
            self.forwards += len(s)
        for k in keys:
            idx, locs = where[k]
            self.cache.put(k, float(vals[idx]))
            for (ii, pi) in locs:
                res[ii][pi] = vals[idx]
        return res

    def scores(self, im, patches):
        return self.scores_many([im], patches)[0]

    def values(self, ims, target):
        return [float(s.max()) if len(s) else float("nan") for s in self.scores_many(ims, target.patches)]

    def value(self, im, target):
        return self.values([im], target)[0]


class YoloFull:
    """phase3_yolo/infer_to_maps.py protokolü: natif 1024 kare / 768 adım, conf 0.001, kareler arası NMS (iou 0.6);
    '-full-' modellerde tam görüntü tek geçiş (imgsz 1280). Kareler görüntüler arası toplu ve önbellekli."""

    def __init__(self, deps, model, cache=512, full_imgsz=0, batch=16):
        self.deps, self.model = deps, model
        self.cache = LRU(cache)
        self.forwards = 0
        self.batch = batch
        self.full_imgsz = full_imgsz

    def _kw(self, imgsz):
        dev = self.deps.device
        cuda = dev is not None and getattr(dev, "type", "") == "cuda"
        return dict(conf=YOLO_CONF, iou=YOLO_IOU, verbose=False, device=0 if cuda else "cpu", half=cuda, imgsz=imgsz)

    def _predict(self, tiles, imgsz):
        import logging
        try:
            from ultralytics.utils import LOGGER
            lvl = LOGGER.level; LOGGER.setLevel(logging.ERROR)
        except Exception:
            LOGGER, lvl = None, None
        try:
            out = []
            for j in range(0, len(tiles), self.batch):
                rs = self.model.predict(tiles[j:j + self.batch], **self._kw(imgsz))
                out += [(r.boxes.xyxy.float().cpu().numpy(), r.boxes.conf.float().cpu().numpy()) for r in rs]
                self.forwards += len(rs)
            return out
        finally:
            if LOGGER is not None:
                LOGGER.setLevel(lvl)

    def detect_many(self, ims):
        torch = _torch()
        from torchvision.ops import nms
        T, S = YOLO_TILE, YOLO_STRIDE
        plans, todo, where = [], [], {}
        for ii, im in enumerate(ims):
            H0, W0 = im.shape[:2]
            bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            if self.full_imgsz:
                wins, img = [(0, 0)], bgr
            else:
                ph, pw = max(0, T - H0), max(0, T - W0)
                img = cv2.copyMakeBorder(bgr, 0, ph, 0, pw, cv2.BORDER_REPLICATE) if (ph or pw) else bgr
                wins = self.deps.windows(img.shape[0], img.shape[1], T, S)
            got = []
            for (y, x) in wins:
                tile = np.ascontiguousarray(img if self.full_imgsz else img[y:y + T, x:x + T])
                k = _digest(tile)
                v = self.cache.get(k)
                if v is None:
                    if k not in where:
                        where[k] = len(todo); todo.append(tile)
                    got.append((y, x, ("q", where[k])))
                else:
                    got.append((y, x, ("c", v)))
            plans.append(((H0, W0), got))
        outs = self._predict(todo, self.full_imgsz or T) if todo else []
        for k, idx in where.items():
            self.cache.put(k, outs[idx])
        res = []
        for (H0, W0), got in plans:
            bl, cl = [], []
            for (y, x, (kind, v)) in got:
                b, c = outs[v] if kind == "q" else v
                if len(b):
                    bl.append(b + np.array([x, y, x, y], np.float32)); cl.append(c)
            if not bl:
                res.append((np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))); continue
            b = np.concatenate(bl); c = np.concatenate(cl)
            if len(got) > 1:
                keep = nms(torch.from_numpy(b), torch.from_numpy(c), YOLO_IOU).numpy(); b, c = b[keep], c[keep]
            b = b.copy(); b[:, [0, 2]] = b[:, [0, 2]].clip(0, W0); b[:, [1, 3]] = b[:, [1, 3]].clip(0, H0)
            res.append((b, c))
        return res

    def detect(self, im):
        return self.detect_many([im])[0]

    @staticmethod
    def _match(b, c, ref):
        if not len(b):
            return 0.0
        ious = np.array([_iou(list(bb), ref) for bb in b.tolist()])
        m = ious >= 0.5
        return float(c[m].max()) if m.any() else 0.0

    def values(self, ims, target):
        return [self._match(b, c, target.patches[0]) for (b, c) in self.detect_many(ims)]

    def value(self, im, target):
        return self.values([im], target)[0]


# ------------------------------------------------------------------ ızgara maskeli Partition açıklayıcısı
def grid_explain(evaluator, target, x, ref, max_evals, batch_size=32, grid_long=GRID_LONG, cell=None):
    """x, ref: (h, w, 3) uint8 — aynı koordinatlar. Dönüş: (piksel başına SHAP (h, w), f(x), E[f], n_eval, hücre).
    Hücre değerleri hücredeki piksel sayısına bölünerek dağıtılır → ΣSHAP korunur."""
    import shap
    h, w = x.shape[:2]
    cell = int(cell or max(2, math.ceil(max(h, w) / grid_long)))
    gh, gw = math.ceil(h / cell), math.ceil(w / cell)
    counter = {"n": 0}
    x = np.ascontiguousarray(x); ref = np.ascontiguousarray(ref)

    def compose(ind):
        keep = ind.reshape(gh, gw) > 0.5
        K = np.repeat(np.repeat(keep, cell, 0), cell, 1)[:h, :w]
        return np.where(K[..., None], x, ref)

    def f(I):
        I = np.asarray(I)
        counter["n"] += len(I)
        return np.asarray(evaluator.values([compose(ind) for ind in I], target), np.float64)[:, None]

    masker = shap.maskers.Image(np.zeros((gh, gw, 1), np.float32), (gh, gw, 1))
    ex = shap.PartitionExplainer(f, masker)
    ev = ex(np.ones((1, gh, gw, 1), np.float32), max_evals=int(max_evals), batch_size=int(batch_size), silent=True)
    v = np.asarray(ev.values, dtype=np.float64).reshape(gh, gw)
    base = float(np.asarray(ev.base_values, dtype=np.float64).ravel()[0])
    ch = np.minimum(cell, h - np.arange(gh) * cell); cw = np.minimum(cell, w - np.arange(gw) * cell)
    per_px = v / np.outer(ch, cw)
    full = np.repeat(np.repeat(per_px, cell, 0), cell, 1)[:h, :w]
    fx = float(evaluator.value(x, target))
    return full, fx, base, counter["n"], cell, (gh, gw)


def _to_original(m, H0, W0):
    """Hazırlanmış (ölçeklenmiş) koordinattaki piksel başına SHAP → orijinal; toplam korunur."""
    if m.shape == (H0, W0):
        return m
    tot = float(m.sum())
    r = cv2.resize(m.astype(np.float32), (W0, H0), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    s = float(r.sum())
    return r * (tot / s) if abs(s) > 1e-12 else r * (m.size / r.size)


def _cosine_weight(h, w, ramp=64):
    def ramp1(n):
        a = np.ones(n, np.float64)
        k = min(ramp, n // 2)
        if k > 0:
            t = 0.5 - 0.5 * np.cos(np.pi * (np.arange(k) + 0.5) / k)
            a[:k] = t; a[n - k:] = t[::-1]
        return a
    return np.outer(ramp1(h), ramp1(w)) + 1e-6


# ------------------------------------------------------------------ yürütme
def _prepared(t, rgb, auth, deps):
    """Açıklama uzayı: prep='unet' → ölçeklenmiş, dolgusuz görüntü; diğerleri → orijinal."""
    if t.prep == "unet":
        img, (h, w), _ = deps.unet_prepare(rgb, deps.unet_win, deps.unet_max_side)
        a = deps.unet_prepare(auth, deps.unet_win, deps.unet_max_side)[0][:h, :w] if auth is not None else None
        return np.ascontiguousarray(img[:h, :w]), (np.ascontiguousarray(a) if a is not None else None)
    return rgb, auth


def _reference(x, auth_p, ref_kind, res):
    if ref_kind == "authentic" and auth_p is not None:
        return auth_p
    if ref_kind == "inpaint":
        res.notes.append("tam görüntüde inpaint referansı maskeye bağlı yeniden hesaplanamaz → blur(32,32)")
        res.reference = REF_LABEL["blur"]
    return cv2.blur(x, (32, 32))


def _evaluator(spec, deps):
    model, in_ch, _ = deps.get_model(spec.name)
    if spec.family == "resnet":
        return ResNetFull(deps, model)
    if spec.family == "yolo":
        return YoloFull(deps, model, full_imgsz=1280 if "-full-" in spec.name else 0)
    return UNetFull(deps, model, in_ch)


def explain_full_job(t, spec, rgb, auth, deps, cfg, ref_kind):
    torch = _torch()
    H0, W0 = rgb.shape[:2]
    res = ExplanationResult(spec.name, spec.label, spec.family, "partition", t.to_dict(), REF_LABEL.get(ref_kind, ref_kind),
                            notes=list(t.notes))
    if cfg.method == "gradient":
        res.notes.append("tam görüntüde Gradient desteklenmez (ızgara maskeli Partition kullanıldı)")
    t0 = time.time(); _peak_reset(torch, deps.device)
    x, auth_p = _prepared(t, rgb, auth, deps)
    ref = _reference(x, auth_p, ref_kind, res)
    ev = _evaluator(spec, deps)
    budget = int(t.budget or cfg.max_evals) if getattr(t, "budget", None) else int(cfg.max_evals)
    if budget < int(cfg.max_evals):
        res.notes.append(f"değerlendirme bütçesi süre sınırı nedeniyle {cfg.max_evals} → {budget}")
    m, fx, base, n, cell, (gh, gw) = grid_explain(ev, t, x, ref, budget, cfg.batch_size)
    fref = float(ev.value(ref, t))
    m = _to_original(m, H0, W0)
    res.values = m[..., None].astype(np.float32)
    res.fx, res.base = fx, base
    res.sum_shap = float(m.sum())
    res.add_err = abs(base + res.sum_shap - fx)
    res.n_evals = n
    res.seconds = time.time() - t0
    res.peak_mem_mb = _peak_mb(torch, deps.device)
    td = dict(res.target); td.update(window=[0, 0, H0, W0], scale=1.0, full=True, grid=[gh, gw], cell_px=cell,
                                     model_forwards=ev.forwards, f_reference=fref)
    res.target = td
    res.notes.append(f"tam görüntü: {W0}×{H0} px · ızgara {gw}×{gh} hücre ({cell} px) · {ev.forwards} model ileri geçişi "
                     f"(önbellekle) · E[f] = f(referans) = {fref:.4f} (açıklayıcı tabanı {base:.4f})")
    return res


def explain_tiles_job(t, spec, rgb, auth, deps, cfg, ref_kind, overlap=128, cell=4):
    """Döşemeli yerel SHAP: her pencere kendi pencere skorunu (U-Net: geçerli piksel top-%1; ResNet: pencere içi maks
    patch) aynı referans türüyle açıklar; haritalar kosinüs ağırlıklı ortalamayla orijinal koordinatlarda birleşir."""
    torch = _torch()
    H0, W0 = rgb.shape[:2]
    res = ExplanationResult(spec.name, spec.label, spec.family, "partition", t.to_dict(), REF_LABEL.get(ref_kind, ref_kind),
                            notes=list(t.notes))
    t0 = time.time(); _peak_reset(torch, deps.device)
    x, auth_p = _prepared(t, rgb, auth, deps)
    ref_full = _reference(x, auth_p, ref_kind, res)
    h, w = x.shape[:2]
    win = deps.unet_win
    ph, pw = max(0, win - h), max(0, win - w)
    pad = (lambda a: cv2.copyMakeBorder(a, 0, ph, 0, pw, cv2.BORDER_REPLICATE)) if (ph or pw) else (lambda a: a)
    xp, rp = pad(x), pad(ref_full)
    Hp, Wp = xp.shape[:2]
    model, in_ch, _ = deps.get_model(spec.name)
    acc = np.zeros((Hp, Wp), np.float64); wsum = np.zeros((Hp, Wp), np.float64)
    tiles, bud = [], int(t.budget or cfg.max_evals) if getattr(t, "budget", None) else int(cfg.max_evals)
    rn = ResNetFull(deps, model) if spec.family == "resnet" else None
    grid_all = deps.windows(Hp, Wp, deps.resnet_patch, deps.resnet_stride) if rn else None
    for (y, xx) in tile_grid(Hp, Wp, win, overlap):
        valid = np.zeros((win, win), bool); valid[:max(0, min(win, h - y)), :max(0, min(win, w - xx))] = True
        tt = TargetSpec(spec.name, "window_image_score" if rn is None else "window_max_patch", t.prep, y, xx, win, win,
                        valid=valid)
        if rn is None:
            ev = UNetWindow(deps, model, in_ch, context=xp if in_ch == 5 else None, origin=(y, xx))
        else:
            P = deps.resnet_patch
            tt.patches = [(gy - y, gx - xx) for (gy, gx) in grid_all if gy >= y and gx >= xx and gy + P <= y + win and gx + P <= xx + win]
            ev = rn
        xw, rw = xp[y:y + win, xx:xx + win], rp[y:y + win, xx:xx + win]
        m, fx, base, n, _, _ = grid_explain(ev, tt, xw, rw, bud, cfg.batch_size, cell=cell)
        wt = _cosine_weight(win, win, overlap // 2)
        acc[y:y + win, xx:xx + win] += m * wt; wsum[y:y + win, xx:xx + win] += wt
        tiles.append(dict(y=int(y), x=int(xx), fx=fx, base=base, sum=float(m.sum()), add_err=abs(base + m.sum() - fx), evals=n))
    mosaic = (acc / np.maximum(wsum, 1e-12))[:h, :w]
    covered = bool((wsum[:h, :w] > 0).all())
    mosaic = _to_original(mosaic, H0, W0)
    res.values = mosaic[..., None].astype(np.float32)
    res.fx = float(np.mean([d["fx"] for d in tiles])); res.base = float(np.mean([d["base"] for d in tiles]))
    res.sum_shap = float(mosaic.sum())
    res.add_err = float(max(d["add_err"] for d in tiles))
    res.n_evals = int(sum(d["evals"] for d in tiles))
    res.seconds = time.time() - t0
    res.peak_mem_mb = _peak_mb(torch, deps.device)
    bases = [d["base"] for d in tiles]
    td = dict(res.target); td.update(window=[0, 0, H0, W0], scale=1.0, full=False, mosaic=True, tiles=tiles, covered=covered,
                                     base_spread=float(np.ptp(bases)) if bases else 0.0)
    res.target = td
    res.notes.append(f"{len(tiles)} pencere · kapsama {'tam' if covered else 'EKSİK'} · pencere tabanları E[f] "
                     f"{min(bases):.3f}–{max(bases):.3f} (fark {np.ptp(bases):.3f}) · pencere başına toplamsallık hatası ≤ {res.add_err:.1e}"
                     " · f(x)/E[f] sütunları pencere ORTALAMASIDIR; harita yerel açıklamaların mozaiğidir, tam görüntü SHAP değildir")
    return res


def run_full_jobs(jobs, rgb, auth, deps, specs, cfg):
    """Tam görüntü ve döşemeli işler (GPU fonksiyonunun içinde çağrılır). Dönüş: [ExplanationResult] (değerler orijinal
    koordinatlarda (H0, W0, 1))."""
    torch = _torch()
    out = []
    has_auth = auth is not None and auth.shape == rgb.shape
    for t in jobs:
        spec = specs.get(t.model)
        try:
            if spec is None or spec.status != "OK":
                raise XAIUnsupported(f"{t.model}: {getattr(spec, 'reason', '') or 'desteklenmiyor'}")
            ref_kind, ref_note = resolve_reference(cfg.reference, has_auth)
            if t.kind == TILE_KIND:
                r = explain_tiles_job(t, spec, rgb, auth if has_auth else None, deps, cfg, ref_kind)
            else:
                r = explain_full_job(t, spec, rgb, auth if has_auth else None, deps, cfg, ref_kind)
            if ref_note:
                r.notes.append(ref_note)
        except XAIUnsupported as e:
            r = ExplanationResult(t.model, spec.label if spec else t.model, spec.family if spec else "?", cfg.method,
                                  t.to_dict(), "—", status="UNSUPPORTED", message=str(e))
        except Exception as e:
            r = ExplanationResult(t.model, spec.label if spec else t.model, spec.family if spec else "?", cfg.method,
                                  t.to_dict(), "—", status="FAIL", message=f"{type(e).__name__}: {e}")
        out.append(r)
        if deps.device is not None and getattr(deps.device, "type", "") == "cuda":
            torch.cuda.empty_cache()
    return out


def region_orig(t: TargetSpec, H0, W0):
    """Hedef bölgesini (hazırlanmış koordinat) orijinal koordinata taşı (en yakın komşu)."""
    if t.region is None:
        return None
    r = t.region.astype(np.uint8)
    if r.shape != (H0, W0):
        r = cv2.resize(r, (W0, H0), interpolation=cv2.INTER_NEAREST)
    return r > 0


def box_mask(box, H0, W0):
    m = np.zeros((H0, W0), bool)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    m[max(0, y0):min(H0, y1), max(0, x0):min(W0, x1)] = True
    return m


def target_region_orig(t: TargetSpec, H0, W0, deps):
    """Görselleştirme için hedefin orijinal koordinattaki alanı (bölge / patch'ler / kutu)."""
    if t.kind in ("full_region_mean", "full_gt_bbox_mean"):
        return region_orig(t, H0, W0)
    if t.kind == "yolo_box_conf" and t.patches:
        return box_mask(t.patches[0], H0, W0)
    if t.kind in ("full_patch",) and t.patches:
        y, x = t.patches[0]
        return box_mask((x, y, x + deps.resnet_patch, y + deps.resnet_patch), H0, W0)
    return None
