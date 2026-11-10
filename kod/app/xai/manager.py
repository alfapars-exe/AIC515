"""XAIManager: kayıt + adaptör seçimi + LRU önbellek. Yalnız önbellekte olmayan (model, hedef, yöntem, referans,
bütçe) birleşimleri hesaplanır; hesaplama `compute_fn` ile yapılır (Space'te @spaces.GPU sarmalı _gpu_explain)."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import numpy as np

from .adapters import ExplainConfig, resolve_reference


def image_hash(rgb, auth=None):
    h = hashlib.sha1()
    for a in (rgb, auth):
        if a is not None:
            a = np.ascontiguousarray(a)
            h.update(str(a.shape).encode()); h.update(a.tobytes())
        else:
            h.update(b"none")
    return h.hexdigest()[:16]


class XAIManager:
    def __init__(self, specs_fn, compute_fn, cache_size=96):
        self.specs_fn = specs_fn          # () → {ad: ModelSpec}
        self.compute_fn = compute_fn      # (rgb, auth, jobs, cfg) → [ExplanationResult]
        self.cache: "OrderedDict[str, object]" = OrderedDict()
        self.cache_size = cache_size
        self.stats = {"hit": 0, "miss": 0}

    def key(self, img_h, target, cfg: ExplainConfig, has_auth):
        ref, _ = resolve_reference(cfg.reference, has_auth)
        c = ExplainConfig(cfg.method, ref, cfg.max_evals, cfg.batch_size, cfg.grad_nsamples, cfg.seed)
        return f"{img_h}|{target.key()}|{c.key()}"

    def explain(self, rgb, auth, jobs, cfg: ExplainConfig):
        has_auth = auth is not None and auth.shape == rgb.shape
        img_h = image_hash(rgb, auth if has_auth else None)
        keys = [self.key(img_h, t, cfg, has_auth) for t in jobs]
        missing = [(k, t) for k, t in zip(keys, jobs) if k not in self.cache]
        missing = list(OrderedDict(missing).items())     # aynı anahtar bir kez
        self.stats["hit"] += len(jobs) - len(missing); self.stats["miss"] += len(missing)
        if missing:
            new = self.compute_fn(rgb, auth if has_auth else None, [t for _, t in missing], cfg)
            for (k, _), r in zip(missing, new):
                r.key = k
                if r.status == "OK":         # hatalar önbelleğe alınmaz (yeniden denenebilir)
                    self.cache[k] = r
                    self.cache.move_to_end(k)
                else:
                    self.cache[k + "|tmp"] = r
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        missing_keys = {k for k, _ in missing}
        out = []
        for k in keys:
            r = self.cache.get(k) or self.cache.pop(k + "|tmp", None)
            if r is None:
                continue
            r.cached = k not in missing_keys
            out.append(r)
        return out

    def clear(self):
        self.cache.clear()
