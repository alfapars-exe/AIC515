"""SHAP adaptörleri: her model ailesi için 'tahmin adaptörü' (görüntü → skaler hedef) + açıklayıcı.

* predict_batch(NHWC, 0–255) → (N, 1) olasılık: uygulamanın KENDİ ön işlemesi ve ileri geçişiyle (Deps).
* PartitionExplainer (birincil): modelden bağımsız, hiyerarşik Owen değerleri; maskers.Image ile referans
  (authentic eş dizisi / blur / inpaint). Toplamsallık: ΣSHAP = f(x) − f(referans) (kesin).
* GradientExplainer (ikincil): beklenen gradyanlar, fp32, model-girdi uzayında; 5 kanallı U-Net'te kanal grubu
  katkıları (RGB / FFT-HP / Laplacian) ayrı raporlanır. Model parametrelerinin gradyanı tutulmaz.
* DeepExplainer kullanılmaz: torchvision BasicBlock aynı ReLU(inplace) modülünü iki kez çağırır + artık toplama;
  shap'ın DeepLIFT kancaları bunu doğru işlemez.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import time
import numpy as np
import cv2

from .targets import TargetSpec
from .service import ExplanationResult


class XAIUnsupported(RuntimeError):
    """Bu model için SHAP üretilemez (sahte çıktı yerine açık hata)."""


@dataclass
class ExplainConfig:
    method: str = "partition"       # partition | gradient
    reference: str = "auto"         # auto | authentic | blur | inpaint
    max_evals: int = 1000           # Partition değerlendirme bütçesi
    batch_size: int = 16
    grad_nsamples: int = 64         # Gradient: örnek sayısı
    seed: int = 0

    def key(self):
        return f"{self.method}|{self.reference}|{self.max_evals}|{self.grad_nsamples}|{self.seed}"


REF_LABEL = {"authentic": "authentic eş", "blur": "blur(32,32)", "inpaint": "inpaint_telea"}


# ------------------------------------------------------------------ yardımcılar
def _torch():
    import torch
    return torch


def to_u8_batch(X):
    X = np.asarray(X)
    if X.dtype == np.uint8:
        return X
    return np.clip(np.rint(X), 0, 255).astype(np.uint8)


def model_fingerprint(model):
    """state_dict sha256 — SHAP öncesi/sonrası ağırlıkların değişmediğini doğrulamak için."""
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def frozen_params(model):
    """Gradyan açıklaması sırasında parametre gradyanı birikmesin (param.grad None kalır); bayraklar geri yüklenir."""
    flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, f in zip(model.parameters(), flags):
            p.requires_grad_(f)


def prepare_for(prep, rgb, deps):
    """prep='unet' → app._unet_prepare (ölçek + dolgu); prep='resnet' → yalnız < 224 ise dolgu (app._predict_resnet)."""
    if prep == "unet":
        return deps.unet_prepare(rgb, deps.unet_win, deps.unet_max_side)
    P = deps.resnet_patch
    h, w = rgb.shape[:2]
    ph, pw = max(0, P - h), max(0, P - w)
    img = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REPLICATE) if (ph or pw) else rgb
    return img, (h, w), 1.0


def _peak_reset(torch, device):
    if device is not None and getattr(device, "type", "") == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(torch, device):
    if device is not None and getattr(device, "type", "") == "cuda":
        return float(torch.cuda.max_memory_allocated() / 2 ** 20)
    return None


# ------------------------------------------------------------------ adaptörler
class BaseShapAdapter:
    family = "base"

    def __init__(self, spec, deps):
        self.spec, self.deps = spec, deps
        self.model, self.in_ch, _ = deps.get_model(spec.name)

    # --- uygulamaya özdeş tahmin (numpy yolu; Partition ve f(x) için)
    def predict_batch(self, X, target: TargetSpec, ctx: dict):
        raise NotImplementedError

    # --- türevlenebilir hedef (torch yolu; Gradient için). x: model-girdi uzayı (N, C, h, w)
    def torch_input(self, win_u8, target, ctx):
        raise NotImplementedError

    def torch_target(self, x, target):
        raise NotImplementedError

    # --- açıklama
    def explain(self, win_u8, ref_u8, ref_kind, target: TargetSpec, cfg: ExplainConfig, ctx: dict) -> ExplanationResult:
        torch = _torch()
        res = ExplanationResult(self.spec.name, self.spec.label, self.spec.family, cfg.method, target.to_dict(),
                                REF_LABEL.get(ref_kind, ref_kind), notes=list(target.notes))
        t0 = time.time(); _peak_reset(torch, self.deps.device)
        if cfg.method == "gradient":
            self._explain_gradient(res, win_u8, ref_u8, ref_kind, target, cfg, ctx)
        else:
            self._explain_partition(res, win_u8, ref_u8, ref_kind, target, cfg, ctx)
        res.seconds = time.time() - t0
        res.peak_mem_mb = _peak_mb(torch, self.deps.device)
        return res

    def _explain_partition(self, res, win_u8, ref_u8, ref_kind, target, cfg, ctx):
        import shap
        counter = {"n": 0}

        def f(X):
            counter["n"] += len(X)
            return self.predict_batch(X, target, ctx)

        x = win_u8.astype(np.float32)
        if ref_kind == "authentic":
            mv = ref_u8.astype(np.float32)
        elif ref_kind == "inpaint":
            mv = "inpaint_telea"
        else:
            mv = "blur(32,32)"
        masker = shap.maskers.Image(mv, x.shape)
        explainer = shap.PartitionExplainer(f, masker)
        ev = explainer(x[None], max_evals=int(cfg.max_evals), batch_size=int(cfg.batch_size), silent=True)
        vals = np.asarray(ev.values, dtype=np.float32)[0]
        if vals.ndim == 4:
            vals = vals[..., 0]
        base = float(np.asarray(ev.base_values, dtype=np.float64).ravel()[0])
        fx = float(self.predict_batch(x[None], target, ctx)[0, 0])
        res.values = vals.astype(np.float16)
        res.fx, res.base = fx, base
        res.sum_shap = float(vals.astype(np.float64).sum())
        res.add_err = abs(base + res.sum_shap - fx)
        res.n_evals = counter["n"]

    def _explain_gradient(self, res, win_u8, ref_u8, ref_kind, target, cfg, ctx):
        import shap
        torch = _torch()
        if ref_kind == "inpaint":
            res.notes.append("Gradient için inpaint referansı tanımsız → blur(32,32) kullanıldı")
            ref_kind = "blur"; res.reference = REF_LABEL["blur"]
        if ref_kind != "authentic":
            ref_u8 = cv2.blur(win_u8, (32, 32))
        adapter = self

        class _Wrap(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = adapter.model

            def forward(self, x):
                return adapter.torch_target(x, target)

        wrap = _Wrap().eval()
        x = self.torch_input(win_u8, target, ctx)
        bg = self.torch_input(ref_u8, target, ctx)
        with frozen_params(self.model), torch.enable_grad():
            ge = shap.GradientExplainer(wrap, bg, batch_size=max(1, int(cfg.batch_size)))
            sv = ge.shap_values(x, nsamples=int(cfg.grad_nsamples), rseed=int(cfg.seed))
            with torch.no_grad():
                fx = float(wrap(x).float().reshape(-1)[0]); base = float(wrap(bg).float().reshape(-1)[0])
        sv = np.asarray(sv[0] if isinstance(sv, list) else sv, dtype=np.float32)
        if sv.ndim == 5:
            sv = sv[..., 0]
        sv = sv[0]                                   # (C, h, w)
        vals = np.transpose(sv, (1, 2, 0))           # (h, w, C)
        res.values = vals.astype(np.float16)
        res.fx, res.base = fx, base
        res.sum_shap = float(vals.astype(np.float64).sum())
        res.add_err = abs(base + res.sum_shap - fx)
        res.n_evals = int(cfg.grad_nsamples)
        res.notes.append("Gradient: beklenen gradyanlar (fp32, model-girdi uzayı); toplamsallık yaklaşık")
        if vals.shape[-1] == 5:
            res.groups = {"RGB": float(vals[..., :3].sum()), "FFT-HP": float(vals[..., 3].sum()),
                          "Laplacian": float(vals[..., 4].sum())}
        else:
            res.groups = {"RGB": float(vals.sum())}


class ResNetPatchAdapter(BaseShapAdapter):
    family = "resnet"

    def _forward(self, xb):
        torch = _torch()
        with torch.no_grad():
            with self.deps.autocast():
                return torch.sigmoid(self.model(xb).float()).squeeze(1)

    def predict_batch(self, X, target, ctx):
        torch = _torch()
        X = to_u8_batch(X)
        dev = self.deps.device
        if target.kind == "patch":
            xb = torch.from_numpy(np.stack([self.deps.norm_chw(im) for im in X])).to(dev)
            out = [self._forward(xb[i:i + 64]).cpu().numpy() for i in range(0, len(xb), 64)]
            return np.concatenate(out)[:, None].astype(np.float64)
        if target.kind == "window_max_patch":
            P = self.deps.resnet_patch
            res = []
            for im in X:
                x = torch.from_numpy(self.deps.norm_chw(im)).to(dev)
                xb = torch.stack([x[:, dy:dy + P, dx:dx + P] for (dy, dx) in target.patches])
                s = np.concatenate([self._forward(xb[i:i + 64]).cpu().numpy() for i in range(0, len(xb), 64)])
                res.append(s.max())
            return np.asarray(res, np.float64)[:, None]
        raise XAIUnsupported(f"ResNet için hedef türü desteklenmiyor: {target.kind}")

    def torch_input(self, win_u8, target, ctx):
        torch = _torch()
        return torch.from_numpy(self.deps.norm_chw(win_u8))[None].to(self.deps.device).float()

    def torch_target(self, x, target):
        torch = _torch()
        if target.kind == "patch":
            return torch.sigmoid(self.model(x).float())
        P = self.deps.resnet_patch
        outs = []
        for (dy, dx) in target.patches:
            outs.append(torch.sigmoid(self.model(x[:, :, dy:dy + P, dx:dx + P]).float()))
        return torch.cat(outs, 1).max(dim=1, keepdim=True).values


class UNetWindowAdapter(BaseShapAdapter):
    """512×512 natif pencere. 5 kanallı modelde frekans kanalları her değerlendirmede maskelenmiş RGB'den yeniden
    hesaplanır (uygulamadaki project_freq_channels: tüm görüntü üzerinde; büyük görüntüde pencere ± bağlam)."""
    family = "unet"
    FULL_CONTEXT_MAX_PX = 2_500_000
    CONTEXT_MARGIN = 256

    def _context(self, target, ctx):
        """Frekans kanalları için bağlam: (bağlam görüntüsü, pencerenin bağlam içindeki konumu, not)."""
        img = ctx["prepared"]
        Hp, Wp = img.shape[:2]
        if Hp * Wp <= self.FULL_CONTEXT_MAX_PX:
            return img, (target.y, target.x), "tam görüntü"
        m = self.CONTEXT_MARGIN
        y0, x0 = max(0, target.y - m), max(0, target.x - m)
        y1, x1 = min(Hp, target.y + target.h + m), min(Wp, target.x + target.w + m)
        return img[y0:y1, x0:x1], (target.y - y0, target.x - x0), f"pencere ± {m} px bağlam"

    def _inputs(self, X, target, ctx):
        """NHWC uint8 pencereler → model girdisi (N, C, h, w) (frekans kanalları global, pencere-std. ileri geçişte)."""
        torch = _torch()
        xs = []
        for im in X:
            x = self.deps.norm_chw(im)
            if self.in_ch == 5:
                cimg, (oy, ox), _ = self._context(target, ctx)
                c = cimg.copy()
                c[oy:oy + target.h, ox:ox + target.w] = im
                fc = self.deps.project_freq_channels(c)[:, oy:oy + target.h, ox:ox + target.w]
                x = np.concatenate([x, fc], 0)
            xs.append(x)
        return torch.from_numpy(np.stack(xs)).to(self.deps.device)

    def _reduce_np(self, prob, target):
        if target.kind == "window_image_score":
            v = prob[target.valid] if target.valid is not None else prob
            return self.deps.image_score(v)
        if target.kind in ("region_mean", "gt_bbox_mean"):
            r = target.region
            return float(prob[r].mean()) if r is not None and r.any() else float("nan")
        raise XAIUnsupported(f"U-Net için hedef türü desteklenmiyor: {target.kind}")

    def predict_batch(self, X, target, ctx):
        torch = _torch()
        X = to_u8_batch(X)
        out = []
        bs = 8
        for i in range(0, len(X), bs):
            xb = self._inputs(X[i:i + bs], target, ctx)
            with torch.no_grad():
                pb = self.deps.unet_window_forward(self.model, self.in_ch, xb).float().cpu().numpy()
            out += [self._reduce_np(p, target) for p in pb]
        return np.asarray(out, np.float64)[:, None]

    def torch_input(self, win_u8, target, ctx):
        return self._inputs(win_u8[None], target, ctx).float()

    def torch_target(self, x, target):
        torch = _torch()
        prob = self.deps.unet_window_forward(self.model, self.in_ch, x, amp=False)   # (N, h, w), fp32
        if target.kind == "window_image_score":
            v = prob[:, torch.from_numpy(target.valid).to(prob.device)] if target.valid is not None else prob.flatten(1)
            k = max(1, int(0.01 * v.shape[1]))
            return v.topk(k, dim=1).values.mean(dim=1, keepdim=True)
        r = torch.from_numpy(target.region).to(prob.device)
        return prob[:, r].mean(dim=1, keepdim=True)


class UnsupportedAdapter(BaseShapAdapter):
    def __init__(self, spec, deps):
        self.spec, self.deps = spec, deps

    def explain(self, *a, **k):
        raise XAIUnsupported(f"{self.spec.label}: {self.spec.reason or 'desteklenmiyor'}")


def adapter_for(spec, deps):
    if spec.status != "OK":
        return UnsupportedAdapter(spec, deps)
    if spec.family == "resnet":
        return ResNetPatchAdapter(spec, deps)
    if spec.family == "unet":
        return UNetWindowAdapter(spec, deps)
    return UnsupportedAdapter(spec, deps)


# ------------------------------------------------------------------ toplu yürütme (GPU fonksiyonunun içinde çağrılır)
def resolve_reference(cfg_ref, has_auth):
    if cfg_ref in ("auto", "authentic"):
        if has_auth:
            return "authentic", ""
        return "blur", ("authentic eş yok → blur(32,32)" if cfg_ref == "authentic" else "")
    return cfg_ref, ""


def run_jobs(jobs, rgb, auth, deps, specs, cfg: ExplainConfig, cleanup=True):
    """jobs: [TargetSpec] → [ExplanationResult]. Tüm açıklayıcı nesneleri burada kurulur; dönüş yalnız numpy + meta."""
    torch = _torch()
    preps, out = {}, []
    has_auth = auth is not None and auth.shape == rgb.shape
    for t in jobs:
        spec = specs.get(t.model)
        try:
            if spec is None:
                raise XAIUnsupported(f"{t.model}: kayıtta yok")
            ad = adapter_for(spec, deps)
            if isinstance(ad, UnsupportedAdapter):
                ad.explain()
            if t.prep not in preps:
                img_p, hw, s = prepare_for(t.prep, rgb, deps)
                auth_p = prepare_for(t.prep, auth, deps)[0] if has_auth else None
                preps[t.prep] = (img_p, auth_p)
            img_p, auth_p = preps[t.prep]
            win = np.ascontiguousarray(img_p[t.y:t.y + t.h, t.x:t.x + t.w])
            ref_kind, ref_note = resolve_reference(cfg.reference, has_auth)
            ref = np.ascontiguousarray(auth_p[t.y:t.y + t.h, t.x:t.x + t.w]) if ref_kind == "authentic" else None
            ctx = {"prepared": img_p}
            r = ad.explain(win, ref, ref_kind, t, cfg, ctx)
            if ref_note:
                r.notes.append(ref_note)
            if isinstance(ad, UNetWindowAdapter) and ad.in_ch == 5:
                r.notes.append("frekans kanalları: " + ad._context(t, ctx)[2])
        except XAIUnsupported as e:
            r = ExplanationResult(t.model, spec.label if spec else t.model, spec.family if spec else "?", cfg.method,
                                  t.to_dict(), "—", status="UNSUPPORTED", message=str(e))
        except Exception as e:  # bir modelin hatası diğerlerini durdurmaz
            r = ExplanationResult(t.model, spec.label if spec else t.model, spec.family if spec else "?", cfg.method,
                                  t.to_dict(), "—", status="FAIL", message=f"{type(e).__name__}: {e}")
        out.append(r)
        if cleanup and deps.device is not None and getattr(deps.device, "type", "") == "cuda":
            torch.cuda.empty_cache()
    return out
