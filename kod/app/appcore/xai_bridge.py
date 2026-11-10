"""appcore.xai_bridge — XAI köprüsü: SHAP kurulumu, tahmin, açıklama, model karşılaştırma, dışa aktarım (xai/ paketi)."""
from __future__ import annotations
from ._common import *  # noqa: F401,F403 — ortak içe aktarmalar
from .config import (  # noqa: F401
    APP_DIR, DEVICE, ZERO_GPU, gpu
)
from .imaging import (  # noqa: F401
    green_marker_mask, project_freq_channels, q_u8, to_gray, to_rgb
)
from .models import (  # noqa: F401
    _REG, _autocast, _unet_prepare, _unet_window_forward, _windows, friendly, get_model, gpu_infer, list_resnet_ckpts,
    list_seg_ckpts, list_unet_ckpts, norm_chw
)
from .metrics import (  # noqa: F401
    default_tau, image_score, image_threshold, postprocess, resnet_decision
)

# ================================================================== XAI · SHAP AÇIKLAMALARI (sekme 8 › SHAP Açıklamaları)
# Ayrı paket: app/xai (registry · targets · adapters · manager · viz · compare · service). Ayrıntı: docs/XAI.md
try:
    sys.path.insert(0, str(APP_DIR))
    import xai as XAI
    XAI_OK, _XAI_ERR = True, ""
except Exception as _e:  # xai yoksa diğer sekmeler etkilenmez
    XAI, XAI_OK, _XAI_ERR = None, False, str(_e)


try:
    import shap as _shap  # noqa: F401
    SHAP_OK, _SHAP_ERR = True, ""
except Exception as _e:
    SHAP_OK, _SHAP_ERR = False, str(_e)

XAI_SOURCES = ["Girdi sekmesi (orijinal)", "Ön işleme pipeline çıktısı", "Son simülasyon girdisi", "Yeni yükleme"]


# Tam görüntü hedefleri (xai/fullimage.py): açıklama orijinal görüntünün TAMAMINI tek hedef + tek referansla kapsar.
# Döşemeli mozaik yerel pencere açıklamalarının birleşimidir (tabanlar farklı) → tam görüntü SHAP olarak sunulmaz.
XT_FULL_DEC = "Tam görüntü · karar hedefi (önerilen)"


XT_FULL_DET = "Tam görüntü · tespitler (ilk k bölge / patch / kutu)"


XT_FULL_GT = "Tam görüntü · GT kutusu"


XT_TILES = "Döşemeli yerel mozaik (512 px pencereler · tam görüntü SHAP DEĞİLDİR)"


XT_WIN_DEC = "Pencere · karar hedefi (512 px pencere / 224 patch)"


XT_WIN_DET = "Pencere · tespitler (ilk k bölge / patch)"


XT_WIN_GT = "Pencere · GT kutusu"


XT_COMMON = "Ortak pencere (karşılaştırma)"


XAI_TARGETS = [XT_FULL_DEC, XT_FULL_DET, XT_FULL_GT, XT_TILES, XT_WIN_DEC, XT_WIN_DET, XT_WIN_GT, XT_COMMON]


XAI_METHODS = ["Partition (önerilen)", "Gradient (ince doku)"]


XAI_REFS = ["Otomatik (authentic eş varsa o, yoksa blur)", "authentic eş", "blur(32,32)", "inpaint_telea"]


_XAI_REF_KEY = {XAI_REFS[0]: "auto", XAI_REFS[1]: "authentic", XAI_REFS[2]: "blur", XAI_REFS[3]: "inpaint"}


XAI_DEPS = None


XAI_SPECS: dict = {}


XAI_MGR = None


def xai_setup():
    """Modeller yüklendikten sonra çağrılır: bağımlılık enjeksiyonu + model kaydı + yönetici."""
    global XAI_DEPS, XAI_SPECS, XAI_MGR
    if not (XAI_OK and TORCH_OK):
        return
    XAI_DEPS = XAI.Deps(get_model=get_model, norm_chw=norm_chw, project_freq_channels=project_freq_channels,
                        windows=_windows, autocast=_autocast, device=DEVICE, unet_window_forward=_unet_window_forward,
                        unet_prepare=_unet_prepare, image_score=image_score, q_u8=q_u8, postprocess=postprocess,
                        default_tau=default_tau, image_threshold=image_threshold, friendly=friendly,
                        list_unet=list_seg_ckpts, list_resnet=list_resnet_ckpts, green_marker_mask=green_marker_mask)
    XAI_SPECS = XAI.build_registry(XAI_DEPS, _REG)
    try:  # YOLO11 kutu dedektörleri: tam görüntü kutu-güveni hedefleri (pencere adaptörleri piksel/patch modelleri içindir)
        XAI_SPECS.update(XAI.fullimage.yolo_specs([n for n in list_unet_ckpts() if n.startswith("YOLO/")], XAI_DEPS))
    except Exception as _e:
        print("[XAI] YOLO kaydı yapılamadı:", _e)
    XAI_MGR = XAI.XAIManager(lambda: XAI_SPECS, gpu_explain)


def _xai_is_full(t):
    return XAI.fullimage.is_full(t.kind) or t.kind == XAI.fullimage.TILE_KIND


def _xai_full_per_eval(t, sp):
    """Tam görüntü işinde değerlendirme başına tahmini süre (s) — ölçüm (RTX 5080): 3 kanal U-Net ≈ 6 ms/pencere,
    5 kanal U-Net ≈ 0.25 s/MP (tam görüntü frekans kanalları, CPU), ResNet ≈ 0.4 ms/patch, YOLO ≈ 25 ms/kare."""
    F = XAI.fullimage
    if t.kind == F.TILE_KIND:
        n = len(F.tile_grid(max(t.h, 512), max(t.w, 512)))
        return n * (0.003 if sp.family == "resnet" else (0.016 if sp.in_ch == 5 else 0.004))
    if sp.family == "resnet":
        return 0.0004 * max(1, len(t.patches or []))
    if sp.family == "yolo":
        n = 1 if "-full-" in t.model else len(_windows(max(t.h, 1024), max(t.w, 1024), 1024, 768))
        return 0.025 * n
    n = len(_windows(max(t.h, 512), max(t.w, 512), 512, 448))
    if sp.in_ch == 5:  # frekans kanalları (FFT-HP + Laplacian) her değerlendirmede tüm görüntüde CPU'da yeniden hesaplanır
        return n * 0.006 + 0.25 * t.h * t.w / 1e6
    return n * 0.006


def _xai_job_cost(t, cfg):
    sp = XAI_SPECS.get(t.model)
    if sp is None:
        return 1.0
    if _xai_is_full(t):
        return 5 + int(getattr(t, "budget", None) or cfg.max_evals) * _xai_full_per_eval(t, sp)
    if cfg.method == "gradient":
        per = 0.03 if sp.family == "unet" else 0.01 * max(1, len(t.patches or [1]))
        return 2 + cfg.grad_nsamples * per
    if sp.family == "resnet":
        per = 0.004 * (len(t.patches) if t.kind == "window_max_patch" else 1)
    else:
        per = 0.012 + (0.05 if sp.in_ch == 5 else 0.0)
    return 2 + cfg.max_evals * (per + 0.002)


def _dur_explain(rgb, auth, jobs, cfg):
    return int(min(120, max(30, 15 + sum(_xai_job_cost(t, cfg) for t in jobs))))


@gpu(duration=_dur_explain)
def _gpu_explain(rgb, auth, jobs, cfg):
    """SHAP açıklayıcıları bu fonksiyonun İÇİNDE kurulur (ZeroGPU); dönüş yalnız numpy + meta veri.
    Tam görüntü / döşemeli işler xai.fullimage'e, pencere işleri mevcut adaptörlere gider (sıra korunur)."""
    out = [None] * len(jobs)
    idx_f = [i for i, t in enumerate(jobs) if _xai_is_full(t)]
    idx_w = [i for i in range(len(jobs)) if i not in idx_f]
    if idx_f:
        for i, r in zip(idx_f, XAI.fullimage.run_full_jobs([jobs[i] for i in idx_f], rgb, auth, XAI_DEPS, XAI_SPECS, cfg)):
            out[i] = r
    if idx_w:
        for i, r in zip(idx_w, XAI.run_jobs([jobs[i] for i in idx_w], rgb, auth, XAI_DEPS, XAI_SPECS, cfg)):
            out[i] = r
    return out


def gpu_explain(rgb, auth, jobs, cfg):
    """İşleri ZeroGPU süre sınırına (≤120 s) sığacak parçalara bölerek çalıştırır."""
    out, chunk, est = [], [], 0.0
    for t in jobs:
        c = _xai_job_cost(t, cfg)
        if chunk and est + c > 100:
            out += _gpu_explain(rgb, auth, chunk, cfg); chunk, est = [], 0.0
        chunk.append(t); est += c
    if chunk:
        out += _gpu_explain(rgb, auth, chunk, cfg)
    return out


def xai_unsupported_md():
    if not XAI_OK:
        return f"⚠️ XAI paketi yüklenemedi: `{_XAI_ERR}`"
    lines = ["**SHAP üretilmeyen modeller:**"]
    for sp in XAI.UNSUPPORTED_MODELS:
        lines.append(f"- {sp.label} — *{sp.status}*: {sp.reason}")
    for n, sp in XAI_SPECS.items():
        if sp.status != "OK":
            lines.append(f"- {sp.label} — *{sp.status}*: {sp.reason}")
    if not SHAP_OK:
        lines.append(f"- ⚠️ `shap` paketi yok: {_SHAP_ERR}")
    return "\n".join(lines)


def _xai_source(src, orig, cur, mask, auth, sim, up_img, up_auth, up_gt):
    note = []
    if src == XAI_SOURCES[0]:
        rgb, a, g, desc = orig, auth, mask, "Girdi sekmesi (orijinal)"
    elif src == XAI_SOURCES[1]:
        rgb, a, g, desc = cur, auth, mask, "Ön işleme pipeline çıktısı"
        if cur is not None and orig is not None and cur is not orig:
            note.append("referans authentic eş işlenmemiş haliyle kullanılır")
    elif src == XAI_SOURCES[2]:
        if not sim:
            raise gr.Error("Henüz simülasyon çalıştırılmadı (sekme 0).")
        rgb, a, g, desc = sim.get("inp"), sim.get("auth"), sim.get("mask"), f"Son simülasyon girdisi ({sim.get('key')})"
    else:
        rgb, a, g, desc = up_img, up_auth, up_gt, "Yeni yükleme"
    if rgb is None:
        raise gr.Error("Görüntü yok: Girdi sekmesinden yükleyin, Simülasyon'dan aktarın ya da 'Yeni yükleme' ile ekleyin.")
    rgb = np.ascontiguousarray(to_rgb(rgb).astype(np.uint8))
    if a is not None:
        a = np.ascontiguousarray(to_rgb(a).astype(np.uint8))
        if a.shape != rgb.shape:
            note.append("authentic eş farklı boyutta → kullanılmadı"); a = None
        elif np.array_equal(a, rgb):
            note.append("girdi authentic eşin kendisi → referans olarak kullanılmadı"); a = None
    if g is not None:
        g = to_gray(g) if g.ndim == 3 else g
        if g.shape != rgb.shape[:2]:
            note.append("GT maskesi farklı boyutta → kullanılmadı"); g = None
    return rgb, a, g, desc + (" · " + "; ".join(note) if note else "")


def xai_predict(src, unets, resnets, orig, cur, mask, auth, sim, up_img, up_auth, up_gt):
    """① Tahmin: mevcut gpu_infer DEĞİŞTİRİLMEDEN çağrılır; skor, eşik, karar ve tespit listesi."""
    if not (XAI_OK and TORCH_OK):
        raise gr.Error(f"XAI kullanılamıyor: {_XAI_ERR or _TORCH_ERR}")
    unets, resnets = list(unets or []), list(resnets or [])
    if not unets and not resnets:
        raise gr.Error("En az bir model seçin.")
    rgb, a, g, desc = _xai_source(src, orig, cur, mask, auth, sim, up_img, up_auth, up_gt)
    t0 = time.time()
    res = gpu_infer(unets, resnets, [rgb], 112)[0]
    preds, rows = {}, []
    for n in resnets:
        hm, info = res[n]
        preds[n] = (hm, info)
        smax, rthr, _ = resnet_decision(n, info["scores"])
        rows.append({"Model": friendly(n), "Aile": "ResNet-18", "Görüntü skoru": smax, "Eşik": rthr,
                     "Karar": "SAHTE" if smax >= rthr else "GERÇEK", "Patch sayısı": len(info["scores"])})
    for n in unets:
        prob, info = res[n]
        preds[n] = (prob, info)
        sc = image_score(prob); it, cal = image_threshold(n)
        sp = XAI_SPECS.get(n)
        if n.startswith("YOLO/"):
            bx = info.get("boxes") or []
            rows.append({"Model": friendly(n), "Aile": "YOLO11", "Görüntü skoru": sc, "Eşik": it,
                         "Karar": "SAHTE" if sc >= it else "GERÇEK", "Patch sayısı": None,
                         "Tespit bölgesi (τ*)": (f"{info.get('n_boxes', len(bx))} kutu · en yüksek güven {bx[0][0]:.3f}" if bx
                                                 else "kutu yok") + ("" if cal else " · eşik kalibre değil")})
            continue
        tau = default_tau(n, (sp.train_gt if sp else "") or "bbox")
        nreg = len(XAI.targets.unet_regions(prob, tau, XAI_DEPS))
        rows.append({"Model": friendly(n), "Aile": "U-Net", "Görüntü skoru": sc, "Eşik": it,
                     "Karar": "SAHTE" if sc >= it else "GERÇEK", "Patch sayısı": None,
                     "Tespit bölgesi (τ*)": f"{nreg} (τ={tau:.2f})" + ("" if cal else " · eşik kalibre değil")})
    st = dict(rgb=rgb, auth=a, gt=g, preds=preds, desc=desc, models=resnets + unets)
    md = (f"**Kaynak:** {desc} · {rgb.shape[1]}×{rgb.shape[0]} px · GT {'var' if g is not None else 'yok'} · "
          f"authentic eş {'var' if a is not None else 'yok'} · tahmin {time.time() - t0:.1f} s. "
          "Şimdi **② SHAP analizini başlat** ile açıklayın (tahminler tekrar hesaplanmaz).")
    return pd.DataFrame(rows), md, st


def _xai_cfg(method, ref, evals):
    return XAI.ExplainConfig(method="gradient" if str(method).startswith("Gradient") else "partition",
                             reference=_XAI_REF_KEY.get(ref, "auto"), max_evals=int(evals),
                             batch_size=16, grad_nsamples=64)


def _xai_jobs(st, mode, k, cfg=None):
    shape = st["rgb"].shape
    preds = st["preds"]
    F = XAI.fullimage
    fam = lambda n: getattr(XAI_SPECS.get(n), "family", "")
    if mode in (XT_FULL_DEC, XT_FULL_DET, XT_FULL_GT, XT_TILES):
        if mode == XT_FULL_DET:
            jobs = F.full_detection_targets(preds, XAI_SPECS, shape, XAI_DEPS, k=int(k))
        elif mode == XT_FULL_GT:
            if st["gt"] is None:
                raise gr.Error("Bu görüntüde GT maskesi yok; başka bir hedef seçin.")
            jobs = F.full_gt_targets(st["models"], XAI_SPECS, shape, st["gt"], XAI_DEPS, preds)
        elif mode == XT_TILES:
            jobs = F.tile_targets(st["models"], XAI_SPECS, shape, XAI_DEPS)
        else:
            jobs = F.full_decision_targets(preds, XAI_SPECS, shape, XAI_DEPS)
        if ZERO_GPU and cfg is not None:  # ZeroGPU çağrı sınırı (≤120 s): bütçe gerekirse düşürülür (sonuç notunda yazılır)
            keep = []
            for t in jobs:
                c = _xai_job_cost(t, cfg)
                if c > 100:
                    b = int(cfg.max_evals * 95 / c)
                    if b < 60:  # bu çözünürlükte anlamlı bir tam görüntü açıklaması süre sınırına sığmıyor → sahte/kaba sonuç üretme
                        gr.Warning(f"{friendly(t.model)}: tam görüntü SHAP bu çözünürlükte ZeroGPU 120 s sınırına sığmıyor "
                                   f"(tahmini {c:.0f} s) — 'Döşemeli yerel mozaik' ya da 'Pencere' hedeflerini kullanın.")
                        continue
                    t.budget = b
                keep.append(t)
            jobs = keep
    else:
        preds = {n: p for n, p in preds.items() if fam(n) != "yolo"}
        models = [n for n in st["models"] if fam(n) != "yolo"]
        if not models:
            raise gr.Error("Pencere hedefleri piksel/patch modelleri içindir; YOLO11 için 'Tam görüntü' hedeflerini seçin.")
        if mode == XT_WIN_DET:
            jobs = XAI.detection_targets(preds, XAI_SPECS, shape, XAI_DEPS, k=int(k))
        elif mode == XT_WIN_GT:
            if st["gt"] is None:
                raise gr.Error("Bu görüntüde GT maskesi yok; başka bir hedef seçin.")
            jobs = XAI.gt_targets(models, XAI_SPECS, shape, st["gt"], XAI_DEPS)
        elif mode == XT_COMMON:
            jobs = XAI.common_window_targets(models, preds, XAI_SPECS, shape, st["gt"], XAI_DEPS)
        else:
            jobs = XAI.decision_targets(preds, XAI_SPECS, shape, XAI_DEPS)
    if not jobs:
        raise gr.Error("Açıklanacak hedef bulunamadı (ör. eşik üstü tespit bölgesi yok). Başka bir hedef türü deneyin.")
    return jobs


def _xai_window_views(st, t):
    img = XAI.prepare_for(t.prep, st["rgb"], XAI_DEPS)[0][t.y:t.y + t.h, t.x:t.x + t.w]
    gt_w = XAI.window_mask(st["gt"], t, XAI_DEPS) if st["gt"] is not None else None
    gm = green_marker_mask(st["rgb"])
    if st["auth"] is not None:
        gm &= ~green_marker_mask(st["auth"])
    mk_w = XAI.window_mask(gm, t, XAI_DEPS)
    return img, gt_w, mk_w


def _xai_full_views(st, t):
    """Tam görüntü sonuçları için ekran çözünürlüğünde (en-boy korunur, ≤2400 px) orijinal + GT + işaretçi + hedef alanı;
    tam çözünürlüklü maskeler de döner (kütle ve dağılım hesapları orijinal çözünürlükte yapılır)."""
    V = XAI.viewer
    rgb = st["rgb"]
    H0, W0 = rgb.shape[:2]
    img = V.display(rgb)
    hw = img.shape[:2]
    g = st["gt"]
    gt = None if g is None else (np.asarray(g) > (127 if np.asarray(g).dtype == np.uint8 else 0))
    gm = green_marker_mask(rgb)
    if st["auth"] is not None:
        gm &= ~green_marker_mask(st["auth"])
    reg = XAI.fullimage.target_region_orig(t, H0, W0, XAI_DEPS)
    return img, gt, gm, reg, V.display_mask(gt, hw), V.display_mask(gm, hw), V.display_mask(reg, hw)


def _xai_full_badge(st, r, t):
    """Tam görüntü iddiası yalnız doğrulanırsa: harita boyutu = orijinal, tek hedef/tek referans, E[f]+ΣSHAP=f(x)."""
    H0, W0 = st["rgb"].shape[:2]
    m = r.shap_map()
    if t.kind == XAI.fullimage.TILE_KIND:
        tg = r.target
        return (f"**DÖŞEMELİ YEREL MOZAİK — tam görüntü SHAP DEĞİLDİR.** {len(tg.get('tiles', []))} pencere · kapsama "
                f"{'tam' if tg.get('covered') else 'EKSİK'} · pencere tabanları farklı (fark {tg.get('base_spread', 0):.3f}); "
                "her pencere yalnız kendi skorunu açıklar.  \n")
    ok = bool(r.target.get("full")) and m.shape == (H0, W0) and np.isfinite(r.add_err) and r.add_err < 1e-3
    if ok:
        return (f"**TAM GÖRÜNTÜ SHAP (doğrulandı):** harita {W0}×{H0} px = orijinal görüntü, kırpma yok · tek hedef, tek referans · "
                f"E[f] + ΣSHAP = f(x) (hata {r.add_err:.1e}) · ızgara {r.target.get('grid', ['?', '?'])[1]}×"
                f"{r.target.get('grid', ['?', '?'])[0]} hücre ({r.target.get('cell_px', '?')} px).  \n")
    return (f"⚠️ **Tam görüntü doğrulanamadı** (harita {m.shape[1]}×{m.shape[0]}, toplamsallık hatası {r.add_err:.1e}) — "
            "sonuç tam görüntü SHAP açıklaması olarak SUNULMUYOR.  \n")


def _xai_app_score(st, model):
    """Uygulamanın tahmin tablosundaki görüntü skoru (U-Net: tam görüntü top-%1; ResNet: maks patch)."""
    pr = st["preds"].get(model)
    if pr is None:
        return None
    m, info = pr
    if "scores" in info:
        return max(s_ for s_, _, _ in info["scores"])
    return image_score(m)


def _xai_label(i, r):
    return f"{i + 1} · {r.label} · {r.target.get('label', '')}"[:150]


def _xai_render(st, results, jobs):
    """Tüm sonuçlar için paneller, bölge kütleleri ve dışa aktarım dosyaları."""
    panels, masses, gal, sums = {}, {}, [], {}
    for i, (r, t) in enumerate(zip(results, jobs)):
        lab = _xai_label(i, r)
        if not r.ok:
            sums[lab] = f"**{r.label}** — {r.status}: {r.message}"
            continue
        badge = ""
        if _xai_is_full(t):
            import copy as _copy
            m = r.shap_map()
            img, gt, gm, reg, gt_d, gm_d, reg_d = _xai_full_views(st, t)
            rd = _copy.copy(r); rd.values = XAI.viewer.display_map(m)[..., None]
            pn = XAI.viz.panels(rd, img, gt_d, gm_d, reg_d)
            pn["Özet şekil"] = XAI.viz.summary_figure(rd, img, gt_d, gm_d)
            pn["Katkı dağılımı"] = XAI.viewer.distribution_figure(
                m, gt, r.fx, r.base, f"{r.label} · açıklanan hedef: SAHTE yönü (+) · {r.target.get('label', '')}"[:160])
            masses[f"{i + 1} · {r.label}"] = XAI.viz.region_masses(m, gt, gm)
            badge = _xai_full_badge(st, r, t)
        else:
            img, gt_w, mk_w = _xai_window_views(st, t)
            pn = XAI.viz.panels(r, img, gt_w, mk_w, t.region if t.kind in ("region_mean", "gt_bbox_mean") else None)
            pn["Özet şekil"] = XAI.viz.summary_figure(r, img, gt_w, mk_w)
            pn["Katkı dağılımı"] = XAI.viewer.distribution_figure(r.shap_map(), gt_w, r.fx, r.base,
                                                                  f"{r.label} · pencere açıklaması · {r.target.get('label', '')}"[:160])
            masses[f"{i + 1} · {r.label}"] = XAI.viz.region_masses(r.shap_map(), gt_w, mk_w)
        panels[lab] = pn
        for pname, im in pn.items():
            if pname != "Özet şekil":
                gal.append((im, f"{i + 1} · {r.label} — {pname}"))
        g = "".join(f" · {k_} {v:+.4f}" for k_, v in r.groups.items()) if r.groups else ""
        app_sc = _xai_app_score(st, r.model)
        cmp_note = ""
        if app_sc is not None and t.kind == "window_image_score":
            cmp_note = (f"  \nUygulamanın tam görüntü skoru (448 px adımlı pencerelerin örtüşme ortalaması): {app_sc:.4f}"
                        + (" — açıklanan pencere tek başına çok daha yüksek skor veriyor; örtüşme ortalaması sinyali seyreltiyor."
                           if r.fx - app_sc > 0.25 else ""))
        sums[lab] = (badge + f"**{r.label}** · hedef: {r.target.get('label', '')}  \n"
                     f"f(x) = **{r.fx:.4f}** · E[f] (referans: {r.reference}) = {r.base:.4f} · ΣSHAP = {r.sum_shap:+.4f} · "
                     f"toplamsallık hatası {r.add_err:.2e} · {r.n_evals} değerlendirme · {r.seconds:.1f} s"
                     + (f" · tepe {r.peak_mem_mb:.0f} MB" if r.peak_mem_mb else "") + g
                     + cmp_note
                     + ("  \n_" + "; ".join(r.notes) + "_" if r.notes else "")
                     + ("  \n(önbellekten)" if r.cached else ""))
    mass_img = XAI.viz.mass_chart(masses)
    loc = XAI.viz.location_image(st["rgb"], [r for r in results])
    return panels, masses, gal, sums, mass_img, loc


def _xai_export(st, results, panels, mass_img, loc, cfg, cmp=None, cmp_img=None, tag="shap"):
    outdir = Path(tempfile.mkdtemp(prefix=f"ai515_{tag}_"))
    meta = dict(title=st["desc"], image=f"{st['rgb'].shape[1]}×{st['rgb'].shape[0]}", source=st["desc"],
                config=dict(method=cfg.method, reference=cfg.reference, max_evals=cfg.max_evals, grad_nsamples=cfg.grad_nsamples),
                device=str(DEVICE), zero_gpu=ZERO_GPU)
    extra = {"konum.png": loc, "bolge_kutlesi.png": mass_img}
    if cmp_img is not None:
        extra["karsilastirma.png"] = cmp_img
    return XAI.export_results(results, outdir, meta, panels, cmp, extra)


def _xai_hires(st, results, jobs, outdir):
    """Tam görüntü sonuçları: natif çözünürlükte ısı haritası + bindirme PNG, float32 SHAP NPY, 300 dpi PNG + PDF yayın
    şekli (renk çubuğu, piksel eksenleri, GT / hedef konturu)."""
    V, F = XAI.viewer, XAI.fullimage
    files = []
    rgb = st["rgb"]
    H0, W0 = rgb.shape[:2]
    for i, (r, t) in enumerate(zip(results, jobs)):
        if not r.ok or not _xai_is_full(t):
            continue
        try:
            m = r.shap_map()
            _, gt, _, reg, _, _, _ = _xai_full_views(st, t)
            v = XAI.viz.sym_vmax(m)
            heat = XAI.viz.diverging(m, v)
            ov = XAI.viz.blend(rgb, heat, 0.75, np.clip(np.abs(m) / v, 0, 1))
            stem = f"r{i + 1:02d}_" + re.sub(r"[^A-Za-z0-9]+", "_", r.model).strip("_")
            files += V.export_native(outdir, stem, {"shap_isi_natif": heat, "bindirme_natif": ov}, {"shap_float32": m})
            kind = ("döşemeli yerel mozaik (tam görüntü SHAP değil)" if t.kind == F.TILE_KIND else "tam görüntü SHAP")
            files += V.publication_figure(rgb, m, f"{r.label} · {kind} · {W0}×{H0} px · f(x)={r.fx:.3f} · E[f]={r.base:.3f} · "
                                                  f"ΣSHAP={r.sum_shap:+.3f}", outdir, stem, gt=gt, region=reg)
        except Exception as e:
            print("[XAI] yüksek çözünürlüklü dışa aktarım:", r.model, e)
    return files


_XAI_PANEL_NAMES = ["Orijinal (açıklanan pencere)", "SHAP ısı haritası (kırmızı: SAHTE yönü · mavi: GERÇEK yönü)", "Bindirme",
                    "Pozitif katkı (SAHTE'yi artıran)", "Negatif katkı (GERÇEK'e iten)", "Sınıf panelleri: SAHTE | GERÇEK (= −SAHTE)"]


def _xai_show_outputs(view, choice):
    """[özet, 6 panel, kaydırıcı (orijinal ↔ bindirme), katkı dağılımı]"""
    if not view or choice not in view.get("panels", {}):
        md = view.get("sums", {}).get(choice, " ") if view else " "
        return [md] + [None] * (len(_XAI_PANEL_NAMES) + 2)
    pn = view["panels"][choice]
    a, b = pn.get(_XAI_PANEL_NAMES[0]), pn.get(_XAI_PANEL_NAMES[2])
    sl = (a, b) if a is not None and b is not None and a.shape == b.shape else None
    return [view["sums"].get(choice, " ")] + [pn.get(n) for n in _XAI_PANEL_NAMES] + [sl, pn.get("Katkı dağılımı")]


def xai_run(st, mode, k, method, ref, evals):
    """② SHAP: tahminden bağımsız; önbellekte olan (model, hedef, yöntem, referans, bütçe) yeniden hesaplanmaz."""
    if not st:
        raise gr.Error("Önce ① Tahmin yap.")
    if not SHAP_OK:
        raise gr.Error(f"shap paketi yüklü değil: {_SHAP_ERR}")
    cfg = _xai_cfg(method, ref, evals)
    jobs = _xai_jobs(st, mode, k, cfg)
    t0 = time.time()
    results = XAI_MGR.explain(st["rgb"], st["auth"], jobs, cfg)
    panels, masses, gal, sums, mass_img, loc = _xai_render(st, results, jobs)
    files = _xai_export(st, results, panels, mass_img, loc, cfg)
    if files:
        files += _xai_hires(st, results, jobs, Path(files[0]).parent)
    df = pd.DataFrame(XAI.service.result_table_rows(results))
    if len(df):
        df.insert(5, "Uygulama skoru", [_xai_app_score(st, r.model) for r in results])
    choices = list(sums)
    view = dict(panels=panels, sums=sums)
    n_ok = sum(r.ok for r in results)
    status = (f"**SHAP tamam:** {n_ok}/{len(results)} açıklama · {time.time() - t0:.1f} s · önbellek isabeti "
              f"{sum(r.cached for r in results)} · hedef: {mode} · yöntem: {cfg.method} · bütçe: "
              f"{cfg.max_evals if cfg.method == 'partition' else cfg.grad_nsamples}")
    first = choices[0] if choices else None
    return [status, df, loc, gr.update(choices=choices, value=first), view, mass_img, gal, files] + _xai_show_outputs(view, first)


def xai_compare(st, method, ref, evals):
    """③ Karşılaştırma: tüm modeller AYNI pencereyi açıklar (önbellek kullanılır); IoU/Dice, Spearman, ortak/özgü bölge."""
    if not st:
        raise gr.Error("Önce ① Tahmin yap.")
    if not SHAP_OK:
        raise gr.Error(f"shap paketi yüklü değil: {_SHAP_ERR}")
    cfg = _xai_cfg(method, ref, evals)
    jobs = _xai_jobs(st, XT_COMMON, 1, cfg)
    results = XAI_MGR.explain(st["rgb"], st["auth"], jobs, cfg)
    cmp = XAI.compare.compare_results(results, st["rgb"])
    fig = XAI.compare.comparison_figure(cmp)
    rows = []
    for i, r in enumerate(results):
        rows.append({"#": i + 1, "Model": r.label, "Hedef türü": r.target.get("kind"), "f(x)": r.fx, "E[f]": r.base,
                     "ΣSHAP": r.sum_shap, "max|SHAP| (normalizasyon)": float(np.abs(r.shap_map()).max()) if r.ok else None,
                     "Durum": r.status + (" (önbellek)" if r.cached else "")})
    md = "**Ölçek / hedef uyumu:** " + cmp["note"]
    if "iou" in cmp:
        n = len(cmp["names"])
        off = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if off:
            md += (f"  \nOrtalama top-%5 IoU {np.nanmean([cmp['iou'][i, j] for i, j in off]):.3f} · ortalama Spearman ρ "
                   f"{np.nanmean([cmp['spearman'][i, j] for i, j in off]):.3f} · tüm modellerde ortak bölge payı "
                   f"%{100 * cmp['common_frac']:.1f}")
    panels, masses, gal, sums, mass_img, loc = _xai_render(st, results, jobs)
    files = _xai_export(st, results, panels, mass_img, loc, cfg, cmp, fig, tag="shap_cmp")
    return md, fig, pd.DataFrame(rows), files
