"""XAI (SHAP) birim ve bütünlük testleri.

Çalıştırma (proje kökü, `docforensics` ortamı):   pytest tests/test_xai.py -q
Not: app.py içe aktarılır (modeller yüklenir). Model/veri yoksa ilgili testler atlanır. CPU'da ~3–6 dk.
"""
from __future__ import annotations
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(os.environ.get("AI515_APP_DIR", ROOT / "app"))
sys.path.insert(0, str(APP_DIR))


@pytest.fixture(scope="session")
def app():
    os.environ.setdefault("APP_NO_BROWSER", "1")
    import app as A  # noqa: E402
    if not A.TORCH_OK:
        pytest.skip("torch yok")
    if not A._REG:
        pytest.skip("checkpoint bulunamadı (models/)")
    if not A.XAI_OK:
        pytest.fail(f"xai paketi yüklenemedi: {A._XAI_ERR}")
    return A


@pytest.fixture(scope="session")
def sample(app):
    """Eşli küçük bir test örneği (yoksa sentetik)."""
    idx = app.dataset_index()
    cand = idx[(idx.split == "testing") & idx.has_pair & (idx.megapixels <= 0.3)]
    for k in cand.index:
        if app.sample_exists(k):
            f, m, a, meta = app.load_dataset_sample(k)
            if a is not None:
                return dict(key=k, rgb=f, mask=m, auth=a)
    rng = np.random.default_rng(0)
    rgb = (rng.random((400, 300, 3)) * 255).astype(np.uint8)
    return dict(key="synthetic", rgb=rgb, mask=None, auth=None)


def _pick(app, family, in_ch=None):
    for n, sp in app.XAI_SPECS.items():
        if sp.family == family and sp.status == "OK" and (in_ch is None or sp.in_ch == in_ch):
            return n
    pytest.skip(f"{family} (in_ch={in_ch}) modeli yok")


# ------------------------------------------------------------------ kayıt
def test_registry(app):
    specs = app.XAI_SPECS
    # pencere adaptörleri: segmentasyon + ResNet; YOLO11 kutu dedektörleri yalnız tam görüntü hedefleriyle (xai/fullimage.py)
    assert len(specs) == len(app.list_unet_ckpts()) + len(app.list_resnet_ckpts())
    assert all(sp.status == "OK" for n, sp in specs.items() if n in app._REG)
    fams = {sp.family for sp in specs.values()}
    assert fams <= {"unet", "resnet", "yolo"}
    assert all(sp.family == "yolo" for n, sp in specs.items() if n.startswith("YOLO/"))
    unsup = {sp.name: sp.status for sp in app.XAI.UNSUPPORTED_MODELS}
    assert unsup["CLASSICAL/svm_rbf"] == "UNSUPPORTED" and unsup["CLASSICAL/random_forest"] == "UNSUPPORTED"
    rows = app.XAI.registry.inventory_rows(specs)
    assert len(rows) == len(specs) + len(app.XAI.UNSUPPORTED_MODELS)


# ------------------------------------------------------------------ refaktör: _predict_unet davranışı değişmedi
OLD_PREDICT_UNET = textwrap.dedent('''
def _old_predict_unet(name, rgb_u8, win=512, stride=448, max_side=4096, batch=8):
    model, in_ch, _ = get_model(name)
    H0, W0 = rgb_u8.shape[:2]
    resized = max(H0, W0) > max_side
    img = rgb_u8
    if resized:
        s = max_side / max(H0, W0)
        img = cv2.resize(img, (max(1, int(round(W0 * s))), max(1, int(round(H0 * s)))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    ph, pw = max(0, win - h), max(0, win - w)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    H, W = img.shape[:2]
    x = torch.from_numpy(norm_chw(img))
    if in_ch == 5:
        x = torch.cat([x, torch.from_numpy(project_freq_channels(img))], 0)
    x = x.to(DEVICE)
    wins = _windows(H, W, win, stride)
    acc = torch.zeros((H, W), dtype=torch.float32, device=DEVICE); cnt = torch.zeros_like(acc)
    with torch.no_grad():
        for i in range(0, len(wins), batch):
            chunk = wins[i:i + batch]
            xb = torch.stack([x[:, y:y + win, xx:xx + win] for (y, xx) in chunk])
            if in_ch == 5:
                fc = xb[:, 3:]
                mu = fc.mean(dim=(2, 3), keepdim=True); sd = fc.std(dim=(2, 3), keepdim=True) + 1e-6
                xb = torch.cat([xb[:, :3], (fc - mu) / sd], 1)
            with _autocast():
                out = torch.sigmoid(model(xb).float())[:, 0]
            for j, (y, xx) in enumerate(chunk):
                acc[y:y + win, xx:xx + win] += out[j]; cnt[y:y + win, xx:xx + win] += 1
    p = (acc / cnt).cpu().numpy()[:h, :w]
    if resized:
        p = cv2.resize(p, (W0, H0), interpolation=cv2.INTER_LINEAR)
    return p.astype(np.float32)
''')


def _old_fn(app):
    ns = dict(vars(app))
    exec(OLD_PREDICT_UNET, ns)
    return ns["_old_predict_unet"]


@pytest.mark.parametrize("in_ch", [3, 5])
def test_unet_refactor_bitwise(app, sample, in_ch):
    n = _pick(app, "unet", in_ch)
    old = _old_fn(app)
    for img in (sample["rgb"], np.ascontiguousarray(sample["rgb"][:300, :200])):   # normal + dolgu gerektiren
        new, _ = app._predict_unet(n, img)
        assert np.array_equal(new, old(n, img)), f"{n}: _predict_unet çıktısı değişti"


def test_unet_refactor_bitwise_resized(app):
    """Uzun kenar > 4096 → küçültme yolu (sentetik, ince görüntü)."""
    n = _pick(app, "unet", 3)
    rng = np.random.default_rng(1)
    img = (rng.random((520, 4300, 3)) * 255).astype(np.uint8)
    new, info = app._predict_unet(n, img)
    assert info["resized"] == 1
    assert np.array_equal(new, _old_fn(app)(n, img))


# ------------------------------------------------------------------ adaptörler
def _targets(app, sample, names):
    res = app.gpu_infer([n for n in names if "resnet" not in n], [n for n in names if "resnet" in n], [sample["rgb"]], 112)[0]
    return res, app.XAI.decision_targets(res, app.XAI_SPECS, sample["rgb"].shape, app.XAI_DEPS)


def _tol(app):
    """Adaptör f(x) ile uygulama skoru arasındaki izin verilen fark.
    CPU'da çıkarım fp32'dir → 1e-5. CUDA'da uygulama bf16 autocast kullanır (appcore/models.py: _autocast);
    bf16 ~3 anlamlı basamak taşır ve parti bileşimi yuvarlamayı etkiler → 5e-3."""
    return 1e-5 if str(app.DEVICE).startswith("cpu") else 5e-3


def test_adapter_shapes_and_fx_resnet(app, sample):
    n = _pick(app, "resnet")
    res, tg = _targets(app, sample, [n])
    t = tg[0]
    assert t.kind == "patch" and (t.h, t.w) == (224, 224)
    ad = app.XAI.adapter_for(app.XAI_SPECS[n], app.XAI_DEPS)
    img_p = app.XAI.prepare_for("resnet", sample["rgb"], app.XAI_DEPS)[0]
    win = img_p[t.y:t.y + 224, t.x:t.x + 224]
    out = ad.predict_batch(np.stack([win, win, win]), t, {"prepared": img_p})
    assert out.shape == (3, 1)
    smax = max(s for s, _, _ in res[n][1]["scores"])
    assert abs(out[0, 0] - smax) < _tol(app), "adaptör f(x) ≠ uygulamanın maks patch skoru"


def test_window_max_patch_matches_app(app, sample):
    n = _pick(app, "resnet")
    res = app.gpu_infer([], [n], [sample["rgb"]], 112)[0]
    t = app.XAI.common_window_targets([n], res, app.XAI_SPECS, sample["rgb"].shape, sample["mask"], app.XAI_DEPS)[0]
    ad = app.XAI.adapter_for(app.XAI_SPECS[n], app.XAI_DEPS)
    img_p = app.XAI.prepare_for("resnet", sample["rgb"], app.XAI_DEPS)[0]
    fx = ad.predict_batch(img_p[t.y:t.y + t.h, t.x:t.x + t.w][None], t, {"prepared": img_p})[0, 0]
    inside = [s for s, y, x in res[n][1]["scores"] if (y - t.y, x - t.x) in set(map(tuple, t.patches))]
    assert inside and abs(fx - max(inside)) < _tol(app)


@pytest.mark.parametrize("in_ch", [3, 5])
def test_adapter_fx_unet_window(app, sample, in_ch):
    """Adaptör f(x) = aynı pencerede uygulamanın pencere ileri geçişi (_unet_window_forward) + aynı indirgeme."""
    torch = pytest.importorskip("torch")
    n = _pick(app, "unet", in_ch)
    res, tg = _targets(app, sample, [n])
    t = tg[0]
    assert t.kind == "window_image_score" and (t.h, t.w) == (512, 512)
    ad = app.XAI.adapter_for(app.XAI_SPECS[n], app.XAI_DEPS)
    img_p = app.XAI.prepare_for("unet", sample["rgb"], app.XAI_DEPS)[0]
    win = img_p[t.y:t.y + t.h, t.x:t.x + t.w]
    out = ad.predict_batch(np.stack([win, win]), t, {"prepared": img_p})
    assert out.shape == (2, 1)
    model, c, _ = app.get_model(n)
    x = torch.from_numpy(app.norm_chw(img_p))
    if c == 5:
        x = torch.cat([x, torch.from_numpy(app.project_freq_channels(img_p))], 0)
    with torch.no_grad():
        p = app._unet_window_forward(model, c, x[None, :, t.y:t.y + t.h, t.x:t.x + t.w].to(app.DEVICE)).float().cpu().numpy()[0]
    ref = app.image_score(p[t.valid])
    assert abs(out[0, 0] - ref) < _tol(app)
    # Not: tam görüntü haritası (gpu_infer) 448 adımlı pencerelerin örtüşme ORTALAMASIDIR; açıklanan pencere hedefe
    # ortalandığı için o haritayla birebir aynı olması beklenmez — eşitlik pencere ileri geçişi düzeyinde test edilir.


# ------------------------------------------------------------------ SHAP: toplamsallık, desteklenmeyen model, bütünlük
def test_partition_additivity_resnet(app, sample):
    n = _pick(app, "resnet")
    _, tg = _targets(app, sample, [n])
    cfg = app.XAI.ExplainConfig(method="partition", reference="auto", max_evals=200, batch_size=64)
    r = app.XAI.run_jobs(tg, sample["rgb"], sample["auth"], app.XAI_DEPS, app.XAI_SPECS, cfg)[0]
    assert r.status == "OK", r.message
    assert r.values.shape == (224, 224, 3)
    assert r.add_err < 1e-3, f"toplamsallık hatası {r.add_err}"
    if sample["auth"] is not None:
        assert r.reference == "authentic eş"


def test_partition_additivity_unet(app, sample):
    n = _pick(app, "unet", 3)
    _, tg = _targets(app, sample, [n])
    cfg = app.XAI.ExplainConfig(method="partition", reference="blur", max_evals=60, batch_size=8)
    r = app.XAI.run_jobs(tg, sample["rgb"], None, app.XAI_DEPS, app.XAI_SPECS, cfg)[0]
    assert r.status == "OK", r.message
    assert r.values.shape == (512, 512, 3)
    assert r.add_err < 1e-3


def test_unsupported_model_no_fake_output(app, sample):
    sp = app.XAI.UNSUPPORTED_MODELS[0]
    with pytest.raises(app.XAI.XAIUnsupported):
        app.XAI.adapter_for(sp, app.XAI_DEPS).explain()
    specs = dict(app.XAI_SPECS); specs[sp.name] = sp
    t = app.XAI.TargetSpec(sp.name, "patch", "resnet", 0, 0, 224, 224)
    r = app.XAI.run_jobs([t], sample["rgb"], None, app.XAI_DEPS, specs, app.XAI.ExplainConfig(max_evals=50))[0]
    assert r.status == "UNSUPPORTED" and r.values is None


def test_integrity_weights_grads_predictions(app, sample):
    """SHAP (Partition + Gradient) öncesi/sonrası: state_dict sha256 aynı, param.grad None, gpu_infer bit düzeyinde aynı."""
    un, rn = _pick(app, "unet", 3), _pick(app, "resnet")
    fps = {n: app.XAI.model_fingerprint(app.get_model(n)[0]) for n in (un, rn)}
    before = app.gpu_infer([un], [rn], [sample["rgb"]], 112)[0]
    res, tg = _targets(app, sample, [un, rn])
    for method, me in (("partition", 40), ("gradient", 2)):
        cfg = app.XAI.ExplainConfig(method=method, reference="blur", max_evals=me, grad_nsamples=2, batch_size=8)
        out = app.XAI.run_jobs(tg, sample["rgb"], None, app.XAI_DEPS, app.XAI_SPECS, cfg)
        assert all(r.status == "OK" for r in out), [r.message for r in out]
    for n in (un, rn):
        m = app.get_model(n)[0]
        assert app.XAI.model_fingerprint(m) == fps[n], f"{n}: ağırlıklar değişti"
        assert all(p.grad is None for p in m.parameters()), f"{n}: param.grad dolu"
        assert not m.training
    after = app.gpu_infer([un], [rn], [sample["rgb"]], 112)[0]
    for n in (un, rn):
        assert np.array_equal(before[n][0], after[n][0])


def test_gradient_channel_groups_rgbfreq(app, sample):
    n = _pick(app, "unet", 5)
    _, tg = _targets(app, sample, [n])
    cfg = app.XAI.ExplainConfig(method="gradient", reference="blur", grad_nsamples=2, batch_size=2)
    r = app.XAI.run_jobs(tg, sample["rgb"], None, app.XAI_DEPS, app.XAI_SPECS, cfg)[0]
    assert r.status == "OK", r.message
    assert r.values.shape == (512, 512, 5)
    assert set(r.groups) == {"RGB", "FFT-HP", "Laplacian"}


# ------------------------------------------------------------------ yönetici, karşılaştırma, dışa aktarım
def test_manager_cache(app, sample):
    calls = {"n": 0}

    def compute(rgb, auth, jobs, cfg):
        calls["n"] += len(jobs)
        return app.XAI.run_jobs(jobs, rgb, auth, app.XAI_DEPS, app.XAI_SPECS, cfg)

    mgr = app.XAI.XAIManager(lambda: app.XAI_SPECS, compute)
    n = _pick(app, "resnet")
    _, tg = _targets(app, sample, [n])
    cfg = app.XAI.ExplainConfig(max_evals=60, batch_size=64)
    a = mgr.explain(sample["rgb"], sample["auth"], tg, cfg)
    b = mgr.explain(sample["rgb"], sample["auth"], tg, cfg)
    assert calls["n"] == 1 and b[0].cached and np.array_equal(a[0].values, b[0].values)
    mgr.explain(sample["rgb"], sample["auth"], tg, app.XAI.ExplainConfig(max_evals=80, batch_size=64))
    assert calls["n"] == 2   # farklı bütçe → yeniden hesap


def test_compare_and_export(app, sample, tmp_path):
    S = app.XAI.service.ExplanationResult
    rng = np.random.default_rng(0)
    base = rng.normal(size=(64, 64, 3)).astype(np.float16)
    mk = lambda v, lab, kind: S("m" + lab, lab, "unet", "partition", dict(kind=kind, window=[0, 0, 64, 64], scale=1.0, label=lab),
                                "blur(32,32)", values=v, fx=0.5, base=0.1, sum_shap=0.4, add_err=0.0)
    rs = [mk(base, "A", "window_image_score"), mk(base, "B", "window_image_score"),
          mk(rng.normal(size=(64, 64, 3)).astype(np.float16), "C", "window_max_patch")]
    cmp = app.XAI.compare.compare_results(rs)
    assert np.allclose(np.diag(cmp["iou"]), 1) and np.allclose(cmp["iou"], cmp["iou"].T)
    assert cmp["iou"][0, 1] == pytest.approx(1.0) and cmp["spearman"][0, 1] == pytest.approx(1.0)
    assert not cmp["check"]["same_kind"] and "⚠️" in cmp["note"]
    files = app.XAI.export_results(rs, tmp_path, {"title": "t"}, comparison=cmp,
                                   extra_images={"k.png": app.XAI.compare.comparison_figure(cmp)})
    names = {Path(f).name for f in files}
    assert {"shap_report.json", "shap_report.md", "shap_values.npz", "k.png"} <= names
