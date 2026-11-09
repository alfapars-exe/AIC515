"""
tests/test_xai_filters.py — XAI · Katman Filtreleri doğrulama testleri (gerçek checkpoint'lerle).

  python tests\\test_xai_filters.py            # tüm kontroller (CPU; GPU boşsa ek cuda kontrolü)
  python tests\\test_xai_filters.py --quick    # model başına küçük bölge

Kontroller
  T1  Sentetik: bilinen kernel (Sobel) → modelin conv çıktısı = cv2.filter2D (sıfır dolgu, çapraz korelasyon)
  T2  Conv katmanı olmayan model / checkpoint → NoConvError (sahte filtre üretilmez)
  T3  Her kayıtlı model: mimari otomatik tespiti, ilk-katman kuralı (adım ≤ 4), checkpoint kernel'i = modül ağırlığı,
      bağımsız F.conv2d = modelin conv çıktısı, ilk conv girdisi = ön işlenmiş tensör, kanal katkılarının toplamı = conv,
      ağırlık SHA-256 ve checkpoint dosyası analiz sonrası değişmedi, aşama zinciri conv→BN→aktivasyon
  T4  YOLO: export_yolo_stem (ultralytics'siz dosya) çıktıları = tam checkpoint çıktıları
  T5  GPU (boş bellek ≥ 1 GB ise): cuda haritaları ≈ CPU haritaları
  T6  Görselleştirme fonksiyonları her aile için görüntü üretir; ham dışa aktarım dosyaları yazılır
Sonuç: reports/xai_filters/test_results.json
"""
from __future__ import annotations
import glob, hashlib, json, math, os, sys, tempfile, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import numpy as np
import cv2
import torch
import torch.nn as nn
import xai_filters as XF

QUICK = "--quick" in sys.argv
OUT = ROOT / "reports" / "xai_filters"
OUT.mkdir(parents=True, exist_ok=True)
RES = dict(checks=[], models=[], started=time.strftime("%Y-%m-%d %H:%M:%S"))
FAIL = []


def check(name, cond, detail=""):
    RES["checks"].append(dict(name=name, ok=bool(cond), detail=str(detail)))
    if not cond:
        FAIL.append(name)
    print(("  OK  " if cond else "  FAIL") + f" {name}" + (f" — {detail}" if detail else ""), flush=True)


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_image():
    ims = sorted(glob.glob(str(ROOT / "data" / "AIForge-Doc-v1" / "TestingSet" / "images" / "*.png")))
    p = ims[0]
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    m = cv2.imread(p.replace("images", "masks"), cv2.IMREAD_GRAYSCALE)
    return img, m, Path(p).name


def manual_lm(module, in_ch=3):
    spec = XF.ModelSpec("TEST/synth", "sentetik", "-", "state_dict", "TEST")
    ck = {n: (m.weight.detach().float().clone(), None if m.bias is None else m.bias.detach().float().clone())
          for n, m in module.named_modules() if isinstance(m, nn.Conv2d)}
    return XF.LoadedModel(spec, module.eval(), "sentetik", "test", in_ch, "imagenet", ck, XF.state_sha(module), 0, "", ["R", "G", "B"][:in_ch])


# ---------------------------------------------------------------- T1
def t1_synthetic():
    print("T1 sentetik bilinen kernel", flush=True)
    conv = nn.Conv2d(1, 2, 3, 1, 1, bias=False)
    sob = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    with torch.no_grad():
        conv.weight[0, 0] = torch.from_numpy(sob); conv.weight[1, 0] = torch.from_numpy(sob.T)
    model = nn.Sequential(conv, nn.BatchNorm2d(2).eval(), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(2, 4, 3, 2, 1))
    lm = manual_lm(model, 1)
    g = np.random.default_rng(0).random((64, 80)).astype(np.float32)
    x = torch.from_numpy(g)[None, None]
    layers, ieq, _, _ = XF.analyze(lm, x, (64, 80), "cpu", 6, 4.0)
    ref0 = cv2.filter2D(g, cv2.CV_32F, sob, borderType=cv2.BORDER_CONSTANT)
    ref1 = cv2.filter2D(g, cv2.CV_32F, sob.T, borderType=cv2.BORDER_CONSTANT)
    d = max(np.abs(layers[0].stages["conv"][0] - ref0).max(), np.abs(layers[0].stages["conv"][1] - ref1).max())
    check("T1 Sobel kernel: model conv çıktısı = cv2.filter2D", d < 1e-4, f"max|Δ| {d:.2e}")
    check("T1 aşama zinciri conv→BN→ReLU yakalandı", list(layers[0].stages) == ["conv", "norm", "act"], list(layers[0].stages))
    check("T1 MaxPool sonrası 2. conv (1/4) dahil, ilk conv girdisi = tensör", len(layers) == 2 and ieq and layers[1].out_stride == 4,
          [(L.name, L.out_stride) for L in layers])
    # adım kuralı: 1/8'e inen conv alınmaz ve forward orada durur
    m2 = nn.Sequential(nn.Conv2d(3, 4, 3, 2, 1), nn.Conv2d(4, 4, 3, 2, 1), nn.Conv2d(4, 4, 3, 2, 1), nn.Conv2d(4, 4, 3, 1, 1))
    L2, _, _, _ = XF.analyze(manual_lm(m2), torch.zeros(1, 3, 64, 64), (64, 64), "cpu", 6, 4.0)
    check("T1 birikimli adım > 4 olan katmanlar alınmadı (derin katman yok)", [L.name for L in L2] == ["0", "1"], [L.name for L in L2])
    # residual ekleme sonrası ReLU zincire dahil edilmez
    class Res(nn.Module):
        def __init__(s):
            super().__init__(); s.c = nn.Conv2d(3, 3, 3, 1, 1); s.bn = nn.BatchNorm2d(3).eval(); s.relu = nn.ReLU(inplace=True)
        def forward(s, x):
            o = s.bn(s.c(x)); o += x; return s.relu(o)
    L3, _, _, _ = XF.analyze(manual_lm(Res()), torch.randn(1, 3, 32, 32), (32, 32), "cpu", 6, 4.0)
    check("T1 artık (residual) toplama sonrası aktivasyon aşaması eklenmedi", list(L3[0].stages) == ["conv", "norm"], list(L3[0].stages))


# ---------------------------------------------------------------- T2
def t2_noconv():
    print("T2 conv'suz model", flush=True)
    try:
        XF.analyze(manual_lm(nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))), torch.zeros(1, 3, 8, 8), (8, 8))
        check("T2 conv'suz forward → NoConvError", False, "hata yükseltilmedi")
    except XF.NoConvError as e:
        check("T2 conv'suz forward → NoConvError", True, e)
    tmp = Path(tempfile.mkdtemp()) / "mlp.pt"
    torch.save(nn.Sequential(nn.Linear(4, 4)).state_dict(), tmp)
    try:
        XF.load_model(XF.ModelSpec("TEST/mlp", "mlp", str(tmp), "state_dict", "TEST"))
        check("T2 conv'suz checkpoint → NoConvError", False)
    except XF.NoConvError as e:
        check("T2 conv'suz checkpoint → NoConvError", True, e)


# ---------------------------------------------------------------- T3
def t3_models(img, mask, iname):
    print("T3 kayıtlı modeller", flush=True)
    specs = XF.discover_models(ROOT / "models", ROOT / "phase3_yolo" / "runs")
    check("T3 model keşfi", len(specs) >= 30, f"{len(specs)} checkpoint")
    size = 256 if QUICK else 512
    for sp in specs:
        row = dict(key=sp.key, label=sp.label, file=Path(sp.path).name)
        try:
            fsha0 = file_sha(sp.path)
            t0 = time.time()
            an = XF.run_analysis(sp, img, mask, XF.REGION_MODES[0], size, "CPU", 6)
            lm = XF.load_model(sp)
            L0 = an.layers[0]
            ok_w = all(L.verify["weight_is_checkpoint"] for L in an.layers)
            worst = max(L.verify["rel"] for L in an.layers)
            strides_ok = all(L.out_stride <= 4 + 1e-6 for L in an.layers[1:])
            chain_ok = all(list(L.stages)[0] == "conv" and list(L.stages) in (["conv"], ["conv", "norm"], ["conv", "norm", "act"], ["conv", "act"])
                           for L in an.layers)
            c, tot = XF.channel_contributions(an, L0, 0)
            cdiff = float(np.abs(tot - L0.stages["conv"][0].astype(np.float32)).max() / (np.abs(L0.stages["conv"][0]).max() + 1e-9)) if c is not None else None
            fsha1 = file_sha(sp.path)
            # aşamalar birbirinden bağımsız kopyalar mı? aktivasyon = f(BN) birebir (yerinde ReLU BN'yi ezmemeli)
            act_err, bn_neg = 0.0, False
            for L in an.layers:
                if "norm" in L.stages and "act" in L.stages:
                    z = L.stages["norm"].astype(np.float32); y = L.stages["act"].astype(np.float32)
                    bn_neg |= bool((z < 0).any())
                    op = L.stage_ops["act"]
                    ref = np.maximum(z, 0) if "(ReLU" in op else (z / (1 + np.exp(-z)) if "(SiLU" in op else y)
                    act_err = max(act_err, float(np.abs(ref - y).max()))
            ok = (ok_w and worst <= 1e-4 and an.input_equal and an.sha_before == an.sha_after and fsha0 == fsha1 and strides_ok
                  and chain_ok and L0.weight.shape[1] == lm.in_ch and (cdiff is None or cdiff < 1e-4) and act_err < 1e-4 and bn_neg)
            row.update(ok=ok, arch=an.arch, family=an.family, preprocess=an.preprocess, in_ch=lm.in_ch,
                       layers=[dict(name=L.name, shape=list(L.weight.shape), kernel=list(L.conv["kernel"]), stride=list(L.conv["stride"]),
                                    padding=str(L.conv["padding"]), out_stride=L.out_stride, out_shape=list(L.out_shape),
                                    stages=list(L.stages), ops=L.stage_ops, rel=L.verify["rel"], max_abs=L.verify["max_abs"])
                               for L in an.layers],
                       n_maps=int(sum(L.n_filters * len(L.stages) for L in an.layers)), worst_rel=worst, input_equal=an.input_equal,
                       weights_sha_unchanged=an.sha_before == an.sha_after, file_unchanged=fsha0 == fsha1, channel_sum_rel=cdiff,
                       sec=round(time.time() - t0, 2), region=an.region_desc, sha=an.sha_after)
            check(f"T3 {sp.key}", ok, f"{an.arch} · {len(an.layers)} katman · en kötü göreli {worst:.1e} · kanal-toplam {cdiff if cdiff is None else f'{cdiff:.1e}'}")
        except Exception as e:
            row.update(ok=False, error=f"{type(e).__name__}: {e}")
            check(f"T3 {sp.key}", False, row["error"]); traceback.print_exc()
        RES["models"].append(row)
        XF._MODEL_CACHE.clear(); XF._RESULTS.clear()


# ---------------------------------------------------------------- T4
def t4_yolo_stem(img, mask):
    print("T4 YOLO ilk-katman dışa aktarımı", flush=True)
    runs = [p for p in sorted(glob.glob(str(ROOT / "phase3_yolo" / "runs" / "*" / "weights" / "best.pt")))
            if (Path(p).parent.parent / "summary.json").exists() and "dryrun" not in p]
    if not runs:
        check("T4 YOLO checkpoint yok (atlandı)", True); return
    best = runs[0]
    run = Path(best).parent.parent.name
    out = Path(tempfile.mkdtemp()) / f"{run}_stem.pt"
    XF.export_yolo_stem(best, out)
    full = XF.run_analysis(XF.ModelSpec("YOLO/x", "full", best, "yolo", "P3"), img, mask, XF.REGION_MODES[0], 256, "CPU", 6)
    stem = XF.run_analysis(XF.ModelSpec("YOLOSTEM/x", "stem", str(out), "yolo_stem", "P3"), img, mask, XF.REGION_MODES[0], 256, "CPU", 6)
    d = 0.0
    for a, b in zip(full.layers[:len(stem.layers)], stem.layers):
        for k in b.stages:
            d = max(d, float(np.abs(a.stages[k].astype(np.float32) - b.stages[k].astype(np.float32)).max()))
        d = max(d, float(np.abs(a.weight - b.weight).max()))
    check("T4 çıkarılmış YOLO ilk katmanları = tam checkpoint (ağırlık + tüm aşamalar)", d < 1e-5 and len(stem.layers) == 2,
          f"{len(stem.layers)} katman · max|Δ| {d:.1e}")


# ---------------------------------------------------------------- T5
def t5_gpu(img, mask):
    print("T5 GPU", flush=True)
    if not torch.cuda.is_available():
        check("T5 cuda yok (atlandı)", True); return
    free = torch.cuda.mem_get_info()[0] / 2 ** 30
    if free < 1.0:
        check(f"T5 GPU dolu ({free:.1f} GiB boş; atlandı)", True); return
    sp = [s for s in XF.discover_models(ROOT / "models") if s.key == "PHASE2/unet_bbox_s42"][0]
    a = XF.run_analysis(sp, img, mask, XF.REGION_MODES[0], 256, "CPU", 6)
    b = XF.run_analysis(sp, img, mask, XF.REGION_MODES[0], 256, "GPU (cuda)", 6)
    rel = max(float(np.abs(x.stages[k] - y.stages[k]).max() / (np.abs(x.stages[k]).max() + 1e-9)) for x, y in zip(a.layers, b.layers) for k in x.stages)
    check("T5 cuda haritaları ≈ CPU (TF32 toleransı)", rel < 5e-3 and b.device == "cuda" and all(XF.verify_ok(L, 5e-3) for L in b.layers),
          f"göreli {rel:.1e} · tepe {b.peak_gpu_mb:.0f} MiB · {b.sec:.2f} s")
    RES["gpu"] = dict(rel=rel, peak_mb=b.peak_gpu_mb, sec=b.sec, free_gib=free)


# ---------------------------------------------------------------- T6
def t6_render(img, mask):
    print("T6 görselleştirme", flush=True)
    specs = {s.key: s for s in XF.discover_models(ROOT / "models", ROOT / "phase3_yolo" / "runs")}
    keys = [k for k in ("PHASE2/unet_bbox_s42", "PHASE1/unet_rgbfreq_best", "PHASE1/resnet18_ft_best", "PHASE2/dlv3p_bbox_s42") if k in specs]
    keys += [k for k in specs if k.startswith("YOLO")][:1]
    for k in keys:
        an = XF.run_analysis(specs[k], img, mask, XF.REGION_MODES[1], 384, "CPU", 6)
        n = 0
        for L in an.layers[:2]:
            for st in L.stages:
                for norm in XF.NORMS:
                    _ = XF.norm_map(L.stages[st][0], norm, L, st); n += 1
            st = list(L.stages)[-1]
            ims = [XF.fig_kernel_grid(L), XF.fig_kernel_grid(L, False), XF.fig_pairs(an, L, st, 1, XF.NORMS[1]),
                   XF.fig_act_grid(an, L, st, XF.NORMS[2]), XF.fig_kernel_detail(an, L, 1), XF.fig_act_detail(an, L, "conv", 1),
                   XF.fig_compare(an, L, 1, st, XF.NORMS[0]), XF.fig_top_filters(an, L, st, XF.NORMS[3])]
            n += sum(1 for i in ims if isinstance(i, np.ndarray) and i.ndim == 3 and i.shape[0] > 50)
        files = XF.export_raw(an, an.layers[0], 0)
        ok = all(Path(f).exists() and Path(f).stat().st_size > 0 for f in files)
        z = np.load(files[0])
        ok &= np.array_equal(z["weight"], an.layers[0].weight)
        check(f"T6 {k}: görseller + ham dışa aktarım", ok and n >= 16, f"{n} görsel/normalizasyon · {len(files)} dosya")
        XF._RESULTS.clear()


if __name__ == "__main__":
    t_all = time.time()
    img, mask, iname = test_image()
    RES["image"] = iname
    for fn in (t1_synthetic, t2_noconv, lambda: t3_models(img, mask, iname), lambda: t4_yolo_stem(img, mask),
               lambda: t5_gpu(img, mask), lambda: t6_render(img, mask)):
        try:
            fn()
        except Exception as e:
            check(f"{getattr(fn, '__name__', 'test')} çöktü", False, f"{type(e).__name__}: {e}"); traceback.print_exc()
    RES["n_checks"] = len(RES["checks"]); RES["n_fail"] = len(FAIL); RES["failed"] = FAIL
    RES["sec"] = round(time.time() - t_all, 1)
    RES["env"] = dict(torch=torch.__version__, cuda=torch.cuda.is_available())
    (OUT / "test_results.json").write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n{RES['n_checks']} kontrol · {len(FAIL)} hata · {RES['sec']} s → {OUT / 'test_results.json'}")
    sys.exit(1 if FAIL else 0)
