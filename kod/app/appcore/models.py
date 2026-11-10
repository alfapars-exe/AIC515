"""appcore.models — model kaydı, checkpoint yükleme, U-Net / DeepLabV3+ / YOLO / ResNet çıkarımı, gpu_infer."""
from __future__ import annotations
from ._common import *  # noqa: F401,F403 — ortak içe aktarmalar
from .config import (  # noqa: F401
    DEVICE, IMAGENET_MEAN, IMAGENET_STD, MODELS, ZERO_GPU, gpu
)
from .imaging import (  # noqa: F401
    project_freq_channels, q_u8
)

# ================================================================== MODELLER
_REG: dict = {}          # ad → (model, in_ch, tür)


_LOAD_ERR: dict = {}


# [FAZ3] model kaydı: PHASE2/ (Faz 2 + Faz 3 segmentasyon), PHASE1/, YOLO/ (Faz 3 kutu dedektörleri)
_HIDE = re.compile(r"(_e20$|geminionly_s4[34]$|_mr_s4[34]$|^unetjoint_.*_s4[34]$)")  # 20 epoch ve ek tohumlar (held-out, işaretçisiz, birleşik kayıp) arayüzde gösterilmez


def list_unet_ckpts():
    ph2 = []
    for pat in ("unet_*.pt", "dlv3p_*.pt", "unetjoint_*.pt"):
        ph2 += sorted(glob.glob(str(MODELS / "phase2" / pat)))
    out = [f"PHASE2/{Path(p).stem}" for p in ph2 if not _HIDE.search(Path(p).stem)]
    out += [f"PHASE1/{Path(p).stem}" for p in sorted(glob.glob(str(MODELS / "unet_*.pt")))]
    out += [f"YOLO/{Path(p).stem}" for p in sorted(glob.glob(str(MODELS / "phase3_yolo" / "aiforge-*.pt")))
            if not re.search(r"(_stem$|-s4[34]$)", Path(p).stem)]  # tohum tekrarları (s43/s44) ve katman dökümleri gizli
    return out


def list_seg_ckpts():
    """Piksel segmentasyon modelleri (XAI adaptörleri için; YOLO hariç)."""
    return [n for n in list_unet_ckpts() if not n.startswith("YOLO/")]


def list_resnet_ckpts():
    out = [f"PHASE1/{Path(p).stem}" for p in sorted(glob.glob(str(MODELS / "resnet18_*.pt")))]
    out += [f"PHASE2/{Path(p).stem}" for p in sorted(glob.glob(str(MODELS / "phase2" / "h4_resnet18*.pt")))]
    # [FAZ4] tam veriyle yeniden eğitilmiş parça sınıflandırıcıları (tohum tekrarları s43/s44 arayüzde gizli)
    out += [f"PHASE4/{Path(p).stem}" for p in sorted(glob.glob(str(MODELS / "phase4" / "resnet18_*.pt")))
            if not re.search(r"(_s4[34]$|dryrun)", Path(p).stem)]
    return out


def ckpt_path(name: str) -> Path:
    phase, stem = name.split("/", 1)
    if phase == "YOLO":
        return MODELS / "phase3_yolo" / f"{stem}.pt"
    if phase == "PHASE4":
        return MODELS / "phase4" / f"{stem}.pt"
    return (MODELS / f"{stem}.pt") if phase == "PHASE1" else (MODELS / "phase2" / f"{stem}.pt")


def friendly(name: str) -> str:
    stem = name.split("/", 1)[-1]
    seed = stem.rsplit("_s", 1)[-1] if "_s4" in stem else ""
    gem = " · yalnız Gemini" if "geminionly" in stem else ""
    if name.startswith("YOLO/"):
        m = re.match(r"aiforge-(?P<d>[a-z_]+)-yolo11(?P<a>[nsm])-img(?P<i>\d+)-(?P<g>[a-z]+)-s(?P<s>\d+)", stem)
        if not m:
            return f"YOLO11 · {stem}"
        d = m.group("d")
        how = ("tam görüntü 1280" if d == "full" else "kare 1024") + (" · yalnız Gemini" if "geminionly" in d else "") + \
              (" · +authentic kare" if "authneg" in d else "") + (" · varsayılan aug" if m.group("g") == "default" else "")
        return f"YOLO11{m.group('a')} · kutu tespiti, {how} (Faz 3, s{m.group('s')})"
    if stem.startswith("resnet18_ft_full"):  # [FAZ4] "_mr_" U-Net dalından ÖNCE ele alınmalı
        return f"ResNet-18 · tam veri{' · işaretçisiz eğitim' if '_mr_' in stem else ''} (Faz 4, s{seed})"
    if stem.startswith("dlv3p_"):
        g = "bbox-GT" if "bbox" in stem else ("piksel-GT" if "pixel" in stem else "glif-GT")
        return f"DeepLabV3+ · {g} (Faz 3, s{seed})"
    if stem.startswith("unetjoint_"):
        g = "piksel-GT" if "pixel" in stem else "glif-GT"
        return f"U-Net · {g} + kutu yardımcı kaybı (Faz 3, s{seed})"
    if "_mr_" in stem:
        g = "bbox-GT" if "bbox" in stem else ("piksel-GT" if "pixel" in stem else "glif-GT")
        return f"U-Net · {g} · işaretçisiz eğitim (Faz 3, s{seed})"
    if stem.startswith("unet_bbox"):
        return f"U-Net · bbox-GT{gem} (s{seed})"
    if stem.startswith("unet_pixel"):
        return f"U-Net · piksel-GT{gem} (s{seed})"
    if stem.startswith("unet_glyph"):
        return f"U-Net · glif-GT{gem} (s{seed})"
    if stem == "unet_rgb_best":
        return "U-Net · RGB (PHASE1, 400 örnek)"
    if stem == "unet_rgbfreq_best":
        return "U-Net · RGB+FFT+Laplacian (PHASE1)"
    if stem.startswith("resnet18_"):
        return {"resnet18_ft_best": "ResNet-18 · tam ince ayar (Faz 1, 400 örnek)", "resnet18_probe_best": "ResNet-18 · linear probe",
                "resnet18_scratch_best": "ResNet-18 · sıfırdan"}.get(stem, stem)
    if stem.startswith("h4_resnet18"):
        return "ResNet-18 · H4 global-iz sondası" + (" (poz. kontrol)" if "poscontrol" in stem else "")
    return stem


def _build_unet(name):
    sd = torch.load(ckpt_path(name), map_location="cpu")
    if any(k.startswith("net.") for k in sd):  # PHASE1 UNetWrapper
        sd = {k[4:]: v for k, v in sd.items() if k.startswith("net.")}
    in_ch = int(sd["encoder.conv1.weight"].shape[1])
    stem = name.split("/", 1)[-1]
    if stem.startswith("dlv3p_"):  # Faz 3: ikinci mimari
        m = smp.DeepLabV3Plus(encoder_name="resnet18", encoder_weights=None, in_channels=in_ch, classes=1)
    elif stem.startswith("unetjoint_"):  # Faz 3: kanal 0 = piksel/glif maske, kanal 1 = kutu yardımcı (çıkarımda kanal 0)
        m = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=in_ch, classes=2)
    else:
        m = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=in_ch, classes=1)
    m.load_state_dict(sd)
    if stem.startswith("unetjoint_"):  # kutu yardımcı başını at: yalnız kanal 0 (birebir aynı çıktı, tek kanal → XAI/çıkarım uyumlu)
        c = m.segmentation_head[0]
        h = nn.Conv2d(c.in_channels, 1, c.kernel_size, padding=c.padding)
        with torch.no_grad():
            h.weight.copy_(c.weight[:1]); h.bias.copy_(c.bias[:1])
        m.segmentation_head[0] = h
    return m.to(DEVICE).eval(), in_ch


def _build_resnet(name):
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(512, 1)
    m.load_state_dict(torch.load(ckpt_path(name), map_location="cpu"))
    return m.to(DEVICE).eval()


def _build_yolo(name):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "Ultralytics"))
    from ultralytics import YOLO  # yalnız YOLO seçildiğinde gerekir
    m = YOLO(str(ckpt_path(name)))
    if DEVICE is not None:
        m.to(DEVICE)
    return m


def get_model(name):
    if name not in _REG:
        if name.startswith("YOLO/"):
            _REG[name] = (_build_yolo(name), 3, "yolo")
        elif "resnet18" in name:
            _REG[name] = (_build_resnet(name), 3, "resnet")
        else:
            m, c = _build_unet(name)
            _REG[name] = (m, c, "unet")
    return _REG[name]


def preload_models():
    """ZeroGPU kuralı: modeller modül düzeyinde cuda'ya yerleştirilir (emülasyon); çıkarım @spaces.GPU içinde."""
    if not TORCH_OK:
        return
    t0 = time.time()
    for n in list_unet_ckpts() + list_resnet_ckpts():
        try:
            get_model(n)
        except Exception as e:  # bozuk/eksik ağırlık uygulamayı durdurmasın
            _LOAD_ERR[n] = str(e)
    print(f"[ai515] {len(_REG)} model yüklendi ({DEVICE}, ZeroGPU={ZERO_GPU}) {time.time() - t0:.1f}s; hata: {list(_LOAD_ERR)}", flush=True)


def _autocast():
    if DEVICE is not None and DEVICE.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def norm_chw(rgb_u8: np.ndarray) -> np.ndarray:
    x = rgb_u8.astype(np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def _windows(H, W, win, stride):
    ys = list(range(0, max(H - win, 0) + 1, stride)); xs = list(range(0, max(W - win, 0) + 1, stride))
    if ys[-1] + win < H: ys.append(H - win)
    if xs[-1] + win < W: xs.append(W - win)
    return [(y, x) for y in ys for x in xs]


def _unet_prepare(rgb_u8, win=512, max_side=4096):
    """U-Net girdi hazırlığı: uzun kenar > max_side ise INTER_AREA ile küçült; kısa kenar < win ise sağ/alttan replicate
    dolgu. Dönüş: (görüntü, (h, w) dolgusuz boyut, ölçek). _predict_unet ve XAI adaptörü aynı fonksiyonu kullanır."""
    H0, W0 = rgb_u8.shape[:2]
    s = 1.0
    img = rgb_u8
    if max(H0, W0) > max_side:
        s = max_side / max(H0, W0)
        img = cv2.resize(img, (max(1, int(round(W0 * s))), max(1, int(round(H0 * s)))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    ph, pw = max(0, win - h), max(0, win - w)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    return img, (h, w), s


def _unet_window_forward(model, in_ch, xb, amp=True):
    """Pencere toplu ileri geçişi → (N, h, w) olasılık. 5 kanallı modelde frekans kanalları pencere başına
    standartlaştırılır (projedeki tanım). amp=False: fp32 (XAI gradyan açıklaması). Tek doğruluk kaynağı."""
    if in_ch == 5:  # projede frekans kanalları pencere başına standartlaştırılır
        fc = xb[:, 3:]
        mu = fc.mean(dim=(2, 3), keepdim=True); sd = fc.std(dim=(2, 3), keepdim=True) + 1e-6
        xb = torch.cat([xb[:, :3], (fc - mu) / sd], 1)
    if amp:
        with _autocast():
            return torch.sigmoid(model(xb).float())[:, 0]
    return torch.sigmoid(model(xb.float()).float())[:, 0]


def _predict_yolo(name, rgb_u8, tile=1024, stride=768, conf=0.001, iou=0.6, batch=8):
    """phase3_yolo/infer_to_maps.py protokolü: natif çözünürlükte 1024 kare / 768 adım (ya da '-full-' modelde tam görüntü 1280),
    kareler arası NMS; her kutu güveniyle haritaya boyanır (örtüşmede maksimum) → U-Net ile aynı metrikler."""
    from torchvision.ops import nms
    model, _, _ = get_model(name)
    stem = name.split("/", 1)[-1]
    H0, W0 = rgb_u8.shape[:2]
    t0 = time.time()
    dev = 0 if (DEVICE is not None and DEVICE.type == "cuda") else "cpu"
    kw = dict(conf=conf, iou=iou, verbose=False, device=dev, half=(dev != "cpu"))
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    if "-full-" in stem:
        r = model.predict(bgr, imgsz=1280, **kw)[0]
        b = r.boxes.xyxy.float().cpu().numpy(); c = r.boxes.conf.float().cpu().numpy(); nw = 1
    else:
        ph, pw = max(0, tile - H0), max(0, tile - W0)
        img = cv2.copyMakeBorder(bgr, 0, ph, 0, pw, cv2.BORDER_REPLICATE) if (ph or pw) else bgr
        H, W = img.shape[:2]
        wins = _windows(H, W, tile, stride); nw = len(wins)
        bl, cl = [], []
        for i in range(0, len(wins), batch):
            chunk = wins[i:i + batch]
            outs = model.predict([np.ascontiguousarray(img[y:y + tile, x:x + tile]) for (y, x) in chunk], imgsz=tile, **kw)
            for (y, x), r in zip(chunk, outs):
                bb = r.boxes.xyxy.float().cpu().numpy()
                if len(bb):
                    bl.append(bb + np.array([x, y, x, y], np.float32)); cl.append(r.boxes.conf.float().cpu().numpy())
        if bl:
            b = np.concatenate(bl); c = np.concatenate(cl)
            keep = nms(torch.from_numpy(b), torch.from_numpy(c), iou).numpy(); b, c = b[keep], c[keep]
        else:
            b, c = np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    if len(b):
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W0); b[:, [1, 3]] = b[:, [1, 3]].clip(0, H0)
    p = np.zeros((H0, W0), np.float32)
    for (x0, y0, x1, y1), s_ in sorted(zip(b.tolist(), c.tolist()), key=lambda t: t[1]):
        ys_, xs_ = slice(int(y0), int(np.ceil(y1))), slice(int(x0), int(np.ceil(x1)))
        p[ys_, xs_] = np.maximum(p[ys_, xs_], s_)
    top = sorted(zip(c.tolist(), b.tolist()), reverse=True)[:10]
    return p, dict(n_windows=nw, resized=0, sec=time.time() - t0, n_boxes=int(len(b)),
                   boxes=[(round(s_, 3), [int(v) for v in bx]) for s_, bx in top])


def _predict_unet(name, rgb_u8, win=512, stride=448, max_side=4096, batch=8):
    """infer_fullimage.py protokolü: 512 pencere, 448 adım, örtüşme ortalaması; uzun kenar > 4096 ise küçült → harita geri büyüt."""
    if name.startswith("YOLO/"):
        return _predict_yolo(name, rgb_u8)
    model, in_ch, _ = get_model(name)
    H0, W0 = rgb_u8.shape[:2]
    img, (h, w), _s = _unet_prepare(rgb_u8, win, max_side)
    resized = _s != 1.0
    H, W = img.shape[:2]
    x = torch.from_numpy(norm_chw(img))
    if in_ch == 5:
        x = torch.cat([x, torch.from_numpy(project_freq_channels(img))], 0)
    x = x.to(DEVICE)
    wins = _windows(H, W, win, stride)
    acc = torch.zeros((H, W), dtype=torch.float32, device=DEVICE); cnt = torch.zeros_like(acc)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(wins), batch):
            chunk = wins[i:i + batch]
            xb = torch.stack([x[:, y:y + win, xx:xx + win] for (y, xx) in chunk])
            out = _unet_window_forward(model, in_ch, xb)
            for j, (y, xx) in enumerate(chunk):
                acc[y:y + win, xx:xx + win] += out[j]; cnt[y:y + win, xx:xx + win] += 1
    p = (acc / cnt).cpu().numpy()[:h, :w]
    if resized:
        p = cv2.resize(p, (W0, H0), interpolation=cv2.INTER_LINEAR)
    return p.astype(np.float32), dict(n_windows=len(wins), resized=int(resized), sec=time.time() - t0)


def _predict_resnet(name, rgb_u8, patch=224, stride=112, batch=32):
    """224 natif patch kayan pencere → patch skorları (örtüşme ortalaması haritası) + patch listesi."""
    model, _, _ = get_model(name)
    h, w = rgb_u8.shape[:2]
    img = rgb_u8
    ph, pw = max(0, patch - h), max(0, patch - w)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    H, W = img.shape[:2]
    x = torch.from_numpy(norm_chw(img)).to(DEVICE)
    wins = _windows(H, W, patch, stride)
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32); scores = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(wins), batch):
            chunk = wins[i:i + batch]
            xb = torch.stack([x[:, y:y + patch, xx:xx + patch] for (y, xx) in chunk])
            with _autocast():
                s = torch.sigmoid(model(xb).float()).squeeze(1).cpu().numpy()
            for j, (y, xx) in enumerate(chunk):
                acc[y:y + patch, xx:xx + patch] += s[j]; cnt[y:y + patch, xx:xx + patch] += 1
                scores.append((float(s[j]), int(y), int(xx)))
    return (acc / np.maximum(cnt, 1))[:h, :w], scores, time.time() - t0


def _mp(img):
    return min(img.shape[0] * img.shape[1], 4096 * 4096) / 1e6


def _dur_infer(unet_names, resnet_names, imgs, stride=112):
    mp = sum(_mp(i) for i in imgs)
    return int(min(240, max(20, 8 + mp * (1.2 * len(unet_names) + 0.8 * len(resnet_names)))))


@gpu(duration=_dur_infer)
def _gpu_infer(unet_names, resnet_names, imgs, stride=112):
    """Tek GPU çağrısında birden çok görüntü × model. Olasılıklar 8-bit döner (metrikler zaten 8-bit nicemlemeyle
    hesaplanır; ZeroGPU süreçler arası aktarımı 4× küçülür)."""
    out = []
    for img in imgs:
        res = {}
        for n in unet_names:
            p, info = _predict_unet(n, img)
            res[n] = (q_u8(p), info)
        for n in resnet_names:
            hm, sc, sec = _predict_resnet(n, img, stride=int(stride))
            res[n] = (q_u8(hm), dict(scores=sc, sec=sec))
        out.append(res)
    return out


def gpu_infer(unet_names, resnet_names, imgs, stride=112):
    raw = _gpu_infer(list(unet_names), list(resnet_names), [np.ascontiguousarray(i) for i in imgs], int(stride))
    return [{k: (u.astype(np.float32) / 255.0, info) for k, (u, info) in r.items()} for r in raw]
