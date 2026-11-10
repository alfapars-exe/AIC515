"""patch_app_phase3.py — app/app.py'ye Faz 3 modellerini ve sondalarını ekler (TR). İdempotent: işaret satırı varsa çıkar.
Eklenenler:
  * DeepLabV3+ (dlv3p_*), birleşik kayıplı U-Net (unetjoint_*), işaretçi-onarımlı U-Net'ler (*_mr_*) — models/phase2
  * YOLO11 kutu dedektörleri (YOLO/<run>) — models/phase3_yolo/<run>.pt; kareli 1024/768 çıkarım + kareler arası NMS,
    kutular güvenle olasılık haritasına boyanır → U-Net ile AYNI metrik hattı (bildiri §IV-C)
  * Senaryolar: "yalnız yeniden-render" ve "yalnız içerik" (bildiri Tablo III); bozulmalar: CLAHE, gri seviye
  * Sonuçlar sekmesi: Faz 3 tabloları (meta/phase3_tables.md)
  * XAI: YOLO hariç tutulur (kutu dedektörü için piksel açıklayıcı tanımlı değil)
Kullanım: python patch_app_phase3.py <app.py yolu>
"""
import sys
from pathlib import Path

P = Path(sys.argv[1]); s = P.read_text(encoding="utf-8")
MARK = "# [FAZ3] model kaydı"
if MARK in s:
    print("zaten yamalı"); sys.exit(0)


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:80], n)
    s = s.replace(old, new)


# ---------------------------------------------------------------- A) model listesi
rep('''def list_unet_ckpts():
    out = [f"PHASE2/{Path(p).stem}" for p in sorted(glob.glob(str(MODELS / "phase2" / "unet_*.pt")))]
    out += [f"PHASE1/{Path(p).stem}" for p in sorted(glob.glob(str(MODELS / "unet_*.pt")))]
    return out
''', '''# [FAZ3] model kaydı: PHASE2/ (Faz 2 + Faz 3 segmentasyon), PHASE1/, YOLO/ (Faz 3 kutu dedektörleri)
_HIDE = re.compile(r"(_e20$|geminionly_s4[34]$)")  # 20 epoch ve ek held-out tohumları arayüzde gösterilmez


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
''')

# ---------------------------------------------------------------- B) checkpoint yolu
rep('''    phase, stem = name.split("/", 1)
    return (MODELS / f"{stem}.pt") if phase == "PHASE1" else (MODELS / "phase2" / f"{stem}.pt")''',
    '''    phase, stem = name.split("/", 1)
    if phase == "YOLO":
        return MODELS / "phase3_yolo" / f"{stem}.pt"
    return (MODELS / f"{stem}.pt") if phase == "PHASE1" else (MODELS / "phase2" / f"{stem}.pt")''')

# ---------------------------------------------------------------- C) okunur adlar
rep('''    gem = " · yalnız Gemini" if "geminionly" in stem else ""
    if stem.startswith("unet_bbox"):''',
    '''    gem = " · yalnız Gemini" if "geminionly" in stem else ""
    if name.startswith("YOLO/"):
        m = re.match(r"aiforge-(?P<d>[a-z_]+)-yolo11(?P<a>[nsm])-img(?P<i>\\d+)-(?P<g>[a-z]+)-s(?P<s>\\d+)", stem)
        if not m:
            return f"YOLO11 · {stem}"
        d = m.group("d")
        how = ("tam görüntü 1280" if d == "full" else "kare 1024") + (" · yalnız Gemini" if "geminionly" in d else "") + \\
              (" · +authentic kare" if "authneg" in d else "") + (" · varsayılan aug" if m.group("g") == "default" else "")
        return f"YOLO11{m.group('a')} · kutu tespiti, {how} (Faz 3, s{m.group('s')})"
    if stem.startswith("dlv3p_"):
        g = "bbox-GT" if "bbox" in stem else ("piksel-GT" if "pixel" in stem else "glif-GT")
        return f"DeepLabV3+ · {g} (Faz 3, s{seed})"
    if stem.startswith("unetjoint_"):
        g = "piksel-GT" if "pixel" in stem else "glif-GT"
        return f"U-Net · {g} + kutu yardımcı kaybı (Faz 3, s{seed})"
    if "_mr_" in stem:
        g = "bbox-GT" if "bbox" in stem else ("piksel-GT" if "pixel" in stem else "glif-GT")
        return f"U-Net · {g} · işaretçisiz eğitim (Faz 3, s{seed})"
    if stem.startswith("unet_bbox"):''')

# ---------------------------------------------------------------- D) mimari ayrımı
rep('''    in_ch = int(sd["encoder.conv1.weight"].shape[1])
    m = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=in_ch, classes=1)''',
    '''    in_ch = int(sd["encoder.conv1.weight"].shape[1])
    stem = name.split("/", 1)[-1]
    if stem.startswith("dlv3p_"):  # Faz 3: ikinci mimari
        m = smp.DeepLabV3Plus(encoder_name="resnet18", encoder_weights=None, in_channels=in_ch, classes=1)
    elif stem.startswith("unetjoint_"):  # Faz 3: kanal 0 = piksel/glif maske, kanal 1 = kutu yardımcı (çıkarımda kanal 0)
        m = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=in_ch, classes=2)
    else:
        m = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=in_ch, classes=1)''')
rep('''    m.load_state_dict(sd)
    return m.to(DEVICE).eval(), in_ch
''', '''    m.load_state_dict(sd)
    if stem.startswith("unetjoint_"):  # kutu yardımcı başını at: yalnız kanal 0 (birebir aynı çıktı, tek kanal → XAI/çıkarım uyumlu)
        c = m.segmentation_head[0]
        h = nn.Conv2d(c.in_channels, 1, c.kernel_size, padding=c.padding)
        with torch.no_grad():
            h.weight.copy_(c.weight[:1]); h.bias.copy_(c.bias[:1])
        m.segmentation_head[0] = h
    return m.to(DEVICE).eval(), in_ch
''')

rep('''def get_model(name):
    if name not in _REG:
        if "resnet18" in name:''',
    '''def _build_yolo(name):
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
        elif "resnet18" in name:''')

# ---------------------------------------------------------------- E) YOLO çıkarımı
rep('''def _predict_unet(name, rgb_u8, win=512, stride=448, max_side=4096, batch=8):
    """infer_fullimage.py protokolü: 512 pencere, 448 adım, örtüşme ortalaması; uzun kenar > 4096 ise küçült → harita geri büyüt."""
    model, in_ch, _ = get_model(name)''',
    '''def _predict_yolo(name, rgb_u8, tile=1024, stride=768, conf=0.001, iou=0.6, batch=8):
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
    model, in_ch, _ = get_model(name)''')

# ---------------------------------------------------------------- F) senaryolar ve bozulmalar
rep('''               "Authentic + yalnız yeşil işaretçi (kısayol testi)"]''',
    '''               "Authentic + yalnız yeşil işaretçi (kısayol testi)",
               "Yalnız yeniden-render (karakterler geri alınmış · Faz 3 sondası)",
               "Yalnız içerik (render ve işaretçi yok · Faz 3 sondası)"]''')
rep('''                "Gauss bulanıklık σ=1", "Gauss gürültüsü σ=5"]''',
    '''                "Gauss bulanıklık σ=1", "Gauss gürültüsü σ=5", "CLAHE (L kanalı, Faz 3)", "Gri seviye (Faz 3)"]''')
rep('''    if "gürültü" in kind:
        return clip_u8(img.astype(np.float32) + np.random.default_rng(0).normal(0, 5, img.shape))
    return img''',
    '''    if "gürültü" in kind:
        return clip_u8(img.astype(np.float32) + np.random.default_rng(0).normal(0, 5, img.shape))
    if kind.startswith("CLAHE"):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if kind.startswith("Gri"):
        return np.repeat(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
    return img''')
rep('''    gm, rep, tra = marker_ops(forged, auth)
    if input_type == INPUT_TYPES[2]:''',
    '''    gm, rep, tra = marker_ops(forged, auth)
    if input_type in INPUT_TYPES[4:6]:  # Faz 3 bölge-görünümü sondası (probes_p3.py tanımı; tek fark: işaretçi kalıntısı bbox halkasıyla sınırlanmaz)
        r = derive_pixel_gt(forged, auth)
        if r is None:
            raise gr.Error("Bu senaryo aynı boyutta bir authentic eş gerektirir.")
        pg = r[0]
        if input_type == INPUT_TYPES[4]:
            sel = cv2.dilate((pg | gm).astype(np.uint8), KX, iterations=2) > 0  # KX: 4-komşu haç = ndi.binary_dilation varsayılanı
            out = forged.copy(); out[sel] = auth[sel]
            return out, 0, (f"sahte belge; değişen karakterler ve işaretçi ({int(sel.sum())} px) authentic'e döndürüldü — "
                            "yalnız üreticinin alanı yeniden çizme izi kalır, içerik değişmedi")
        sel = (cv2.dilate((pg & ~gm).astype(np.uint8), KX, iterations=2) > 0) & ~gm
        out = auth.copy(); out[sel] = forged[sel]
        return out, 1, (f"orijinal belgeye yalnız değişen karakterler ({int(sel.sum())} px) yapıştırıldı — "
                        "yeniden-render izi ve işaretçi yok, içerik sahte")
    if input_type == INPUT_TYPES[2]:''')

rep('K3 = np.ones((3, 3), np.uint8)\n',
    'K3 = np.ones((3, 3), np.uint8)\nKX = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Faz 3 sondaları (scipy.ndimage varsayılan yapı elemanı)\n')

# ---------------------------------------------------------------- G) XAI'den YOLO'yu çıkar
rep('''list_unet=list_unet_ckpts, list_resnet=list_resnet_ckpts, green_marker_mask=green_marker_mask)''',
    '''list_unet=list_seg_ckpts, list_resnet=list_resnet_ckpts, green_marker_mask=green_marker_mask)''')
rep('''x_unet = gr.CheckboxGroup(choices=[(friendly(n), n) for n in _UNETS], value=_DEF_UNETS, label="U-Net", elem_id="tip-x_unet")''',
    '''x_unet = gr.CheckboxGroup(choices=[(friendly(n), n) for n in _UNETS if not n.startswith("YOLO/")], value=[n for n in _DEF_UNETS if not n.startswith("YOLO/")], label="U-Net / DeepLabV3+ (YOLO kutu dedektörü XAI dışı)", elem_id="tip-x_unet")''')

# ---------------------------------------------------------------- H) etiketler + varsayılanlar
rep('''label="U-Net (piksel lokalizasyonu)"''', '''label="Lokalizasyon modelleri (U-Net · DeepLabV3+ · YOLO11 kutu tespiti)"''')
rep('''label="U-Net", elem_id="tip-t_unet"''', '''label="Lokalizasyon (U-Net · DeepLabV3+ · YOLO11)", elem_id="tip-t_unet"''')
rep('''label="U-Net checkpoint"''', '''label="Model (U-Net · DeepLabV3+ · YOLO11)"''')
rep('''_DEF_UNETS = [n for n in ("PHASE2/unet_bbox_s42", "PHASE2/unet_pixel_eps8_morph_s42", "PHASE2/unet_glyph_v2_s42") if n in _UNETS] or _UNETS[:1]''',
    '''_DEF_UNETS = [n for n in ("PHASE2/unet_bbox_s42", "PHASE2/unet_pixel_eps8_morph_s42", "PHASE2/unet_glyph_v2_s42",
                             "YOLO/aiforge-tiles-yolo11s-img1024-conserv-s42") if n in _UNETS] or _UNETS[:1]''')

# ---------------------------------------------------------------- I) Faz 3 sonuç tabloları
rep('''def results_markdown():
    parts = []''', '''def results_markdown():
    parts = []
    p3 = META / "phase3_tables.md"
    if p3.exists():  # Faz 3: YOLO11, DeepLabV3+, işaretçisiz eğitim, sondalar (bildiri: paper/main.pdf)
        parts.append(p3.read_text(encoding="utf-8") + "\\n\\n---\\n")''')

P.write_text(s, encoding="utf-8")
print("yamalandı:", P)
