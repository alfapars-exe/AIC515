"""appcore.handlers — diğer sekmelerin olay işleyicileri (girdi, taslaklar, ön işleme, U-Net, ResNet, işaretçi, sonuçlar)."""
from __future__ import annotations
from ._common import *  # noqa: F401,F403 — ortak içe aktarmalar
from .config import (  # noqa: F401
    DATA_ROOT, DEVICE, FC, FC_OK, IN_SPACE, META, P2_SUMMARY, RESULTS_MD, ROOT, ZERO_GPU
)
from .imaging import (  # noqa: F401
    K3, derive_pixel_gt, green_marker_mask, hist_figure, to_gray, to_rgb
)
from .ops import (  # noqa: F401
    OPS, PRESETS, replay, run_op
)
from .dataset import (  # noqa: F401
    dataset_index, filter_index, load_dataset_sample, meta_md, sample_choices, sample_exists
)
from .models import (  # noqa: F401
    friendly, gpu_infer
)
from .metrics import (  # noqa: F401
    IMG_TAU, RESNET_TAU, disp, heatmap_rgb, hist_metrics, image_score, image_threshold, overlay, postprocess, rectangularity,
    resnet_decision
)
from .scenarios import (  # noqa: F401
    bbox_of, fmt
)

# ================================================================== DİĞER SEKMELERİN MANTIĞI
def _info_md(img, mask, auth):
    if img is None:
        return "Görüntü yok."
    h, w = img.shape[:2]
    s = f"**Boyut:** {w}×{h} px ({w * h / 1e6:.2f} MP) · **kanal:** {1 if img.ndim == 2 else img.shape[2]}"
    if mask is not None:
        m = mask > 127
        s += f"  \n**GT maskesi:** {int(m.sum())} px (%{100 * m.mean():.2f} alan)"
        rect, bb = rectangularity(m)
        if bb:
            s += f", bbox {bb[2] - bb[0]}×{bb[3] - bb[1]} (dikdörtgensellik {rect:.2f})"
    if auth is not None:
        s += f"  \n**Authentic eş:** {auth.shape[1]}×{auth.shape[0]} " + ("✓ aynı boyut" if auth.shape[:2] == img.shape[:2] else "⚠ boyut farklı")
    return s


def on_upload(img, mask, auth):
    if img is None:
        raise gr.Error("Önce bir görüntü yükleyin.")
    img = to_rgb(img)
    if mask is not None:
        mask = to_gray(mask)
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if auth is not None:
        auth = to_rgb(auth)
        if auth.shape != img.shape:
            gr.Warning("Authentic eş farklı boyutta; piksel-değişim GT'si ve işaretçi işlemleri devre dışı.")
            auth = None
    return img, img, mask, auth, [], _info_md(img, mask, auth), img, img, mask, auth


def on_sample(key):
    key = (key or "").strip()
    if not key or key not in dataset_index().index:
        raise gr.Error("Geçerli bir örnek seçin (ör. testing:000000045).")
    img, mask, auth, meta = load_dataset_sample(key)
    return img, img, mask, auth, [], meta_md(meta, img, mask, auth) + "  \n" + _info_md(img, mask, auth), img, img, mask, auth



# ------------------------------------------------------------------ 2 · Girdi › TASLAKLAR
# Her taslak bir klasör: image.png · mask.png (ops.) · auth.png (ops.) · meta.json (ad, tarih, kaynak, pipeline adımları).
# Konum: AI515_DRAFTS_DIR > yerelde <proje>/drafts > Space'te /data/drafts (yazılabilirse) > geçici klasör.
def _drafts_dir() -> Path:
    cands = [os.environ.get("AI515_DRAFTS_DIR")]
    cands += [str(DATA_ROOT / "drafts"), str(Path(tempfile.gettempdir()) / "ai515_drafts")] if IN_SPACE else [str(ROOT / "drafts")]
    for c in cands:
        if not c:
            continue
        d = Path(c)
        try:
            d.mkdir(parents=True, exist_ok=True)
            t = d / ".write_test"; t.write_text("ok"); t.unlink()
            return d
        except Exception:
            continue
    raise RuntimeError("Taslak klasörü oluşturulamadı.")


DRAFTS = None


_DRAFT_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")


def _drafts() -> Path:
    global DRAFTS
    if DRAFTS is None:
        DRAFTS = _drafts_dir()
    return DRAFTS


def _slug(name: str) -> str:
    tr = str.maketrans("çğıöşüÇĞİÖŞÜ ", "cgiosuCGIOSU_")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", (name or "").translate(tr))[:40].strip("_-")
    return s or "taslak"


def _png_write(path: Path, arr: np.ndarray):
    a = arr if arr.ndim == 2 else cv2.cvtColor(to_rgb(arr), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", a)
    if not ok:
        raise gr.Error(f"PNG yazılamadı: {path.name}")
    path.write_bytes(buf.tobytes())


def _png_read(path: Path, gray=False):
    if not path.exists():
        return None
    buf = np.frombuffer(path.read_bytes(), np.uint8)
    a = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    return a if gray else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def draft_list():
    """[(etiket, kimlik)] — en yeni önce."""
    out = []
    try:
        root = _drafts()
    except Exception:
        return out
    for d in root.iterdir():
        mj = d / "meta.json"
        if not (d.is_dir() and _DRAFT_ID_RE.match(d.name) and mj.exists()):
            continue
        try:
            m = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            continue
        tags = [t for t, k in (("maske", "has_mask"), ("eş", "has_auth")) if m.get(k)]
        n = len(m.get("steps") or [])
        if n:
            tags.append(f"{n} adım")
        lbl = f"{m.get('name', d.name)} · {m.get('created', '')[:16].replace('T', ' ')} · {m.get('w')}×{m.get('h')}" + (f" · {', '.join(tags)}" if tags else "")
        out.append((m.get("created", ""), lbl, d.name))
    out.sort(reverse=True)
    return [(lbl, i) for _, lbl, i in out]


def _draft_dd(value=None):
    ch = draft_list()
    ids = [i for _, i in ch]
    return gr.Dropdown(choices=ch, value=value if value in ids else (ids[0] if ids else None))


def draft_save(name, orig, mask, auth, steps, up_img, up_mask, up_auth, sample_key):
    """Girdi sekmesinde yüklü olanı (yoksa yükleme alanlarındakini) taslak olarak kaydeder."""
    src = "yüklü girdi"
    if orig is None:
        if up_img is None:
            raise gr.Error("Kaydedilecek görüntü yok: önce görüntü yükleyin ya da örnek seçin.")
        orig, _, mask, auth, steps, *_ = on_upload(up_img, up_mask, up_auth)
        src = "yükleme alanları"
    did = f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{_slug(name)}"
    d, k = _drafts() / did, 1
    while d.exists():
        k += 1; d = _drafts() / f"{did}_{k}"
    did = d.name
    d.mkdir(parents=True)
    _png_write(d / "image.png", orig)
    if mask is not None:
        _png_write(d / "mask.png", to_gray(mask))
    if auth is not None:
        _png_write(d / "auth.png", auth)
    h, w = orig.shape[:2]
    meta = dict(name=(name or "").strip() or did, created=datetime.datetime.now().isoformat(timespec="seconds"),
                w=w, h=h, has_mask=mask is not None, has_auth=auth is not None,
                sample_key=(sample_key or "").strip() or None, steps=list(steps or []), source=src)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    msg = f"💾 Taslak kaydedildi: **{meta['name']}** ({w}×{h}" + (", maske" if meta["has_mask"] else "") + \
          (", authentic eş" if meta["has_auth"] else "") + (f", {len(meta['steps'])} pipeline adımı" if meta["steps"] else "") + \
          f") · kaynak: {src} · `{d}`"
    return _draft_dd(did), msg


def draft_load(did):
    if not did or not _DRAFT_ID_RE.match(str(did)) or not (_drafts() / did / "meta.json").exists():
        raise gr.Error("Geçerli bir taslak seçin.")
    d = _drafts() / did
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    img, mask, auth = _png_read(d / "image.png"), _png_read(d / "mask.png", gray=True), _png_read(d / "auth.png")
    if img is None:
        raise gr.Error("Taslağın görüntü dosyası eksik.")
    steps = list(meta.get("steps") or [])
    cur, note = img, ""
    if steps:
        try:
            cur = replay(img, steps, auth)
        except Exception as e:
            steps, note = [], f"  \n⚠ Pipeline adımları yeniden uygulanamadı ({e}); orijinal görüntü yüklendi."
    info = f"📂 **Taslak:** {meta.get('name', did)} · {meta.get('created', '')[:16].replace('T', ' ')}" + \
           (f" · örnek `{meta['sample_key']}`" if meta.get("sample_key") else "") + "  \n" + _info_md(img, mask, auth) + note
    status = f"📂 Yüklendi: **{meta.get('name', did)}**" + (f" ({len(steps)} pipeline adımı geri yüklendi)" if steps else "")
    return (img, cur, mask, auth, steps, info, img, cur, mask, auth, steps_md(steps), img, mask, auth, status)


def draft_delete(did):
    if not did or not _DRAFT_ID_RE.match(str(did)):
        raise gr.Error("Silinecek taslağı seçin.")
    import shutil
    d = _drafts() / did
    if d.is_dir():
        shutil.rmtree(d)
    return _draft_dd(), f"🗑 Taslak silindi: `{did}`"


def op_choices(cat):
    names = list(OPS[cat].keys())
    return gr.Dropdown(choices=names, value=names[0]), *param_updates(cat, names[0])


def param_updates(cat, name):
    spec = OPS[cat][name]
    ups = []
    for i in range(3):
        if i < len(spec["params"]):
            _, label, mn, mx, st, dv = spec["params"][i]
            ups.append(gr.Slider(label=label, minimum=mn, maximum=mx, step=st, value=dv, visible=True))
        else:
            ups.append(gr.Slider(visible=False))
    if spec["modes"]:
        ups.append(gr.Dropdown(choices=spec["modes"], value=spec["modes"][0], visible=True, label="mod"))
    else:
        ups.append(gr.Dropdown(choices=[], value=None, visible=False))
    ups.append(spec["doc"] or " ")
    return tuple(ups)


def preview(cur, cat, name, p1, p2, p3, mode, auth):
    out, info = run_op(cur, cat, name, p1, p2, p3, mode, auth)
    return out, hist_figure(cur, out), f"Önizleme: {name} — {info}"


def apply_step(cur, orig, steps, cat, name, p1, p2, p3, mode, auth):
    out, info = run_op(cur, cat, name, p1, p2, p3, mode, auth)
    steps = list(steps) + [dict(cat=cat, name=name, info=info, p=[p1, p2, p3], mode=mode)]
    return out, out, steps, steps_md(steps), hist_figure(cur, out), f"Uygulandı: {name} — {info}"


def steps_md(steps):
    if not steps:
        return "_Pipeline boş — orijinal görüntü._"
    return "\n".join(f"{i + 1}. **{s['name']}** ({s['cat'].split('·')[1].strip()}) — {s['info']}" for i, s in enumerate(steps))


def undo(orig, steps, auth):
    steps = list(steps)[:-1]
    cur = replay(orig, steps, auth) if orig is not None else None
    return cur, cur, steps, steps_md(steps), "Son adım geri alındı."


def reset(orig):
    return orig, orig, [], steps_md([]), "Pipeline sıfırlandı."


def apply_preset(orig, preset, auth):
    seq = PRESETS.get(preset, [])
    if orig is None:
        raise gr.Error("Önce bir görüntü yükleyin.")
    if not seq:
        return orig, orig, [], steps_md([]), None, "Preset seçilmedi."
    cur, steps = orig, []
    for cat, name, kw in seq:
        spec = OPS[cat][name]
        p = [sp[5] for sp in spec["params"]] + [0] * (3 - len(spec["params"]))
        for i, key in enumerate(("p1", "p2", "p3")):
            if key in kw:
                p[i] = kw[key]
        mode = kw.get("mode", spec["modes"][0] if spec["modes"] else None)
        cur, info = run_op(cur, cat, name, *p, mode, auth)
        steps.append(dict(cat=cat, name=name, info=info, p=p, mode=mode))
    return cur, cur, steps, steps_md(steps), hist_figure(orig, cur), f"Preset uygulandı: {preset}"


def classical_features(cur, mask):
    if cur is None:
        raise gr.Error("Görüntü yok.")
    if not FC_OK:
        return pd.DataFrame({"hata": ["baselines/features_classical.py bulunamadı"]})
    rgb = to_rgb(cur)
    h, w = rgb.shape[:2]
    S = min(256, h, w)
    bb = bbox_of(mask) if (mask is not None and mask.shape == rgb.shape[:2]) else None
    if bb:
        cx, cy = (bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2
    else:
        cx, cy = w // 2, h // 2
    x0, y0 = int(np.clip(cx - S // 2, 0, w - S)), int(np.clip(cy - S // 2, 0, h - S))
    f = FC.extract_features(rgb[y0:y0 + S, x0:x0 + S])
    df = pd.DataFrame({"öznitelik": FC.feature_names(), "değer": np.round(f, 5)})
    df.insert(0, "grup", df["öznitelik"].str.split("_").str[0])
    return df


def run_unet(orig, cur, mask, auth, model_name, source, thr, use_otsu, morph_k, gt_kind):
    if not TORCH_OK:
        raise gr.Error(f"torch yüklü değil: {_TORCH_ERR}")
    if orig is None:
        raise gr.Error("Görüntü yok. Girdi sekmesinden yükleyin ya da Simülasyon'dan aktarın.")
    if not model_name:
        raise gr.Error("Model seçin (models/ klasöründe unet_*.pt bulunamadı).")
    img = to_rgb(orig if source == "Orijinal görüntü" else cur)
    prob, info = gpu_infer([model_name], [], [img])[0][model_name]
    binm, thr_used = postprocess(prob, thr, int(morph_k), use_otsu)
    rect, bb = rectangularity(binm)
    sc = image_score(prob); it, cal = image_threshold(model_name)
    gt, gt_note = None, ""
    if gt_kind.startswith("piksel"):
        if auth is not None and auth.shape[:2] == img.shape[:2]:
            gt, _ = derive_pixel_gt(to_rgb(orig), auth)
            gt_note = "GT: piksel-değişim (D>8, morfolojik)"
        else:
            gt_note = "⚠ piksel GT için aynı boyutta authentic gerekir; GT yok."
    elif mask is not None and mask.shape == img.shape[:2]:
        gt = mask > 127; gt_note = "GT: resmî bbox maskesi"
    elif mask is not None:
        gt_note = "⚠ GT boyutu görüntüyle uyuşmuyor (geometrik adım?) — metrik hesaplanmadı."
    rows = [("model", friendly(model_name)), ("girdi", source), ("cihaz", "ZeroGPU" if ZERO_GPU else str(DEVICE)),
            ("pencere sayısı", info["n_windows"]), ("süre (s)", f"{info['sec']:.2f}"),
            ("eşik", f"{thr_used:.2f}" + (" (Otsu)" if use_otsu else "")),
            ("pozitif piksel", f"{int(binm.sum())} (%{100 * binm.mean():.3f})"),
            ("tahmin dikdörtgenselliği", fmt(rect, 2)), ("görüntü skoru (top-%1)", f"{sc:.3f}"),
            ("görüntü kararı", f"{'SAHTE' if sc >= it else 'GERÇEK'} (eşik {it:.3f}{'' if cal else ', kalibre değil'})")]
    if gt is not None:
        m = hist_metrics(prob, gt, thr_used)
        rows += [("F1@eşik", fmt(m["F1"])), ("IoU@eşik", fmt(m["IoU"])), ("AP (görüntü, kesin)", fmt(m["AP"])),
                 ("Oracle-F1", fmt(m["OracleF1"])), ("TP/FP/FN", f"{m['TP']}/{m['FP']}/{m['FN']}")]
    df = pd.DataFrame(rows, columns=["ölçüt", "değer"])
    ov = overlay(img, binm, gt, bb)
    return disp(heatmap_rgb(prob)), disp(ov), disp((binm * 255).astype(np.uint8)), df, "Kırmızı: tahmin · Mavi: GT · Yeşil kutu: tahmin bbox. " + gt_note, None


def compare_unet(orig, cur, mask, auth, model_name, thr, use_otsu, morph_k, gt_kind, steps):
    if not steps:
        raise gr.Error("Pipeline boş; karşılaştırma için önce ön işleme adımı ekleyin.")
    if to_rgb(cur).shape[:2] != to_rgb(orig).shape[:2]:
        raise gr.Error("Pipeline çıktısı farklı boyutta; geometrik adımlarla karşılaştırma desteklenmiyor.")
    a = run_unet(orig, cur, mask, auth, model_name, "Orijinal görüntü", thr, use_otsu, morph_k, gt_kind)
    b = run_unet(orig, cur, mask, auth, model_name, "Pipeline çıktısı", thr, use_otsu, morph_k, gt_kind)
    da, db = a[3].set_index("ölçüt")["değer"], b[3].set_index("ölçüt")["değer"]
    df = pd.DataFrame({"ölçüt": da.index, "orijinal": da.values, "pipeline": db.reindex(da.index).values})
    return a[1], b[1], df, f"Pipeline: {steps_md(steps)}"


def run_resnet(orig, cur, mask, model_name, source, stride):
    if not TORCH_OK:
        raise gr.Error(f"torch yüklü değil: {_TORCH_ERR}")
    if orig is None:
        raise gr.Error("Görüntü yok.")
    if not model_name:
        raise gr.Error("ResNet checkpoint'i bulunamadı.")
    img = to_rgb(orig if source == "Orijinal görüntü" else cur)
    hm, info = gpu_infer([], [model_name], [img], int(stride))[0][model_name]
    scores = info["scores"]
    smax, rthr, rcal = resnet_decision(model_name, scores)
    rows = [("model", friendly(model_name)), ("girdi", source), ("patch sayısı", len(scores)), ("süre (s)", f"{info['sec']:.2f}"),
            ("görüntü skoru (maks patch)", f"{smax:.3f}"),
            (f"karar (eşik {rthr:.3f}{'' if rcal else ', kalibre değil'})", "SAHTE" if smax >= rthr else "GERÇEK"),
            ("ortalama patch skoru", f"{np.mean([s for s, _, _ in scores]):.3f}")]
    if mask is not None and mask.shape == img.shape[:2]:
        bb = bbox_of(mask)
        if bb:
            x0b, y0b, x1b, y1b = bb
            hit = lambda y, x: not (x + 224 <= x0b or x >= x1b or y + 224 <= y0b or y >= y1b)
            inter = [s for s, y, x in scores if hit(y, x)]; clean = [s for s, y, x in scores if not hit(y, x)]
            rows += [("GT ile kesişen patch skoru (ort/maks)", f"{np.mean(inter):.3f} / {np.max(inter):.3f}" if inter else "—"),
                     ("temiz patch skoru (ort/maks)", f"{np.mean(clean):.3f} / {np.max(clean):.3f}" if clean else "—")]
    ov = cv2.addWeighted(img, 0.55, heatmap_rgb(hm), 0.45, 0)
    if mask is not None and mask.shape == img.shape[:2]:
        ov = overlay(ov, None, mask > 127)
    return disp(ov), pd.DataFrame(rows, columns=["ölçüt", "değer"])


def run_green(cur, mask, auth):
    if cur is None:
        raise gr.Error("Görüntü yok.")
    rgb = to_rgb(cur)
    gm = green_marker_mask(rgb)
    if auth is not None and auth.shape == rgb.shape:
        gm &= ~green_marker_mask(auth)
    txt = f"Yeşil işaretçi pikseli: **{int(gm.sum())}**"
    if mask is not None and mask.shape == rgb.shape[:2]:
        m = (mask > 127).astype(np.uint8)
        ring = (cv2.dilate(m, np.ones((13, 13), np.uint8)) > 0) & ~(cv2.erode(m, np.ones((13, 13), np.uint8)) > 0)
        nr = int((gm & ring).sum())
        txt += f" · bbox kenar halkasında (±6 px): **{nr}** · bbox içinde: {int((gm & (m > 0)).sum())}"
        txt += "  \n→ " + ("**kalıntı var** (eşik ≥20 px; sahtelerin %75,4'ünde görülen üretim-hattı artefaktı)" if nr >= 20 else "kalıntı eşiğin altında")
    ov = rgb.copy(); ov[cv2.dilate(gm.astype(np.uint8), K3) > 0] = [255, 0, 255]
    return disp(ov), txt


def results_markdown():
    parts = []
    p3 = META / "phase3_tables.md"
    if p3.exists():  # Faz 3: YOLO11, DeepLabV3+, işaretçisiz eğitim, sondalar (bildiri: paper/main.pdf)
        parts.append(p3.read_text(encoding="utf-8") + "\n\n---\n")
    v2 = META / "v2_tables.md"
    if v2.exists():  # AIForge-Doc v2 (GPT-Image-2) harici test: v1 modelleri, yeniden eğitim yok
        parts.append(v2.read_text(encoding="utf-8") + "\n\n---\n")
    if RESULTS_MD.exists():
        parts.append("## Küçük-veri (smoke) baseline sonuçları\n")
        parts.append(RESULTS_MD.read_text(encoding="utf-8"))
    if P2_SUMMARY.exists():
        df = pd.read_csv(P2_SUMMARY)
        df = df[df["subset"] == "all"][["train_gt", "gt", "auprc_pooled_mean", "auprc_pooled_std", "f1_tau_mean", "f1_tau_std", "iou_tau_mean", "n_images"]].round(3)
        parts.append("\n\n## PHASE 2 — resmî test (812), 3 tohum: eğitim GT × değerlendirme GT — `summary_test_agg.csv`\n")
        parts.append(df.to_markdown(index=False))
    if IMG_TAU:
        rows = [(friendly('PHASE2/' + k), v["thr"], v["auroc"], v["tpr"], v["fpr"]) for k, v in IMG_TAU.items() if isinstance(v, dict)]
        parts.append("\n\n## Görüntü-düzeyi karar eşikleri (val 487 sahte + 487 eşleşen authentic, Youden-J)\n")
        parts.append(pd.DataFrame(rows, columns=["model", "eşik", "AUROC", "TPR", "FPR"]).to_markdown(index=False))
    if RESNET_TAU:
        rows = [(friendly(('PHASE1/' if k.endswith('_best') else 'PHASE4/') + k), v["thr"], v["auroc"], v["tpr"], v["fpr"])
                for k, v in RESNET_TAU.items() if isinstance(v, dict) and "thr" in v]
        parts.append("\n\n## ResNet-18 görüntü-düzeyi karar eşikleri (skor = maks patch; val 487 + 487, Youden-J) — Faz 4\n")
        parts.append(pd.DataFrame(rows, columns=["model", "eşik", "AUROC", "TPR", "FPR"]).to_markdown(index=False))
    return "\n".join(parts) if parts else "Sonuç dosyaları bulunamadı."


# ---------------- simülasyon sekmesi yardımcıları
def sim_filter(split, source, tool, area, pair_only, marker, max_mp):
    df = filter_index(split, source, tool, area, pair_only, marker, max_mp)
    ch = sample_choices(df)
    return gr.Dropdown(choices=ch, value=None), f"{len(df)} örnek filtreye uyuyor."


def sim_random(split, source, tool, area, pair_only, marker, max_mp):
    df = filter_index(split, source, tool, area, pair_only, marker, max_mp)
    keys = list(df.index)
    random.shuffle(keys)
    for k in keys[:60]:
        if sample_exists(k):
            return gr.Dropdown(choices=sample_choices(df), value=k), f"{len(df)} örnek filtreye uyuyor · rastgele seçildi: `{k}`"
    raise gr.Error("Filtreye uyan, bucket'ta mevcut örnek bulunamadı.")


def sim_preview(key):
    key = (key or "").strip()
    if not key:
        return None, None, "Örnek seçilmedi."
    if key not in dataset_index().index:
        return None, None, f"⚠ '{key}' veri setinde yok (biçim: testing:000000045)."
    if not sample_exists(key):
        return None, None, f"⚠ '{key}' dosyası veri kökünde bulunamadı."
    img, mask, auth, meta = load_dataset_sample(key)
    return disp(img, 900), disp(overlay(img, None, mask > 127) if mask is not None else img, 900), meta_md(meta, img, mask, auth)


def sim_to_tabs(state):
    if not state:
        raise gr.Error("Önce bir simülasyon çalıştırın.")
    img, mask, auth = state["forged"], state["mask"], state["auth"]
    meta = load_dataset_sample(state["key"])[3]
    gr.Info(f"{state['key']} diğer sekmelere aktarıldı.")
    return img, img, mask, auth, [], meta_md(meta, img, mask, auth), img, img, mask, auth, steps_md([])
