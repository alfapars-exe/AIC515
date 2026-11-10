"""appcore.simulation — tek örnek simülasyonu: simulate, insights, rapor dosyaları."""
from __future__ import annotations
from ._common import *  # noqa: F401,F403 — ortak içe aktarmalar
from .config import (  # noqa: F401
    DEVICE, ZERO_GPU
)
from .imaging import (  # noqa: F401
    K3, derive_pixel_gt, green_marker_mask, to_rgb
)
from .ops import (  # noqa: F401
    replay
)
from .dataset import (  # noqa: F401
    dataset_index, load_dataset_sample
)
from .models import (  # noqa: F401
    friendly, gpu_infer
)
from .metrics import (  # noqa: F401
    default_tau, disp, heatmap_rgb, hist_metrics, image_score, image_threshold, overlay, postprocess, rectangularity,
    ref_lookup, resnet_decision
)
from .scenarios import (  # noqa: F401
    INPUT_TYPES, bbox_of, build_input, degrade, montage, show_df, zoom_box
)

# ================================================================== SİMÜLASYON (tek örnek)
SIM_COLS = ["Model", "Görüntü skoru", "Görüntü eşiği", "Karar", "Doğru", "τ piksel", "F1 bbox-GT", "IoU bbox-GT", "AP bbox-GT",
            "Oracle-F1", "F1 piksel-GT", "AP piksel-GT", "Kutu kapsama", "Pozitif piksel %", "Dikdörtgensellik",
            "Ref F1@0.5 (boru hattı)", "Uyg. F1@0.5", "Süre s"]


def evaluate_unet(name, prob, label, gt_bbox, gt_pixel, meta, clean_forged):
    stem = name.split("/", 1)[-1]
    tb = default_tau(name, "bbox"); tp = default_tau(name, "pixel_eps8_morph")
    sc = image_score(prob); it, calibrated = image_threshold(name)
    dec = int(sc >= it)
    binm, _ = postprocess(prob, tb, 0, False)
    rect, pbb = rectangularity(binm)
    row = {"Model": friendly(name), "Görüntü skoru": sc, "Görüntü eşiği": f"{it:.3f}" + ("" if calibrated else " (kalibre değil)"),
           "Karar": "SAHTE" if dec else "GERÇEK", "Doğru": "✓" if dec == label else "✗", "τ piksel": tb,
           "Pozitif piksel %": 100 * binm.mean(), "Dikdörtgensellik": rect}
    if gt_bbox is not None:
        gb = gt_bbox > 127
        row["Kutu kapsama"] = float((binm & gb).sum() / max(gb.sum(), 1))
        if label == 1:
            m = hist_metrics(prob, gb, tb)
            row.update({"F1 bbox-GT": m["F1"], "IoU bbox-GT": m["IoU"], "AP bbox-GT": m["AP"], "Oracle-F1": m["OracleF1"]})
            row["Uyg. F1@0.5"] = hist_metrics(prob, gb, 0.5)["F1"]
            ref = ref_lookup(meta["split"], meta["new_id"], stem, "bbox") if clean_forged else None
            if ref is not None:
                row["Ref F1@0.5 (boru hattı)"] = float(ref["f1_0p5"])
    if gt_pixel is not None and label == 1:
        m = hist_metrics(prob, gt_pixel, tp)
        row.update({"F1 piksel-GT": m["F1"], "AP piksel-GT": m["AP"]})
    return row, binm, pbb


def simulate(key, input_type, degradation, use_pipeline, steps, unet_sel, resnet_sel, stride):
    """Adım adım simülasyon (generator). Her adımda arayüz güncellenir."""
    log, gallery, t_start = [], [], time.time()
    empty = pd.DataFrame(columns=SIM_COLS)
    sent = {"n": -1}

    def emit(df=empty, verdict="", zoom=None, files=None, state=None, final=False):
        # Galeri yalnız yeni görüntü eklendiyse gönderilir (ağ yükünü azaltır); diğer adımlarda gr.skip().
        g = gr.skip() if (len(gallery) == sent["n"] and not final) else list(gallery)
        sent["n"] = len(gallery)
        return ("\n".join(log), g, show_df(df), verdict, zoom, files, state)

    def step(msg):
        log.append(f"- ✓ **{msg}** · {time.time() - t_start:.1f} s")

    key = (key or "").strip()
    if not key:
        raise gr.Error("Önce bir örnek seçin (filtre + 'Rastgele seç' ya da listeden).")
    if key not in dataset_index().index:
        raise gr.Error(f"'{key}' veri setinde yok. Biçim: training:000000123 ya da testing:000000045.")
    if not unet_sel and not resnet_sel:
        raise gr.Error("En az bir model seçin.")
    log.append(f"### Simülasyon · `{key}`")
    log.append("- ⏳ veri yükleniyor…"); yield emit()
    forged, mask, auth, meta = load_dataset_sample(key)
    log.pop(); step(f"1/7 Veri yüklendi — {forged.shape[1]}×{forged.shape[0]} px, GT {'var' if mask is not None else 'yok'}, authentic {'var' if auth is not None else 'yok'}")
    bb = bbox_of(mask)
    gallery.append((disp(forged), "1 · Sahte görüntü (forged)"))
    if mask is not None:
        gallery.append((disp(overlay(forged, None, mask > 127)), "1 · Resmî GT: alan kutusu (bbox)"))
    yield emit()

    # 2 · GT analizi
    gt_pixel, D, gt_note = None, None, ""
    if auth is not None:
        gallery.append((disp(auth), "2 · Authentic eş (orijinal)"))
        gt_pixel, D = derive_pixel_gt(forged, auth)
        if mask is not None:
            gb = mask > 127
            inside = float((D[gb] > 8).mean()) if gb.any() else float("nan")
            outside = int(((D > 8) & ~gb).sum())
            gt_note = f"bbox piksellerinin %{100 * inside:.0f}'i gerçekten değişmiş (ε=8); bbox dışında {outside} değişen piksel"
        gallery.append((disp(cv2.cvtColor(cv2.applyColorMap(np.clip(D.astype(np.int32) * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)), "2 · |sahte − authentic| × 4"))
        gallery.append((disp(overlay(forged, None, gt_pixel, gt_color=(255, 200, 0))), "2 · Türetilmiş piksel-değişim GT (eps8_morph)"))
    step("2/7 Ground truth analizi — " + (gt_note or "authentic eş yok → yalnız bbox GT"))
    yield emit()

    # 3 · yeşil işaretçi
    gm = green_marker_mask(forged)
    if auth is not None:
        gm &= ~green_marker_mask(auth)
    n_ring = 0
    if mask is not None:
        m = (mask > 127).astype(np.uint8)
        ring = (cv2.dilate(m, np.ones((13, 13), np.uint8)) > 0) & ~(cv2.erode(m, np.ones((13, 13), np.uint8)) > 0)
        n_ring = int((gm & ring).sum())
    gv = forged.copy(); gv[cv2.dilate(gm.astype(np.uint8), K3) > 0] = [255, 0, 255]
    gallery.append((disp(gv), f"3 · Yeşil işaretçi pikselleri (magenta): {int(gm.sum())} px"))
    step(f"3/7 Yeşil işaretçi taraması — kutu kenarında {n_ring} px → " + ("**kalıntı VAR**" if n_ring >= 20 else "kalıntı yok"))
    yield emit()

    # 4 · girdi
    inp, label, note = build_input(forged, auth, input_type)
    if degradation and not degradation.startswith("Yok"):
        inp = degrade(inp, degradation); note += f" + {degradation}"
    if use_pipeline and steps:
        inp = to_rgb(replay(inp, steps, auth)); note += f" + ön işleme pipeline'ı ({len(steps)} adım)"
    same_geom = inp.shape[:2] == forged.shape[:2]
    gallery.append((disp(inp), f"4 · Model girdisi: {note}"))
    clean_forged = input_type == INPUT_TYPES[0] and degradation.startswith("Yok") and not (use_pipeline and steps)
    step(f"4/7 Model girdisi hazır — {note}; gerçek etiket **{'SAHTE' if label else 'GERÇEK'}**")
    yield emit()

    # 5–6 · çıkarım (tek GPU çağrısı)
    log.append(f"- ⏳ {len(unet_sel)} U-Net + {len(resnet_sel)} ResNet çalışıyor ({'ZeroGPU' if ZERO_GPU else DEVICE})…"); yield emit()
    t_inf = time.time()
    res = gpu_infer(list(unet_sel), list(resnet_sel), [inp], int(stride))[0]
    log.pop(); step(f"5/7 Çıkarım tamamlandı — {time.time() - t_inf:.1f} s ({'ZeroGPU' if ZERO_GPU else DEVICE})")
    rows, panels_zoom, verdicts = [], [], []
    gtb = mask if (mask is not None and same_geom) else None
    gtp = gt_pixel if (gt_pixel is not None and same_geom) else None
    for n in resnet_sel:
        hm, info = res[n]
        smax, rthr, rcal = resnet_decision(n, info["scores"])
        dec = int(smax >= rthr)
        rows.append({"Model": friendly(n), "Görüntü skoru": smax, "Görüntü eşiği": f"{rthr:.3f}" + ("" if rcal else " (kalibre değil)"), "Karar": "SAHTE" if dec else "GERÇEK",
                     "Doğru": "✓" if dec == label else "✗", "Süre s": info["sec"]})
        ov = cv2.addWeighted(inp, 0.55, heatmap_rgb(hm), 0.45, 0)
        gallery.append((disp(overlay(ov, None, gtb > 127 if gtb is not None else None)), f"5 · {friendly(n)} — patch skor haritası (maks {smax:.2f})"))
        panels_zoom.append((f"{friendly(n)}\npatch skoru", ov))
        verdicts.append((friendly(n), dec, smax, rthr))
    for n in unet_sel:
        prob, info = res[n]
        row, binm, pbb = evaluate_unet(n, prob, label, gtb, gtp, meta, clean_forged)
        row["Süre s"] = info["sec"]
        rows.append(row)
        gallery.append((disp(heatmap_rgb(prob)), f"6 · {friendly(n)} — olasılık haritası"))
        gallery.append((disp(overlay(inp, binm, gtb > 127 if gtb is not None else None, pbb)), f"6 · {friendly(n)} — τ={row['τ piksel']:.2f} (kırmızı tahmin · mavi GT · yeşil tahmin kutusu)"))
        panels_zoom.append((f"{friendly(n)}\nτ={row['τ piksel']:.2f}", overlay(inp, binm, None, None)))
        verdicts.append((friendly(n), int(row["Karar"] == "SAHTE"), row["Görüntü skoru"], float(row["Görüntü eşiği"].split()[0])))
    df = pd.DataFrame(rows, columns=SIM_COLS)
    step("6/7 Değerlendirme — metrikler hesaplandı")
    yield emit(df=df)

    # 7 · yakın plan + karar + rapor
    zb = zoom_box(bb if same_geom else None, inp.shape)
    pz = [("Model girdisi", inp)]
    if auth is not None and input_type == INPUT_TYPES[0] and same_geom:
        pz.append(("Authentic", auth))
        pz.append(("|sahte − authentic| × 4", cv2.cvtColor(cv2.applyColorMap(np.clip(D.astype(np.int32) * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)))
    if gtb is not None:
        pz.append(("GT: bbox", overlay(inp, None, gtb > 127)))
    if gtp is not None:
        pz.append(("GT: piksel-değişim", overlay(inp, None, gtp, gt_color=(255, 200, 0))))
    pz += panels_zoom
    zoom = montage(pz, zb, f"{key} · yakın plan (alan kutusu çevresi) · girdi: {input_type}")
    truth = "SAHTE" if label else "GERÇEK"
    vlines = [f"### Gerçek etiket: **{truth}** — {note}"]
    for name, dec, sc, th in verdicts:
        ok = dec == label
        vlines.append(f"- {'✅' if ok else '❌'} **{name}** → **{'SAHTE' if dec else 'GERÇEK'}** (skor {sc:.3f} {'≥' if dec else '<'} eşik {th:.3f})")
    vlines.append("")
    vlines += insights(df, label, input_type, n_ring, gt_note, degradation, auth is not None, clean_forged)
    verdict = "\n".join(vlines)
    files = write_sim_report(key, meta, input_type, degradation, steps if use_pipeline else [], df, zoom, gallery, verdict)
    step(f"7/7 Rapor hazır — toplam {time.time() - t_start:.1f} s")
    print(f"[sim] {key} · {input_type} · {degradation} · {len(unet_sel)}U+{len(resnet_sel)}R · çıkarım {time.time() - t_inf:.1f}s · toplam {time.time() - t_start:.1f}s", flush=True)
    state = dict(key=key, inp=inp, forged=forged, mask=mask, auth=auth)
    yield emit(df=df, verdict=verdict, zoom=zoom, files=files, state=state, final=True)


def insights(df, label, input_type, n_ring, gt_note, degradation, has_auth, clean_forged):
    out = ["#### Yorum"]
    get = lambda name_part, col: next((r[col] for _, r in df.iterrows() if name_part in r["Model"] and pd.notna(r.get(col))), None)
    rb = get("bbox-GT (s", "Dikdörtgensellik")
    if rb is not None and not math.isnan(rb):
        out.append(f"- **Kutu davranışı:** bbox-GT modelinin tahmini dikdörtgensellik {rb:.2f} — " +
                   ("kutunun tamamını işaretliyor (kutu maskesiyle eğitimin beklenen sonucu)." if rb > 0.7 else "tahmin glif düzeyinde parçalı."))
    f1b, f1p = get("bbox-GT (s", "F1 bbox-GT"), get("bbox-GT (s", "F1 piksel-GT")
    if f1b is not None and f1p is not None:
        out.append(f"- **H1 (GT seçimi):** aynı tahmin bbox-GT ile F1 {f1b:.3f}, piksel-değişim GT ile F1 {f1p:.3f} → fark {f1b - f1p:+.3f}. "
                   "Kutu maskesi, değişmeyen kutu piksellerini de doğru sayar; raporlanan skor GT tanımına bağlıdır.")
    if gt_note:
        out.append(f"- **GT analizi:** {gt_note}.")
    if n_ring >= 20:
        out.append(f"- **Yeşil işaretçi:** kutu kenarında {n_ring} px kalıntı var. 'Sahte, yeşil işaretçi onarılmış' senaryosuyla tekrar çalıştırıp skorun ne kadar düştüğüne bakın (PHASE 2 Ek-C: AP 0,988 → 0,763).")
    if input_type == INPUT_TYPES[3]:
        cov = get("bbox-GT (s", "Kutu kapsama")
        if cov is not None:
            out.append(f"- **Kısayol testi:** içerik değişmediği hâlde bbox-GT modeli kutunun %{100 * cov:.0f}'ini işaretledi" +
                       (" → model içeriği değil işaretçiyi kullanıyor." if cov > 0.3 else " → bu örnekte işaretçi tek başına yeterli olmadı."))
    if input_type == INPUT_TYPES[1]:
        out.append("- **Negatif kontrol:** girdi orijinal belgedir; 'GERÇEK' kararı ve düşük pozitif piksel oranı beklenir.")
    if degradation and not degradation.startswith("Yok"):
        out.append(f"- **Bozulma ({degradation}):** PHASE 2'de ΔAUPRC (bbox modeli) JPEG Q75 −0,154, 0,75× −0,112, blur σ=1 −0,190; 'Yok' ile karşılaştırın.")
    if clean_forged:
        rf, uf = get("bbox-GT (s42)", "Ref F1@0.5 (boru hattı)"), get("bbox-GT (s42)", "Uyg. F1@0.5")
        if rf is not None and uf is not None:
            out.append(f"- **Yeniden üretim:** boru hattının kayıtlı F1 (τ = 0,5) = {rf:.3f}, uygulamanın hesapladığı = {uf:.3f} (fark {uf - rf:+.3f}).")
    if not has_auth:
        out.append("- Authentic eş olmadığı için piksel-değişim GT'si ve işaretçi senaryoları bu örnekte kullanılamaz.")
    return out


def _outdir(prefix):
    d = Path(tempfile.mkdtemp(prefix=prefix))
    return d


def write_sim_report(key, meta, input_type, degradation, steps, df, zoom, gallery, verdict):
    d = _outdir("ai515_sim_")
    stem = key.replace(":", "_")
    df.to_csv(d / f"{stem}_metrikler.csv", index=False)
    cv2.imwrite(str(d / f"{stem}_yakin_plan.png"), cv2.cvtColor(zoom, cv2.COLOR_RGB2BGR))
    rep = dict(zaman=datetime.datetime.now().isoformat(timespec="seconds"), ornek=meta, girdi=input_type, bozulma=degradation,
               on_isleme=[f"{s['name']} — {s['info']}" for s in (steps or [])], cihaz=str(DEVICE), zero_gpu=ZERO_GPU,
               metrikler=json.loads(df.to_json(orient="records", force_ascii=False)), yorum=verdict)
    (d / f"{stem}_rapor.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    md = [f"# Simülasyon raporu — {key}", "", f"- Zaman: {rep['zaman']}", f"- Girdi senaryosu: {input_type}", f"- Bozulma: {degradation}",
          f"- Ön işleme: {', '.join(rep['on_isleme']) or 'yok'}", "", verdict, "", df.to_markdown(index=False, floatfmt=".3f"), "",
          f"![yakın plan]({stem}_yakin_plan.png)"]
    (d / f"{stem}_rapor.md").write_text("\n".join(md), encoding="utf-8")
    return [str(d / f"{stem}_rapor.md"), str(d / f"{stem}_rapor.json"), str(d / f"{stem}_metrikler.csv"), str(d / f"{stem}_yakin_plan.png")]
