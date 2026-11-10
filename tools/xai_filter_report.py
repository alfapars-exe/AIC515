"""
tools/xai_filter_report.py — XAI · Katman Filtreleri model raporu.

  python tests\\test_xai_filters.py          # önce: reports/xai_filters/test_results.json
  python tools\\xai_filter_report.py         # → reports/xai_filters/RAPOR.md + figs/*.png + rapor.json
  python tools\\xai_filter_report.py --export-yolo-stems   # Space için YOLO ilk-katman dosyaları → models/phase3_yolo/

Her model için: tespit edilen ilk katmanlar, filtre sayısı, kernel boyutları, üretilen aktivasyon haritaları ve test sonuçları.
Temsilî modellerde (aile başına bir) kernel grid'i, filtre↔harita çiftleri, aktivasyon grid'i ve orijinal karşılaştırması kaydedilir.
"""
from __future__ import annotations
import glob, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import cv2
import numpy as np
import xai_filters as XF

OUT = ROOT / "reports" / "xai_filters"
FIG = OUT / "figs"
FIG.mkdir(parents=True, exist_ok=True)
REPR = ["PHASE2/unet_bbox_s42", "PHASE1/unet_rgbfreq_best", "PHASE1/resnet18_ft_best", "PHASE2/dlv3p_bbox_s42",
        "YOLO/aiforge-tiles-yolo11s-img1024-conserv-s42"]


def save(img, name):
    cv2.imwrite(str(FIG / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return f"figs/{name}"


def export_stems():
    dst = ROOT / "models" / "phase3_yolo"
    dst.mkdir(parents=True, exist_ok=True)
    for best in sorted(glob.glob(str(ROOT / "phase3_yolo" / "runs" / "*" / "weights" / "best.pt"))):
        run = Path(best).parent.parent.name
        if "dryrun" in run or not (Path(best).parent.parent / "summary.json").exists():
            continue
        p = XF.export_yolo_stem(best, dst / f"{run}_stem.pt")
        print("stem:", p)


def main():
    if "--export-yolo-stems" in sys.argv:
        export_stems()
    tr = json.loads((OUT / "test_results.json").read_text(encoding="utf-8")) if (OUT / "test_results.json").exists() else {}
    ims = sorted(glob.glob(str(ROOT / "data" / "AIForge-Doc-v1" / "TestingSet" / "images" / "*.png")))
    p = ims[0]
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(p.replace("images", "masks"), cv2.IMREAD_GRAYSCALE)
    specs = {s.key: s for s in XF.discover_models(ROOT / "models", ROOT / "phase3_yolo" / "runs")}
    figs = {}
    for k in REPR:
        if k not in specs:
            continue
        an = XF.run_analysis(specs[k], img, mask, XF.MODE_CROP, 512, "CPU", 6)
        L = an.layers[0]
        st = list(L.stages)[-1]
        tag = k.replace("/", "_")
        df = XF.stats_df(an, L, st)
        f_best = int(df.sort_values("iç/dış oranı" if "iç/dış oranı" in df else "ort|a|", ascending=False).iloc[0]["filtre"])
        figs[k] = dict(label=an.model_label, region=an.region_desc, best=f_best, files=[
            save(XF.fig_kernel_grid(L), f"{tag}__1_kernel_grid.png"),
            save(XF.fig_pairs(an, L, st, 1, XF.NORMS[0]), f"{tag}__2_filtre_harita_ciftleri.png"),
            save(XF.fig_act_grid(an, L, st, XF.NORMS[0]), f"{tag}__3_aktivasyon_grid.png"),
            save(XF.fig_kernel_detail(an, L, f_best), f"{tag}__4_kernel_detay_f{f_best}.png"),
            save(XF.fig_compare(an, L, f_best, st, XF.NORMS[0]), f"{tag}__5_karsilastirma_f{f_best}.png"),
            save(XF.fig_top_filters(an, L, st, XF.NORMS[0]), f"{tag}__6_en_guclu_filtreler.png")])
        if len(an.layers) > 1:
            L2 = an.layers[1]
            figs[k]["files"].append(save(XF.fig_pairs(an, L2, list(L2.stages)[-1], 1, XF.NORMS[0]), f"{tag}__7_katman2_ciftler.png"))
        XF._RESULTS.clear()

    rows = tr.get("models", [])
    md = ["# XAI · Katman Filtreleri (Filter Visualization) — model raporu", "",
          "> Not: Veri setinden türetilen görseller (filtre↔harita çiftleri, aktivasyon grid'leri, karşılaştırmalar) AIForge-Doc lisansı",
          "> gereği depoda tutulmaz; yalnız kernel ağırlık görselleri vardır. Tümü `python tools\\xai_filter_report.py` ile yerelde üretilir.", "",
          f"Tarih: {time.strftime('%d.%m.%Y %H:%M')} · test görüntüsü `{tr.get('image', Path(p).name)}` (TestingSet, GT kutusu merkezli 512 px natif kırpma) · "
          f"rapor görselleri CPU'da (GPU doğrulaması test T5'te) · test: **{tr.get('n_checks', '?')} kontrol / {tr.get('n_fail', '?')} hata** ({tr.get('sec', '?')} s)", "",
          "Kernel'ler checkpoint dosyalarından okunur; yeni filtre eğitilmez/üretilmez. Ara çıktılar modelin kendi forward'ından "
          "kancalarla alınır; her conv katmanı checkpoint kernel'iyle bağımsız `F.conv2d` ile yeniden hesaplanıp modelin çıktısıyla "
          "karşılaştırılır. İlk katman kuralı: yürütme sırasındaki Conv2d'ler, çıkış çözünürlüğü ≥ girdinin 1/4'ü olduğu sürece "
          "(en fazla 6); forward bu noktada durdurulur.", "",
          "## Model başına özet", "",
          "| Model | Mimari (otomatik tespit) | Ön işleme | İlk katmanlar (ad · filtre×giriş×kernel · stride · çözünürlük) | Filtre sayısı | Kernel boyutları | Aktivasyon haritası | En kötü göreli Δ | Ağırlık/dosya değişmedi | Test |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "layers" not in r:
            md.append(f"| `{r['key']}` | — | — | — | — | — | — | — | — | ❌ {r.get('error', '')} |")
            continue
        lay = "<br>".join(f"`{L['name']}` {'×'.join(map(str, L['shape']))} · s{L['stride'][0]} · 1/{L['out_stride']:g} · {'→'.join(L['stages'])}" for L in r["layers"])
        nf = " + ".join(str(L["shape"][0]) for L in r["layers"])
        ks = ", ".join(sorted({f"{L['kernel'][0]}×{L['kernel'][1]}" for L in r["layers"]}))
        md.append(f"| `{r['key']}` | {r['arch']} | {r['preprocess']} ({r['in_ch']} kanal) | {lay} | {nf} | {ks} | {r['n_maps']} | "
                  f"{r['worst_rel']:.1e} | {'✓' if r['weights_sha_unchanged'] and r['file_unchanged'] else '✗'} | {'✅' if r['ok'] else '❌'} |")
    md += ["", "## Test sonuçları", ""]
    for c in tr.get("checks", []):
        if c["name"].startswith("T3 ") and c["name"] != "T3 model keşfi":
            continue
        md.append(f"- {'✅' if c['ok'] else '❌'} **{c['name']}** — {c['detail']}")
    n3 = [c for c in tr.get("checks", []) if c["name"].startswith("T3 ") and c["name"] != "T3 model keşfi"]
    md.append(f"- {'✅' if all(c['ok'] for c in n3) else '❌'} **T3 kayıtlı modeller** — {sum(c['ok'] for c in n3)}/{len(n3)} model tüm kontrolleri geçti "
              "(checkpoint kernel'i = modül ağırlığı; bağımsız F.conv2d = model conv çıktısı; ilk conv girdisi = ön işlenmiş tensör; "
              "kanal katkıları toplamı = conv; aktivasyon = f(BN) birebir; ağırlık SHA-256 ve .pt dosyası değişmedi; adım ≤ 4)")
    if "gpu" in tr:
        g = tr["gpu"]
        md.append(f"- GPU: cuda ↔ CPU göreli fark {g['rel']:.1e}, tepe bellek {g['peak_mb']:.0f} MiB, {g['sec']:.2f} s (eğitim sürerken {g['free_gib']:.1f} GiB boş)")
    md += ["", "## Temsilî görseller", ""]
    for k, v in figs.items():
        md.append(f"### {v['label']}")
        md.append(f"Bölge: {v['region']} · karşılaştırmada filtre #{v['best']} (GT içi/dışı yanıt oranı en yüksek)")
        md.append("")
        for f in v["files"]:
            md.append(f"![{Path(f).stem}]({f})")
        md.append("")
    md += ["## Bulgular (kısa)", "",
           "- Tüm segmentasyon/sınıflandırma modelleri (U-Net, DeepLabV3+, birleşik U-Net, ResNet-18) aynı ResNet-18 gövdesini paylaşır: "
           "ilk katman `conv1` 64 filtre 7×7, stride 2 (1/2), ardından maxpool ile 1/4'te `layer1` içinde dört 64×64×3×3 conv. "
           "5 kanallı PHASE1 U-Net'in ilk katmanı RGB'ye ek FFT-HP ve Laplacian kanallarına da ağırlık verir (5×7×7).",
           "- YOLO11s gövdesi farklıdır: `model.0` 32 filtre 3×3 s2 (1/2) → `model.1` 64×32×3×3 s2 (1/4) → C3k2 içinde 1×1 ve 3×3 conv'lar; "
           "aktivasyon SiLU (Ultralytics'te tüm Conv blokları tek SiLU modülünü paylaşır, raporda 'paylaşımlı modül' olarak işaretli).",
           "- ImageNet ön-eğitimli başlatılan modellerde ilk katman kernel'leri klasik Gabor-benzeri kenar/çizgi ve renk-karşıtlık "
           "desenleri gösterir; belge görüntüsünde en güçlü yanıtlar metin kenarlarında ve sahte alan çevresindedir (GT içi/dışı oranı > 1).",
           "- ReLU sonrası bazı ilk-katman haritaları bu görüntüde tamamen 0'dır (renk filtreleri, gri belge) — bu gerçek model davranışıdır, hata değildir."]
    (OUT / "RAPOR.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "rapor.json").write_text(json.dumps(dict(models=rows, figures=figs), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("→", OUT / "RAPOR.md")


if __name__ == "__main__":
    main()
