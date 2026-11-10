"""appcore.ui — Gradio arayüzü: açılış (model ön yükleme, XAI kurulumu), sekmeler ve olay bağlantıları; `demo` nesnesi."""
from __future__ import annotations
from ._common import *  # noqa: F401,F403 — ortak içe aktarmalar
from .config import (  # noqa: F401
    DATA, DEVICE, MODELS, ROOT, ZERO_GPU, gpu
)
from .imaging import (  # noqa: F401
    project_freq_channels
)
from .ops import (  # noqa: F401
    CATEGORIES, OPS, PRESETS
)
from .dataset import (  # noqa: F401
    dataset_index, filter_index, load_dataset_sample, sample_choices
)
from .models import (  # noqa: F401
    _REG, friendly, list_resnet_ckpts, list_unet_ckpts, preload_models
)
from .metrics import (  # noqa: F401
    default_tau
)
from .scenarios import (  # noqa: F401
    DEGRADATIONS, INPUT_TYPES
)
from .simulation import (  # noqa: F401
    simulate
)
from .batch import (  # noqa: F401
    BATCH_MODES, batch_test
)
from .handlers import (  # noqa: F401
    _draft_dd, apply_preset, apply_step, classical_features, compare_unet, draft_delete, draft_list, draft_load,
    draft_save, on_sample, on_upload, op_choices, param_updates, preview, reset, results_markdown, run_green,
    run_resnet, run_unet, sim_filter, sim_preview, sim_random, sim_to_tabs, steps_md, undo
)
from .xai_bridge import (  # noqa: F401
    XAI, XAI_METHODS, XAI_OK, XAI_REFS, XAI_SOURCES, XAI_TARGETS, _xai_show_outputs, xai_compare, xai_predict,
    xai_run, xai_setup, xai_unsupported_md
)
from .theme import (  # noqa: F401
    H_LG, H_MD, H_SM, H_XL, LEGEND_MD, TIP_JS
)

# ================================================================== XAI · KATMAN FİLTRELERİ (sekme 8 › Katman Filtreleri)
# SHAP'tan bağımsız ikinci XAI özelliği: app/xai_filters.py — ilk conv katmanlarının checkpoint'teki gerçek kernel'leri
# + modelin kendi forward'ından alınan aktivasyon haritaları (bağımsız F.conv2d ile doğrulanır). Ayrıntı: docs/XAI_FILTERS.md
try:
    import xai_filters as XFILT
    XFILT_OK, _XFILT_ERR = True, ""
except Exception as _e:  # paket yoksa diğer sekmeler etkilenmez
    XFILT, XFILT_OK, _XFILT_ERR = None, False, str(_e)

preload_models()
xai_setup()
_IDX = dataset_index()
_UNETS = [n for n in list_unet_ckpts() if n in _REG or not _REG]
_RESNETS = [n for n in list_resnet_ckpts() if n in _REG or not _REG]
_DEF_UNETS = [n for n in ("PHASE2/unet_bbox_s42", "PHASE2/unet_pixel_eps8_morph_s42", "PHASE2/unet_glyph_v2_s42",
                             "YOLO/aiforge-tiles-yolo11s-img1024-conserv-s42") if n in _UNETS] or _UNETS[:1]
_DEF_RESNET = [n for n in ("PHASE1/resnet18_ft_best",) if n in _RESNETS]
_SOURCES = ["hepsi"] + sorted(x for x in _IDX["source_dataset"].unique() if x)
_TOOLS = ["hepsi"] + sorted(x for x in _IDX["tool"].unique() if x)
_AREAS = ["hepsi"] + [a for a in ("<1%", "1-5%", "5-20%", ">20%") if a in set(_IDX["area_bin"])]


SIM_SUMMARY_COLS = ["Model", "Karar", "Doğru", "Görüntü skoru", "F1 bbox-GT", "IoU bbox-GT", "F1 piksel-GT", "Süre s"]


def sim_summary(df):
    """Simülasyon tablosunun özet görünümü (aynı satırlar, yalnız temel sütunlar)."""
    if df is None or not len(df):
        return None
    return df[[c for c in SIM_SUMMARY_COLS if c in df.columns]]


def status_line() -> str:
    """Başlık altındaki durum satırı: uygulamanın gerçekten çalışır durumda olduğunu tek bakışta gösterir."""
    dev = {"cuda": "GPU (CUDA)", "cpu": "CPU"}.get(str(DEVICE), str(DEVICE)) + (" · ZeroGPU" if ZERO_GPU else "")
    return (f"**Durum** · {len(_REG)} model yüklü ({len(_UNETS)} lokalizasyon · {len(_RESNETS)} sınıflandırma) · cihaz: {dev} · "
            f"veri seti: {len(_IDX):,} örnek".replace(",", ".") + f" · veri kökü: `{DATA.parent.parent if DATA.exists() else 'bulunamadı'}`")


with gr.Blocks(title="AIC515 · AIForge-Doc Sahtecilik Lokalizasyonu") as demo:
    gr.Markdown("# AIC515 Görüntü İşleme · AIForge-Doc — Belge Sahteciliği Tespiti ve Lokalizasyonu")
    gr.Markdown(status_line(), elem_classes="status-md")

    st_orig = gr.State(None); st_cur = gr.State(None); st_mask = gr.State(None); st_auth = gr.State(None); st_steps = gr.State([])
    st_sim = gr.State(None)

    # ---------------------------------------------------------------- 0 · SİMÜLASYON
    with gr.Tab("0 · Simülasyon"):
        with gr.Row():
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("#### 1) Örnek seçimi")
                with gr.Row():
                    f_split = gr.Dropdown(["hepsi", "eğitim (training)", "test (testing)"], value="test (testing)", label="Bölüm", elem_id="tip-f_split")
                    f_source = gr.Dropdown(_SOURCES, value="hepsi", label="Kaynak veri seti", elem_id="tip-f_source")
                with gr.Row():
                    f_tool = gr.Dropdown(_TOOLS, value="hepsi", label="Inpainting üreticisi", elem_id="tip-f_tool")
                    f_area = gr.Dropdown(_AREAS, value="hepsi", label="Kurcalanan alan oranı", elem_id="tip-f_area")
                with gr.Row():
                    f_marker = gr.Dropdown(["hepsi", "var", "yok"], value="hepsi", label="Yeşil işaretçi kalıntısı", elem_id="tip-f_marker")
                    f_mp = gr.Slider(0.1, 40, value=12, step=0.1, label="En fazla megapiksel", elem_id="tip-f_mp")
                f_pair = gr.Checkbox(False, label="Yalnız authentic eşi olanlar (piksel-GT ve işaretçi senaryoları için gerekli)", elem_id="tip-f_pair")
                with gr.Row():
                    b_filter = gr.Button("Filtrele", elem_id="tip-b_filter")
                    b_rand = gr.Button("🎲 Rastgele seç", variant="secondary", elem_id="tip-b_rand")
                f_count = gr.Markdown(" ", elem_classes="small-md")
                dd_key = gr.Dropdown(choices=sample_choices(filter_index("test (testing)", max_mp=12)), label="Örnek (bölüm:kimlik)", filterable=True, allow_custom_value=True, elem_id="tip-dd_key")
                gr.Markdown("#### 2) Senaryo")
                rd_input = gr.Radio(INPUT_TYPES, value=INPUT_TYPES[0], label="Girdi senaryosu", elem_id="tip-rd_input")
                dd_deg = gr.Dropdown(DEGRADATIONS, value=DEGRADATIONS[0], label="Test-zamanı bozulma", elem_id="tip-dd_deg")
                ck_pipe = gr.Checkbox(False, label="Ön işleme sekmesindeki pipeline'ı da uygula", elem_id="tip-ck_pipe")
                gr.Markdown("#### 3) Modeller")
                cg_unet = gr.Dropdown(choices=[(friendly(n), n) for n in _UNETS], value=_DEF_UNETS, multiselect=True, filterable=True, label=f"Lokalizasyon modelleri (U-Net · DeepLabV3+ · YOLO11) — {len(_UNETS)} seçenek", elem_id="tip-cg_unet")
                cg_res = gr.Dropdown(choices=[(friendly(n), n) for n in _RESNETS], value=_DEF_RESNET, multiselect=True, label="ResNet-18 (patch sınıflandırma)", elem_id="tip-cg_res")
                sl_stride_sim = gr.Slider(56, 224, value=112, step=8, label="ResNet kayan pencere adımı (px)", elem_id="tip-sl_stride_sim")
                b_sim = gr.Button("▶ Simülasyonu başlat", variant="primary", size="lg", elem_id="tip-b_sim")
                b_to_tabs = gr.Button("Bu örneği ön işleme / model sekmelerine aktar", elem_id="tip-b_to_tabs")
            with gr.Column(scale=2):
                with gr.Row():
                    pv_img = gr.Image(label="Seçilen örnek", type="numpy", height=H_MD, interactive=False)
                    pv_gt = gr.Image(label="Resmî GT (mavi kutu)", type="numpy", height=H_MD, interactive=False)
                pv_info = gr.Markdown("Bir örnek seçin ya da 🎲 ile rastgele seçin.")
                sim_verdict = gr.Markdown(" ", elem_id="verdict")
                gr.Markdown(LEGEND_MD, elem_classes="legend-md")
                with gr.Tabs():
                    with gr.Tab("Sonuç tablosu"):
                        sim_sum = gr.Dataframe(label="Özet: karar ve temel ölçütler", wrap=True, interactive=False)
                        with gr.Accordion("Tüm ölçütler (18 sütun)", open=False):
                            sim_df = gr.Dataframe(label="Model bazında sonuçlar", wrap=True, interactive=False)
                        sim_log = gr.Markdown(" ")
                    with gr.Tab("Yakın plan"):
                        sim_zoom = gr.Image(label="Yakın plan: alan kutusu çevresi (girdi · GT · model tahminleri)", type="numpy", height=H_XL, interactive=False)
                    with gr.Tab("Adım adım çıktılar"):
                        sim_gal = gr.Gallery(label="Adım adım çıktılar", columns=3, height=H_XL, object_fit="contain", preview=False, format="jpeg")
                    with gr.Tab("Rapor dosyaları"):
                        sim_files = gr.File(label="Simülasyon raporu (MD · JSON · CSV · PNG)", file_count="multiple")

    # ---------------------------------------------------------------- 1 · TOPLU TEST
    with gr.Tab("1 · Toplu test"):
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                with gr.Row():
                    t_split = gr.Dropdown(["hepsi", "eğitim (training)", "test (testing)"], value="test (testing)", label="Bölüm", elem_id="tip-t_split")
                    t_source = gr.Dropdown(_SOURCES, value="hepsi", label="Kaynak", elem_id="tip-t_source")
                with gr.Row():
                    t_tool = gr.Dropdown(_TOOLS, value="hepsi", label="Üretici", elem_id="tip-t_tool")
                    t_area = gr.Dropdown(_AREAS, value="hepsi", label="Alan", elem_id="tip-t_area")
                with gr.Row():
                    t_marker = gr.Dropdown(["hepsi", "var", "yok"], value="hepsi", label="İşaretçi", elem_id="tip-t_marker")
                    t_mp = gr.Slider(0.1, 40, value=6, step=0.1, label="En fazla MP", elem_id="tip-t_mp")
                with gr.Row():
                    t_n = gr.Slider(1, 100, value=16, step=1, label="Örnek sayısı N", elem_id="tip-t_n")
                    t_seed = gr.Number(42, label="Tohum", precision=0, elem_id="tip-t_seed")
                t_mode = gr.Radio(BATCH_MODES, value=BATCH_MODES[1], label="Mod", elem_id="tip-t_mode")
                t_deg = gr.Dropdown(DEGRADATIONS, value=DEGRADATIONS[0], label="Bozulma", elem_id="tip-t_deg")
                t_unet = gr.Dropdown(choices=[(friendly(n), n) for n in _UNETS], value=_DEF_UNETS[:2], multiselect=True, filterable=True, label="Lokalizasyon (U-Net · DeepLabV3+ · YOLO11)", elem_id="tip-t_unet")
                t_res = gr.Dropdown(choices=[(friendly(n), n) for n in _RESNETS], value=[], multiselect=True, label="ResNet-18", elem_id="tip-t_res")
                b_batch = gr.Button("▶ Toplu testi başlat", variant="primary", size="lg", elem_id="tip-b_batch")
            with gr.Column(scale=2):
                t_prog = gr.Markdown(" ")
                t_agg = gr.Dataframe(label="Model bazında özet", wrap=True, interactive=False)
                t_fig = gr.Plot(label="Özet grafik")
                t_per = gr.Dataframe(label="Örnek bazında sonuçlar", wrap=True, interactive=False)
                t_files = gr.File(label="Toplu test çıktıları", file_count="multiple")

    # ---------------------------------------------------------------- 2 · GİRDİ
    with gr.Tab("2 · Girdi"):
        with gr.Row():
            with gr.Column(scale=1):
                in_img = gr.Image(label="Sahte (forged) görüntü — PNG/JPG", type="numpy", height=H_MD, elem_id="tip-in_img")
                in_mask = gr.Image(label="GT maskesi (isteğe bağlı, 0/255)", type="numpy", height=H_SM, image_mode="L", elem_id="tip-in_mask")
                in_auth = gr.Image(label="Authentic eş (isteğe bağlı; piksel-değişim GT için)", type="numpy", height=H_SM, elem_id="tip-in_auth")
                btn_load = gr.Button("Yüklenenleri kullan", variant="primary", elem_id="tip-btn_load")
                dd_sample = gr.Dropdown(choices=sample_choices(), label="örnek (bölüm:kimlik)", filterable=True, allow_custom_value=True, elem_id="tip-dd_sample")
                btn_sample = gr.Button("Örneği yükle", elem_id="tip-btn_sample")
                with gr.Accordion("💾 Taslaklar (sonra kullanmak için kaydet)", open=True):
                    tx_draft = gr.Textbox(label="Taslak adı", placeholder="ör. cord fişi · indirim alanı", elem_id="tip-tx_draft")
                    btn_dsave = gr.Button("💾 Taslak olarak kaydet", variant="secondary", elem_id="tip-btn_dsave")
                    dd_draft = gr.Dropdown(choices=draft_list(), label="Kayıtlı taslaklar", filterable=True, elem_id="tip-dd_draft")
                    with gr.Row():
                        btn_dload = gr.Button("📂 Taslağı yükle", variant="primary", elem_id="tip-btn_dload")
                        btn_drefresh = gr.Button("↻", scale=0, min_width=48)
                        btn_ddel = gr.Button("🗑 Sil", variant="stop", scale=0, min_width=72, elem_id="tip-btn_ddel")
                    draft_md = gr.Markdown(" ", elem_classes="small-md")
            with gr.Column(scale=1):
                info_md = gr.Markdown("Görüntü yükleyin ya da örnek seçin.")
                view_orig = gr.Image(label="Orijinal", type="numpy", height=H_LG, interactive=False)
                with gr.Row():
                    view_gt = gr.Image(label="GT maskesi", type="numpy", height=H_SM, interactive=False)
                    view_auth = gr.Image(label="Authentic eş", type="numpy", height=H_SM, interactive=False)

    # ---------------------------------------------------------------- 3 · ÖN İŞLEME
    with gr.Tab("3 · Ön işleme (aşamalı pipeline)"):
        with gr.Row():
            with gr.Column(scale=1):
                dd_cat = gr.Dropdown(choices=CATEGORIES, value=CATEGORIES[0], label="Kategori (ders haftası)", elem_id="tip-dd_cat")
                first_ops = list(OPS[CATEGORIES[0]].keys())
                dd_op = gr.Dropdown(choices=first_ops, value=first_ops[0], label="İşlem", elem_id="tip-dd_op")
                sl1 = gr.Slider(visible=False, elem_id="tip-sl1"); sl2 = gr.Slider(visible=False, elem_id="tip-sl2"); sl3 = gr.Slider(visible=False, elem_id="tip-sl3")
                dd_mode = gr.Dropdown(choices=[], visible=False, label="mod", elem_id="tip-dd_mode")
                op_doc = gr.Markdown(" ", elem_classes="small-md")
                with gr.Row():
                    btn_prev = gr.Button("Önizle", elem_id="tip-btn_prev"); btn_apply = gr.Button("Pipeline'a ekle (uygula)", variant="primary", elem_id="tip-btn_apply")
                with gr.Row():
                    btn_undo = gr.Button("Geri al", elem_id="tip-btn_undo"); btn_reset = gr.Button("Sıfırla", elem_id="tip-btn_reset")
                dd_preset = gr.Dropdown(choices=list(PRESETS), value="— seçiniz —", label="Hazır pipeline (preset)", elem_id="tip-dd_preset")
                btn_preset = gr.Button("Preseti uygula", elem_id="tip-btn_preset")
                steps_box = gr.Markdown(steps_md([]))
            with gr.Column(scale=2):
                with gr.Row():
                    img_before = gr.Image(label="Önce (mevcut pipeline çıktısı)", type="numpy", height=H_LG, interactive=False)
                    img_after = gr.Image(label="Sonra", type="numpy", height=H_LG, interactive=False)
                hist_plot = gr.Plot(label="Histogram")
                status = gr.Markdown(" ")
                with gr.Accordion("Klasik öznitelikler (56; LBP · GLCM · DCT · Laplacian · FFT) — 256 px, alan kutusu merkezli", open=False):
                    btn_feat = gr.Button("Öznitelikleri hesapla")
                    feat_df = gr.Dataframe(label="öznitelik vektörü", wrap=True)

    # ---------------------------------------------------------------- 4 · U-NET
    with gr.Tab("4 · U-Net lokalizasyonu"):
        with gr.Row():
            with gr.Column(scale=1):
                dd_unet = gr.Dropdown(choices=[(friendly(n), n) for n in _UNETS], value=_DEF_UNETS[0] if _DEF_UNETS else None, label="Model (U-Net · DeepLabV3+ · YOLO11)", elem_id="tip-dd_unet")
                rd_src = gr.Radio(["Orijinal görüntü", "Pipeline çıktısı"], value="Orijinal görüntü", label="Model girdisi", elem_id="tip-rd_src")
                rd_gt = gr.Radio(["resmî bbox maskesi", "piksel-değişim (authentic'ten türet, eps8_morph)"], value="resmî bbox maskesi", label="Değerlendirme GT'si", elem_id="tip-rd_gt")
                sl_thr = gr.Slider(0.05, 0.95, value=default_tau(_DEF_UNETS[0], "bbox") if _DEF_UNETS else 0.5, step=0.05, label="eşik τ", elem_id="tip-sl_thr")
                ck_otsu = gr.Checkbox(False, label="Eşik yerine Otsu (olasılık haritası üzerinde)", elem_id="tip-ck_otsu")
                sl_morph = gr.Slider(0, 15, value=0, step=1, label="Son-işleme morfoloji (açma+kapama) çekirdeği; 0 = kapalı", elem_id="tip-sl_morph")
                btn_unet = gr.Button("U-Net çalıştır", variant="primary", elem_id="tip-btn_unet")
                btn_cmp = gr.Button("Karşılaştır: orijinal vs pipeline çıktısı", elem_id="tip-btn_cmp")
                gr.Markdown(LEGEND_MD, elem_classes="legend-md")
            with gr.Column(scale=2):
                with gr.Row():
                    unet_heat = gr.Image(label="Olasılık haritası", type="numpy", height=H_MD, interactive=False)
                    unet_ov = gr.Image(label="Bindirme (kırmızı tahmin · mavi GT · yeşil tahmin kutusu)", type="numpy", height=H_MD, interactive=False)
                with gr.Row():
                    unet_bin = gr.Image(label="İkili maske (eşik sonrası)", type="numpy", height=H_MD, interactive=False)
                    with gr.Column():
                        unet_df = gr.Dataframe(label="Ölçütler", wrap=True)
                        unet_note = gr.Markdown(" ")
                with gr.Accordion("Karşılaştırma sonucu (orijinal vs pipeline)", open=False):
                    with gr.Row():
                        cmp_a = gr.Image(label="Orijinal girdi → tahmin", type="numpy", height=H_MD, interactive=False)
                        cmp_b = gr.Image(label="Pipeline çıktısı → tahmin", type="numpy", height=H_MD, interactive=False)
                    cmp_df = gr.Dataframe(label="Ölçüt karşılaştırması", wrap=True)
                    cmp_note = gr.Markdown(" ")
        st_dummy = gr.State(None)

    # ---------------------------------------------------------------- 5 · RESNET
    with gr.Tab("5 · ResNet-18 patch sınıflandırma"):
        with gr.Row():
            with gr.Column(scale=1):
                dd_rn = gr.Dropdown(choices=[(friendly(n), n) for n in _RESNETS], value=_DEF_RESNET[0] if _DEF_RESNET else (_RESNETS[0] if _RESNETS else None), label="ResNet-18 checkpoint", elem_id="tip-dd_rn")
                rd_src2 = gr.Radio(["Orijinal görüntü", "Pipeline çıktısı"], value="Orijinal görüntü", label="Model girdisi", elem_id="tip-rd_src2")
                sl_stride = gr.Slider(32, 224, value=112, step=16, label="kayan pencere adımı (patch 224, natif çözünürlük)", elem_id="tip-sl_stride")
                btn_rn = gr.Button("ResNet-18 çalıştır", variant="primary", elem_id="tip-btn_rn")
                gr.Markdown(LEGEND_MD, elem_classes="legend-md")
            with gr.Column(scale=2):
                rn_ov = gr.Image(label="Patch skor ısı haritası (mavi: GT alan kutusu)", type="numpy", height=H_LG, interactive=False)
                rn_df = gr.Dataframe(label="Ölçütler", wrap=True)

    # ---------------------------------------------------------------- 6 · YEŞİL İŞARETÇİ
    with gr.Tab("6 · Yeşil işaretçi sondası"):
        gr.Markdown("Veri seti üretim hattının bıraktığı **yeşil işaretçi kalıntısı** (PHASE 2 Ek-C): G − max(R,B) > 50 ve G > 110. "
                    "Sahtelerin %75,4'ünde bbox iç kenarında görülür ve tüm modellerin kullandığı bir kısayoldur. "
                    "Simülasyondaki 'işaretçi onarılmış' ve 'yalnız işaretçi' senaryoları bu etkiyi ölçer.")
        btn_green = gr.Button("Mevcut görüntüde işaretçiyi ara", variant="primary", elem_id="tip-btn_green")
        with gr.Row():
            with gr.Column(scale=2):
                green_ov = gr.Image(label="İşaretçi pikselleri (magenta)", type="numpy", height=H_LG, interactive=False)
            with gr.Column(scale=1):
                green_md = gr.Markdown(" ")

    # ---------------------------------------------------------------- 7 · SONUÇLAR
    with gr.Tab("7 · Sonuçlar ve model kartı"):
        gr.Markdown(results_markdown())
        gr.Markdown(
            "### Notlar\n"
            "- Modeller **ham PNG, natif çözünürlük, ImageNet normalizasyonu** ile eğitildi; ön işleme ve bozulmalar test-zamanı müdahaleleridir.\n"
            "- Tam görüntü çıkarımı 512 px pencere / 448 px adım / örtüşme ortalaması; uzun kenar > 4096 px ise 4096'ya küçültülür.\n"
            "- Metrikler boru hattıyla aynı tanımla hesaplanır: 8-bit nicemlenmiş olasılık, 256 kutulu histogram, tüm pikseller (alt-örnekleme yok).\n"
            "- Görüntü-düzeyi skor = olasılık haritasının en yüksek %1 piksel ortalaması; eşik val'de Youden-J ile seçildi.\n"
            "- Veri seti: AIForge-Doc v1 (Scam.AI, CC BY-NC-SA 4.0, kapılı); harici test: AIForge-Doc v2 (GPT-Image-2, aynı lisans), yalnız sonuç tabloları.")

    # ---------------------------------------------------------------- 8 · AÇIKLANABİLİRLİK (SHAP açıklamaları + katman filtreleri)
    with gr.Tab("8 · Açıklanabilirlik (XAI)"):
        gr.Markdown("**Açıklanabilirlik (XAI)** — birbirinden bağımsız iki özellik: **SHAP Açıklamaları** modelin tahmin "
                    "kararına görüntü bölgelerinin katkısını açıklar (`xai/`); **Katman Filtreleri** eğitilmiş modelin ilk "
                    "konvolüsyon katmanlarındaki gerçek filtre ağırlıklarını ve bunların görüntüye uygulanmasıyla oluşan aktivasyon "
                    "haritalarını gösterir (`xai_filters.py`; filtre eğitilmez/üretilmez, ağırlıklar ve mimari değişmez).", elem_classes="small-md")
        with gr.Tabs():
            # ------------------------------------------------ 8a · SHAP Açıklamaları
            with gr.Tab("SHAP Açıklamaları"):
                st_xpred = gr.State(None); st_xview = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=1, min_width=340):
                        gr.Markdown("#### 1) Görüntü ve modeller")
                        x_src = gr.Radio(XAI_SOURCES, value=XAI_SOURCES[0], label="Görüntü kaynağı", elem_id="tip-x_src")
                        with gr.Group(visible=False) as x_up_grp:
                            x_up_img = gr.Image(label="Görüntü", type="numpy", height=H_SM, elem_id="tip-x_up_img")
                            with gr.Row():
                                x_up_auth = gr.Image(label="Authentic eş (isteğe bağlı)", type="numpy", height=H_SM, elem_id="tip-x_up_auth")
                                x_up_gt = gr.Image(label="GT maskesi (isteğe bağlı)", type="numpy", height=H_SM, image_mode="L", elem_id="tip-x_up_gt")
                        x_unet = gr.Dropdown(choices=[(friendly(n), n) for n in _UNETS], value=[n for n in _DEF_UNETS if not n.startswith("YOLO/")], multiselect=True, filterable=True, label="U-Net / DeepLabV3+ / YOLO11 (YOLO: yalnız tam görüntü hedefleri)", elem_id="tip-x_unet")
                        x_res = gr.Dropdown(choices=[(friendly(n), n) for n in _RESNETS], value=_DEF_RESNET, multiselect=True, label="ResNet-18", elem_id="tip-x_res")
                        gr.Markdown(xai_unsupported_md(), elem_classes="small-md")
                        b_xpred = gr.Button("① Tahmin yap", variant="primary", elem_id="tip-b_xpred")
                        gr.Markdown("#### 2) SHAP")
                        x_target = gr.Radio(XAI_TARGETS, value=XAI_TARGETS[0], label="Açıklanacak hedef", elem_id="tip-x_target")
                        x_k = gr.Slider(1, 3, value=3, step=1, label="Tespit sayısı k", elem_id="tip-x_k")
                        x_method = gr.Radio(XAI_METHODS, value=XAI_METHODS[0], label="Yöntem", elem_id="tip-x_method")
                        x_ref = gr.Dropdown(XAI_REFS, value=XAI_REFS[0], label="Referans (arka plan)", elem_id="tip-x_ref")
                        x_evals = gr.Slider(200, 3000, value=800, step=100, label="max_evals (Partition bütçesi)", elem_id="tip-x_evals")
                        b_xshap = gr.Button("② SHAP analizini başlat", variant="primary", elem_id="tip-b_xshap")
                        b_xcmp = gr.Button("③ Modelleri karşılaştır", elem_id="tip-b_xcmp")
                    with gr.Column(scale=2):
                        x_status = gr.Markdown("① ile tahmin yapın, ② ile seçilen hedefleri SHAP ile açıklayın, ③ ile modelleri aynı pencerede karşılaştırın.")
                        x_pred_df = gr.Dataframe(label="Tahminler (uygulamanın kendi çıkarımı)", wrap=True, interactive=False)
                        x_res_df = gr.Dataframe(label="SHAP sonuçları", wrap=True, interactive=False)
                        x_loc = gr.Image(label="Açıklanan alanların konumu (tam görüntü hedefinde çerçeve = görüntünün tamamı)", type="numpy", height=H_MD, interactive=False)
                        x_show = gr.Dropdown(choices=[], label="Gösterilecek açıklama", elem_id="tip-x_show")
                        x_sum = gr.Markdown(" ")
                        gr.Markdown("Görüntüler en-boy oranı korunarak gösterilir · **fare tekerleği**: imleç çevresinde yakınlaştır (32×'e kadar) · "
                                    "**sürükle**: kaydır · **çift tık**: sıfırla · sağ üst düğmeler: tam ekran / indir. Natif çözünürlüklü "
                                    "PNG/NPY ve 300 dpi PNG + PDF yayın şekilleri aşağıdaki dışa aktarımdadır.", elem_classes="small-md")
                        x_slider = gr.ImageSlider(label="Yan yana karşılaştırma: orijinal ↔ SHAP bindirmesi (kaydırıcıyı sürükleyin)", type="numpy", height=H_XL)
                        with gr.Row():
                            x_p0 = gr.Image(label="Orijinal — tam görüntü ya da açıklanan pencere (mavi: GT · magenta: işaretçi · yeşil: hedef)", type="numpy", height=H_XL, interactive=False, elem_classes=["xai-zoom"])
                            x_p2 = gr.Image(label="Bindirme (aynı koordinat sistemi)", type="numpy", height=H_XL, interactive=False, elem_classes=["xai-zoom"])
                        with gr.Row():
                            x_p1 = gr.Image(label="SHAP ısı haritası (kırmızı: SAHTE yönü · mavi: GERÇEK yönü)", type="numpy", height=H_LG, interactive=False, elem_classes=["xai-zoom"])
                            x_p5 = gr.Image(label="Sınıf panelleri: SAHTE | GERÇEK (= −SAHTE)", type="numpy", height=H_LG, interactive=False, elem_classes=["xai-zoom"])
                        with gr.Row():
                            x_p3 = gr.Image(label="Pozitif katkı (SAHTE'yi artıran bölgeler)", type="numpy", height=H_LG, interactive=False, elem_classes=["xai-zoom"])
                            x_p4 = gr.Image(label="Negatif katkı (GERÇEK'e iten bölgeler)", type="numpy", height=H_LG, interactive=False, elem_classes=["xai-zoom"])
                        x_dist = gr.Image(label="Açıklanan hedef sınıfına (SAHTE) katkı dağılımı: histogram · birikimli kütle · toplamlar (GT içi/dışı)", type="numpy", height=H_MD, interactive=False)
                        x_mass = gr.Image(label="Pozitif SHAP kütlesinin bölgelere dağılımı (GT kutusu / kenar halkası / işaretçi / geri kalan)", type="numpy", interactive=False)
                        with gr.Accordion("Tüm paneller (galeri)", open=False):
                            x_gal = gr.Gallery(columns=3, height=H_LG, object_fit="contain", preview=False, format="png")
                        with gr.Accordion("Model karşılaştırması (③)", open=True):
                            x_cmp_md = gr.Markdown(" ")
                            x_cmp_img = gr.Image(label="IoU · Spearman · ortak/özgü bölge", type="numpy", interactive=False)
                            x_cmp_df = gr.Dataframe(label="Karşılaştırılan açıklamalar", wrap=True, interactive=False)
                        x_files = gr.File(label="Dışa aktarım (PNG paneller · JSON · NPZ · MD rapor)", file_count="multiple")

            # ------------------------------------------------ 8b · Katman Filtreleri
            with gr.Tab("Katman Filtreleri"):
                if XFILT_OK and TORCH_OK:
                    def _xf_sample(key):
                        try:
                            return load_dataset_sample(key)
                        except Exception as e:
                            raise gr.Error(f"Örnek yüklenemedi: {e}")
                    XFILT.build_tab(gr, models_dir=MODELS, yolo_runs_dir=ROOT / "phase3_yolo" / "runs", gpu=gpu,
                                    freq_fn=project_freq_channels, load_sample=_xf_sample,
                                    sample_choices=sample_choices(filter_index("test (testing)", max_mp=12)),
                                    image_states=(st_orig, st_mask), default_key="PHASE2/unet_bbox_s42", store_fp16=True if ZERO_GPU else None)
                else:
                    gr.Markdown(f"⚠️ Katman filtreleri kullanılamıyor: `{_XFILT_ERR or _TORCH_ERR}`")


    # ---------------------------------------------------------------- olaylar
    filt_in = [f_split, f_source, f_tool, f_area, f_pair, f_marker, f_mp]
    b_filter.click(sim_filter, filt_in, [dd_key, f_count])
    b_rand.click(sim_random, filt_in, [dd_key, f_count])
    dd_key.change(sim_preview, [dd_key], [pv_img, pv_gt, pv_info])
    b_sim.click(simulate, [dd_key, rd_input, dd_deg, ck_pipe, st_steps, cg_unet, cg_res, sl_stride_sim],
                [sim_log, sim_gal, sim_df, sim_verdict, sim_zoom, sim_files, st_sim]).then(sim_summary, [sim_df], [sim_sum])
    b_to_tabs.click(sim_to_tabs, [st_sim], [st_orig, st_cur, st_mask, st_auth, st_steps, info_md, view_orig, img_before, view_gt, view_auth, steps_box])

    b_batch.click(batch_test, [t_split, t_source, t_tool, t_area, t_marker, t_n, t_seed, t_mode, t_deg, t_unet, t_res, t_mp],
                  [t_prog, t_per, t_agg, t_fig, t_files])

    btn_load.click(on_upload, [in_img, in_mask, in_auth], [st_orig, st_cur, st_mask, st_auth, st_steps, info_md, view_orig, img_before, view_gt, view_auth]).then(
        lambda: steps_md([]), None, steps_box)
    btn_sample.click(on_sample, [dd_sample], [st_orig, st_cur, st_mask, st_auth, st_steps, info_md, view_orig, img_before, view_gt, view_auth]).then(
        lambda: steps_md([]), None, steps_box)
    btn_dsave.click(draft_save, [tx_draft, st_orig, st_mask, st_auth, st_steps, in_img, in_mask, in_auth, dd_sample], [dd_draft, draft_md])
    btn_dload.click(draft_load, [dd_draft], [st_orig, st_cur, st_mask, st_auth, st_steps, info_md, view_orig, img_before, view_gt, view_auth,
                                             steps_box, in_img, in_mask, in_auth, draft_md]).then(lambda c: c, [st_cur], [img_after])
    btn_ddel.click(draft_delete, [dd_draft], [dd_draft, draft_md])
    btn_drefresh.click(lambda v: _draft_dd(v), [dd_draft], [dd_draft])
    demo.load(lambda: _draft_dd(), None, [dd_draft])

    dd_cat.change(op_choices, [dd_cat], [dd_op, sl1, sl2, sl3, dd_mode, op_doc])
    dd_op.change(param_updates, [dd_cat, dd_op], [sl1, sl2, sl3, dd_mode, op_doc])
    demo.load(param_updates, [dd_cat, dd_op], [sl1, sl2, sl3, dd_mode, op_doc])

    btn_prev.click(preview, [st_cur, dd_cat, dd_op, sl1, sl2, sl3, dd_mode, st_auth], [img_after, hist_plot, status])
    btn_apply.click(apply_step, [st_cur, st_orig, st_steps, dd_cat, dd_op, sl1, sl2, sl3, dd_mode, st_auth],
                    [st_cur, img_after, st_steps, steps_box, hist_plot, status]).then(lambda c: c, [st_cur], [img_before])
    btn_undo.click(undo, [st_orig, st_steps, st_auth], [st_cur, img_before, st_steps, steps_box, status]).then(lambda c: c, [st_cur], [img_after])
    btn_reset.click(reset, [st_orig], [st_cur, img_before, st_steps, steps_box, status]).then(lambda c: c, [st_cur], [img_after])
    btn_preset.click(apply_preset, [st_orig, dd_preset, st_auth], [st_cur, img_after, st_steps, steps_box, hist_plot, status]).then(lambda c: c, [st_cur], [img_before])
    btn_feat.click(classical_features, [st_cur, st_mask], [feat_df])

    btn_unet.click(run_unet, [st_orig, st_cur, st_mask, st_auth, dd_unet, rd_src, sl_thr, ck_otsu, sl_morph, rd_gt],
                   [unet_heat, unet_ov, unet_bin, unet_df, unet_note, st_dummy])
    btn_cmp.click(compare_unet, [st_orig, st_cur, st_mask, st_auth, dd_unet, sl_thr, ck_otsu, sl_morph, rd_gt, st_steps],
                  [cmp_a, cmp_b, cmp_df, cmp_note])
    dd_unet.change(lambda m, g: default_tau(m, "bbox" if g.startswith("resmî") else "pixel_eps8_morph") if m else 0.5, [dd_unet, rd_gt], [sl_thr])
    rd_gt.change(lambda m, g: default_tau(m, "bbox" if g.startswith("resmî") else "pixel_eps8_morph") if m else 0.5, [dd_unet, rd_gt], [sl_thr])

    btn_rn.click(run_resnet, [st_orig, st_cur, st_mask, dd_rn, rd_src2, sl_stride], [rn_ov, rn_df])
    btn_green.click(run_green, [st_cur, st_mask, st_auth], [green_ov, green_md])


    # ---------------------------------------------------------------- 8 · Açıklanabilirlik (XAI) › SHAP olayları
    x_src.change(lambda v: gr.update(visible=v == XAI_SOURCES[3]), [x_src], [x_up_grp])
    b_xpred.click(xai_predict, [x_src, x_unet, x_res, st_orig, st_cur, st_mask, st_auth, st_sim, x_up_img, x_up_auth, x_up_gt],
                  [x_pred_df, x_status, st_xpred])
    _x_panels = [x_sum, x_p0, x_p1, x_p2, x_p3, x_p4, x_p5, x_slider, x_dist]
    b_xshap.click(xai_run, [st_xpred, x_target, x_k, x_method, x_ref, x_evals],
                  [x_status, x_res_df, x_loc, x_show, st_xview, x_mass, x_gal, x_files] + _x_panels)
    x_show.change(_xai_show_outputs, [st_xview, x_show], _x_panels)
    b_xcmp.click(xai_compare, [st_xpred, x_method, x_ref, x_evals], [x_cmp_md, x_cmp_img, x_cmp_df, x_files])

    demo.load(None, None, None, js=TIP_JS)  # sol panel açıklamaları
    if XAI_OK:
        demo.load(None, None, None, js=XAI.viewer.ZOOM_JS)  # XAI görüntüleri: tekerlek yakınlaştırma · sürükleme · çift tık sıfırlama

demo.queue(default_concurrency_limit=2)
