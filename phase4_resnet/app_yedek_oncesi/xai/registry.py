"""Model kaydı: uygulamadaki her checkpoint için mimari / görev / ön işleme / çıktı / karar / SHAP yöntemi bilgisi.

app.py fonksiyonları `Deps` ile enjekte edilir; bu modül app.py'yi içe aktarmaz (döngüsel import yok).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Deps:
    """app.py'den enjekte edilen bağımlılıklar (tek doğruluk kaynağı uygulamanın kendi fonksiyonlarıdır)."""
    get_model: Callable            # ad → (model, in_ch, tür)
    norm_chw: Callable             # uint8 HWC → float32 CHW (ImageNet)
    project_freq_channels: Callable  # uint8 HWC → (2,H,W) frekans kanalları
    windows: Callable              # (_windows) H, W, win, stride → [(y,x)]
    autocast: Callable             # () → bağlam yöneticisi (cuda'da bf16)
    device: Any
    unet_window_forward: Callable  # (model, in_ch, xb) → (N,H,W) olasılık (pencere başına frekans standardizasyonu dahil)
    unet_prepare: Callable         # (rgb_u8, win, max_side) → (img, (h, w), ölçek)
    image_score: Callable          # olasılık haritası → top-%1 ortalaması (8-bit nicemleme ile)
    q_u8: Callable
    postprocess: Callable          # (prob, thr, morph_k, use_otsu) → (ikili, eşik)
    default_tau: Callable          # (model_adı, gt_türü) → τ*
    image_threshold: Callable      # model_adı → (eşik, kalibre_mi)
    friendly: Callable
    list_unet: Callable
    list_resnet: Callable
    green_marker_mask: Optional[Callable] = None
    unet_win: int = 512
    unet_stride: int = 448
    unet_max_side: int = 4096
    resnet_patch: int = 224
    resnet_stride: int = 112


@dataclass
class ModelSpec:
    name: str                 # uygulamadaki ad (PHASE1/… ya da PHASE2/…)
    label: str                # arayüz adı (friendly)
    family: str               # resnet | unet | classical | none
    task: str                 # patch-classification | segmentation | …
    in_ch: Optional[int]
    native_size: Optional[int]
    preprocess: str
    output: str
    decision: str
    method: str               # birincil SHAP yöntemi
    fallback: str             # ikincil yöntem
    status: str = "OK"        # OK | UNSUPPORTED | N/A | LOAD_ERROR
    reason: str = ""
    train_gt: str = ""        # bbox | pixel_eps8_morph | glyph_v2 | ""
    extra: dict = field(default_factory=dict)

    def to_row(self):
        return {"Model": self.label, "Ad": self.name, "Aile": self.family, "Görev": self.task, "Kanal": self.in_ch,
                "Ön işleme": self.preprocess, "Çıktı / karar": f"{self.output} · {self.decision}",
                "SHAP yöntemi": f"{self.method} (+{self.fallback})" if self.fallback else self.method,
                "Durum": self.status, "Gerekçe": self.reason}


def train_gt_of(name: str) -> str:
    stem = name.split("/", 1)[-1]
    if stem.startswith("unet_bbox"):
        return "bbox"
    if stem.startswith("unet_pixel"):
        return "pixel_eps8_morph"
    if stem.startswith("unet_glyph"):
        return "glyph_v2"
    return ""


# Projede adı geçen ama dosyası olmayan / uygulanamayan modeller: SHAP ÜRETİLMEZ, gerekçesiyle raporlanır.
UNSUPPORTED_MODELS = [
    ModelSpec("CLASSICAL/svm_rbf", "Klasik · SVM-RBF (56 öznitelik)", "classical", "binary-classification", None, None,
              "features_classical.py (56 öznitelik)", "skor", "—", "—", "", "UNSUPPORTED",
              "Model dosyası yok (yalnız results_tables.md'de sonuçları var). Dosya eklenirse KernelExplainer (56 öznitelik)."),
    ModelSpec("CLASSICAL/random_forest", "Klasik · RandomForest (56 öznitelik)", "classical", "binary-classification", None,
              None, "features_classical.py (56 öznitelik)", "olasılık", "—", "—", "", "UNSUPPORTED",
              "Model dosyası yok (bucket'ta .pkl/.joblib yok). Dosya eklenirse TreeExplainer (56 öznitelik)."),
    ModelSpec("N/A/vit", "ViT / EfficientNet", "none", "classification", None, None, "—", "—", "—", "—", "", "N/A",
              "Projede bu mimariler yok."),
]


def build_registry(deps: Deps, loaded: Optional[dict] = None) -> dict:
    """Uygulamanın listelediği checkpoint'lerden ModelSpec sözlüğü. `loaded`: app._REG (ad → (model, in_ch, tür))."""
    reg = {}
    loaded = loaded or {}
    for n in deps.list_resnet():
        st, why = ("OK", "") if (not loaded or n in loaded) else ("LOAD_ERROR", "checkpoint yüklenemedi")
        reg[n] = ModelSpec(
            n, deps.friendly(n), "resnet", "patch-classification", 3, deps.resnet_patch,
            f"{deps.resnet_patch} px natif patch · ImageNet norm · kayan pencere adım {deps.resnet_stride}",
            "sigmoid p(sahte) / patch", "görüntü kararı: maks patch ≥ 0.5",
            "PartitionExplainer + maskers.Image", "GradientExplainer (fp32)", st, why)
    for n in deps.list_unet():
        in_ch = loaded[n][1] if n in loaded else None
        st, why = ("OK", "") if (not loaded or n in loaded) else ("LOAD_ERROR", "checkpoint yüklenemedi")
        pre = f"{deps.unet_win} px pencere / {deps.unet_stride} adım, örtüşme ort., uzun kenar > {deps.unet_max_side} küçült · ImageNet norm"
        if in_ch == 5:
            pre += " · + FFT-HP & Laplacian (pencere başına standartlaştırma)"
        reg[n] = ModelSpec(
            n, deps.friendly(n), "unet", "segmentation", in_ch, deps.unet_win, pre,
            "sigmoid piksel olasılığı", "görüntü skoru top-%1 ort. ≥ IMG_TAU (Youden-J); bölge = olasılık ≥ τ*",
            "PartitionExplainer + maskers.Image" + (" (frekans kanalları maskelenmiş RGB'den yeniden hesaplanır)" if in_ch == 5 else ""),
            "GradientExplainer (fp32" + (", RGB / FFT-HP / Laplacian grup katkısı)" if in_ch == 5 else ")"), st, why,
            train_gt=train_gt_of(n))
    return reg


def inventory_rows(reg: dict):
    rows = [s.to_row() for s in reg.values()]
    rows += [s.to_row() for s in UNSUPPORTED_MODELS]
    return rows
