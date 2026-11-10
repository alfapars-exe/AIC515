"""xai — AIC515 için SHAP tabanlı açıklanabilirlik (XAI) altyapısı.

Tasarım ilkeleri
  * Mevcut tahmin hattına dokunulmaz: adaptörler uygulamanın KENDİ ön işleme / ileri geçiş fonksiyonlarını kullanır
    (bağımlılıklar `Deps` ile enjekte edilir; app.py'ye döngüsel import yoktur).
  * Açıklanan şey her zaman modelin gerçek çıktısından türetilmiş, olasılık ölçeğinde SKALER bir hedeftir
    (patch skoru, pencere görüntü skoru, bölge ortalaması …) — bkz. targets.py.
  * Desteklenmeyen model için sahte SHAP üretilmez; açık bir hata/durum döner.

Modüller: registry · targets · adapters · manager · viz · compare · service  (ayrıntı: docs/XAI.md)
"""
from .registry import Deps, ModelSpec, build_registry, UNSUPPORTED_MODELS
from .targets import (TargetSpec, decision_targets, detection_targets, gt_targets, common_window_targets,
                      window_mask)
from .adapters import (XAIUnsupported, ExplainConfig, adapter_for, prepare_for, run_jobs, model_fingerprint)
from .service import ExplanationResult, export_results
from .manager import XAIManager
from . import viz, compare, fullimage, viewer

__all__ = ["Deps", "ModelSpec", "build_registry", "UNSUPPORTED_MODELS", "TargetSpec", "decision_targets",
           "detection_targets", "gt_targets", "common_window_targets", "window_mask", "XAIUnsupported", "ExplainConfig",
           "adapter_for", "prepare_for", "run_jobs", "model_fingerprint", "ExplanationResult", "export_results",
           "XAIManager", "viz", "compare"]
