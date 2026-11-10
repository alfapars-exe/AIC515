"""
app.py — AIC515 Görüntü İşleme · AIForge-Doc Sahtecilik Lokalizasyonu — Simülasyon + Aşamalı İşlem Hattı (TR)

Sekmeler
  0 · Simülasyon   : veri setinden bir örnek seç (filtre / rastgele / kimlik) → girdi senaryosu (sahte, authentic,
                     işaretçi aktarımı/onarımı) → bozulma → ResNet-18 + U-Net'ler → adım adım çıktı, metrikler
                     (bbox ve piksel-değişim GT'si), görüntü-düzeyi karar, yakın plan paneli, indirilebilir rapor.
  1 · Toplu test   : N örnek üzerinde aynı hat; model başına toplu metrikler, tespit doğruluğu, boru hattının
                     kayıtlı sonuçlarıyla yeniden-üretim karşılaştırması, CSV/JSON.
  2–7              : girdi yükleme, 61 ön işleme işlemi (aşamalı pipeline), U-Net, ResNet-18, yeşil işaretçi, sonuçlar.
  8 · Açıklanabilirlik (XAI) : iki bağımsız alt sekme —
      SHAP Açıklamaları : tahmin → SHAP (Partition / Gradient; authentic / blur / inpaint referansı) → model karşılaştırması;
                     xai/ paketi (docs/XAI.md).
      Katman Filtreleri : SHAP'tan bağımsız — ilk conv katmanlarının checkpoint'teki gerçek kernel'leri, görüntüye
                     uygulanmış aktivasyon haritaları (conv / BN / aktivasyon), orijinalle karşılaştırma; xai_filters.py
                     (docs/XAI_FILTERS.md).

Çalıştırma
  Yerel (proje kökü):   python app/app.py                        → http://127.0.0.1:7860
  Hugging Face Space :  ZeroGPU (@spaces.GPU); veri/modeller özel bucket'tan AI515_DATA_ROOT (/data) altına bağlanır.
"""
from __future__ import annotations
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # appcore/, xai/, xai_filters.py

# Uygulama mantığı appcore/ paketindedir (katmanlar için bkz. appcore/__init__.py). Bu dosya giriş noktasıdır:
# modülleri sırayla yükler ve adlarını `app.<ad>` olarak yeniden dışa aktarır (testler, xai/ ve araçlar bu adları kullanır).
import appcore

_MODS = [importlib.import_module(f"appcore.{_n}") for _n in appcore.MODULES]
_LIVE = {"_INDEX", "_REF", "DRAFTS", "XAI_DEPS", "XAI_SPECS", "XAI_MGR"}  # çalışma anında yeniden bağlanan durum → canlı okunur
for _m in _MODS:
    globals().update({_k: _v for _k, _v in vars(_m).items() if not _k.startswith("__") and _k not in _LIVE})


def __getattr__(name):  # app._INDEX, app.DRAFTS ... her erişimde ait olduğu modülden okunur
    for m in reversed(_MODS):
        if name in vars(m):
            return vars(m)[name]
    raise AttributeError(f"module 'app' has no attribute {name!r}")


if __name__ == "__main__":
    port = int(os.environ.get("APP_PORT", "7860"))
    host = os.environ.get("APP_HOST", "0.0.0.0" if IN_SPACE else "127.0.0.1")
    demo.launch(server_name=host, server_port=port, inbrowser=(os.environ.get("APP_NO_BROWSER") is None and not IN_SPACE),
                css=CSS, show_error=True)
