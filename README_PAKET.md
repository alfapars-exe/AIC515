# TESLİM PAKETİ — AIC515 Görüntü İşleme Dönem Projesi
**Proje:** What Do Document Forgery Localizers Learn from Box Masks? Pixel-Change Ground Truth, Shortcut Audits and a Hidden Marker Artefact in AI-Inpainted Financial Documents (AIForge-Doc v1)
**Oluşturulma:** 20 Ekim 2026 · `tools/make_package.py` · 3.08 GB

## İçerik
| Klasör | İçerik |
|---|---|
| `reports/` | Vize raporu (TR), PHASE 1 araştırma raporu (TR), PHASE 2 metodoloji+sonuçlar (TR, Ek-C işaretçi bulgusu), **Bildiri_Tam_Metin** (EN, tam metin), IEEE bildiri taslağı, `src/` (makale kaynak parçaları) |
| `literature/` | Literatür matrisi (38 çalışma, csv/xlsx/md), arama stratejisi, seed paper incelemesi, `references.bib` |
| `dataset_audit/` | Veri seti denetim raporu, istatistikler, bölünmeler (`splits.json`), figürler |
| `baselines/` | Pilot (vize) kodu, sonuçları, ağırlıkları (SVM/RF/ResNet-18/U-Net) |
| `phase2/gt/` | 7 türetilmiş GT varyantı (maske PNG), `gt_index.csv`, doğrulama paneli PDF, VLM destekli doğrulama (`verification_sheet_vlm.csv`, `vlm_verification/`), yeşil işaretçi kalıntı listeleri |
| `phase2/controls/` | Benign kontrol üretim kodu ve istatistikleri (görüntüler yeniden üretilir) |
| `phase2/experiments/` | Eğitim/değerlendirme kodu, sonuç tabloları, figürler, işaretçi sondası |
| `phase2/literature_baselines/` | TruFor ve ADCD-Net zero-shot çalıştırma kodu, ağırlıkları, lisansları, sonuçları |
| `phase2/stats/` | 463 kontrast (grup bootstrap, Holm), hipotez kararları, figürler |
| `models/` | Model ağırlıkları: pilot (phase1) ve 12 tam-veri U-Net + 2 sonda sınıflandırıcısı (phase2) |
| `tools/` | `download_dataset.py`, `build_manuscript.py`, `md2docx.py`, `make_package.py` |
| `run_everything.ps1` | Uçtan uca yeniden üretim (tek 16 GB GPU, ~6 saat, veri indirme hariç) |

## Pakette OLMAYANLAR (ve neden)
- Ham veri seti `data/AIForge-Doc-v1` — CC BY-NC-SA 4.0, gated; `python tools/download_dataset.py` ile indirilir (HF token gerekir).
- Tahmin haritaları (`preds/`) ve kontrol görüntüleri — büyük ve kod ile yeniden üretilebilir.

## Ortam
`conda create -n docforensics python=3.11` + `pip install -r requirements.txt` (torch cu128, segmentation-models-pytorch 0.5, timm, scikit-learn, scikit-image, opencv-python-headless, albumentations, torchmetrics, huggingface_hub, python-docx). `VIZE_ROOT` ortam değişkeni paket köküne ayarlanmalıdır.

## Önemli notlar
- GT doğrulaması **VLM destekli denetimdir, insan doğrulaması değildir**; insan sütunları boştur.
- Yeşil işaretçi sondası **keşifseldir (ön-kayıtlı değil)**; raporlarda ön-kayıtlı kararların yanında ayrı sunulur.
- Yazar adları ve bazı klasik kaynak alanları bildiride yer tutucudur.
