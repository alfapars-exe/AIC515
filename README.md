# AIC515 Görüntü İşleme — Dönem Projesi: AI-Generated Financial Document Forgery Localization (AIForge-Doc v1)

**Durum (10 Kasım 2026):** PHASE 1–3 tamamlandı. IEEE bildirisi: `paper/main.pdf` (kaynak `paper/`). Faz 3 kapanış raporu: `reports/PHASE3_Sonuclar_ve_Kapanis.md`; hakem hazırlığı: `reports/Hakem_Hazirligi_Faz3.md`; otomatik tablolar: `phase3/results/tables_p3.md`.
**Araştırma yönü:** *What Do Document Forgery Localizers Learn from Box Masks? Pixel-Change Ground Truth and Shortcut Audits for AI-Inpainted Financial Documents*. Ana sonuç: H1 doğrulandı (aynı tahminler bbox-GT AUPRC 0.923 → piksel-GT 0.374), H2 kısmen, H3/H4 yanlışlandı (bkz. `reports/PHASE2_Metodoloji_ve_Sonuclar.md`).


## Durum (20 Ekim 2026) ve son eklenenler
- **Tam metin bildiri (İngilizce):** `reports/Bildiri_Tam_Metin.md/.docx` — 9 bölüm + 2 ek, 21 tablo, 28 şekil, 50 doğrulanmış kaynak; `tools/build_manuscript.py` ile sonuç dosyalarından yeniden üretilir.
- **Yeni bulgu — yeşil işaretçi artefaktı:** sahtelerin %75'inde kutu iç kenarında işaretçi kalıntısı; tüm modeller kullanıyor (authentic + işaretçi → AP 0,906; onarım → Δ −0,225). Ayrıntı: PHASE2 raporu Ek-C, bildiri §6.6-bis. Kod: `phase2/experiments/probe_green_marker.py`, `analyze_green_marker.py`.
- **GT doğrulaması:** insan doğrulaması yerine VLM destekli denetim (100 çift; `phase2/gt/verification_sheet_vlm.csv`, `vlm_verification_summary.json`); insan sütunları boş bırakıldı, "insan doğrulaması" olarak **sunulmamalıdır**.
- **Checkpoint değerlendirmesi:** `phase2/experiments/results/checkpoint_summary.csv`.
- **Teslim paketi:** `TESLIM_PAKETI/` (kod + ağırlıklar + GT maskeleri + sonuçlar + raporlar; ham veri seti hariç — lisans gereği `tools/download_dataset.py` ile indirilir). Uçtan uca yeniden üretim: `run_everything.ps1` (tek 16 GB GPU, ~6 saat).

## Dizin yapısı
| Klasör / dosya | İçerik |
|---|---|
| `reports/PHASE1_Arastirma_Raporu.{md,docx}` | Şartnamedeki A–J çıktıları: yönetici özeti, kaynak doğrulama, literatür matrisi özeti, Gap A–D kararları, veri seti audit özeti, 3 aday, karar matrisi, seçilen yön, RQ + falsifiable hipotezler, sonraki aşama gereksinimleri, AIC515 ilişkisi, küçük-veri baseline sonuçları |
| `reports/Vize_Raporu_Taslak.{md,docx}` | Vize raporu taslağı: Proje Adı, Literatür, Materyal ve Metot (veri tabanı, modeller, pseudo kod, matematiksel altyapı), Sonuçlar (smoke alt küme), Tartışma, Kaynaklar |
| `literature/` | `seed_papers_review.md` + `seed_papers.json` (6 seed makale tam metin incelemesi), `literature_matrix.{csv,xlsx,md}` (38 çalışma × 15+ alan), `search_strategy.md`, `references.bib`, `seed_sources/` (kaynak PDF'ler, API dökümleri) |
| `dataset_audit/` | `audit_dataset.py` (yeniden çalıştırılabilir), `audit_report.md`, `audit_stats.json`, `splits.json` (group-aware train/val + smoke alt küme), `samples_forged.csv`, `samples_authentic.csv`, `phash_pairs.csv`, `dataset_card_snapshot.md`, figürler |
| `data/AIForge-Doc-v1/` | Hugging Face'ten indirilen veri seti (12.187 dosya, 9.98 GB; **CC BY-NC-SA 4.0 + gated — yeniden dağıtmayın**) |
| `baselines/` | Kod (`common/data.py`, `features_classical.py`, `train_classical.py`, `train_resnet.py`, `train_unet.py`, `evaluate.py`, `run_all.ps1`), `results/`, `figures/`, `logs/`, `README.md` (tüm sonuç tabloları), `ENV_SETUP.md`, `requirements_freeze.txt` |
| `models/` | PHASE 1 checkpoint'leri: `resnet18_{probe,ft,scratch}_best.pt`, `unet_{rgb,rgbfreq}_best.pt` |
| `reports/PHASE2_Metodoloji_ve_Sonuclar.{md,docx}` | PHASE 2: GT türetimi, kontroller, protokol, tam-veri sonuçları, istatistiksel kararlar (CI/Holm), adversarial review, yayın planı |
| `reports/Bildiri_Taslagi_IEEE.{md,docx}` | İngilizce konferans bildirisi taslağı (IEEE yapısı; yalnız ölçülen değerler) |
| `phase2/gt/` | 7 piksel-değişim/glif GT varyantı (3.247 train + 89 test çifti), gt_report.md, ε-duyarlılık figürleri, verification_grid.pdf + verification_sheet.csv (insan doğrulaması için), make_gt.py |
| `phase2/controls/` | benign_pasteback / fullimage_resample / reencode_png kontrol setleri (487 val), controls_report.md, make_controls.py |
| `phase2/literature_baselines/` | TruFor ve ADCD-Net kurulumu, LICENSES.md, zero_shot_results.csv, preds/ |
| `phase2/experiments/` | train_unet_p2.py, infer_fullimage.py, evaluate_p2.py, probes.py, h4_global_trace.py, run_phase2.ps1, results/, figures/, logs/, preds/, README.md (tüm tablolar) |
| `phase2/stats/` | stats_analysis.py, stats_summary.csv (463 kontrast), hypothesis_decisions.md, figures/ |
| `models/phase2/` | 12 U-Net checkpoint (3 GT × 3 seed + Gemini-only × 3) |

## Hızlı başlangıç (kendi makinenizde)
1. Ortam: `baselines/ENV_SETUP.md` (conda + pip; RTX 5080 için PyTorch cu128 zorunlu).
2. Veri audit'i: `python dataset_audit/audit_dataset.py --data data/AIForge-Doc-v1 --out dataset_audit --workers 24`
3. Baseline'lar: `powershell baselines/run_all.ps1` (smoke alt küme; ~25 dk GPU).

## Küçük-veri baseline özeti (smoke_train 400 / smoke_val 100, tek seed — provizyonel)
| Model | Ana metrik |
|---|---|
| SVM-RBF / RandomForest (56 klasik öznitelik) | AUROC 0.646 / 0.706 |
| ResNet-18 linear probe / full fine-tune / scratch | AUROC 0.726 / **0.899** / 0.679 |
| U-Net RGB | Pixel-F1@0.5 0.707, IoU 0.637, AUPRC 0.947 |
| U-Net RGB+FFT-HP+Laplacian | Pixel-F1@0.5 0.674, AUPRC 0.939 |
| Dayanıklılık (U-Net RGB) | JPEG Q75 F1 0.636; 0.75× rescale F1 0.638 |

## Veri seti hakkında bilinmesi gerekenler (audit)
- Maskeler alan **bbox** maskesidir (piksel-değişim değil); eşleşen authentic ile fark alındığında bbox piksellerinin medyan %49'u (eps=8) değişir; çiftlerin %33.5'inde bbox dışı değişim vardır.
- `TestingSet/authentic` test sahtelerinin aynası değildir → image-level değerlendirmede kullanmayın; group-aware negatifler `splits.json` ile tanımlıdır.
- Test sahtelerinin %8.9'u train kaynak belgesini paylaşır → "temiz test" alt kümesi (740) ayrıca raporlanmalı.
- 12 metadata satırında bbox hatalı → GT olarak maske dosyasını kullanın.

## PHASE 2 ana sonuçları (tam veri, 3 seed)
| Bulgu | Değer |
|---|---|
| Resmî test, bbox-GT (U-Net bbox eğitimi) | AUPRC 0.950 ± 0.011, F1@τ* 0.846 ± 0.015 (temiz 740: aynı) |
| Aynı tahminler, piksel-GT | AUPRC 0.374; ΔF1 0.398 [0.377, 0.418] → **H1 doğrulandı** |
| Tahmin dikdörtgenselliği (bbox / piksel / glif modeli) | 0.87 / 0.32 / 0.05 |
| Sınır-blur relΔAP (bbox / piksel) | −1.5 % / −3.0 % → sınır shortcut'ı yanlışlandı; **H2 kısmen** |
| Held-out Ideogram, piksel − bbox ΔAP | +0.025 [−0.039, 0.088] → **H3 yanlışlandı** |
| Bbox-dışı crop forged/authentic AUROC | 0.516 (pozitif kontrol 0.980) → **H4 yanlışlandı** |
| Zero-shot TruFor / ADCD-Net (test, F1) | 0.403 / 0.182 |

## PHASE 3 — sonuçlar (10 Kasım 2026)

![Dual-GT](phase3/figures/fig_dual_gt_families.png)
*Val pooled piksel AUPRC: eğitim maskesi (x) × değerlendirme GT'si (renk). U-Net 3 tohum, DeepLabV3+ 1 tohum, YOLO11s 3 tohum.*

| Bulgu | Değer | Tohum |
|---|---|---|
| H1: aynı tahminler, box-GT → piksel-GT (U-Net box modeli, val) | 0.923 → 0.374 | 3 |
| H1 tekrarı: DeepLabV3+ / YOLO11s | 0.939 → 0.382 / 0.965 → 0.368 | 1 / 3 |
| Kutu dedektörü (YOLO11s kare 1024), test box AUPRC / F1@τ* | 0.969 / 0.946 (U-Net 0.950 / 0.846) | 3 |
| İşaretçi onarımı ΔAP (U-Net box) → işaretçisiz yeniden eğitimde | −0.222 → +0.019 | 3 → 1 |
| Test AP işaretçili / işaretçisiz (U-Net box) | 0.972 / 0.690 | 3 |
| Held-out: Gemini-only → Ideogram AP (U-Net box / YOLO11s) | 0.412 / 0.752 | 3 / 1 |
| Birleşik kayıp (piksel + 0.5·box), piksel-GT AUPRC | 0.610 → 0.622 | 1 |
| Gri seviye test girdisi, relΔ AUPRC | −47 … −67 % | 1–3 |
| Faz 1 AUPRC (kesin) | 0.947 → 0.805 | — |

### Yeniden üretim (Windows, tek 16 GB GPU, ≈14 GPU-saat)
```powershell
conda activate docforensics            # Python 3.11
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r phase3/requirements_phase3.txt --no-deps
$env:P3_PYTHON = (Get-Command python).Source
$env:VIZE_ROOT = (Get-Location).Path   # proje kökü
python tools/download_dataset.py --out data/AIForge-Doc-v1      # HF kapılı erişim
powershell -File run_everything.ps1 -SkipDownload                # Faz 1–2
powershell -File phase3/q0_dataprep.ps1; powershell -File phase3/q1_unet_train.ps1
powershell -File phase3/q2_yolo_train.ps1 -Epochs 30
powershell -File phase3/q3a_eval_unet.ps1; powershell -File phase3/q3b_eval_yolo.ps1
python phase3/synthesize_p3.py                                   # tablolar + şekiller
```
Ağırlıklar ve tahmin haritaları repoda yoktur (boyut/lisans); yukarıdaki komutlarla üretilir. Faz 2 checkpoint'leri yeni ortamda birebir aynı çıktıyı verir (`phase3/results/repro_check_env.json`).

### Atıf
```bibtex
@misc{benli2026boxmasks,
  author = {Harun Benli},
  title  = {What Do Document Forgery Localizers Learn from Box Masks? Ground-Truth Geometry, a Hidden Marker Shortcut, and Box Detection on AI-Inpainted Financial Documents},
  year   = {2026},
  note   = {AIC515 Image Processing term project},
  howpublished = {\url{https://github.com/alfapars-exe/AIC515}}
}
```

### Faz 2'den devralınan işlerin kapanışı
| Açık iş (PHASE 2 / Lab Rat inceleme) | Faz 3'teki karşılığı | Kod |
|---|---|---|
| W2 tek mimari | DeepLabV3+ (aynı ResNet-18 kodlayıcı) × 3 GT; YOLO11 kutu dedektörü | `phase3/train_unet_p3.py --arch dlv3p`, `phase3_yolo/` |
| W4/W5 işaretçi; GT işaretçiyi içeriyor | İşaretçi-onarımlı eğitim + işaretçisiz piksel/glif GT (`*_nm`) | `phase3/make_marker_repaired.py`, `--marker-repair` |
| PHASE3-a birleşik kayıp | U-Net 2 kanal: piksel/glif + 0.5·bbox yardımcı | `--aux-bbox 0.5` |
| PHASE3-b 20 epoch | 3 GT × s42, 20 epoch | `train_unet_p2.py --epochs 20 --tag e20` |
| W3 held-out tek seed | Gemini-only × 3 GT × s43/s44 (toplam 3 seed) | `train_unet_p2.py --train-filter gemini_only` |
| PHASE3-d bölge görünümü | render-only / content-only sondası | `phase3/probes_p3.py` |
| PHASE3-e kutu formülasyonu | YOLO11 n/s/m, kareli 1024 vs tam 1280, aug, authentic negatif | `phase3_yolo/` |
| §6.4 "test edilmedi" (CLAHE, gri) | Test-zamanı koşulları tüm modellerde | `phase3/p3ext.py` |
| W6 Faz 1 AUPRC şişkin | Kesin AUPRC: 0.947 → **0.805** (rgb), 0.939 → **0.792** (rgbfreq) | `phase3/phase1_exact_auprc.py` |
| Veri bias analizi | işaretçi × üretici Cramér's V 0.61; split χ² p ≥ 0.31 | `phase3/dataset_curation.py` |
| Literatür | 77 doğrulanmış kaynak (38 yeni) | `literature/references_v3.bib`, `lit_update_phase3.md` |
| W1 insan GT doğrulaması | **Yapılamadı** — insan emeği gerektirir; `phase2/gt/verification_sheet.csv` hazır | — |
| W7 TruFor ince ayar | **Yapılamadı** — yayımlanan kodda eğitim betiği yok; mimari bağımsızlık DeepLabV3+/YOLO ile | — |

Ortam: torch 2.11.0+cu128, smp 0.6.0, ultralytics 8.4.208; Faz 2 checkpoint'leri yeni ortamda birebir aynı çıktıyı verir (`phase3/results/repro_check_env.json`).

## Veri ve lisans
AIForge-Doc v1 (CC BY-NC-SA 4.0, kapılı erişim) bu repoda **yeniden dağıtılmaz**; görüntüler, türetilmiş maskeler, onarılmış görüntüler, kontrol setleri, model ağırlıkları ve tahmin haritaları repoya dahil değildir. Veri `tools/download_dataset.py` ile indirilir, tüm türevler betiklerle yeniden üretilir (`run_everything.ps1`, `phase3/q*.ps1`). Üçüncü taraf referans modeller (TruFor, ADCD-Net) kendi depolarından kurulur (`phase2/literature_baselines/README.md`, `LICENSES.md`).
