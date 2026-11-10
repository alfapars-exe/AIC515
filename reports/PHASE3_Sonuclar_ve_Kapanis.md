# PHASE 3 — Sonuçlar ve Proje Kapanışı (AIC515, AIForge-Doc v1)

**Tarih:** 10 Kasım 2026 · **Yazar:** Harun Benli · **Donanım:** RTX 5080 16 GB · torch 2.11.0+cu128, smp 0.6.0, ultralytics 8.4.208
**Bağlayıcı metin:** `paper/main.pdf` (IEEE bildirisi). Tüm sayılar `phase3/results/tables_p3.md` ve `phase3_yolo/results/*.csv`'den otomatik üretildi (`phase3/synthesize_p3.py`).
**Tohum etiketi:** Aksi belirtilmedikçe Faz 3'teki tek-faktör deneyleri **tek tohumdur (42)** ve öncül kanıt sayılmalıdır. Üç tohumlu olanlar: Faz 2 U-Net ana ızgarası (3 GT × s42–44), Gemini-only held-out U-Net'ler (3 GT × s42–44), YOLO11s ana model (s42–44).

## 1. Faz 2'den devralınan açık işler — kapanış tablosu

| Açık iş | Durum | Sonuç (kısa) | Kanıt |
|---|---|---|---|
| W2 tek mimari | ✅ Kapandı | DeepLabV3+ H1'i tekrarlıyor: box modeli box-GT 0.939 → piksel-GT 0.382; her GT kendi modelini birinci sıralıyor (tek tohum) | T1 |
| W4 işaretçisiz yeniden eğitim | ✅ Kapandı | İşaretçi bağımlılığı kalkıyor (onarım Δ −0.222 [3 tohum] → +0.019 [1 tohum]); box skoru büyük ölçüde korunuyor (val 0.916 vs 0.923, test 0.939 vs 0.950) (tek tohum) | T1–T3 |
| W5 piksel GT işaretçiyi içeriyor | ✅ Kapandı | `*_nm` GT varyantları üretildi ve eğitimde kullanıldı | `make_marker_repaired.py` |
| W3 held-out tek tohum | ✅ Kapandı | 3 tohum: Gemini-only box U-Net Ideogram 0.412 vs Gemini 0.909; piksel denetimi açığı kapatmıyor (H3 reddi teyit) | T4 |
| W6 Faz 1 AUPRC şişkin | ✅ Kapandı | Kesin AUPRC 0.805 (rgb) / 0.792 (rgbfreq); raporlanan 0.947/0.939 alt-örneklemeden | T8 |
| W9/W10 tek tohum etiketleri, eksik kaynaklar | ✅ Kapandı | Errata eski raporlara eklendi; kaynakça 77 doğrulanmış kayıt | `references_v3.bib` |
| PHASE3-a birleşik kayıp | ✅ Kapandı | Piksel+0.5·box: pix-GT 0.610→0.622, box-GT 0.579→0.647, blur −59 %→−43 % (tek tohum) | T1, T6 |
| PHASE3-b 20 epoch | ✅ Kapandı | Fayda yok (box 0.926, pix 0.594, glyph 0.645 vs 0.923/0.610/0.676) | T1 |
| PHASE3-d bölge görünümü | ✅ Kapandı | Render-only/content-only ayrıştırması; standart modeller ikisinin birlikteliğine (ve işaretçiye) tepki veriyor; işaretçisiz eğitim içeriğe kaydırıyor | T3 |
| PHASE3-e kutu formülasyonu (YOLO) | ✅ Kapandı | Bkz. §2 | T2, T5 |
| §6.4 "test edilmedi" ön işleme | ✅ Kapandı | Gri seviye tüm modellerde AUPRC −47…−67 % (en zararlı; YOLO −59 %); CLAHE ≤ ±3 % | T3, T6 |
| Veri bias analizi | ✅ Kapandı | İşaretçi × üretici Cramér's V 0.61; split χ² p ≥ 0.30 | `dataset_curation.md` |
| W1 100 çift GT doğrulaması | ⚠ Kısmen | İkinci bağımsız denetim tamamlandı (değerlendirici türü doğrulanmadı → insan etiketlemesi sayılmaz): kutu 99/100 OK; piksel GT 36 OK / 64 PARTIAL / 0 WRONG — PARTIAL'ların 61'i değişmeyen, yeniden çizilmiş metni fazladan kapsamaktan (WildReceipt 39/39, XFUND 0/11); glyph_v2 42/54/4; VLM ile uyum zayıf (κ = 0.07 [−0.08, 0.21]), VLM 43 çiftte daha iyimser. Bildiri sınırlaması güncellendi | `phase2/gt/human_verification_summary.md` |
| W7 TruFor ince ayar | ❌ Yapılamadı | Yayımlanan kodda eğitim betiği yok (README "TODO"); mimari bağımsızlık DeepLabV3+ ve YOLO ile gösterildi | — |
| AIForge-Doc v2 harici test | ✅ Kapandı | 567 sızıntısız çift (v1 test karşılığı, kaynak belge eğitimde yok), yeniden eğitim/kalibrasyon yok: box U-Net 0.950 → 0.520±0.018 AUPRC; YOLO11s 0.887±0.006 (görüntü AUROC 0.919); işaretçisiz box U-Net 0.712 ve kalıntısız spesifikasyonlarda kayıpsız (+0.003, tek tohum). Kaybın çoğu v1'de işaretçi taşıyan spesifikasyonlarda (§6) | `phase3_v2/results/v2_tables.md` |

## 2. YOLO11 hipotez kararları (plan: `phase3_yolo/README.md`)

| Hipotez | Beklenti | Sonuç | Karar |
|---|---|---|---|
| H5a kutu dedektörü box-GT'de U-Net'e eşit/üstün, ≥3× hızlı | AP50 ≥ 0.90; ±0.03 | Test box AUPRC 0.969±0.006 (U-Net 0.950), F1@τ* 0.946 (0.846), IoU 0.918 (0.789); AP50 0.880 (U-Net 0.873, s42/43). **Hız:** 98 ms vs U-Net 81 ms | **Doğruluk kısmı desteklendi; hız kısmı reddedildi** |
| H5b piksel-GT'de belirgin düşük | pix AUPRC ≈ 0.35–0.45 | val 0.368±0.015 (U-Net box 0.374; U-Net pix 0.610) | **Desteklendi** (H1'in model-bağımsız teyidi) |
| H5c küçük hedeflerde kare ≫ tam 1280 | <1 % alanda ΔAP ≥ 0.10 | Kare: AUPRC 0.969 vs 0.932, işaretçisiz AP 0.878 vs 0.756; **ama** kutu AP50 0.880 vs 0.960 (<1 %: 0.872 vs 0.933) — olası neden kare sınırında bölünen kutular (karesiz model de ~5 kutu/görüntü ürettiği için mekanizma kesin değil) | **Karışık** |
| H5d held-out açığı YOLO'da daha küçük | ≤ 0.51 | Gemini-only YOLO Ideogram 0.752 vs Gemini 0.938 (açık 0.19; U-Net 0.50) — tek tohum | **Desteklendi (öncül)** |
| H5e muhafazakâr aug ≥ varsayılan | conserv ≥ default | 0.969 vs 0.971 test AUPRC; AP50 0.882 vs 0.864 | **Desteklenmedi** (fark yok) |

Ek: YOLO işaretçiye daha az bağımlı (onarım Δ −0.062 vs U-Net −0.222) ama "yalnız yeniden-render" girdisini authentic'ten AUROC 0.865 ile ayırıyor (U-Net 0.637, 3 tohum; YOLO tek tohum) → dedektör üreticinin alanı yeniden çizme dokusunu öğrenmiş; bu, üreticiler arası daha iyi aktarımı da açıklayabilir. JPEG Q75'e kırılgan (−49 %), blur/CLAHE/ölçeğe dayanıklı (≤1.4 %).

## 3. Ana mesaj (bildiri)
1. **H1 (3 tohum):** Kutu GT, değişen pikselleri değil "düzenlenen alanı" ödüllendirir; aynı tahminler box-GT 0.923 → piksel-GT 0.374. İkinci mimaride ve kutu dedektöründe tekrarlanır.
2. **İşaretçi kısayolu:** Kullanılır ama gerekli değildir; v1'de held-out üretici sonuçları işaretçiyle karışıktır (V = 0.61).
3. **Kutu dedektörü** box metriğinde en iyi modeldir; bu metrik "alan tespiti"ni ölçer, piksel düzeyi kurcalama lokalizasyonunu değil.
4. **Öneriler:** box ve piksel-değişim metriklerini yan yana raporlamak; üretim kalıntılarını denetlemek; kutu dedektörünü taban çizgisi olarak eklemek.

## 4. Operasyon notları
- 9 Kasım 23:30'da tüm kuyruk süreçleri sonlandı (makine yeniden başlamadı; neden belirsiz). Yarım kalan 2 adım (YOLO E1, U-Net s43/s44 sondası) 01:14'te yeniden koşuldu; tamamlanan adımlar `.done` işaretleriyle korundu (`phase3/logs/queue_notes.txt`).
- Plandan sapmalar (lr0 0.01→0.001, epoch 40→30, workers 6, YOLO sondaları s42): `phase3_yolo/README.md` §7.
- Yeniden üretim: `phase3/q0_dataprep.ps1` → `q1_unet_train.ps1` ∥ `q2_yolo_train.ps1` → `q3a_eval_unet.ps1` → `q3b_eval_yolo.ps1` → `synthesize_p3.py`. Toplam ≈ 14 GPU-saat (RTX 5080).

## 5. Hugging Face Space entegrasyonu (infinayazilim/ai515, ZeroGPU)
- **Modeller:** Lokalizasyon listelerine Faz 3 modelleri eklendi — DeepLabV3+ (3 GT), kutu yardımcı kayıplı U-Net (piksel/glif), işaretçisiz eğitilmiş U-Net (3 GT) ve YOLO11 kutu dedektörleri (s/n/m kare 1024, tam görüntü 1280, yalnız Gemini, +authentic kare, varsayılan aug). Tohum tekrarları (YOLO s43/s44, Gemini-only U-Net s43/s44) ve 20 epoch varyantları bucket'ta durur, arayüzde gizlidir. Ağırlıklar özel bucket'ta: `models/phase2/`, `models/phase3_yolo/<run>.pt`.
- **YOLO çıkarımı** `phase3_yolo/infer_to_maps.py` ile aynı: natif çözünürlük, 1024 kare / 768 adım (ya da tam görüntü 1280), conf 0.001, NMS IoU 0.6, kareler arası NMS, kutular güvenle haritaya boyanır → U-Net ile aynı metrikler, τ* ve görüntü kararı.
- **Doğrulama (`tests/test_app_p3.py`, RTX 5080):** 3 val görüntüsü × 9 Faz 3 modelinde uygulama içi harita, boru hattının kaydettiği haritayla karşılaştırıldı: 27 karşılaştırmanın 25'inde görüntü skoru farkı ≤ 0.0005; YOLO'da 2 durumda fp16/NMS sınırındaki kutulardan kaynaklı fark (0.011 ve 0.044; değişen piksel oranı ≤ %0.12). Mevcut `test_app.py` (76 kontrol) ve `test_xai_filters.py` geçti; `test_xai.py` CPU'da geçti (GPU'da bf16 toleransı nedeniyle 2 kesinlik testi 1e-5 eşiğini aşıyor — Faz 3'ten bağımsız).
- **Görüntü-düzeyi eşikler** (`hf_space/meta/image_tau.json`, Youden-J, val 487 sahte vs eşleşen authentic): YOLO11s s42 eşik 0.171, AUROC 0.961 (U-Net bbox s42: 0.921); DeepLabV3+ bbox 0.840; işaretçisiz U-Net bbox 0.890; birleşik kayıplı piksel U-Net 0.897. Authentic haritası bulunmayan YOLO varyantlarında görüntü kararı "kalibre değil" olarak işaretlenir.
- **Yeni senaryolar:** *Yalnız yeniden-render* (değişen karakterler + işaretçi authentic'e döndürülür; etiket GERÇEK) ve *yalnız içerik* (authentic'e yalnız değişen karakterler yapıştırılır; etiket SAHTE) — `probes_p3.py` tanımı (4-komşu haç, 2 px genişletme); tek fark işaretçi kalıntısının bbox halkasıyla sınırlanmaması. Yeni bozulmalar: CLAHE (L kanalı), gri seviye.
- **Sonuçlar sekmesi** Faz 3 sentez tablolarını (`meta/phase3_tables.md`) gösterir. XAI SHAP sekmesi piksel segmentasyon modelleriyle sınırlıdır (YOLO için tam görüntü hedefleri ayrı bir çalışmada geliştiriliyor).
- **Canlı test** (`hf_space/live_test.py` → `reports/canli_test_p3.json`): 6 senaryo × 6 model + 24 çiftli toplu test ZeroGPU'da tamamlandı; toplu testte YOLO11s F1 bbox 0.948, IoU 0.910, AP bbox 0.939, piksel-GT F1 0.534 (U-Net bbox s42: 0.925 / 0.871 / 0.964 / 0.528). Bağımlılıklar: `ultralytics==8.4.208` (AGPL-3.0), `segmentation-models-pytorch==0.6.0`, `packages.txt` (libgl1).

## 6. AIForge-Doc v2 (GPT-Image-2) — harici test, yeniden eğitim yok

567 çiftlik birincil küme: v1 karşılığı test bölümünde, kaynak belgesi v1 eğitiminde hiç görülmemiş. v2'de yalnız düzenlenen kutu geri yapıştırılır; yeşil işaretçi kalıntısı yoktur (3066 görüntünün 0'ı). Eşikler v1 doğrulamadan, değişmeden. ΔAP: aynı spesifikasyonun v1 test görüntüsünden v2 görüntüsüne görüntü-başı AP değişimi; v1 görüntüsünde işaretçi kalıntısı olan (397) ve olmayan (170) çiftler ayrı.

| model (v1'de eğitildi)        |   box AUPRC |   piksel AUPRC |   glif AUPRC |   görüntü AUROC |   ΔAP v1 kalıntılı |   ΔAP v1 kalıntısız |
|:------------------------------|------------:|---------------:|-------------:|----------------:|-------------------:|--------------------:|
| U-Net box (3 tohum)           |       0.52  |          0.272 |        0.209 |           0.717 |             -0.49  |              -0.169 |
| U-Net piksel (3)              |       0.227 |          0.246 |        0.252 |           0.693 |             -0.398 |              -0.172 |
| U-Net glif (3)                |       0.147 |          0.192 |        0.304 |           0.688 |             -0.171 |              -0.112 |
| DeepLabV3+ box                |       0.539 |          0.28  |        0.218 |           0.724 |             -0.375 |              -0.141 |
| U-Net box, işaretçisiz eğitim |       0.712 |          0.392 |        0.296 |           0.835 |             -0.19  |               0.003 |
| U-Net piksel, işaretçisiz     |       0.353 |          0.43  |        0.416 |           0.826 |             -0.106 |              -0.078 |
| U-Net glif, işaretçisiz       |       0.214 |          0.287 |        0.446 |           0.764 |             -0.076 |              -0.065 |
| U-Net box, yalnız Gemini      |       0.449 |          0.248 |        0.175 |           0.742 |             -0.54  |              -0.189 |
| YOLO11s kare (3)              |       0.887 |          0.354 |        0.244 |           0.919 |             -0.149 |              -0.033 |
| YOLO11s tam 1280              |       0.697 |          0.328 |        0.215 |           0.8   |             -0.202 |              -0.083 |
| YOLO11s yalnız Gemini         |       0.843 |          0.338 |        0.229 |           0.865 |             -0.212 |              -0.055 |
| YOLO11s +authentic kare       |       0.863 |          0.345 |        0.241 |           0.937 |             -0.191 |              -0.055 |

**Okuma:** box U-Net v1'de 0.95 iken v2'de 0.52'ye düşüyor; kaybın çoğu v1'de işaretçi taşıyan spesifikasyonlarda. İşaretçisiz eğitilen box U-Net kalıntısız spesifikasyonlarda hiç kaybetmiyor (tek tohum); kareli YOLO11s 0.89'da kalıyor.

**Tüm v2 (3066 çift) görüntü AUROC** — 'seen': kaynak belge v1 eğitiminde vardı; README zero-shot taban çizgileri: TruFor 0.599, DocTamper 0.585, insan 0.501.

| model                                     | subset   |   n_pairs |   auroc_top1pct |
|:------------------------------------------|:---------|----------:|----------------:|
| unet_bbox_s42                             | all      |      3066 |           0.742 |
| unet_bbox_s42                             | primary  |       567 |           0.736 |
| unet_bbox_s42                             | seen     |      2494 |           0.743 |
| unet_pixel_eps8_morph_s42                 | all      |      3066 |           0.712 |
| unet_pixel_eps8_morph_s42                 | primary  |       567 |           0.712 |
| unet_pixel_eps8_morph_s42                 | seen     |      2494 |           0.712 |
| unet_glyph_v2_s42                         | all      |      3066 |           0.703 |
| unet_glyph_v2_s42                         | primary  |       567 |           0.694 |
| unet_glyph_v2_s42                         | seen     |      2494 |           0.705 |
| aiforge-tiles-yolo11s-img1024-conserv-s42 | all      |      3066 |           0.912 |
| aiforge-tiles-yolo11s-img1024-conserv-s42 | primary  |       567 |           0.915 |
| aiforge-tiles-yolo11s-img1024-conserv-s42 | seen     |      2494 |           0.911 |

Yöntem: `phase3_v2/v2_eval.py` (prep → infer → eval; sızıntı kuralı koşmadan önce sabitlendi), `v2_report.py`, `v2_marker_split.py`, kuyruk `run_v2.ps1`. GT varyantları v1 ile birebir (`make_gt.process_pair`). Veri: tam paket (authentic dahil) Google Drive'dan; CC BY-NC-SA 4.0, yeniden dağıtılmaz.
