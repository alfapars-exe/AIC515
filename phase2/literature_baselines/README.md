# Literatür baseline'ları — TruFor ve ADCD-Net zero-shot (AIForge-Doc v1)

Tarih: 20 Ekim 2026 · Donanım: RTX 5080 16 GB (sandbox) · Ortam: `docforensics` (Python 3.11, torch 2.11.0+cu128, torchvision 0.26.0+cu128).
Lisans/kullanım koşulları: [LICENSES.md](LICENSES.md). Tüm sayılar bu klasördeki `zero_shot_results.csv` / `zero_shot_per_image.csv` dosyalarından okunmuştur.

## 1. Kurulum (yapılanlar)

```text
# git yerel olarak yok -> GitHub API zipball (credential GitHub)
grip-unina/TruFor  @ main  ae54475  -> TruFor/           (124 MB arşiv; test_docker/src kullanıldı)
KAHIMWONG/ACDC-Net @ master 169d04d -> ADCD-Net/         (MIT)
# ağırlıklar
TruFor:   https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip  (260,878,690 B, MD5 7bee48f3476c75616c3c5721ab256ff8 = README) -> TruFor/test_docker/weights/trufor.pth.tar
ADCD-Net: gdown id 1-5BU3Bavs6SGJpaByua_FhDuUJGoo-iS -> ADCD-Net/weights/ADCD-Net_exp_data.zip (612,670,649 B); açılan: ADCDNet.pth (278.7 MB, step 200000), docres.pkl (183.3 MB)
          (DocTamperOCR/ maskeleri açılmadı; bu veri setinde kullanılamaz)
# pip (docforensics; torch sürümü requirements'lardan BAĞIMSIZ kuruldu, değiştirilmedi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # 2.8 GB, 140 s
pip install requests gdown timm yacs efficientnet_pytorch einops scikit-learn jpeglib pypdf
```
Repo requirements'ları kurulmadı (TruFor Docker: torch 1.11/cuda 11.3; ADCD-Net: torch 2.3.0+cu121, albumentations 2.0.8) — RTX 5080 (sm_120) bu sürümlerle çalışmaz. Gerekli modüller tek tek eklendi; `jpeg2dct` yerine **jpeglib 1.0.2** kullanıldı (aynı kuantize DCT katsayıları; `normalized=False` eşdeğeri). Ortam bozulmadı.
Allowlist onayları: www.grip.unina.it, drive.google.com, doc-08-1g-docs.googleusercontent.com.

## 2. Veri ve GT

- Val: `splits.json['val']` 487 görüntü (TrainingSet). Test: resmî 812 (TestingSet); "temiz 740" = `nn_train_any_dist` NaN veya >20.
- GT: maske PNG'leri (0/255, tek dikdörtgen bbox). **Dikkat:** TrainingSet ve TestingSet id'leri aynı 9 haneli uzayı paylaşır (ör. `000000012` ikisinde de var) → tahminler `preds/<model>/<split>/<id>.png` altında tutulur. İlk koşuda düz klasör kullanılmış, çakışma fark edilince koşular durdurulup yeniden yapıldı; geçersiz çıktılar `preds/_INVALID_flat_id_collision/` altında (silinebilir, değerlendirmede kullanılmadı).
- Alan dilimleri (test): <1 %: 429, 1–5 %: 381, 5–20 %: 2. Araç: gemini-nano (Gemini 2.5 Flash Image) 727, qwen-inpaint (Ideogram v2 Edit, kart beyanı) 85.

## 3. Çıkarım koşulları

| Koşul | Girdi | Not |
|---|---|---|
| `trufor` | Tam RGB; H·W > 2 MP ise en-boy korunarak H·W = 2 MP'ye küçült (bilinear), haritayı orijinal boyuta bilinear büyüt (DOCFORGE-BENCH protokolü). 302/1299 görüntü küçültüldü (test: 197/812; min ölçek 0.240). | Resmî `trufor_test.py` mantığı (`/256`, softmax sınıf-1), tile yok. Çıktı: olasılık haritası + güven haritası (`trufor_conf/`) + image-level skor. |
| `trufor_1024pad` | AIForge-Doc makalesi §5.1 protokolü: en uzun kenar 1024, sıfır (siyah) dolgu ile 1024², tahmin içerik bölgesine kırpılıp orijinale büyütülür. | Makalenin dolgu rengi/konumu belirtilmemiş; sol-üst hizalı siyah dolgu varsayıldı. |
| `adcd_q100_ocrproxy` | Repo cross-domain yolu (`GeneralValDs`+`pad_collate`): BGR oku → LongestMaxSize 512 → gri ('L') → **JPEG Q=100** → RGB(3×gri) + ImageNet normalizasyonu; DCT = o JPEG'in kuantize luma katsayıları (clip 0..20); qt = JPEG'in luma tablosu (Q100'de tümü 1); kare 16-katı dolgu. OCR maskesi: Otsu ikilileştirme + morfolojik kapama/genişletme ile metin-bölgesi **proxy'si** (ort. kapsama %32). | Repo'nun OCR aracı PaddleOCR (PP-OCRv5_server_det); kurulmadı. Görev metnindeki "CRAFT" repo ile uyuşmuyor. |
| `adcd_q95_ocrproxy` | Aynı, **JPEG Q=95** (görev şartı). | qt tablosu Q95 standart luma tablosu (jpeglib'den okundu). |
| `adcd_q100_noaln` | Q100, hizalama skoru **ŝ_aln = 0** zorlandı → DCT dalı katkısı (toplamsal gating) kapalı. | Model forward'ı yeniden yazılarak (`run_adcd.py::forward`) uygulandı. |
| `adcd_q100_emptyocr` | Q100, **boş OCR maskesi** (pristine prototype tüm görüntüden). | Proxy maskenin etkisini ölçmek için. |

ADCD-Net için tüm görüntüler 512 piksel uzun kenara küçültülür (1299'un %80'i küçültüldü); harita orijinal boyuta bilinear büyütülür. Bu, repo'nun kendi cross-domain protokolüdür (`cfg.val_max_size=512`).

Çalışma süreleri / GPU belleği (`timings.json`): TruFor 2 MP: 1299 görüntü 574 s duvar süresi, çıkarım medyanı 0.22 s/görüntü (ilk çağrı 15 s cuDNN ısınma), peak 4.36 GB. TruFor 1024²: 528 s, peak 2.41 GB. ADCD-Net 4 koşul tek geçişte: 1083 s (I/O ağırlıklı; çıkarım ≈0.12–0.16 s/koşul/görüntü), peak 1.82 GB. Değerlendirme (6 koşul × 1299 harita): 83 s.

## 4. Metrik tanımları (`evaluate_zero_shot.py`)

uint8 haritalardan 256-kutulu (pozitif/negatif) histogramlarla **kesin** hesaplanır (sklearn ile sentetik veride birebir doğrulandı): eşik "0.5" = u8 ≥ 128.
- F1@0.5, IoU@0.5: görüntü başına, ortalama. Oracle-F1/IoU: görüntü başına 256 eşik üzerinden en iyi. "F1@val-eşik": val'de ortalama-F1'i maksimize eden tek eşik test'e uygulanır.
- Px-AUROC/AUPRC "havuz, kesin": tüm pikseller (test ≈ 2.5 G piksel) havuzlanır. "görüntü ort.": görüntü başına AUROC/AP ortalaması. "≤50k neg/img": görev protokolü (tüm pozitifler + görüntü başına ≤50k rastgele negatif) — sınıf oranını değiştirdiği için AUPRC'yi şişirir (ör. TruFor test 0.741 vs kesin 0.080); yalnızca karşılaştırma için.
- Image-level: test-authentic geçersiz olduğundan AUC **hesaplanamadı**; yalnız forged görüntülerde skor dağılımı verilir.

## 5. Sonuçlar — ana tablo

| Model/koşul | split | alt küme | n | Px-AUROC (havuz, kesin) | Px-AUROC (görüntü ort.) | Px-AUROC (≤50k neg/img) | Px-AUPRC (havuz, kesin) | Px-AUPRC (≤50k neg/img) | F1@0.5 | IoU@0.5 | Oracle-F1 | Oracle-IoU | val eşiği | F1@val-eşik | tahmin + oranı |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TruFor (≤2 MP native) | val | all | 487 | 0.839 | 0.912 | 0.924 | 0.088 | 0.771 | 0.426 | 0.350 | 0.478 | 0.401 | — | — | 0.040 |
| TruFor (≤2 MP native) | test | full | 812 | 0.822 | 0.903 | 0.913 | 0.080 | 0.741 | 0.403 | 0.328 | 0.456 | 0.379 | 0.447 | 0.405 | 0.037 |
| TruFor (≤2 MP native) | test | clean | 740 | 0.828 | 0.905 | 0.913 | 0.084 | 0.741 | 0.401 | 0.326 | 0.455 | 0.377 | 0.447 | 0.403 | 0.037 |
| TruFor (1024² letterbox, makale protokolü) | val | all | 487 | 0.717 | 0.763 | 0.684 | 0.021 | 0.374 | 0.126 | 0.091 | 0.215 | 0.161 | — | — | 0.159 |
| TruFor (1024² letterbox, makale protokolü) | test | full | 812 | 0.749 | 0.779 | 0.709 | 0.028 | 0.416 | 0.134 | 0.100 | 0.216 | 0.161 | 0.753 | 0.137 | 0.161 |
| TruFor (1024² letterbox, makale protokolü) | test | clean | 740 | 0.758 | 0.782 | 0.705 | 0.030 | 0.413 | 0.133 | 0.099 | 0.218 | 0.163 | 0.753 | 0.137 | 0.169 |
| ADCD-Net Q100 + OCR-proxy | val | all | 487 | 0.821 | 0.862 | 0.814 | 0.065 | 0.635 | 0.190 | 0.157 | 0.332 | 0.265 | — | — | 0.004 |
| ADCD-Net Q100 + OCR-proxy | test | full | 812 | 0.860 | 0.868 | 0.856 | 0.130 | 0.699 | 0.182 | 0.149 | 0.328 | 0.256 | 0.169 | 0.244 | 0.004 |
| ADCD-Net Q100 + OCR-proxy | test | clean | 740 | 0.855 | 0.869 | 0.854 | 0.133 | 0.695 | 0.189 | 0.155 | 0.333 | 0.263 | 0.169 | 0.250 | 0.004 |
| ADCD-Net Q95 + OCR-proxy | val | all | 487 | 0.823 | 0.842 | 0.811 | 0.061 | 0.626 | 0.121 | 0.097 | 0.275 | 0.204 | — | — | 0.003 |
| ADCD-Net Q95 + OCR-proxy | test | full | 812 | 0.859 | 0.851 | 0.848 | 0.090 | 0.675 | 0.114 | 0.089 | 0.269 | 0.195 | 0.165 | 0.177 | 0.003 |
| ADCD-Net Q95 + OCR-proxy | test | clean | 740 | 0.855 | 0.850 | 0.847 | 0.092 | 0.672 | 0.117 | 0.091 | 0.272 | 0.198 | 0.165 | 0.179 | 0.003 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | val | all | 487 | 0.787 | 0.831 | 0.804 | 0.044 | 0.624 | 0.115 | 0.091 | 0.262 | 0.194 | — | — | 0.003 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | test | full | 812 | 0.830 | 0.834 | 0.846 | 0.114 | 0.694 | 0.100 | 0.079 | 0.247 | 0.178 | 0.153 | 0.155 | 0.003 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | test | clean | 740 | 0.826 | 0.834 | 0.845 | 0.123 | 0.691 | 0.101 | 0.079 | 0.246 | 0.178 | 0.153 | 0.156 | 0.003 |
| ADCD-Net Q100, boş OCR maskesi | val | all | 487 | 0.816 | 0.857 | 0.809 | 0.066 | 0.634 | 0.187 | 0.154 | 0.332 | 0.265 | — | — | 0.004 |
| ADCD-Net Q100, boş OCR maskesi | test | full | 812 | 0.855 | 0.864 | 0.850 | 0.131 | 0.697 | 0.179 | 0.147 | 0.328 | 0.257 | 0.161 | 0.243 | 0.004 |
| ADCD-Net Q100, boş OCR maskesi | test | clean | 740 | 0.850 | 0.865 | 0.848 | 0.135 | 0.692 | 0.186 | 0.153 | 0.333 | 0.263 | 0.161 | 0.250 | 0.004 |

## 6. Alan dilimi kırılımı (test, tam 812)

| Model/koşul | alt küme (test, tam 812) | n | Px-AUROC (havuz) | Px-AUPRC (havuz) | F1@0.5 | IoU@0.5 | Oracle-F1 | F1@val-eşik |
|---|---|---|---|---|---|---|---|---|
| TruFor (≤2 MP native) | area=1-5% | 381 | 0.964 | 0.527 | 0.652 | 0.538 | 0.721 | 0.657 |
| TruFor (≤2 MP native) | area=5-20% | 2 | 0.999 | 0.990 | 0.931 | 0.872 | 0.948 | 0.938 |
| TruFor (≤2 MP native) | area=<1% | 429 | 0.709 | 0.014 | 0.179 | 0.139 | 0.218 | 0.178 |
| TruFor (1024² letterbox, makale protokolü) | area=1-5% | 381 | 0.778 | 0.142 | 0.215 | 0.163 | 0.325 | 0.216 |
| TruFor (1024² letterbox, makale protokolü) | area=5-20% | 2 | 0.998 | 0.981 | 0.902 | 0.822 | 0.934 | 0.835 |
| TruFor (1024² letterbox, makale protokolü) | area=<1% | 429 | 0.684 | 0.007 | 0.058 | 0.041 | 0.116 | 0.064 |
| ADCD-Net Q100 + OCR-proxy | area=1-5% | 381 | 0.801 | 0.225 | 0.220 | 0.182 | 0.382 | 0.309 |
| ADCD-Net Q100 + OCR-proxy | area=5-20% | 2 | 0.594 | 0.109 | 0.000 | 0.000 | 0.151 | 0.055 |
| ADCD-Net Q100 + OCR-proxy | area=<1% | 429 | 0.903 | 0.063 | 0.148 | 0.121 | 0.281 | 0.186 |
| ADCD-Net Q95 + OCR-proxy | area=1-5% | 381 | 0.793 | 0.147 | 0.100 | 0.074 | 0.281 | 0.198 |
| ADCD-Net Q95 + OCR-proxy | area=5-20% | 2 | 0.583 | 0.099 | 0.014 | 0.007 | 0.144 | 0.037 |
| ADCD-Net Q95 + OCR-proxy | area=<1% | 429 | 0.906 | 0.062 | 0.127 | 0.103 | 0.260 | 0.159 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | area=1-5% | 381 | 0.791 | 0.244 | 0.099 | 0.076 | 0.264 | 0.180 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | area=5-20% | 2 | 0.634 | 0.191 | 0.075 | 0.041 | 0.196 | 0.111 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | area=<1% | 429 | 0.882 | 0.032 | 0.102 | 0.081 | 0.231 | 0.132 |
| ADCD-Net Q100, boş OCR maskesi | area=1-5% | 381 | 0.792 | 0.226 | 0.216 | 0.179 | 0.381 | 0.310 |
| ADCD-Net Q100, boş OCR maskesi | area=5-20% | 2 | 0.588 | 0.107 | 0.000 | 0.000 | 0.148 | 0.056 |
| ADCD-Net Q100, boş OCR maskesi | area=<1% | 429 | 0.902 | 0.066 | 0.147 | 0.120 | 0.281 | 0.184 |

## 7. Araç kırılımı (test, tam 812)

| Model/koşul | alt küme (test, tam 812) | n | Px-AUROC (havuz) | Px-AUPRC (havuz) | F1@0.5 | IoU@0.5 | Oracle-F1 | F1@val-eşik |
|---|---|---|---|---|---|---|---|---|
| TruFor (≤2 MP native) | tool=gemini-nano | 727 | 0.838 | 0.090 | 0.428 | 0.349 | 0.484 | 0.430 |
| TruFor (≤2 MP native) | tool=qwen-inpaint | 85 | 0.701 | 0.020 | 0.183 | 0.145 | 0.217 | 0.185 |
| TruFor (1024² letterbox, makale protokolü) | tool=gemini-nano | 727 | 0.747 | 0.032 | 0.143 | 0.107 | 0.231 | 0.147 |
| TruFor (1024² letterbox, makale protokolü) | tool=qwen-inpaint | 85 | 0.727 | 0.007 | 0.054 | 0.036 | 0.085 | 0.046 |
| ADCD-Net Q100 + OCR-proxy | tool=gemini-nano | 727 | 0.861 | 0.143 | 0.196 | 0.161 | 0.349 | 0.262 |
| ADCD-Net Q100 + OCR-proxy | tool=qwen-inpaint | 85 | 0.877 | 0.047 | 0.055 | 0.044 | 0.150 | 0.088 |
| ADCD-Net Q95 + OCR-proxy | tool=gemini-nano | 727 | 0.860 | 0.100 | 0.119 | 0.093 | 0.282 | 0.188 |
| ADCD-Net Q95 + OCR-proxy | tool=qwen-inpaint | 85 | 0.876 | 0.045 | 0.072 | 0.056 | 0.163 | 0.082 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | tool=gemini-nano | 727 | 0.832 | 0.131 | 0.107 | 0.084 | 0.260 | 0.165 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | tool=qwen-inpaint | 85 | 0.851 | 0.014 | 0.042 | 0.031 | 0.132 | 0.071 |
| ADCD-Net Q100, boş OCR maskesi | tool=gemini-nano | 727 | 0.855 | 0.144 | 0.194 | 0.159 | 0.348 | 0.261 |
| ADCD-Net Q100, boş OCR maskesi | tool=qwen-inpaint | 85 | 0.876 | 0.047 | 0.054 | 0.043 | 0.150 | 0.089 |

## 8. Temiz 740 alt kümeleri

| Model/koşul | alt küme (test, temiz 740) | n | Px-AUROC (havuz) | Px-AUPRC (havuz) | F1@0.5 | IoU@0.5 | Oracle-F1 | F1@val-eşik |
|---|---|---|---|---|---|---|---|---|
| TruFor (≤2 MP native) | clean&tool=gemini-nano | 662 | 0.843 | 0.096 | 0.430 | 0.350 | 0.485 | 0.431 |
| TruFor (≤2 MP native) | clean&tool=qwen-inpaint | 78 | 0.707 | 0.016 | 0.159 | 0.124 | 0.193 | 0.161 |
| TruFor (≤2 MP native) | clean&area=1-5% | 350 | 0.961 | 0.513 | 0.644 | 0.530 | 0.713 | 0.649 |
| TruFor (≤2 MP native) | clean&area=5-20% | 2 | 0.999 | 0.990 | 0.931 | 0.872 | 0.948 | 0.938 |
| TruFor (≤2 MP native) | clean&area=<1% | 388 | 0.716 | 0.014 | 0.179 | 0.139 | 0.219 | 0.178 |
| TruFor (1024² letterbox, makale protokolü) | clean&tool=gemini-nano | 662 | 0.756 | 0.034 | 0.143 | 0.107 | 0.234 | 0.149 |
| TruFor (1024² letterbox, makale protokolü) | clean&tool=qwen-inpaint | 78 | 0.743 | 0.007 | 0.048 | 0.032 | 0.079 | 0.041 |
| TruFor (1024² letterbox, makale protokolü) | clean&area=1-5% | 350 | 0.779 | 0.141 | 0.213 | 0.160 | 0.326 | 0.214 |
| TruFor (1024² letterbox, makale protokolü) | clean&area=5-20% | 2 | 0.998 | 0.981 | 0.902 | 0.822 | 0.934 | 0.835 |
| TruFor (1024² letterbox, makale protokolü) | clean&area=<1% | 388 | 0.691 | 0.007 | 0.057 | 0.040 | 0.116 | 0.064 |
| ADCD-Net Q100 + OCR-proxy | clean&tool=gemini-nano | 662 | 0.855 | 0.146 | 0.204 | 0.168 | 0.354 | 0.269 |
| ADCD-Net Q100 + OCR-proxy | clean&tool=qwen-inpaint | 78 | 0.885 | 0.051 | 0.060 | 0.048 | 0.156 | 0.092 |
| ADCD-Net Q100 + OCR-proxy | clean&area=1-5% | 350 | 0.802 | 0.235 | 0.235 | 0.195 | 0.396 | 0.325 |
| ADCD-Net Q100 + OCR-proxy | clean&area=5-20% | 2 | 0.594 | 0.109 | 0.000 | 0.000 | 0.151 | 0.055 |
| ADCD-Net Q100 + OCR-proxy | clean&area=<1% | 388 | 0.900 | 0.058 | 0.148 | 0.120 | 0.277 | 0.184 |
| ADCD-Net Q95 + OCR-proxy | clean&tool=gemini-nano | 662 | 0.855 | 0.102 | 0.122 | 0.095 | 0.284 | 0.190 |
| ADCD-Net Q95 + OCR-proxy | clean&tool=qwen-inpaint | 78 | 0.884 | 0.049 | 0.078 | 0.061 | 0.169 | 0.084 |
| ADCD-Net Q95 + OCR-proxy | clean&area=1-5% | 350 | 0.795 | 0.153 | 0.106 | 0.079 | 0.290 | 0.207 |
| ADCD-Net Q95 + OCR-proxy | clean&area=5-20% | 2 | 0.583 | 0.099 | 0.014 | 0.007 | 0.144 | 0.037 |
| ADCD-Net Q95 + OCR-proxy | clean&area=<1% | 388 | 0.903 | 0.061 | 0.128 | 0.103 | 0.256 | 0.154 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | clean&tool=gemini-nano | 662 | 0.828 | 0.140 | 0.108 | 0.085 | 0.259 | 0.166 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | clean&tool=qwen-inpaint | 78 | 0.858 | 0.015 | 0.045 | 0.034 | 0.138 | 0.074 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | clean&area=1-5% | 350 | 0.794 | 0.256 | 0.105 | 0.081 | 0.270 | 0.187 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | clean&area=5-20% | 2 | 0.634 | 0.191 | 0.075 | 0.041 | 0.196 | 0.111 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | clean&area=<1% | 388 | 0.879 | 0.031 | 0.098 | 0.078 | 0.225 | 0.129 |
| ADCD-Net Q100, boş OCR maskesi | clean&tool=gemini-nano | 662 | 0.849 | 0.147 | 0.201 | 0.166 | 0.354 | 0.269 |
| ADCD-Net Q100, boş OCR maskesi | clean&tool=qwen-inpaint | 78 | 0.885 | 0.053 | 0.058 | 0.047 | 0.156 | 0.092 |
| ADCD-Net Q100, boş OCR maskesi | clean&area=1-5% | 350 | 0.793 | 0.236 | 0.232 | 0.192 | 0.396 | 0.326 |
| ADCD-Net Q100, boş OCR maskesi | clean&area=5-20% | 2 | 0.588 | 0.107 | 0.000 | 0.000 | 0.148 | 0.056 |
| ADCD-Net Q100, boş OCR maskesi | clean&area=<1% | 388 | 0.898 | 0.061 | 0.147 | 0.119 | 0.277 | 0.183 |

## 9. Val (487) kırılımları

| Model/koşul | alt küme (val 487) | n | Px-AUROC (havuz) | Px-AUPRC (havuz) | F1@0.5 | IoU@0.5 | Oracle-F1 | F1@val-eşik (=kendi) |
|---|---|---|---|---|---|---|---|---|
| TruFor (≤2 MP native) | tool=gemini-nano | 446 | 0.861 | 0.098 | 0.454 | 0.373 | 0.509 | 0.455 |
| TruFor (≤2 MP native) | tool=qwen-inpaint | 41 | 0.619 | 0.016 | 0.113 | 0.091 | 0.146 | 0.115 |
| TruFor (≤2 MP native) | area=1-5% | 254 | 0.973 | 0.669 | 0.693 | 0.577 | 0.751 | 0.696 |
| TruFor (≤2 MP native) | area=5-20% | 2 | 0.963 | 0.861 | 0.767 | 0.623 | 0.774 | 0.766 |
| TruFor (≤2 MP native) | area=<1% | 231 | 0.648 | 0.007 | 0.128 | 0.098 | 0.176 | 0.127 |
| TruFor (1024² letterbox, makale protokolü) | tool=gemini-nano | 446 | 0.719 | 0.023 | 0.134 | 0.097 | 0.229 | 0.142 |
| TruFor (1024² letterbox, makale protokolü) | tool=qwen-inpaint | 41 | 0.671 | 0.005 | 0.042 | 0.026 | 0.063 | 0.023 |
| TruFor (1024² letterbox, makale protokolü) | area=1-5% | 254 | 0.768 | 0.105 | 0.195 | 0.141 | 0.316 | 0.202 |
| TruFor (1024² letterbox, makale protokolü) | area=5-20% | 2 | 0.927 | 0.621 | 0.280 | 0.178 | 0.430 | 0.304 |
| TruFor (1024² letterbox, makale protokolü) | area=<1% | 231 | 0.625 | 0.004 | 0.049 | 0.036 | 0.103 | 0.054 |
| ADCD-Net Q100 + OCR-proxy | tool=gemini-nano | 446 | 0.818 | 0.071 | 0.202 | 0.167 | 0.349 | 0.267 |
| ADCD-Net Q100 + OCR-proxy | tool=qwen-inpaint | 41 | 0.864 | 0.036 | 0.052 | 0.041 | 0.151 | 0.086 |
| ADCD-Net Q100 + OCR-proxy | area=1-5% | 254 | 0.752 | 0.112 | 0.224 | 0.185 | 0.371 | 0.307 |
| ADCD-Net Q100 + OCR-proxy | area=5-20% | 2 | 0.738 | 0.316 | 0.420 | 0.361 | 0.584 | 0.476 |
| ADCD-Net Q100 + OCR-proxy | area=<1% | 231 | 0.898 | 0.041 | 0.150 | 0.124 | 0.287 | 0.189 |
| ADCD-Net Q95 + OCR-proxy | tool=gemini-nano | 446 | 0.821 | 0.067 | 0.126 | 0.102 | 0.287 | 0.198 |
| ADCD-Net Q95 + OCR-proxy | tool=qwen-inpaint | 41 | 0.860 | 0.028 | 0.058 | 0.048 | 0.150 | 0.088 |
| ADCD-Net Q95 + OCR-proxy | area=1-5% | 254 | 0.747 | 0.101 | 0.120 | 0.092 | 0.285 | 0.213 |
| ADCD-Net Q95 + OCR-proxy | area=5-20% | 2 | 0.690 | 0.145 | 0.067 | 0.036 | 0.395 | 0.273 |
| ADCD-Net Q95 + OCR-proxy | area=<1% | 231 | 0.907 | 0.038 | 0.122 | 0.103 | 0.263 | 0.161 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | tool=gemini-nano | 446 | 0.783 | 0.048 | 0.122 | 0.096 | 0.273 | 0.181 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | tool=qwen-inpaint | 41 | 0.857 | 0.028 | 0.040 | 0.032 | 0.140 | 0.075 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | area=1-5% | 254 | 0.735 | 0.101 | 0.108 | 0.082 | 0.269 | 0.196 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | area=5-20% | 2 | 0.710 | 0.195 | 0.112 | 0.063 | 0.384 | 0.372 |
| ADCD-Net Q100, ŝ_aln=0 (DCT kapalı) | area=<1% | 231 | 0.889 | 0.026 | 0.123 | 0.100 | 0.253 | 0.144 |
| ADCD-Net Q100, boş OCR maskesi | tool=gemini-nano | 446 | 0.812 | 0.072 | 0.199 | 0.165 | 0.348 | 0.267 |
| ADCD-Net Q100, boş OCR maskesi | tool=qwen-inpaint | 41 | 0.863 | 0.038 | 0.051 | 0.041 | 0.150 | 0.084 |
| ADCD-Net Q100, boş OCR maskesi | area=1-5% | 254 | 0.745 | 0.113 | 0.220 | 0.181 | 0.371 | 0.307 |
| ADCD-Net Q100, boş OCR maskesi | area=5-20% | 2 | 0.727 | 0.311 | 0.421 | 0.363 | 0.576 | 0.479 |
| ADCD-Net Q100, boş OCR maskesi | area=<1% | 231 | 0.898 | 0.043 | 0.149 | 0.123 | 0.287 | 0.189 |

## 10. TruFor image-level skor (yalnız forged görüntüler; negatif yok)

TruFor ≤2 MP:

| split | alt küme | n | ort. | q05 | q25 | medyan | q75 | q95 | oran ≥0.5 |
|---|---|---|---|---|---|---|---|---|---|
| val | all | 487 | 0.754 | 0.087 | 0.566 | 0.948 | 0.997 | 1.000 | 0.766 |
| val | tool=gemini-nano | 446 | 0.783 | 0.103 | 0.638 | 0.960 | 0.997 | 1.000 | 0.803 |
| val | tool=qwen-inpaint | 41 | 0.439 | 0.051 | 0.102 | 0.203 | 0.954 | 0.999 | 0.366 |
| val | area=1-5% | 254 | 0.821 | 0.324 | 0.721 | 0.940 | 0.996 | 1.000 | 0.854 |
| val | area=5-20% | 2 | 0.985 | 0.981 | 0.983 | 0.985 | 0.987 | 0.988 | 1.000 |
| val | area=<1% | 231 | 0.679 | 0.071 | 0.210 | 0.949 | 0.998 | 1.000 | 0.667 |
| test | all | 812 | 0.721 | 0.089 | 0.431 | 0.921 | 0.997 | 1.000 | 0.728 |
| test | clean | 740 | 0.713 | 0.081 | 0.420 | 0.914 | 0.997 | 1.000 | 0.720 |
| test | tool=gemini-nano | 727 | 0.753 | 0.124 | 0.536 | 0.940 | 0.997 | 1.000 | 0.768 |
| test | tool=qwen-inpaint | 85 | 0.449 | 0.045 | 0.094 | 0.222 | 0.976 | 0.999 | 0.388 |
| test | area=1-5% | 381 | 0.784 | 0.237 | 0.592 | 0.925 | 0.996 | 1.000 | 0.816 |
| test | area=5-20% | 2 | 0.995 | 0.993 | 0.994 | 0.995 | 0.997 | 0.998 | 1.000 |
| test | area=<1% | 429 | 0.664 | 0.059 | 0.223 | 0.913 | 0.997 | 1.000 | 0.648 |

TruFor 1024² letterbox:

| split | alt küme | n | ort. | medyan | oran ≥0.5 |
|---|---|---|---|---|---|
| val | all | 487 | 0.590 | 0.568 | 0.552 |
| test | all | 812 | 0.578 | 0.544 | 0.541 |
| test | clean | 740 | 0.573 | 0.542 | 0.541 |

## 11. ADCD-Net hizalama skoru ŝ_aln (DCT dalı gating'i)

| koşul | split | n | ort. | q05 | medyan | q95 | oran ≥0.5 |
|---|---|---|---|---|---|---|---|
| q100_ocrproxy | val | 487 | 0.428 | 0.033 | 0.412 | 0.931 | 0.435 |
| q100_ocrproxy | test | 812 | 0.412 | 0.031 | 0.369 | 0.933 | 0.397 |
| q95_ocrproxy | val | 487 | 0.509 | 0.108 | 0.530 | 0.854 | 0.561 |
| q95_ocrproxy | test | 812 | 0.483 | 0.111 | 0.500 | 0.851 | 0.500 |

Not: ŝ_aln, modelin "DCT ızgarası hizalı mı?" tahminidir; Q100'de ortanca 0.37, Q95'te 0.50. Girdi PNG'den sentetik olarak yeniden kodlandığı için DCT dalının katkısı (ŝ_aln·feat) veri setine özgü bir kodlama kararına bağlı hâle gelir.

## 12. AIForge-Doc makalesi (arXiv 2602.20569, Tablo 2) ile karşılaştırma

| Kaynak | Protokol | IoU | F1 | AUCpx |
|---|---|---|---|---|
| Makale, TruFor zero-shot, test 812 | 1024² aspect-ratio padding, eşik 0.5, "per-image, micro-averaged" | 0.358 | 0.434 | 0.916 |
| Bu çalışma, `trufor` (≤2 MP native), test 812 | eşik 0.5, görüntü-ortalaması | 0.328 | 0.403 | 0.903 (görüntü ort.) / 0.913 (≤50k) / 0.822 (kesin havuz) |
| Bu çalışma, `trufor_1024pad` (makale protokolü replikası), test 812 | eşik 0.5, görüntü-ortalaması | 0.100 | 0.134 | 0.779 (görüntü ort.) |

Yorum (olası nedenler, kanıt sınıfı HYPOTHESIS):
1. Native-çözünürlük koşulumuz makaleye yakın (IoU −0.030, F1 −0.031; görüntü-ortalaması AUROC 0.903 ≈ 0.916). Makalenin AUCpx'inin havuzlanmış değil görüntü-başına (veya alt-örneklenmiş) olduğu anlaşılıyor; kesin havuz AUROC 0.822'dir.
2. Makalenin yazılı protokolünü (1024² letterbox) birebir uyguladığımızda sonuç çok daha düşük (F1 0.134): sıfır dolgu + 1024'e küçültme Noiseprint++ izlerini bozuyor ve pozitif tahmin oranını 4× artırıyor (0.161 vs 0.037). Makalenin dolgu türü/renk ve yeniden-boyutlandırma ayrıntıları belirtilmediğinden fark buradan kaynaklanabilir; makale ayrıca TruFor'u "CLIP-pretrained ViT-L" diye tanımlar, oysa resmî config `mit_b2` SegFormer'dır.
3. Kalan farklar: torch 2.11/cu128 vs makalenin ortamı (TruFor README skorların sürümle değişebileceğini belirtir), eşik/yuvarlama (u8 ≥128), split aynı (resmî 812).
4. ADCD-Net için makalede/literatürde AIForge-Doc sonucu yok; buradaki değerler ilk referanstır.

## 13. Kısıtlamalar ve sapmalar

- ADCD-Net OCR maskesi PaddleOCR yerine ikilileştirme proxy'si; boş-maske koşulu ile fark ihmal edilebilir düzeyde (test F1 0.182 vs 0.179) → PPE modülü bu veri setinde maskeye duyarsız; PaddleOCR'ın sonucu değiştirmesi beklenmez ama **doğrulanmadı**.
- ADCD-Net tasarım gereği 512-uzun-kenar çalışır; <1 % alan diliminde sahte bölge 512 ölçeğinde birkaç piksele düşer. Tile/sliding-window denenmedi (zaman bütçesi).
- TruFor 2 MP sınırı DOCFORGE-BENCH protokolüne göre; 302/1299 görüntü küçültüldü (XFUND A4 taramaları ölçek 0.24'e kadar).
- Image-level AUC hesaplanamadı (test-authentic negatifler geçersiz; audit).
- `≤50k neg/img` AUPRC protokolü sınıf oranını değiştirdiğinden mutlak değerleri yorumlanmamalı; kesin havuz değerleri verilmiştir.
- Oracle-F1 kendi başına üst sınırdır (görüntü başına eşik); F1@val-eşik gerçekçi tek-eşik performansıdır.
- Yalnızca tek koşu (deterministik çıkarım); CI verilmedi.

## 14. Dosyalar

- `LICENSES.md`, `README.md` (bu dosya)
- `run_trufor.py`, `run_trufor_1024pad.py`, `run_adcd.py`, `evaluate_zero_shot.py`
- `zero_shot_results.csv` (koşul × split × alt küme, 108 satır), `zero_shot_per_image.csv` (7794 satır), `trufor_image_scores_summary.csv`, `adcd_alignment_scores_summary.csv`, `timings.json`
- `preds/trufor/<split>/<id>.png`, `preds/trufor_conf/...`, `preds/trufor_1024pad/...`, `preds/adcd_<koşul>/<split>/<id>.png` (uint8 olasılık, orijinal boyut); `preds/trufor_scores.csv`, `preds/trufor_1024pad_scores.csv`, `preds/adcd_scores.csv` (image-level skor, ŝ_aln, süreler)
- `figures/fig_zero_shot_f1_breakdown.png` (alan/araç/temiz kırılımı; alan etkisi TruFor'da baskın, ADCD-Net'te araç farkı [0.14] alan farkından [0.07] büyük), `figures/fig_trufor_score_adcd_align.png`; `figures/fig_qualitative_examples_LOCAL_ONLY.png` (veri seti görüntüleri içerir, artifact yapılmadı)
- `logs/` koşu günlükleri
