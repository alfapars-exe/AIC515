# Literatür baseline'ları — lisans ve kullanım koşulları

Tarih: 20 Ekim 2026. Kaynaklar GitHub API `zipball` ile indirildi (git yerel olarak kurulu değil); commit SHA'ları arşiv kök adından alındı.

## 1. TruFor (Guillaro ve ark., CVPR 2023)

| Alan | Değer |
|---|---|
| Repo | https://github.com/grip-unina/TruFor — branch `main`, commit `ae54475` (son push 2025-05-29) |
| GitHub API `license` alanı | `null` (kök dizinde SPDX ile tanınan bir LICENSE dosyası yok) |
| Lisans dosyaları (repo içinde) | `test_docker/LICENSE.txt` ve `TruFor_train_test/LICENSE.txt` (aynı metin, GRIP-UNINA özel lisansı); `LICENSE_CMX.txt` (MIT, © 2022 Huayao Liu — CMX backbone kodu için) |
| Lisans özeti (GRIP-UNINA) | (i) yalnızca bilgilendirme ve kâr amacı gütmeyen (nonprofit) kullanım; endüstriyel/ticari kullanım açıkça yasak; (ii) tüm telif/özgün bildirimler korunmalı; (iii) sonuçlar kamuya açıklandığında özgün yazarlara atıf zorunlu. Garanti yok ("as-is"). Kaynak dosya başlıklarında ayrıca http://www.grip.unina.it/download/LICENSE_OPEN.txt adresine atıf var. |
| Ağırlıklar | `TruFor_weights.zip` → `weights/trufor.pth.tar`; https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip ; indirilen boyut 260,878,690 B; MD5 `7bee48f3476c75616c3c5721ab256ff8` (README'deki değerle **birebir eşleşti**). README: "By downloading and/or using any of these files, you implicitly agree to all the terms of the license" → ağırlıklar da aynı nonprofit lisansa tabidir. |
| Kullanılan kod yolu | `test_docker/src/` (trufor_test.py, config.py, trufor.yaml, data_core.py, models/) — Docker yerine doğrudan Python ile çağrıldı. |
| AIForge-Doc v1 (CC BY-NC-SA 4.0, gated) ile uyum | Uyumlu: her iki taraf da ticari olmayan kullanımla sınırlı; bu çalışma ders/akademik amaçlıdır. Üretilen olasılık haritaları veri setinin türev işi sayılır → proje klasörü dışına kopyalanmaz, yayınlanmaz (yalnız özet metrikler/figürler paylaşılır). Bildiri/raporda TruFor atfı zorunlu (lisans md. iii). |

## 2. ADCD-Net (Wong ve ark., ICCV 2025)

| Alan | Değer |
|---|---|
| Repo | https://github.com/KAHIMWONG/ACDC-Net (GitHub yönlendirmesi: KahimWong/ADCD-Net) — branch `master`, commit `169d04d` (son push 2026-10-17) |
| Lisans | **MIT** (`LICENSE`, © 2025 KahimWong; GitHub API `license.spdx_id = MIT`) |
| Ağırlıklar | README'deki Google Drive arşivi (id `1-5BU3Bavs6SGJpaByua_FhDuUJGoo-iS`) → `ADCD-Net_exp_data.zip` 612,670,649 B; içerik: `ADCDNet.pth` (278,747,570 B), `docres.pkl` (183,260,711 B), `DocTamperOCR/` (bu çalışmada kullanılmadı, açılmadı). Arşiv veya README'de ağırlıklar için **ayrı bir lisans beyanı yok**; repo MIT olduğu için ağırlıkların da yazar tarafından aynı koşullarla dağıtıldığı varsayıldı (açık beyan: YOK). |
| Üçüncü taraf bileşenler | `docres.pkl` = DocRes (Zhang ve ark., CVPR 2024) backbone ağırlığı — DocRes'in kendi lisansı burada **doğrulanmadı**. `model/fph.py` `efficientnet_pytorch` (Apache-2.0) kütüphanesini kullanır. Model DocTamper (Qu ve ark., CVPR 2023) eğitim kümesiyle eğitilmiştir; DocTamper veri lisansı burada doğrulanmadı. |
| OCR maskesi aracı | Repo `seg_char.py` ile **PaddleOCR** (PP-OCRv5_server_det) kullanır (görev metnindeki "CRAFT" ifadesi repo ile uyuşmuyor). Bu çalışmada PaddleOCR kurulmadı; yerine ikilileştirme (binarization) tabanlı metin maskesi proxy'si ve boş maske koşulu kullanıldı (README'ye bakınız). |
| AIForge-Doc v1 (CC BY-NC-SA 4.0) ile uyum | Uyumlu: MIT kod/ağırlıklar, ticari olmayan akademik kullanım. Türev olasılık haritaları için TruFor ile aynı kısıt geçerlidir (proje klasörü dışına çıkarılmaz). Bildiride ADCD-Net atfı yapılacak. |

## 3. Veri seti

AIForge-Doc v1 (HF: Scam-AI/AIForge-Doc-v1): **CC BY-NC-SA 4.0**, gated. Görüntüler ve türev maskeler yalnızca `Vize Proje\` altında tutulur; artifact olarak yalnız küçük tablolar/figürler/kod kaydedilir.

## 4. Ortam notu

`docforensics` ortamına pip ile eklenen paketler: requests, gdown, torch/torchvision (cu128 index), timm, yacs, efficientnet_pytorch, jpeglib/jpeg2dct denemeleri (ayrıntı README.md). TruFor'un Docker tabanı (pytorch 1.11 / cuda 11.3) kullanılmadı; README'de belirtildiği gibi farklı torch/cuda sürümünde skorlar küçük ölçüde değişebilir.
