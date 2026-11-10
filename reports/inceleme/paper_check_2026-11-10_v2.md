# Paper Check Report — 10 Kasım 2026, ikinci tur (düzeltmelerden sonra)

**Kapsam:** `paper/main.tex` ve `paper/sec/*.tex` (9 dosya), `paper/refs.bib` (79 kayıt, 62'si atıflı), `paper/main.pdf` (9 sayfa, 10 Kasım 2026 21:03 derlemesi), sonuç dosyaları (`phase2/`, `phase3/results`, `phase3_yolo/results`, `phase3_v2/results`), `kod/`, açık depo `github.com/alfapars-exe/AIC515` (HEAD `553fd7d`).
**Yöntem:** Lab Rat `paper-check` (altı kontrol) ve kod yayını denetimi. Bu belge sabahki denetimin (1 HIGH, 6 MEDIUM, 8 LOW) her maddesinin son durumunu ve bu turda bulunan yeni maddeleri verir. Önceki turlar: `paper_check.md` (9 Kasım), `paper_check_faz3.md` (10 Kasım 00:44, 8 sayfalık sürüm).
**Derleme:** 9 sayfa, taşan kutu yok, BibTeX uyarısı yok, tanımsız atıf yok. Bu turdaki düzeltmeler sayfa sayısını ve sayfa geçişlerini değiştirmedi.

## Özet

| | Toplam | Kapandı | Kısmen | Açık |
|---|---|---|---|---|
| HIGH | 2 | 2 | 0 | 0 |
| MEDIUM | 9 | 5 | 3 | 1 |
| LOW | 11 | 7 | 0 | 4 |

Sabahki denetimde 1 HIGH, 6 MEDIUM ve 8 LOW madde vardı; bu turda 1 HIGH, 2 MEDIUM ve 3 LOW madde eklendi, sabahki bir LOW madde ("61 of the 64") MEDIUM'a yükseltildi ve rapordaki "sekiz sayfa" ifadesi LOW listesine alındı. LOW sütunundaki "kapandı" sayısı değişiklik gerektirmeyen bir maddeyi içerir.

Kısmen kapanan ya da açık kalan sekiz maddenin hiçbiri bildirideki bir sayının yanlış olması değildir: altısı karar bekler (depo gönderimi ve lisans, depodaki üç şekil, mekân ve sayfa sınırı, özet uzunluğu, yazar bloğu, Faz 4 cümlesi), biri ek iştir (iki ek tohum), biri yöntem notudur (DBLP). Doğrulanmış bir insan GT denetimi de hâlâ yapılmamıştır (MEDIUM-9).

## HIGH

1. **[Numbers] v2'de "0 of 3,066" (sabah HIGH-1) — kapandı.**
   - Bildiri diyordu: `results_v2.tex` "carry no marker residue (0 of 3{,}066 images in our colour test)".
   - Kaynak: v2 renk kuralı bir görüntüde tetikleniyor (`phase3_v2/v2_table.csv`, `tr_000000726`, CORD 923 numaralı fiş).
   - Yapılan: v1'deki eşli kural v2 çiftlerinin tamamına uygulandı (`kod/phase3_v2/v2_marker_paired_check.py` → `phase3_v2/results/v2_marker_paired_check.json`). 3.066 çiftin hiçbirinde kutu dışında farklı piksel yok; eşli kural tek çiftte tetikleniyor (96 piksel, tamamı kutunun içinde, baştan sona yeşil tonlu bir fiş); 567 çiftlik birincil kümede tetiklenme yok.
   - Yeni metin: "and carry no marker residue: no pixel outside the box differs in any of the 3{,}066 pairs, and our colour rule fires on a single green-tinted receipt, inside its box."
   - Aynı ifade iki yerde daha vardı ve düzeltildi: otomatik tablo `phase3_v2/results/v2_tables.md` ("0.000" yerine "1/3066 görüntü"; üreteç `v2_report.py`) ve uygulamanın Sonuçlar sekmesindeki özet `kod/app/meta/v2_tables.md` (üreteç `v2_space_meta.py`). Canlı Space yeniden dağıtılana kadar eski cümleyi gösterir. Faz 3 kapanış belgesindeki cümle de düzeltildi.

2. **[Claim / Release] Türetilmiş maskelerin yayımlandığı izlenimi (bu tur) — kapandı.**
   - Bildiri diyordu: özet "releasing derived masks and probes, which we provide"; giriş "We release the audit, derived masks, controls, probes and evaluation code". Depo ise yalnız kod, kimlik listeleri ve görüntü başı istatistik içeriyor (README "Veri ve lisans").
   - Yeni metin: özet "we release the code that derives our masks, controls and probes"; giriş "the images and the masks derived from them are not redistributed and are regenerated from the gated dataset by this code" (depo bağlantısı dipnotta); veri bölümü "beyond the examples shown in figures, we redistribute identifiers and per-image statistics only, not images or masks".

## MEDIUM

| # | Madde | Durum | Kanıt ve yapılan |
|---|---|---|---|
| 1 | [Terms] Sonda alt kümesi 380, veri bölümündeki kural 368 veriyor | Kapandı | `probes_p3.py` kalıntı maskesini 1 piksel genişletip ≥20 piksel arıyor (380); genişletmesiz kural 368 (`phase3/results/marker_residue.csv`, doğrulama bölmesi: `n_residue_dil` ≥ 20 → 380, `n_residue` ≥ 20 → 368; Faz 2 ve Faz 4 368 kullanır). `results.tex` §V-C: "($\ge$20\,px after a one-pixel dilation of the residue mask; 368 pairs without the dilation)". Aynı cümleye tohum etiketi eklendi. |
| 2 | [Claim] "lose up to half of their box score" | Kapandı | v1 test → v2 birincil kutu AUPRC kaybı: U-Net kutu %45.3, piksel %64.4, glif %63.5; DeepLabV3+ %43.4, %62.6, %55.1; işaretçisiz kutu U-Net %24.2; karolu YOLO11s %8.5 (`phase3_v2/results/T_v2.csv`, `phase3/results/T2_test.csv`). Yeni metin: "lose 43--64\,\% of their box AUPRC, the marker-repaired box model (24\,\%) and the tiled detector (8\,\%) far less". |
| 3 | [Claim] "the detector was not evaluated on these controls" | Kapandı | Diskteki dedektör çıkarımları puanlandı: `kod/phase3/yolo_controls.py` → `phase3/results/T7b_yolo_controls.csv` (487 çift, 469 grup, tohum 42, en yüksek kutu güveni). AUROC: özgün 0.995 [0.992, 0.998], geri yapıştırma 0.992 [0.988, 0.996], yeniden örnekleme 0.994 [0.990, 0.997]; kayıp en çok 0.28 puan. Kutu U-Net aynı en-yüksek-skor kuralıyla 0.922 → 0.899 ve 0.900 (2.2–2.3 puan; `T7_controls.csv`). Skor tanımı metinde ve yöntem bölümünde yazıldı. |
| 4 | [Release] Bağlantı, `LICENSE`, sabit yollar, AGPL | Kısmen | Depo bağlantısı girişte dipnot olarak eklendi. Sabit `C:\Users\PARS` yolu kalan iki yerde düzeltildi (`kod/app/README_APP.md` içindeki iki `cd` satırı; `kod/phase3_v2/run_v2.ps1` artık öteki iki kuyruk betiği gibi `P3_PYTHON` değişkenini okuyor). **Açık:** depoda `LICENSE` yok; yerel değişiklikler (Faz 4, yeni betikler, rapor v5, bildirinin son hâli) depoya gönderilmedi. Ultralytics AGPL-3.0 lisanslıdır; kod lisansı seçilirken dikkate alınmalı. |
| 5 | [Citations] DBLP doğrulaması yok; kaynak notu eksik kayıtlar | Kısmen | `cheng2023smallobject` cilt 45, sayı 11, s. 13467–13488 eklendi; `holm1979` ve `loshchilov2019adamw` için kaynak notu yazıldı. Bu turda iki kayıt daha tamamlandı (aşağıda). **Açık (yöntem notu):** doğrulama Crossref, arXiv ve yayıncı sayfalarıyla yapıldı; DBLP kullanılmadı. |
| 6 | [Seed] v2 bölümünün en güçlü iddiası tek tohum | Açık | İşaretçisiz kutu U-Net 0.712 ve aynı-tarif farkı tek tohumludur; metinde etiketli. İki ek tohum GPU koşusu gerektirir. |
| 7 | [Numbers] "in 61 of the 64" izlenemiyor (sabah LOW-3; bu turda yükseltildi) | Kapandı | 64 kısmi kararın notları sınıflandı (`kod/phase2/gt/partial_note_check.py` → `phase2/gt/partial_note_check.json`): 46 not arka planın ya da yamanın, 43 not değişmeyen içeriğin işaretlendiğini söylüyor; 64 notun hepsi bu ikisinden en az birini içeriyor; referansın değişen bölgeyi kaçırdığını bildiren not yok. Kaynak dağılımı: WildReceipt 39/39, SROIE 16/24, CORD 9/26, XFUND 0/11 (`human_verification_summary.json`, `eps8_by_source`). Yeni metin: "in all 64 the auditor's note describes over-coverage … (WildReceipt 39/39 partial, XFUND 0/11)". |
| 8 | [Release / Ethics] Açık depoda kuralla çelişen üç şekil (bu tur) | Kısmen | `phase2/gt/verification_grid_page1.png`, `page2.png` ve `fig_qualitative_examples_LOCAL_ONLY.png` depoda izleniyor; proje kuralı bunları dışarıda tutuyor. Aynı taramada iki denetim tablosunun (`dataset_audit/samples_forged.csv`, `phase3_v2/v2_table.csv`) veri tabanı üstverisindeki metin alanlarını (`original_value`, `forged_value`) bütün örnekler için içerdiği görüldü; bu, bildirideki "identifiers and per-image statistics only" cümlesiyle de çelişiyordu. Eşitleme betiği ve depodaki `.gitignore` düzeltildi: üç şekil depodan kalkar, iki sütun yayımlanan kopyalardan çıkarılır. **Açık:** önceki sürümler git geçmişinde durur. Ayrıntı: `veri_etigi_kontrol_listesi_2026-11-10.md`. |
| 9 | [Eval] İkinci GT denetiminin özeti "insan doğrulaması" başlığı taşıyordu (bu tur) | Kapandı | `score_human_verification.py` artık değerlendirici belgelenmedikçe özeti "İkinci GT denetimi (değerlendirici türü doğrulanmadı)" başlığıyla yazıyor; sayılar değişmedi (JSON'da üç yeni anahtar, ortak anahtarlarda fark yok). Bildiri zaten "neither audit is a verified human annotation" diyor. Doğrulanmış insan denetimi hâlâ yapılmadı. |

## LOW

| # | Madde | Durum | Not |
|---|---|---|---|
| 1 | Özet uzunluğu | Açık | Şu an yaklaşık 282 kelime. 242 ve 150 kelimelik iki hazır sürüm `reports/Hakem_Hazirligi_Faz3.md` ekinde; sınır mekâna bağlı olduğu için uygulanmadı. |
| 2 | Hedef konferans ve sayfa sınırı | Açık | Ders kuralı final haftasında kabul mektubu ve katılım belgesi istiyor; mekânın tarihi sayfa sınırından önce gelen kısıt. Kısaltma sırası hakem hazırlığı belgesinde. |
| 3 | TruFor dipnotu | Kapandı | "Zero-shot TruFor (our run): F1@0.5 0.403." Veri tabanı makalesindeki 0.434 ile fark `phase2/literature_baselines/README.md` içinde açıklanmış. |
| 4 | YOLO etiketlerinde 1 piksel | Kapandı | `method.tex`: "its end coordinates are inclusive, so the labels are one pixel short in width and height". |
| 5 | `T5_yolo_ablation.csv` `best_epoch` ölçütü | Değişiklik gerekmiyor | Bildiride kullanılmıyor. |
| 6 | Yazar bloğu | Açık | "AIC515 Image Processing" bir ders adı; gönderimde bölüm ya da enstitü adı gerekir. |
| 7 | `references_v3.bib` 77 kayıt, `refs.bib` 79 | Kapandı | İki dosya özdeş (79 kayıt); README güncellendi. |
| 8 | Rapor ve sunumda "sekiz sayfa" | Kapandı | Rapor v5 ve sunum "dokuz sayfa" diyor. |
| 9 | `efron1979bootstrap` sayfa numarası yok (bu tur) | Kapandı | s. 1–26 eklendi. |
| 10 | `wang2022tsroie` cilt, sayı, sayfa yok (bu tur) | Kapandı | Cilt 8, sayı 3, s. 29–40 eklendi. |
| 11 | Faz 4 bildiride yok (bu tur) | Açık (isteğe bağlı) | ResNet-18'in tam veriyle yeniden eğitimi aynı kısayolu üçüncü model ailesinde gösteriyor (`reports/PHASE4_ResNet_Sonuclar.md`). Bildiri bu sonucu içermiyor ve içermek zorunda değil; önerilen tek cümle hakem hazırlığı belgesinde. |

## Sayısal denetim

Sabahki denetimde tablolar ve metindeki yaklaşık 250 sayı kaynak dosyalarla karşılaştırılmıştı; uyuşmayan tek sayı HIGH-1 idi. Bu turda değişen ya da yeni giren sayılar:

| Bildirideki ifade | Konum | Kaynak | Kaynak değeri | Uyum |
|---|---|---|---|---|
| no pixel outside the box differs in any of the 3,066 pairs | `results_v2.tex` | `v2_marker_paired_check.json` → `pairs_with_any_difference_outside_box` | 0 / 3066 | Evet |
| colour rule fires on a single … receipt, inside its box | `results_v2.tex` | aynı dosya → `paired_residue_ge20`, `v2_marker_paired_check.csv` | 1 (96 piksel, kutu içinde) | Evet |
| 380 pairs; 368 pairs without the dilation | `results.tex` §V-C | `phase3/results/marker_residue.csv`, doğrulama bölmesi (`n_residue_dil`, `n_residue` ≥ 20) | 380; 368 | Evet |
| AUROC 0.995 / 0.992 / 0.994 (detector) | `results.tex` §V-B | `T7b_yolo_controls.csv` | 0.9952 / 0.9924 / 0.9937 | Evet |
| loses at most 0.3 points | `results.tex` §V-B | aynı dosya → `delta_vs_authentic_pts` | −0.28, −0.15 | Evet |
| box U-Net loses 2.2–2.3 points | `results.tex` §V-B | `T7_controls.csv` → `auroc_max` | 0.9224, 0.8993, 0.9000 | Evet |
| in all 64 … (WildReceipt 39/39, XFUND 0/11) | `discussion.tex` | `partial_note_check.json`, `human_verification_summary.json` | 64 / 64; 39 / 39; 0 / 11 | Evet |
| lose 43–64 %; 24 %; 8 % | `discussion.tex` | `T_v2.csv`, `T2_test.csv` | 43.4–64.4; 24.2; 8.5 | Evet |
| Zero-shot TruFor (our run): F1@0.5 0.403 | Tablo V dipnotu | `phase2/literature_baselines/zero_shot_results.csv` (`trufor`, test, 812 görüntü, `pixel_f1_05`) | 0.4025 | Evet |

Sabah denetlenip değişmeyenlerden bu turda görüntü başı dosyalardan bağımsız yeniden hesaplananlar: Tablo V'in AP, F1 ve IoU sütunları; sınır sondaları (+%1.2, −%2.0, %0.0); referans medyanları (0.35, 0.27; 0.49, çeyrekler 0.37–0.62); 600 özgün görüntüde yanlış alarm çıkmaması; dışlanan 14 çift; işaretçi payı. Hepsi eşleşti.

## Düzeltilen BibTeX kayıtları

```bibtex
% src: Crossref 10.1109/TPAMI.2023.3290594; cilt ve sayfa arXiv journal-ref ve yayıncı sayfasından
@article{cheng2023smallobject, ... volume={45}, number={11}, pages={13467--13488}, year={2023} }

% src: Crossref 10.1214/aos/1176344552 (kayıtta sayfa yok); s. 1-26 Project Euclid ve JSTOR 2958830
@article{efron1979bootstrap, ... volume={7}, number={1}, pages={1--26}, year={1979} }

% src: DOAJ makale kaydı (Haziran 2022, cilt 8, s. 29-40); sayı 3 ve "(in Chinese)":
%      Ju ve ark., Acta Electronica Sinica 54(3), 2026, DOI 10.12263/DZXB.20250830 kaynakçası
@article{wang2022tsroie, ... volume={8}, number={3}, pages={29--40}, year={2022}, note={In Chinese} }
```

`paper/refs.bib` ile `literature/references_v3.bib` özdeştir. Sayfa numarası olmayan üç kayıt kaldı: `loshchilov2019adamw` (ICLR; sayfa numarası yok, olağan), `fsts2025` ve `qu2024omniiml` (bildiride atıf yapılmıyor).

## Doğrulanamayan ya da koşullu iddialar

| Konum | İddia | Durum | Öneri |
|---|---|---|---|
| `discussion.tex` | İkinci denetim "stricter audit" | Değerlendirici türü doğrulanmadı | Metin bunu söylüyor ("neither audit is a verified human annotation"); doğrulanmış bir insan denetimi yapılırsa cümle güncellenir. |
| `results_v2.tex` | İşaretçisiz kutu U-Net 0.712 ve aynı-tarif karşılaştırması | Tek tohum | Metinde etiketli; iki ek tohumla ya doğrulanır ya daraltılır. |
| `intro.tex` | "We release … the code" | Depo geride ve lisanssız | Gönderimden önce eşitleme ve `LICENSE`. |

## Kod yayını denetimi (özet)

| Kontrol | Durum |
|---|---|
| Depo açık ve erişilebilir | Evet (anonim istek HTTP 200) |
| Bildirideki sonuçları üreten betikler depoda | Kısmen: HEAD `kod/` düzeninden önceki sürüm; `yolo_controls.py`, `v2_marker_paired_check.py`, `partial_note_check.py` ve Faz 4 yalnız yerelde |
| Lisans | Yok |
| Sabit kişisel yol | Yalnız varsayılan Python yolu olarak üç kuyruk betiğinde kalıyor (`run_v2.ps1`, `run_phase4.ps1`, `run_all_tests.ps1`); üçü de `P3_PYTHON` ile değiştirilebiliyor |
| Gizli anahtar | Eşitleme betiği her gönderimde tarıyor |
| Veri ve ağırlıklar | Depoda yok; betiklerle yeniden üretiliyor. Üç şekil kuralla çelişiyor (MEDIUM-8) |
| Koşuların commit kimliği | Yok: çalışma ağacı git deposu değil. Faz 4 koşuları betik özetini (SHA-256) kaydediyor; Faz 2–3 koşularında commit kimliği yok |
