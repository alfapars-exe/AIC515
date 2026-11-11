# Veri etiği ve yeniden dağıtım kontrol listesi — AIForge-Doc v1 / v2

**Tarih:** 10 Kasım 2026 · **Kapsam:** `Vize Proje/` çalışma ağacı ve açık depo `github.com/alfapars-exe/AIC515` (HEAD `553fd7d`, 10 Kasım 2026 13:50)
**Yöntem:** Lab Rat `dataset-curation` 6. adım (etik inceleme kontrol listesi). Dağılım, yanlılık, bölme ve etiket kalitesi adımlarının kanıtları zaten projede vardır; aşağıda yalnız dosya adlarıyla anılmıştır.
**Kaynaklar:** `data/AIForge-Doc-v1/README.md` ve `data/AIForge-Doc-v2/README.md` (veri tabanı kartları, yerel kopya), `dataset_audit/audit_report.md` §14, `phase2/literature_baselines/LICENSES.md`, `kod/phase3/sync_repo.ps1`, depodaki `.gitignore`, `git ls-files` çıktısı.

## 1. Kontrol listesi

| Madde | Durum | Kanıt ve not |
|---|---|---|
| İçerik hassasiyeti | Uygun, dikkat gerektirir | Görüntüler market fişleri ve formlardır (3.642 fiş, 419 form; `dataset_audit/audit_report.md` §7). İşletme adı, adres, telefon, kasiyer adı ve kısmen maskelenmiş kart numarası gibi alanlar görülebilir. CORD kaynaklı bazı görüntülerde alanlar kaynak veri tabanında bulanıklaştırılmış olarak gelir (ör. doğrulama ızgarasının ilk sayfası). Proje bu alanları okumaz, saklamaz, kişi ya da işletme düzeyinde çözümleme yapmaz. |
| Rıza | Uygun | Yeni veri toplanmamıştır. Görüntüler dört açık veri tabanından (CORD, WildReceipt, SROIE, XFUND) gelir; toplama ve rıza sorumluluğu bu kaynakların yayıncılarındadır. İkinci GT denetimi insan katılımcı verisi içermez (değerlendirici türü de doğrulanmamıştır; `phase2/gt/human_verification_summary.md`). |
| Gizlilik | Düzeltme gerekti | Depoda kimlik listeleri (`dataset_audit/splits.json`), görüntü başı istatistikler ve ölçüt tabloları vardır; tabloların anahtarı veri tabanının kendi kimlik numarasıdır. İki denetim tablosu ise veri tabanı üstverisindeki metin alanlarını (düzenlenen alanın eski ve yeni değeri) bütün örnekler için içeriyordu; eşitleme betiği artık bu iki sütunu yayımlanan kopyalardan çıkarıyor (2. ve 3. bölüm). 100 çiftlik denetim formları (`phase2/gt/verification_human.xlsx`, `verification_sheet_vlm.csv`, `vlm_verification_pass1.csv`) değişen metni ve notları içerir ve depoda kalmaktadır. |
| Lisans | Uygun, bir tutarsızlık notuyla | v1 ve v2 kartları: `license: cc-by-nc-sa-4.0`, kapılı erişim, "yalnız ticari olmayan araştırma" ve atıf onayı. Kaynak lisansları (kart beyanı): CORD v2 CC BY 4.0, WildReceipt Apache 2.0, SROIE "ICDAR 2019 research use", XFUND CC BY 4.0. v1 kartının kendi içinde tutarsızlığı var (`dataset_info.json` "CC BY 4.0" der, lisans bölümündeki bağlantı da CC BY 4.0'a gider); proje en kısıtlayıcı okumayı esas alır (`audit_report.md` §14). Kullanım ders projesi ve akademik bildiridir, ticari değildir. TruFor ağırlıkları yalnız kâr amacı gütmeyen kullanıma izinlidir; ADCD-Net MIT lisanslıdır (`LICENSES.md`). |
| Türev işler | Uygun | Piksel-değişim ve glif maskeleri, işaretçisi onarılmış görüntüler, kontrol kümeleri ve tahmin haritaları veri tabanının türevidir (aynı lisansla paylaşım koşulu). Hiçbiri depoya ya da Space deposuna konmamıştır; hepsi betiklerle yeniden üretilir. Model ağırlıkları da depoda yoktur; canlı uygulama bunları özel bir depolama kovasından salt okunur bağlar. |
| Olası zararlar | Düşük, açıkça yazılmalı | Bulgular savunma yönlüdür: bir kıyas kümesindeki üretim kalıntısını ve kutu etiketinin neyi ödüllendirdiğini ölçer. İki nokta çift kullanımlıdır: (i) işaretçi kalıntısı olmayan sahtelerde ve gri düzeyde dedektörlerin zayıfladığı bilgisi, (ii) kalıntı piksellerini özgün görüntüden geri kopyalayan onarım betiği. Onarım betiği yalnız özgün eşi elde olan veri tabanı görüntülerinde çalışır ve yeni sahte üretmez; proje sahte üreten bir araç yayımlamaz. |
| Belgeleme | Kısmen | Veri tabanının kendi kartı var. Proje tarafında 14 başlıklı denetim raporu (`dataset_audit/audit_report.md`), bölme tanımı ve türetilmiş referansların raporu (`phase2/gt/gt_report.md`) var. Türetilmiş maskeler paylaşılmadığı için ayrı bir veri kartı yazılmamıştır; paylaşılırlarsa gerekir. |

## 2. Depo içeriği taraması (HEAD `553fd7d`)

Proje kendi kuralını üç yerde yazmıştır: README "Veri ve lisans" ("görüntüler, türetilmiş maskeler … repoya dahil değildir"), `sync_repo.ps1` başlığı ("AIForge-Doc görüntüleri ve türevleri … yeniden dağıtılmaz") ve `audit_report.md` §14 ("yalnızca id listeleri paylaşılmalı"). Depoda izlenen 669 dosyanın 128'i PNG'dir. Bunların çoğu grafik ya da model çekirdeği görselidir; veri tabanı görüntüsü içerenler üç gruptur.

| Grup | Dosyalar | Değerlendirme |
|---|---|---|
| Kuralla çelişen | `phase2/gt/verification_grid_page1.png`, `phase2/gt/verification_grid_page2.png` | Denetim ızgarasının 20 sayfasından ikisi: on çiftin sahte ve özgün kırpıntısı ile dört maskesi. Aynı ızgaranın PDF'i `.gitignore` ile dışlanmış, PNG sayfaları dışlanmamış. |
| Kuralla çelişen | `phase2/literature_baselines/figures/fig_qualitative_examples_LOCAL_ONLY.png` | Şeklin kendi başlığı "NOT exported as artifact" diyor; `phase2/literature_baselines/README.md` de "artifact yapılmadı" diye not düşmüş. Dört test görüntüsü ile TruFor ve ADCD-Net haritalarını içerir. |
| Kuralla çelişen | `dataset_audit/samples_forged.csv` ve `phase3_v2/v2_table.csv` içindeki `original_value` ve `forged_value` sütunları | Veri tabanı üstverisinden gelen metin alanları: 4.061 v1 ve 3.066 v2 örneğinde düzenlenen alanın eski ve yeni değeri (tutar, tarih, adres, telefon; bir satırda e-posta adresi). Sonuçların yeniden üretimi için gerekmez; uzunlukları (`orig_len`, `forg_len`) tabloda ayrıca vardır. |
| Örnek amaçlı şekiller | `app/screenshot_*.png` (7), `dataset_audit/fig_examples_overlay.png`, `dataset_audit/fig_pixel_diff_quads.png`, `phase2/controls/examples/*.png` (2), `phase2/experiments/figures/fig_examples_predictions.png`, `baselines/figures/unet_predictions.png`, `phase3/figures/fig_probe_conditions.png`, `paper/fig/fig_probe_conditions.png`; rapor ve bildiri PDF/DOCX dosyalarındaki şekiller | Az sayıda örnek görüntü; rapor ve bildiri şekilleridir. CC BY-NC-SA 4.0 atıf, ticari olmayan kullanım ve aynı lisans koşuluyla paylaşıma izin verir. Depoda bu koşulu söyleyen bir not ve bir `LICENSE` dosyası henüz yoktur. |

Ayrıca `phase2/gt/verification_human.xlsx`, `verification_sheet*.csv` ve `vlm_verification/vlm_verification_pass1.csv` depoya gitmektedir. Görüntü içermezler; 100 çift için değişen alanın eski ve yeni metnini (ör. tutar), işletme adı ya da adres geçen denetim notlarını ve dil modelinin yanıtlarını içerirler. İkinci denetimin sayıları bu dosyalardan üretildiği için depoda bırakıldılar; yayımlanmaları bir tercih konusudur.

## 3. Bu turda yapılanlar (yalnız yerel)

1. `kod/phase3/sync_repo.ps1`: `verification_grid_page*.png` ve `*_LOCAL_ONLY*` dışlama listesine eklendi; depoda kalmış eski kopyaları silen bir adım eklendi (silme bir sonraki eşitlemede `git add -A` ile kayda girer).
2. `AIC515_repo/.gitignore`: aynı iki desen eklendi.
   Eşitleme betiği ayrıca `dataset_audit/samples_forged.csv` ve `phase3_v2/v2_table.csv` dosyalarının depodaki kopyalarından `original_value` ve `forged_value` sütunlarını çıkarır (çalışma ağacındaki dosyalar değişmez).
3. `README.md` "Veri ve lisans": örnek şekillerin kaynağı ve lisans koşulu yazıldı.
4. Bildiri (`paper/sec/data.tex`): "we redistribute identifiers and per-image statistics only" cümlesi "beyond the examples shown in figures" kaydıyla düzeltildi.

## 4. Karar bekleyenler

| # | Konu | Not |
|---|---|---|
| 1 | Depoya gönderim | **Yapıldı (10 Kasım 2026, commit `4668b08`):** üç dosya HEAD'den kalktı; ayrıca veri tabanı metni taşıyan `original_value` ve `forged_value` sütunları yayımlanan iki tablodan (`dataset_audit/samples_forged.csv`, `phase3_v2/v2_table.csv`) çıkarıldı. Git geçmişinde kalırlar; geçmişin yeniden yazılması ayrı bir karardır (lisans paylaşıma izin verdiğinden zorunlu görünmüyor). |
| 2 | `LICENSE` ve şekil notu | **Yapıldı (10 Kasım 2026):** kod MIT lisanslı (`LICENSE`), README'de lisans bölümü ve şekil notu var. İlk öneri: kod için bir lisans (ör. MIT ya da Apache-2.0) ve "örnek görüntü içeren şekiller AIForge-Doc'tan türetilmiştir, CC BY-NC-SA 4.0" notu. Lisans dosyası yokken kod yasal olarak yeniden kullanılamaz; bildirideki "we release the code" cümlesi bir lisans gerektirir. |
| 3 | Veri tabanı yayıncısına bilgi | İşaretçi kalıntısı v1 makalesinde anlatılan yeşil dikdörtgenin artığıdır ve v2'de yoktur. Yayından önce Scam.AI'ye kısa bir not göndermek iyi uygulamadır; zorunlu değildir. |
| 4 | Denetim formu | **Kapandı (11 Kasım 2026):** yazar formu kendisinin doldurduğunu bildirdi; özet artık "İnsan GT doğrulaması" başlığını taşıyor ve değerlendiriciyi kaydediyor (`score_human_verification.py --evaluator`). Önceki not: `verification_human.xlsx` depoda kalsın mı? Kalacaksa dosya adındaki "human" sözcüğü yanıltıcıdır; özet dosyası artık değerlendiricinin doğrulanmadığını belirtiyor. |

## 5. Diğer `dataset-curation` adımlarının kanıtı

| Adım | Kanıt |
|---|---|
| Dağılım | `dataset_audit/audit_report.md` §4–10 (kaynak, üretici, belge tipi, dil, alan, çözünürlük, alan oranı) |
| Yanlılık ve sahte bağıntı | Yeşil işaretçi kalıntısı (`phase2/gt/green_residue_pairs.csv`, çiftlerin %75.4'ü; Gemini %83.6, Ideogram %1.5); üretici ile işaretçinin karışması raporda sınırlama olarak yazılı |
| Bölme | Kaynak belge gruplarıyla bölme, grup çakışması yok (`dataset_audit/splits.json`); araç, kaynak, alan ve işaretçi dağılımları üç bölmede benzer (χ² p ≥ 0.30); test sahtelerinin %8.9'u eğitimle kaynak paylaşır, 740 sızıntısız görüntü ayrıca raporlanır |
| Etiket kalitesi | İki denetim (dil modeli destekli: 69/96 uygun; insan denetimi, yazar: 36 uygun, 64 kısmi), uyum zayıf (κ = 0.07); insan denetimi tek değerlendiricili |
| Genişletme | v2 birincil kümesi (567 çift) yalnız harici test olarak kullanıldı; v2 üzerinde eğitim yapılmadı |
