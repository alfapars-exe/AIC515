# Hakem Hazırlığı (Lab Rat `rebuttal`) — IEEE bildirisi

**Sürüm:** 10 Kasım 2026 akşamı (v2 harici testi, ikinci GT denetimi, Faz 4 ve tutarlılık düzeltmelerinden sonra). Önceki sürüm (Faz 3 sonrası, 8 sayfalık bildiri): `_arsiv/_backup/2026-11-10_denetim/reports/Hakem_Hazirligi_Faz3.md`.
**Bildiri:** `paper/main.pdf` — 10 sayfa (gövde 8 sayfayı biraz aşıyor, kaynakça 9. sayfada başlıyor; 64 kaynak). Son güncelleme: 11 Kasım 2026 (Faz 4 bölümleri ve ek tohum sonuçları eklendi).
**Hedef mekân:** henüz seçilmedi. Ders kuralı final haftasında kabul mektubu ve katılım belgesi istediğinden, mekânın tarihi sayfa sınırından önce gelen kısıttır. Sayfa sınırı belli olunca "Önerilen son düzenlemeler" bölümündeki kısaltma sırası uygulanır.
Tüm sayılar `phase3/results/tables_p3.md`, `phase3/results/T7b_yolo_controls.csv`, `phase3_v2/results/{v2_tables.md,v2_marker_paired_check.json}` ve `phase4_resnet/results/tables_p4.md` dosyalarından alınmıştır.

## Zayıflıklar

| # | Zayıflık | Tür | Önem | Durum / öneri |
|---|---|---|---|---|
| 1 | Piksel-değişim GT'sinin insan denetimi tek değerlendiricili ve değerlendirici yazarın kendisi; değerlendiriciler arası uyum yok. İki denetim var (VLM: 69 uygun / 26 kısmi / 1 hatalı, n = 96; insan, daha sıkı: 36 uygun / 64 kısmi / 0 hatalı, n = 100) ve uyumları zayıf (κ = 0.07 [−0.08, 0.21]) | deneysel | orta (önce yüksek) | Sınırlama olarak yazıldı. İnsan denetiminin 64 kısmi notunun tamamı fazla kapsamayı anlatıyor (değişimi kaçırma yok). **Güçlendirme:** aynı formla (`phase2/gt/verification_human.xlsx`, yönerge sayfası) ikinci bir bağımsız değerlendirici ve κ (yaklaşık 1–1,5 saat). H1 daha sıkı olan glif GT'siyle de geçerli (0.923 → 0.257) |
| 2 | Tek tohumlu sonuçlar: DeepLabV3+, işaretçisiz piksel-değişim ve glif modelleri, birleşik kayıplı glif modeli, 20 epok, YOLO ablasyonları | istatistik | orta (önce yüksek) | **Büyük ölçüde kapatıldı (11 Kasım 2026):** işaretçisiz kutu modeli ve birleşik kayıplı piksel modeli 43 ve 44 tohumlarıyla yinelendi; öngörüler ve yanlışlama ölçütleri koşudan önce yazılıp depoya gönderildi, altısı da tuttu (`reports/inceleme/deney_plani_ek_tohum_2026-11-10.md`). v2 bölümünün en güçlü iddiası artık üç tohumlu: işaretçisiz kutu U-Net 0.711 ± 0.007; kalıntısız tariflerde −0.008 [−0.069, +0.053]. Kalan tek tohumlu sonuçlar metinde etiketli |
| 3 | v1'deki ayrı tutulan üretici deneyi işaretçi ile üreticiyi ayıramıyor (kalıntı Gemini'de %86, Ideogram'da %2); yalnız Gemini ile işaretçisiz eğitim yapılmadı | tasarım | orta | Metinde açıkça yazıldı. v2 harici testi soruyu dışarıdan kısmen yanıtlıyor: aynı tarifin v1 ve v2 görüntüsü, v1 kalıntısına göre ayrılarak karşılaştırıldı |
| 4 | v2 hem üreticiyi hem geri yapıştırma yöntemini değiştiriyor | tasarım | orta | Sınırlama olarak yazıldı; iki etki yalnız v1 kalıntı ayrımı üzerinden ayrışıyor |
| 5 | TruFor ince ayarı yok; öğrenilmiş modeller yalnız sıfır-atış başvurularla karşılaştırılıyor | taban çizgisi | orta | Eğitim kodu yayımlanmamış; mimari bağımsızlık DeepLabV3+ ve YOLO11 ile gösterildi. Tablo V dipnotundaki TruFor F1 0.403 kendi koşumuz (veri seti makalesi 0.434 veriyor; fark `phase2/literature_baselines/README.md` içinde açıklandı) |
| 6 | Dedektörün zararsız kontrolleri farklı bir görüntü skoruyla ölçüldü (en yüksek kutu güveni; bölütleme modellerinde %99 yüzdelik skoru) | yöntem | düşük | Metinde belirtildi ve aynı skorla kutu U-Net karşılaştırması eklendi (dedektör en çok 0.3 puan, U-Net 2.2–2.3 puan kaybediyor). Dedektör tek tohum |
| 7 | Karolu YOLO'nun kutu AP'si tam görüntü YOLO'dan düşük (0.88 ve 0.96) | yöntem | düşük | Metinde açıklandı; mekanizma kesin değil. Düzeltme: karolar arası kutu birleştirme — gelecek iş |
| 8 | Hız beklentisi doğrulanmadı (98 ms ve 81 ms) | sunum | düşük | Olduğu gibi raporlandı |
| 9 | Görüntü düzeyi tespit testte ölçülemiyor (test özgünleri eşli değil) | deneysel | düşük | Açıklandı; doğrulama çiftlerinde raporlandı. v2'de bütün çiftler eşli olduğundan görüntü AUROC orada verildi |
| 10 | Paylaşım: depo herkese açık ve kodu, kimlikleri, görüntü başı istatistikleri içeriyor; maskeler ve kontrol kümeleri dağıtılmıyor, betiklerle yeniden üretiliyor. Depoda `LICENSE` yok | yeniden üretilebilirlik | orta | Bildirideki ifadeler buna göre düzeltildi (özet, giriş, veri bölümü) ve depo bağlantısı eklendi. `LICENSE` seçimi açık karar |
| 11 | Uzunluk: 10 sayfa; özet yaklaşık 321 kelime | biçim | mekâna bağlı | Kısaltma sırası aşağıda |

## Ablasyon seçimi (Lab Rat `compare` mantığı: aynı protokol, tek etken)

| Koşu | Değişen etken | Ana ölçüt farkı | Anlatı gücü | Yer |
|---|---|---|---|---|
| Eğitim GT'si (kutu / piksel / glif) × 3 tohum | eğitim maskesi | aynı tahminler: kutu-GT 0.923, piksel-GT 0.374 | Ana iddia (H1) | Ana metin, Tablo I |
| DeepLabV3+ | kod çözücü | 0.939 → 0.382 | H1 mimariden bağımsız | Ana metin, Tablo I |
| YOLO11s (3 tohum) | formülasyon | test kutu AUPRC 0.969; piksel 0.368 | Kutu ölçütü alan tespitini ölçer | Ana metin, Tablo V |
| İşaretçisiz eğitim | eğitim girdisi | onarım Δ −0.222 (3 tohum) → +0.017 (3 tohum); test AUPRC 0.950 → 0.940 | İşaretçi kısayoldur, ön koşul değildir | Ana metin, Tablo II |
| v2 harici testi | test verisi (üretici, işaretçisiz) | kutu U-Net 0.950 → 0.520; işaretçisiz kutu U-Net 0.711 (3 tohum); YOLO11s 0.887 | İşaretçi bulgusunun dışarıdan doğrulanması | Ana metin, Tablo VI |
| Yeniden çizim / içerik sondası | girdi | AUROC: yeniden çizim 0.637, içerik 0.733 (kutu U-Net, 3 tohum) | Mekanizma | Ana metin, Tablo III |
| Yalnız Gemini × 3 tohum | eğitim üreticisi | Ideogram 0.412, Gemini 0.909 | H3 reddi | Ana metin, Tablo IV |
| Birleşik kayıp (3 tohum) | kayıp | doğrulama: kutu 0.579 → 0.668, piksel 0.610 → 0.633 (89 test çiftinde piksel farkı yok); bulanıklık −%59 → −%46 | Pratik öneri; piksel kazancı doğrulamayla sınırlı | Ana metin (kısa) |
| Zararsız kontroller (dedektör dahil) | girdi | bölütleme 1.6–6.5 puan; dedektör en çok 0.3 puan | Yapıştırma dikişi itirazına yanıt | Ana metin (tek cümle) |
| 20 epok, YOLO n/m, varsayılan artırım, özgün negatif karo | eğitim ayrıntısı | yaklaşık 0 | "Yetersiz eğitim" itirazına yanıt | Kısa cümle; yer darsa ek |
| ResNet-18 tam veri + eşik kalibrasyonu (Faz 4, 3 tohum) | model ailesi | onarılmış sahtede doğru tespit 0.995 → 0.471; işaretçisiz eğitimde 0.899 | İşaretçi bağımlılığı üçüncü model ailesinde | Ana metin, §V-I ve §V-J, Tablo VII ve VIII (11 Kasım 2026'da eklendi) |

## En olası 10 hakem sorusu

1. **Q:** Your pixel-change ground truth may encode re-rendering noise rather than real edits. Why is it a valid reference?
   *Yanıtlanabilir:* Kısmen. *Taslak:* We do not claim it is the true mask; the claim is that changing only the training mask reverses the ranking. A stricter human audit of 100 stratified pairs by the author found it wrong in none and partial in 64, and in all 64 the auditor's note describes over-coverage (re-rendered background or unchanged content), not a missed edit. The glyph-restricted variant, which removes most of that over-coverage, gives the same ordering (0.923 → 0.257). The human audit has a single annotator, the author, so there is no inter-annotator agreement; this is stated as a limitation.
2. **Q:** Is the box-versus-pixel effect specific to the U-Net? — *Evet, Tablo I:* DeepLabV3+ 0.939 → 0.382 and a YOLO11 detector 0.965 → 0.368.
3. **Q:** If the marker is removed, is your shortcut claim overstated? — *Evet, Tablo II ve VI:* retraining on repaired images keeps 0.940 test AUPRC and removes the dependence; on the marker-free v2 benchmark the repaired model shows no loss on residue-free specifications (−0.008, CI [−0.07, 0.05]) while the standard model loses 0.17. Three seeds for the repaired box model; predictions were committed before the two extra seeds were run.
4. **Q:** v2 changes the generator and the paste-back procedure at once. How do you attribute the loss to the marker? — *Kısmen:* by comparing the same specification on v1 and v2 and splitting by whether the v1 image carried the residue (397 / 170 pairs): −0.49 versus −0.17 for the box U-Net. The two v2 changes themselves are not separated; stated as a limitation.
5. **Q:** Does v2 really carry no marker? — *Evet:* in none of the 3,066 pairs does a pixel outside the box differ from the authentic source, and the colour rule fires on a single green-tinted receipt, inside its box (`v2_marker_paired_check.json`).
6. **Q:** Many results use a single seed. — *Kısmen:* the main U-Net grid, the held-out U-Nets and the main detector use three seeds; the rest are labelled. Two extra seeds for the repaired and joint models are the first item to add.
7. **Q:** Why not fine-tune TruFor or another IML model? — *Kısmen:* its training code is not released; two decoders and a detector show architecture independence.
8. **Q:** Is the held-out-generator test powered, and is it a generator gap or a marker gap? — *Kısmen:* 126 Ideogram images with group-bootstrap intervals; the design cannot separate the two on v1 and the paper says so.
9. **Q:** Did you check for train/test leakage? — *Evet:* source-document group splits, a 740-image leakage-free test subset (0.949 versus 0.950), and on v2 a primary set fixed before scoring (image AUROC 0.915 on unseen versus 0.911 on seen documents).
10. **Q:** Are your masks and probes available? — *Evet, sınırlı:* the code that derives them, the identifiers and per-image statistics are public; images and masks are not redistributed because the dataset is gated (CC BY-NC-SA 4.0) and are regenerated by the released scripts.

Ek sorular: karolu dedektörün kutu AP'si neden düşük (Zayıflık 7); gri düzeye çevirmek işaretçiyi siler mi (hayır: sınanan bütün modellerde AUPRC'nin %47–67'si gidiyor, işaretçisiz eğitilenler dahil).

## Önerilen son düzenlemeler
1. **+2 tohum** (Zayıflık 2): **tamamlandı (11 Kasım 2026).** Altı öngörünün altısı da tuttu; özet, giriş, yöntem, sonuçlar ve tartışmadaki "one seed" kayıtları üç tohumlu değerlerle değiştirildi. Daraltılan tek iddia: birleşik kaybın piksel-değişim ölçütündeki kazancı küçük ve doğrulamayla sınırlı (0.633 ± 0.013 ve 0.610 ± 0.012; 89 test çiftinde fark yok).
2. **Sayfa sınırına göre kısaltma sırası** (az kayıptan çoğa): Tablo III (yeniden çizim / içerik) eke; §V-F dayanıklılık paragrafı tek cümleye; YOLO ablasyon paragrafı ikiye; ilgili çalışmalar bölümünde 2.1 ve 2.3 birleştirilir; kaynakça 64'ten yaklaşık 45'e iner; Faz 4 bölümleri (§V-I, §V-J) eke taşınabilir.
3. **Özet** şu an yaklaşık 321 kelimedir (Faz 4 cümlesi ve üç tohum kayıtlarıyla); IEEE konferanslarında sınır çoğunlukla 150–250 kelimedir (bazı çağrılarda 100–150). Sınır mekâna bağlı olduğundan bildiriye uygulanmadı; iki hazır sürüm bu belgenin sonundaki ekte.
4. **Yazar bloğu:** bölüm adı yerine ders adı yazıyor ("AIC515 Image Processing"); gönderimde bölüm ya da enstitü adıyla değiştirilmeli.
5. **Faz 4:** bildiriye eklendi (özet, katkılar, yöntem, §V-I "A patch classifier", §V-J "Attribution evidence", Tablo VII ve VIII, tartışma ve sınırlamalar). Sayfa sınırı olan bir çağrıda bu iki alt bölüm eke taşınıp ana metinde tek cümle bırakılabilir: "An image-level ResNet-18 patch classifier trained on the full split shows the same dependence: repairing the residue lowers its detection rate from 0.995 to 0.471 (three seeds)."

## Ek — özet için hazır kısa sürümler (bildiriye uygulanmadı)

Her iki sürüm yalnız şu anki özetteki cümleleri ve sayıları kullanır; yeni sayı içermez.

**242 kelime** (çıkarılanlar: işaretçinin tek başına özgün görüntüye yapıştırılması, Ideogram aktarımı, v2 kaybının işaretçili tariflerde yoğunlaşması):

> Benchmarks for AI-inpainted document forgery, such as AIForge-Doc, annotate each edit with the bounding box of the rewritten field. We ask what such labels teach a localizer. Using the authentic source images released with the training forgeries, we derive pixel-change and glyph masks and train the same network on box, pixel-change and glyph masks, changing nothing else. Under its own box metric the box-trained U-Net scores 0.92 pixel AUPRC; against the pixels that actually changed the same predictions score 0.37, and each ground truth ranks its own model first; a second decoder (DeepLabV3+) reproduces the effect. A 1–3 px residue of the green edit marker drawn by the generation pipeline survives in over 80 % of Gemini forgeries: removing it lowers the box model's per-image AP from 0.99 to 0.77, and retraining on marker-repaired images (three seeds) removes this dependence while keeping most of the box score. A tiled YOLO11 detector trained on the same boxes is the best model under the box metric (0.97 test AUPRC) and depends less on the marker, yet reaches only 0.37 validation AUPRC under the pixel-change truth. On the marker-free AIForge-Doc v2 (GPT-Image-2), without retraining, the box U-Net falls from 0.95 to 0.52 AUPRC while the detector keeps 0.89 and the marker-repaired U-Net 0.71. Box scores on this benchmark therefore measure field detection, not pixel-level tamper localization; we recommend reporting both and auditing generation residue, and we release the code that derives our masks, controls and probes.

**150 kelime** (100–150 kelime isteyen çağrılar için; işaretçisiz yeniden eğitim ve öneri cümlesi de çıkar):

> Benchmarks for AI-inpainted document forgery such as AIForge-Doc annotate each edit with the bounding box of the rewritten field. Using the authentic sources of the training forgeries, we derive pixel-change and glyph masks and train the same network on each ground truth. The box-trained U-Net scores 0.92 pixel AUPRC under its own metric but 0.37 against the pixels that actually changed; DeepLabV3+ reproduces the effect. A 1–3 px residue of the generator's green edit marker survives in over 80 % of Gemini forgeries, and removing it lowers the box model's per-image AP from 0.99 to 0.77. A tiled YOLO11 detector is the best model under the box metric (0.97 test AUPRC) yet reaches only 0.37 under the pixel-change truth. On the marker-free AIForge-Doc v2 the box U-Net falls from 0.95 to 0.52 AUPRC while the detector keeps 0.89. Box scores therefore measure field detection, not pixel-level localization; we release the code.
