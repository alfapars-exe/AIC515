# Dönem İçi Proje Raporu

**Ders:** AIC515 Görüntü İşleme — Prof. Dr. Onur Osman, Topkapı Üniversitesi

**Rapor:** Dönem İçi Proje Raporu (sürüm v5)

**Öğrenci:** Harun Benli

**Tarih:** 10 Kasım 2026

**Canlı uygulama:** https://huggingface.co/spaces/infinayazilim/ai515

**Kod deposu:** https://github.com/alfapars-exe/AIC515

## Özet

Bu çalışma, üretken yapay zekâ ile tek bir alanı yeniden yazılmış fiş ve form görüntülerinde sahteciliğin tespitini ve değiştirilen bölgenin yerinin bulunmasını AIForge-Doc v1 veri tabanı üzerinde ele almaktadır. Ders kapsamındaki yöntemler tek bir işlem hattında birleştirilmiştir: 56 klasik öznitelikle SVM ve Random Forest, üç farklı eğitim ayarıyla ResNet-18 yama sınıflandırıcısı, ResNet-18 kodlayıcılı U-Net, ikinci mimari olarak DeepLabV3+ ve kutu dedektörü olarak YOLO11. Sistemin çalıştığı önce 400 eğitim ve 100 doğrulama görüntüsünden oluşan kısıtlı bir alt kümede gösterilmiş (ResNet-18 ince ayar AUROC 0.899; U-Net piksel F1 0.707), ardından tam veride üç tohumla sınanmıştır. Resmî testte kutu maskesiyle eğitilen U-Net 0.950 piksel AUPRC değerine ulaşmakta, ancak aynı tahminler eşleşen özgün görüntüden türetilen piksel-değişim referansıyla puanlandığında doğrulama kümesinde 0.923'ten 0.374'e inmektedir; bu örüntü DeepLabV3+ ve YOLO11 ile yinelenmiştir. Sahtelerin büyük bölümünde üretim hattından kalan yeşil bir işaretçi çizgisi bulunmuş, modellerin bu kalıntıyı kısayol olarak kullandığı ve işaretçisi onarılmış görüntülerle yeniden eğitimde bağımlılığın ortadan kalktığı ölçülmüştür. Karolu YOLO11s kutu ölçütünde en iyi modeldir (test AUPRC 0.969) ve işaretçi taşımayan AIForge-Doc v2 üzerinde yeniden eğitim olmadan 0.887 değerini korurken U-Net 0.520'ye düşmektedir. ResNet-18 yama sınıflandırıcısı tam veriyle yeniden eğitilip görüntü düzeyi karar eşiği doğrulama kümesinde kalibre edildiğinde resmî testte 0.978 AUROC ve 0.022 yanlış alarm oranına ulaşmış, aynı işaretçi bağımlılığı bu model ailesinde de ölçülmüştür. SHAP ve katman filtresi görselleştirmeleriyle model kararları açıklanmış; tüm hat, Hugging Face üzerinde çalışan Gradio tabanlı bir uygulamada yeniden üretilebilir hâle getirilmiştir.

## 1. Proje Adı

**Yapay Zekâ ile Üretilmiş Finansal Belge Sahteciliğinin Tespiti ve Manipüle Edilmiş Bölgelerin Segmentasyonu: AIForge-Doc Veri Setinde Klasik Görüntü İşleme, ResNet, U-Net, DeepLabV3+ ve YOLO11 Karşılaştırması**

Projenin araştırma başlığı şudur: *What Do Document Forgery Localizers Learn from Box Masks? Pixel-Change Ground Truth and Shortcut Audits for AI-Inpainted Financial Documents.* Dönem sonunda hazırlanacak konferans bildirisinin taslağı aynı soruyu taşımakta, alt başlığı ise üç ana bulguyu (referans geometrisi, gizli işaretçi kısayolu, kutu tespiti) adlandırmaktadır.

Proje iki katmanlıdır. Ders katmanında amaç, haftalık konuların (yoğunluk dönüşümleri, histogram, eşikleme, frekans uzayı, morfoloji, öznitelik çıkarımı, ResNet, U-Net, YOLO) gerçek bir veri tabanında uçtan uca uygulanmasıdır. Araştırma katmanında amaç, kutu biçimindeki resmî maskelerin bir lokalizasyon modeline gerçekte ne öğrettiğini ve bu maskelerle hesaplanan başarı ölçütlerinin neyi ölçtüğünü denetlemektir.

## 2. Literatür

### 2.1 Problem ve kapsam

Çok kipli üretici modeller üzerine kurulu görüntü düzenleme servisleri, bir fişteki tutarı ya da bir formdaki tek bir alanı gözle fark edilir bir dikiş bırakmadan yeniden yazabilmektedir. Bu işlem iç-boyama (inpainting) olarak adlandırılır: görüntünün seçilen bölgesi üretici model tarafından baştan sentezlenir, geri kalanı korunur. Klasik yapıştırma ve kopyala-taşı sahteciliklerinde iz bırakan sıkıştırma süreksizliği ya da kamera gürültüsü tutarsızlığı bu tür düzenlemelerde büyük ölçüde kaybolmaktadır [1, 2].

Problem iki düzeyde tanımlanır. Görüntü düzeyinde bir belgenin sahte olup olmadığına karar verilir; bu ikili sınıflandırmadır. Piksel düzeyinde ise değiştirilen bölgenin haritası çıkarılır; bu ikili anlamsal bölütlemedir ve dersin U-Net haftasının doğrudan uygulama alanıdır. Aynı bölge bir sınırlayıcı kutuyla da ifade edilebildiğinden görev tek sınıflı nesne tespiti olarak da kurulabilir; bu da dersin YOLO haftasına karşılık gelir. İncelenen veri tabanında değiştirilen alan görüntünün medyan %0.98'ini kaplar. Ön planın bu kadar küçük olması, sabit eşikli ölçütlerin tek başına yanıltıcı olabileceği ve eşikten bağımsız ölçütlerle birlikte raporlanması gerektiği anlamına gelir [3, 4].

### 2.2 Bu veri tabanıyla yapılmış çalışmalar

AIForge-Doc ailesi üzerinde yayımlanmış çalışmalar Tablo 1'de özetlenmiştir. Veri tabanını tanıtan çalışma, CORD v2, WildReceipt, SROIE ve XFUND kaynaklı 4.061 belgede tek bir alanı Gemini 2.5 Flash Image ve Ideogram v2 Edit servisleriyle değiştirmiş, mevcut dedektörleri hiç eğitmeden (sıfır-atış) sınamıştır [1]. Genel amaçlı TruFor [5] görüntü düzeyinde 0.751 AUC ve piksel düzeyinde 0.358 IoU verirken, belgeye özgü DocTamper modeli [6] 0.563 AUC ve 0.020 IoU düzeyinde kalmıştır. Aynı makale iki tasarım kararını da belgeler: maske, değiştirilen alanın sıkı sınırlayıcı kutusudur ve Gemini yolunda düzenlenecek bölge, bağlam kırpıntısına çizilen 3 piksellik parlak yeşil bir dikdörtgenle işaretlenir. Bu iki karar, projenin iki ana bulgusunun kaynağıdır.

İkinci sürüm aynı düzenleme tariflerini GPT-Image-2 ile yeniden üretmiş ve 3.066 sahte belge yayımlamıştır [7]. İnsan gözlemciler ikili zorunlu seçimde şans düzeyinde kalmış (0.501), TruFor 0.599 ve DocTamper 0.585 AUC vermiştir; aynı dedektörler geleneksel yapıştırma kümelerinde 0.962 ve 0.852 düzeyindedir. Dolayısıyla çöküş alan kaymasına değil, yapay zekâ ile iç-boyamanın kendisine bağlanmaktadır. Üçüncü tur çalışma daha yeni bir üreticiyi ele almakta ve lokalizasyonun zayıf kaldığını bildirmektedir [8]; bu çalışmanın sayısal ayrıntıları bu raporda doğrulanmadığı için yalnız nitel olarak anılmıştır.

Bu veri tabanında model eğiten hakemli bir yayına rastlanmamıştır. Arşivlenmiş ancak hakemsiz iki çalışma vardır: biri v2 üzerinde küçük bir evrişimli ağ eğitip, içerik değişmeden üretim hattından geçirilen özgün görüntülerle bir "süreç ikizi" denetimi yapmıştır [9]; diğeri v1 ve v2 üzerinde karo tabanlı bir sayfa sınıflandırıcısı sunmakta, piksel lokalizasyonu içermemektedir [10]. Ayrıca küçük bölütleme ağları eğiten hakemsiz bir GitHub deposu bulunmaktadır; arşivlenmiş bir kaydı olmadığından kaynakçaya alınmamıştır. Bu nedenle raporda "AIForge-Doc üzerinde ilk denetimli eğitim" iddiası kullanılmamaktadır.

Table: Tablo 1. AIForge-Doc veri tabanı ailesiyle yapılmış çalışmalar, yöntemleri ve bildirdikleri sonuçlar.

| Çalışma | Veri | Yöntem | Bildirilen sonuç |
|---|---|---|---|
| Wu ve ark. [1] | v1, 4.061 sahte (Gemini, Ideogram) | Sıfır-atış: TruFor, DocTamper, GPT-4o | TruFor AUC 0.751, IoU 0.358, F1 0.434; DocTamper AUC 0.563, IoU 0.020; GPT-4o AUC 0.509 |
| Wu ve ark. [7] | v2, 3.066 sahte (GPT-Image-2) | Sıfır-atış dedektörler, insan deneyi, üreticinin öz-yargısı | İnsan 0.501; TruFor 0.599; DocTamper 0.585; öz-yargı 0.532 |
| Raj ve ark. [8] | Üçüncü tur üretici | Sıfır-atış değerlendirme | Lokalizasyon zayıf (nitel aktarım) |
| Helali [9] | v2 | Eğitilmiş küçük evrişimli ağ, süreç ikizi denetimi (hakemsiz) | Görüntü düzeyi tespit ve üretim hattı izi denetimi (sayılar bu raporda doğrulanmadı) |
| DocDet [10] | v1, v2 ve GPT4o-Receipt | Karo tabanlı ResNet-18 sayfa sınıflandırıcısı (hakemsiz) | Yalnız sayfa düzeyi karar; lokalizasyon yok |
| Bu proje | v1 (eğitim ve test), v2 (harici test) | SVM, RF, ResNet-18, U-Net, DeepLabV3+, YOLO11; üç referans türü | U-Net test AUPRC 0.950; YOLO11s 0.969; piksel-değişim referansıyla 0.37 düzeyi |

Kaynak: `literature/seed_papers_review.md`, `literature/lit_update_phase3.md`, `phase3/results/tables_p3.md`

### 2.3 İlgili yöntemler ve kıyaslama tablosu

Belge sahteciliği lokalizasyonu çoğunlukla yoğun tahmin problemi olarak ele alınır. DocTamper çalışması 170 bin sentetik görüntülük bir veri tabanı ile RGB görüntüyü 8×8 blok DCT katsayıları ve JPEG nicemleme tablosuyla birleştiren bir ağ önermiştir [6]. FFDN, frekans bilgisini RGB özelliklerini bozmadan eklemek için sıfırdan başlatılan bir iyileştirme modülü ve dalgacık benzeri bir ayrıştırma kullanır [11]. ADCD-Net, DCT ipucunun blok hizası bozulduğunda kırılganlaştığını ölçmüş ve hizalama skoruna göre uyarlanan bir DCT ağırlığı getirmiştir [12]. Genel amaçlı lokalizasyon tarafında TruFor kamera gürültüsü parmak izini RGB ile birleştirir [5]; CAT-Net JPEG sıkıştırma izlerini [13], MVSS-Net ise gürültü ve kenar görünümlerini kullanır [14]. PSCC-Net, IML-ViT, Mesorch ve SAFIRE aynı ailenin diğer temsilcileridir [15, 16, 17, 18]. Bu yöntemlerin ortak varsayımı görüntünün bir JPEG geçmişi taşımasıdır; AIForge-Doc görüntüleri ise kayıpsız PNG'dir.

DOCFORGE-BENCH, 14 yöntemi 8 belge kümesinde sıfır-atış koşulda karşılaştırmış ve kurcalanmış piksel oranının çok düşük olduğu kümelerde 0.5 eşiğinin kalibre olmadığını, yüksek piksel AUC değerine karşın F1'in sıfıra yaklaştığını göstermiştir [3]. Tablo 2 bu yöntemlerin kendi kıyas kümelerindeki ve AIForge-Doc v1 üzerindeki sonuçlarını bir araya getirmektedir.

Table: Tablo 2. Belge sahteciliği lokalizasyon yöntemleri: ana fikir, DocTamper kıyas kümelerindeki F1 değerleri ve AIForge-Doc v1 üzerindeki sıfır-atış sonuçları.

| Yöntem | Ana fikir | DocTamper-Test F1 | FCD F1 | SCD F1 | AIForge-Doc v1 (sıfır-atış) |
|---|---|---|---|---|---|
| DTD / DocTamper [6] | RGB + blok DCT + nicemleme tablosu | 0.792 (IoU 0.828) | 0.816 | 0.754 | F1 0.030, IoU 0.020, AUC 0.563 |
| FFDN [11] | Sıfır başlatmalı frekans dalı, dalgacık ayrıştırma | 0.86 | 0.92 | 0.81 | bildirilmemiş |
| ADCD-Net [12] | Hizalamaya uyarlanan DCT ağırlığı | 0.806 | 0.815 | 0.740 | bu proje: F1 0.182, IoU 0.149 |
| TruFor [5] | Gürültü parmak izi + RGB | ortalama 0.205 (resmî ağırlık) | — | — | makale: F1 0.434, IoU 0.358; bu proje: 0.403, 0.328 |
| CAT-Net [13] | DCT ve JPEG izi | 0.700 | 0.553 | 0.631 | değerlendirilmemiş |
| MVSS-Net [14] | Gürültü ve kenar görünümleri | 0.431 | 0.424 | 0.414 | değerlendirilmemiş |

Kaynak: `literature/seed_papers_review.md`, `phase2/literature_baselines/zero_shot_results.csv`

Üretici modellerle yapılan düzenlemeler üzerine çalışmalar bu raporun bulgularıyla doğrudan ilişkilidir. TGIF, metin yönlendirmeli iç-boyamada görüntünün tamamı yeniden çözüldüğünde yapıştırma lokalizatörlerinin başarısız olduğunu göstermiştir [2]; genişletilmiş sürümü, ince ayarlanan lokalizatörlerin maskeler anlamsal olduğunda bir nesne önyargısı edindiğini bildirir [19]. INP-X, dedektörlerin düzenlenen bölgeye değil görüntünün tümüne yayılan izlere yaslandığını, bu küresel iz ortadan kalktığında doğruluğun %91'den %55'e indiğini bildirmiştir [20]. Üretici parmak izine dayalı lokalizasyon [21] ve küçük alanlı düzenlemelerin en zor durum olduğuna ilişkin bulgular [22, 23] aynı çizgidedir. Metin odaklı sahtecilikte açıklama üreten çok kipli sistemler de gelişmektedir [24, 25, 26].

Bir ağın, etiketlerin ödüllendirdiği herhangi bir sinyali öğrenmesi ve bu sinyalin amaçlanan kanıt olmaması olgusu kısayol öğrenme olarak adlandırılır [27]. Adli görüntü analizinde bu durum JPEG ızgara hizası [12, 28], kıyas kümesi inşasından kalan önemsiz artefaktlar [29, 30] ve frekans önyargısı [31] için gösterilmiştir. Bu rapordaki soruya en yakın çalışma, nesne biçimli maskelerin değişmemiş pek çok pikseli etiketlediğini ve yalnız maskeye dayalı puanların dedektörleri yanlış sıralayabildiğini savunan çalışmadır [32]; ancak o çalışma doğal görüntüleri ele alır ve denetimli eğitim içermez. Bölütlemede kutu etiketi iyi bilinen bir zayıf denetim biçimidir ve nesne kapsamını yaklaşık olarak vermesi için ek adımlarla inceltilmesi gerekir [33, 34].

### 2.4 Projede kullanılan modellerin kaynak çalışmaları

Projede kullanılan her model ve tekniğin ilk sunulduğu çalışma Tablo 3'te verilmiştir. Klasik öznitelikler dersin ilk on bir haftasının araçlarıdır ve zorunlu ders kitabındaki ilgili bölümlere dayanır [35]: ayrık kosinüs dönüşümü [36], yerel ikili örüntüler [37] ve gri düzey eş-oluşum matrisi dokusu [38]. Sınıflandırıcı olarak destek vektör makinesi [39] ve rastgele orman [40] kullanılmıştır.

Derin modellerin omurgası artık öğrenmeye dayanan ResNet'tir [41]; ağırlıklar ImageNet üzerinde ön eğitimlidir [42]. U-Net, kodlayıcı ve kod çözücü arasındaki atlama bağlantılarıyla ince ayrıntıyı koruyan bölütleme mimarisidir [43]; kayıp fonksiyonundaki Dice terimi hacimsel bölütleme için önerilen V-Net çalışmasından alınmıştır [44]. DeepLabV3+ boşluklu evrişim ve çok ölçekli havuzlama ile bağlamı genişletir [45]. YOLO, tespiti tek ileri geçişte regresyon olarak çözen yaklaşımın kaynağıdır [46]; bu projede sürdürülen güncel sürümü YOLO11 kullanılmıştır [47]. Küçük nesneler için görüntünün doğal çözünürlükte örtüşen karolara bölünmesi dilimleme destekli çıkarım çalışmasına dayanır [48]. Açıklanabilirlik tarafında Shapley değerlerine dayalı SHAP çerçevesi `shap` kütüphanesi üzerinden kullanılmıştır.

Table: Tablo 3. Projede kullanılan model ve tekniklerin sunulduğu çalışmalar ve projedeki rolleri.

| Model veya teknik | Sunulduğu çalışma | Projedeki rolü |
|---|---|---|
| 8×8 blok DCT | Ahmed ve ark. [36] | Frekans öznitelikleri (34 öznitelik) |
| LBP | Ojala ve ark. [37] | Doku histogramı (10 öznitelik) |
| GLCM | Haralick ve ark. [38] | Doku istatistikleri (4 öznitelik) |
| Sauvola ikilileştirme | Sauvola ve Pietikäinen [49] | Glif referansının türetimi |
| SVM | Cortes ve Vapnik [39] | Klasik öznitelik sınıflandırıcısı |
| Random Forest | Breiman [40] | Klasik öznitelik sınıflandırıcısı, öznitelik önemi |
| ResNet-18 | He ve ark. [41] | Yama sınıflandırıcısı; U-Net ve DeepLabV3+ kodlayıcısı |
| ImageNet ön eğitimi | Deng ve ark. [42] | Başlangıç ağırlıkları |
| U-Net | Ronneberger ve ark. [43] | Ana lokalizasyon modeli |
| Dice kaybı | Milletari ve ark. [44] | Sınıf dengesizliğine karşı kayıp terimi |
| AdamW | Loshchilov ve Hutter [50] | Tüm derin modellerde eniyileyici |
| DeepLabV3+ | Chen ve ark. [45] | İkinci bölütleme mimarisi |
| YOLO, YOLO11 | Redmon ve ark. [46]; Jocher ve Qiu [47] | Kutu dedektörü |
| Dilimli çıkarım | Akyon ve ark. [48] | Karolu eğitim ve çıkarım |
| COCO ortalama kesinliği | Lin ve ark. [51] | Kutu düzeyi AP ölçütü |
| Bootstrap, Holm, Kendall | Efron [52]; Holm [53]; Kendall [54] | Güven aralığı, çoklu test düzeltmesi, sıralama uyumu |

Kaynak: `literature/references_v3.bib`, `paper/refs.bib`

### 2.5 Araştırma boşluğu, araştırma soruları ve katkı

Literatürdeki boşluklar ve projenin bunlara verdiği karşılık Tablo 4'te özetlenmiştir. Projede yeni bir mimari önerilmemektedir; katkı yöntemsel ve ampiriktir. Dört başlıkta toplanabilir: yalnız eğitim maskesinin değiştirildiği denetimli bir deney, eşleşen özgün görüntülerden türetilen iki yeni referans, üretim hattı kalıntısının ölçülmesi ve giderilerek yeniden ölçülmesi, son olarak bölütleme ve tespit formülasyonlarının aynı değerlendirme koduyla karşılaştırılması.

Table: Tablo 4. Literatürdeki boşluklar ve projenin katkısı.

| Boşluk | Bu projenin karşılığı |
|---|---|
| Veri tabanında yalnız sıfır-atış sonuçlar var; denetimli ve kontrollü eğitim yok | Üç referans türü × üç tohum U-Net; resmî test ve sızıntısız alt küme değerlendirmesi |
| Kutu maskesi piksel hassasiyetinde kabul ediliyor; modele ne öğrettiği sorgulanmamış | Piksel-değişim ve glif referansları; aynı tahminlerin üç referansla puanlanması |
| Üretim hattı artefaktlarının eğitilmiş modellerce kullanılıp kullanılmadığı bilinmiyor | Zararsız kontroller, sınır ablasyonu, işaretçi aktarma ve onarma sondaları, işaretçisiz yeniden eğitim |
| Tek mimari ve tek formülasyonla sınırlı sonuçlar | DeepLabV3+ ve YOLO11 ile yineleme; kutu düzeyi AP |
| Sabit eşik ve tek koşu | Eşikten bağımsız ölçütler, doğrulamada kalibre edilen eşik, grup bootstrap, Holm düzeltmesi |
| Kaynak belge sızıntısı ölçülmemiş | Grup farkında bölme; 740 görüntülük sızıntısız test alt kümesi |
| Üreticiler arası genelleme tek veri tabanıyla sınırlı | Ayrı tutulan üretici deneyi (üç tohum) ve AIForge-Doc v2 harici testi |

Kaynak: `reports/PHASE2_Metodoloji_ve_Sonuclar.md`, `reports/PHASE3_Sonuclar_ve_Kapanis.md`

Araştırma sorusu şudur: kutu maskesiyle eğitilen bir lokalizasyon modeli gerçekte neyi öğrenir ve kutu maskesiyle hesaplanan puan neyi ölçer? Bu soru, tam veri deneylerinden önce yanlışlama ölçütleriyle birlikte sabitlenen dört hipoteze ayrılmıştır (H1–H4). İkinci mimari, kutu dedektörü ve işaretçi çalışmaları sonradan eklenmiştir; bunlardan kutu dedektörüne ilişkin beş beklenti (H5a–H5e) deneyler koşulmadan önce yazılmış, diğerleri keşifsel olarak raporlanmıştır. Hipotezler ve ölçütleri Tablo 5'te verilmiştir; kararlar ve kanıtları Bölüm 4.2, 4.4 ve 4.5'tedir.

Table: Tablo 5. Hipotezler ve deneylerden önce belirlenen ölçütler.

| Hipotez | İçerik | Ölçüt |
| --- | --- | --- |
| H1 | Referans türü ölçütleri ve model sıralamasını değiştirir | ΔF1 0.05'ten küçük ve Kendall τ en az 0.9 ise yanlışlanır |
| H2 | Kutu modeli kutu sınırına ya da dikdörtgen biçimine yaslanır | Modeller benzer düşer ve zararsız kontrollerde yanlış alarm değişmezse yanlışlanır |
| H3 | Piksel ve glif denetimi görülmemiş üreticiye ve bozulmalara daha iyi aktarılır | Fark yoksa ya da tersse yanlışlanır |
| H4 | Sahteler kutu dışında öğrenilebilir bir küresel iz taşır | AUROC yaklaşık 0.5 ise yanlışlanır |
| H5a | Kutu dedektörü kutu ölçütünde U-Net'e eşit ya da üstündür ve en az üç kat hızlıdır | AP50 en az 0.90; F1 farkı 0.03 içinde; hız |
| H5b | Kutu dedektörü piksel-değişim ölçütünde belirgin düşüktür | Piksel AUPRC 0.35–0.45 aralığı |
| H5c | Küçük hedeflerde karolama tam görüntüden iyidir | Alanı %1'den küçük dilimde AP farkı en az 0.10 |
| H5d | Ayrı tutulan üretici açığı YOLO'da U-Net'ten büyük değildir | Açık 0.51'den küçük |
| H5e | Korumacı artırım varsayılandan iyi ya da eşittir | Test AUPRC karşılaştırması |

Kaynak: `phase2/stats/hypothesis_decisions.md`, `phase3_yolo/README.md`, `reports/PHASE3_Sonuclar_ve_Kapanis.md`

### 2.6 Veri seti seçim gerekçesi

AIForge-Doc v1 dersin örnek veri listesinde yer almamakla birlikte aynı kalıba uymaktadır. Veri tabanı kartı görevleri görüntü sınıflandırma ve görüntü bölütleme olarak etiketler; her sahte görüntü için 0 ve 255 değerli ikili bir maske vardır; küçük ön planlı ikili maske yapısı tıbbi bölütleme kümelerine, anomali lokalizasyonu niteliği ise endüstriyel kusur kümelerine benzer. Seçimi belirleyen dört etken şunlardır. Birincisi, yayımlanmış sıfır-atış sonuçlar karşılaştırma tabanı sağlar. İkincisi, eğitim bölümündeki her sahtenin özgün kaynağı da yayımlanmıştır; bu, piksel düzeyinde değişim haritası türetmeye ve yalnız eğitim maskesini değiştiren bir deney kurmaya olanak verir. Üçüncüsü, aynı tariflerle üretilen ikinci sürüm bağımsız bir harici test kümesi sunar. Dördüncüsü, finansal belge sahteciliği güncel ve ekonomik etkisi yüksek bir uygulama alanıdır. Lisans ticari olmayan akademik kullanıma izin vermektedir; ham veri yeniden dağıtılmamakta, yalnız kod, kimlik listeleri ve türetilmiş sonuçlar paylaşılmaktadır.

## 3. Materyal ve Metot

### 3.1 Veri tabanı ve denetimi

**AIForge-Doc v1.** Veri tabanı Hugging Face üzerinde `Scam-AI/AIForge-Doc-v1` adıyla, erişim onayına bağlı ve CC BY-NC-SA 4.0 lisansıyla yayımlanmaktadır [1]. Yerel kopya 12.187 dosya ve 9.98 GB'tır. Raporda verilen bütün istatistikler veri tabanı kartından alınmamış, dosyalar üzerinde çalışan bir denetim betiğiyle (`kod/dataset_audit/audit_dataset.py`) yeniden hesaplanmıştır. Temel özellikler Tablo 6'da, örnek görüntüler Şekil 1'de verilmiştir.

Table: Tablo 6. AIForge-Doc v1 veri tabanının yerel denetimle doğrulanan özellikleri.

| Özellik | Değer |
|---|---|
| Sahte görüntü | 4.061 (eğitim 3.249, test 812); RGB PNG |
| Maske | 8 bit, 0 ve 255 değerli; her maskede tek dikdörtgen |
| Özgün (authentic) görüntü | Eğitim klasöründe 3.249, test klasöründe 812 dosya |
| Doğrulanan sahte–özgün çifti | 3.336 (eğitim 3.247, test 89) |
| Kaynak veri tabanı | WildReceipt 1.696; CORD 1.000; SROIE 946; XFUND 419 |
| Belge türü ve dil | Fiş %89.7, form %10.3; 9 dil (İngilizce 2.642, Endonezce 1.000) |
| Üretici | Gemini 2.5 Flash Image 3.639 (%89.6); Ideogram v2 Edit 422 (%10.4). Meta veride iç anahtarlarla kodlanır: `gemini-nano` = Gemini 2.5 Flash Image, `qwen-inpaint` = Ideogram v2 Edit |
| Çözünürlük | 150×168 ile 6016×7016 piksel arası; medyan 750×1296; 0.03–34.81 megapiksel |
| Değiştirilen alan oranı | Medyan %0.975; görüntülerin %50.7'sinde %1'in altında |
| Maske kutusu boyutu | Yükseklik medyan 33 piksel; genişlik medyan 191 piksel |

Kaynak: `dataset_audit/audit_report.md`

![Şekil 1. AIForge-Doc v1 örnekleri: dört kaynak veri tabanı ve iki üreticiden sahte görüntüler; kırmızı bölge resmî maske, mavi çerçeve üstveri kutusudur.](dataset_audit/fig_examples_overlay.png)

**Üretim süreci.** Her belgede tek bir alanın değeri (tutar, tarih, telefon, adres gibi) değiştirilir. Alan kutusu her yönde %50 genişletilerek bir bağlam penceresi kesilir, üretici model bu pencereyi yeniden çizer ve yalnız alan kutusuna karşılık gelen pikseller özgün görüntüye geri yapıştırılır. Maske, geri yapıştırılan kutunun kendisidir.

**Ölçek.** Şekil 2, değiştirilen alanın görüntüye oranının dağılımını, Şekil 3 ise çözünürlük dağılımını göstermektedir. Alanların yarısı görüntünün yüzde birinden küçüktür ve tipik bir alan yalnız 33 piksel yüksekliğindedir. Görüntü sabit bir boyuta küçültülürse alan birkaç piksele iner; bu nedenle bütün modeller görüntüyü doğal çözünürlükte, kırpıntı ya da karo temelli işler. Piksel düzeyinde sınıf dengesizliği yaklaşık 1:100'dür.

![Şekil 2. Değiştirilen alanın görüntü alanına oranı: üreticiye göre histogram (solda) ve alan dilimlerine göre görüntü sayısı (sağda). Şekildeki üretici etiketleri veri setinin iç anahtarlarıdır: gemini-nano = Gemini 2.5 Flash Image, qwen-inpaint = Ideogram v2 Edit.](dataset_audit/fig_tamper_ratio.png)

![Şekil 3. Görüntü boyutlarının kaynak veri tabanına göre dağılımı (solda) ve en-boy oranı histogramı (sağda).](dataset_audit/fig_resolution.png)

**Denetim bulguları.** Denetim, deney tasarımını doğrudan belirleyen beş sonuç vermiştir.

1. Maskeler kutu düzeyindedir. Sahte ile özgün görüntü arasındaki kanal bazında en büyük mutlak fark 8'den büyük olan pikseller sayıldığında, kutu piksellerinin medyan %49.2'si gerçekten değişmiştir (çeyrekler arası %37.4–%62.0). Şekil 4 bu farkı örneklerle göstermektedir.
2. Çiftlerin %33.5'inde kutu dışında da değişen piksel vardır (Gemini %35.9, Ideogram %11.9); tipik neden üreticinin alanın bulunduğu satırın tamamını yeniden çizmesidir.
3. Test klasöründeki özgün görüntüler test sahtelerinin eşi değildir: 812 dosyanın yalnız 86'sı bir test sahtesinin kaynağıdır. Bu yüzden görüntü düzeyi tespit ölçütleri resmî test üzerinde değil, doğrulama bölümündeki eşli görüntülerde hesaplanmıştır.
4. Test sahtelerinin 72'si (%8.9) eğitim bölümündeki bir kaynak belgeyle aynı gruptadır. Kalan 740 görüntü sızıntısız test alt kümesi olarak ayrıca raporlanmıştır.
5. Üstverideki 12 satırda kutu koordinatı maskeyle uyuşmamaktadır; referans olarak her zaman maske dosyası kullanılmıştır.

![Şekil 4. Kutu maskesi ile piksel-değişim haritasının karşılaştırması: sahte, özgün, resmî maske ve farkın 8'den büyük olduğu pikseller.](dataset_audit/fig_pixel_diff_quads.png)

**Bölme.** Resmî test bölümüne dokunulmamıştır. Eğitim bölümü, algısal özet uzaklığı ve doğrulanmış çift bağlantıları üzerinden kurulan kaynak belge gruplarına göre ayrılmıştır; aynı grubun görüntüleri hem eğitime hem doğrulamaya düşmez. Dersin "kısıtlı veri" koşulu için ayrıca 400 ve 100 görüntülük iki küçük alt küme tanımlanmıştır; bunlar alan dilimine göre tabakalı, kaynak ve üretici hücrelerine göre dengelidir. Bölmelerin bileşimi Tablo 7'de verilmiştir. Eğitim bölümünden 14 kusurlu çift (sahtesi özgünle neredeyse aynı olan ya da bütün değişimi kutu dışında kalan) çıkarıldığında tam veri deneylerinde 2.748 eğitim görüntüsü kalmaktadır.

Table: Tablo 7. Veri bölmelerinin bileşimi.

| Alt küme | Görüntü | WildReceipt / CORD / SROIE / XFUND | Gemini / Ideogram | Alan dilimi: %1 altı / %1–5 / %5–20 | Grup |
|---|---|---|---|---|---|
| Eğitim | 2.762 | 1.159 / 679 / 638 / 286 | 2.466 / 296 | 1.397 / 1.354 / 11 | 2.619 |
| Doğrulama | 487 | 198 / 121 / 119 / 49 | 446 / 41 | 231 / 254 / 2 | 469 |
| Resmî test | 812 | 339 / 200 / 189 / 84 | 727 / 85 | 429 / 381 / 2 | 795 |
| Küçük eğitim (smoke_train) | 400 | 87 / 85 / 170 / 58 | 254 / 146 | 202 / 196 / 2 | 390 |
| Küçük doğrulama (smoke_val) | 100 | 28 / 20 / 38 / 14 | 75 / 25 | 47 / 52 / 1 | 100 |

Kaynak: `dataset_audit/audit_report.md`, `dataset_audit/splits.json`

**Karıştırıcı değişkenler.** Üretici, kaynak veri tabanı, alan dilimi ve işaretçi kalıntısı (Bölüm 3.9) arasındaki ilişkiler ki-kare tabanlı Cramér V katsayısıyla ölçülmüştür. $n$ gözlemli ve en küçük boyutu $k$ olan bir olumsallık tablosu için

$$V=\sqrt{\frac{\chi^{2}}{n\,(k-1)}}$$

tanımlanır. Tablo 8 en güçlü ilişkinin üretici ile işaretçi arasında olduğunu göstermektedir: kutu çevresinde yeşil piksel Gemini sahtelerinin %85.9'unda, Ideogram sahtelerinin %2.1'inde bulunur. WildReceipt kaynağında hiç Ideogram düzenlemesi yoktur. Üç bölme arasında üretici, kaynak, alan ve işaretçi dağılımları farklı değildir (ki-kare $p\geq 0.30$).

Table: Tablo 8. Veri tabanındaki değişken çiftleri arasında Cramér V katsayıları.

| Değişken çifti | V | p | Yorum |
|---|---|---|---|
| Üretici × işaretçi | 0.608 | 0.001'den küçük | Güçlü |
| Kaynak × alan dilimi | 0.483 | 0.001'den küçük | Orta |
| Üretici × kaynak | 0.429 | 0.001'den küçük | Orta |
| Üretici × dil | 0.391 | 0.001'den küçük | Orta |
| İşaretçi × alan dilimi | 0.324 | 0.001'den küçük | Orta |
| Üretici × alan dilimi | 0.177 | 0.001'den küçük | Zayıf |

Kaynak: `phase3/results/dataset_curation.md`

**AIForge-Doc v2 (harici test kümesi).** İkinci sürüm, v1 düzenleme tariflerini GPT-Image-2 ile yeniden üretir ve yalnız alan kutusunu geri yapıştırır [7]. Bu projede v2 üzerinde hiçbir eğitim ya da eşik ayarı yapılmamış, küme yalnız v1 modellerinin dışarıdan sınanması için kullanılmıştır. v2, v1 ile aynı kaynak belgeleri kullandığından sızıntı kuralı puanlamadan önce sabitlenmiştir: birincil küme, v1 karşılığı test bölümünde olan ve kaynak belgesi v1 eğitiminde hiç görünmeyen 567 çifttir. v2 sahteleri özgün kaynaklarından yalnız kutu içinde ayrılmaktadır: 3.066 çiftin hiçbirinde kutu dışında farklı piksel yoktur ve işaretçi kuralı yalnız baştan sona yeşil tonlu tek bir fişte, kutunun içinde tetiklenmektedir (`phase3_v2/results/v2_marker_paired_check.json`).

### 3.2 Sistem mimarisi ve işlem hattı

Projenin bütün adımları Şekil 5'te gösterilen tek bir hat üzerinde düzenlenmiştir. Hat üç aşamada büyümüştür. Faz 1, dersin kısıtlı veri koşulunu karşılayan küçük ölçekli çalışırlık denemesidir. Faz 2, aynı kodu tam veriye taşır ve dört hipotezi üç tohumla sınar. Faz 3, ikinci mimariyi, kutu dedektörünü, işaretçisiz yeniden eğitimi, v2 harici testini ve açıklanabilirlik modülünü ekler. Bu üç aşamanın ardından eklenen Faz 4 yalnız ResNet-18 yama sınıflandırıcısını ilgilendirir: model tam veriyle yeniden eğitilmiş ve görüntü düzeyi karar eşiği kalibre edilmiştir (Bölüm 3.5 ve 4.9, `kod/phase4_resnet/`). Şekil 5 ilk üç fazı gösterir; Faz 4 şekle ayrı bir kutu olarak eklenmemiştir.

![Şekil 5. Sistemin genel mimarisi: veri, veri denetimi, referans türetimi ve kontroller, üç fazdaki modeller, ortak değerlendirme ve çıktılar (her kutunun altında ilgili kod klasörü).](reports/figures/fig_sistem_mimarisi.png)

Hat, şekildeki numaralarla altı adımdan oluşur.

1. **Veri:** AIForge-Doc v1 (eğitim ve test) ile yalnız harici test için kullanılan v2 birincil kümesi.
2. **Veri denetimi:** dosya düzeyinde denetim, grup farkında eğitim ve doğrulama bölmesi, kısıtlı veri alt kümesi, kusurlu çiftlerin dışlanması.
3. **Referans türetimi ve kontroller:** resmî kutu maskesine ek olarak piksel-değişim ve glif referansları, işaretçisiz varyantları ve zararsız kontrol kümeleri (Bölüm 3.9).
4. **Modeller:** klasik öznitelikler ile SVM ve Random Forest (Bölüm 3.4), ResNet-18 (Bölüm 3.5), U-Net (Bölüm 3.6), DeepLabV3+ (Bölüm 3.7) ve YOLO11 (Bölüm 3.8); eğitilen ağırlıklar hem değerlendirmeye hem uygulamaya girer.
5. **Değerlendirme:** bütün lokalizasyon modelleri tam görüntü için 8 bitlik bir olasılık haritası üretir; kutu dedektörünün kutuları da güven değeriyle haritaya boyanır. Böylece bütün model aileleri aynı koddan geçer: üç referans türüne göre piksel ölçütleri, kutu düzeyi ortalama kesinlik, görüntü düzeyi ayırt edicilik, sondalar, bozulmalar, grup bootstrap ve Holm düzeltmesi (Bölüm 3.10).
6. **Çıktılar:** sonuç tabloları ve şekiller, rapor ve bildiri, açıklanabilirlik modülü (Bölüm 3.11) ve Hugging Face üzerinde çalışan etkileşimli uygulama (Bölüm 3.12).

### 3.3 Ön işleme

Adli görüntü analizinde aranan sinyal düşük düzeyli istatistiklerdedir: gürültü artığı, yeniden örnekleme izi, harf kenarlarının biçimi. Görüntüyü iyileştirmeye yönelik her işlem bu sinyali de değiştirebilir. Bu nedenle model girdisine yalnız izi koruyan ve sahte ile özgün görüntüye aynı biçimde uygulanan adımlar uygulanmıştır. Ders kapsamındaki diğer işlemler (gürültü giderme, histogram eşitleme, keskinleştirme, geometrik dönüşüm) model girdisinde kullanılmamış; etkileri test zamanı bozulması olarak ölçülmüş (Bölüm 4.5) ve uygulamanın ön işleme sekmesinde etkileşimli olarak denenebilir bırakılmıştır. Uygulanan adımlar ve gerekçeleri Tablo 9'da özetlenmiştir.

Table: Tablo 9. Ön işleme adımları, kullanıldıkları yer ve gerekçeleri.

| Adım | Kullanıldığı yer | Gerekçe |
|---|---|---|
| Kayıpsız PNG okuma, BGR'den RGB'ye dönüşüm | Tüm modeller | Sıkıştırma izi eklememek |
| Doğal çözünürlükte kırpma (256, 512 ya da 1024 piksel), küçük görüntüde dolgu | Tüm modeller | Yeniden örnekleme izinden kaçınmak; 33 piksellik alanı korumak |
| ImageNet ortalama ve standart sapmasıyla normalizasyon | ResNet-18, U-Net, DeepLabV3+ | Ön eğitimli ağırlıklarla uyum |
| 255'e bölme | YOLO11 | Kütüphanenin girdi ölçeği |
| Parlaklık ve karşıtlıkta ±%8 değişim | Eğitim | Aydınlatma farklarına karşı hafif artırım |
| Geometrik artırım yok (U-Net, ResNet) | Eğitim | Metin yönünü ve piksel ızgarasını korumak |
| Gri düzeye dönüşüm | Klasik öznitelikler | Doku ve frekans özniteliklerinin tanımı |
| FFT yüksek geçiren ve Laplace kanalları | U-Net beş kanallı varyant | Frekans bilgisinin katkısını sınamak |
| Morfolojik açma ve kapama, Sauvola ikilileştirme | Yalnız referans türetimi | Piksel-değişim ve glif maskelerinin temizlenmesi |
| İşaretçi onarımı | İşaretçisiz yeniden eğitim | Üretim kalıntısının eğitimden çıkarılması |

Kaynak: `baselines/README.md`, `phase2/experiments/README.md`, `kod/phase3/make_marker_repaired.py`

Görüntülerden örnek üretimi Algoritma 1'de özetlenmiştir. Sınıflandırma için üç tür yama çıkarılır: değiştirilen alanı içeren yama, aynı sahte görüntüden alana değmeyen temiz yama ve eşleşen özgün görüntüden aynı koordinatlı yama. Sonuncusu içerik denetimli negatiftir: içerik ve konum aynı kalır, yalnız düzenleme farklıdır.

**Algoritma 1. Doğal çözünürlükte yama ve kırpıntı üretimi**

```
Girdi : sahte görüntü I_f, maske M, varsa eşleşen özgün görüntü I_a, boyut s
Çıktı : sınıflandırma yamaları P, bölütleme kırpıntıları C
 1  B <- M'nin sınırlayıcı kutusu
 2  -- sınıflandırma (s = 256)
 3  c <- B'nin merkezi + rastgele kayma (en çok 32 piksel)
 4  P <- P ∪ { (kırp(I_f, c, s), etiket = 1) }                 # değiştirilmiş yama
 5  eğer B ile kesişmeyen s×s pencere w bulunursa:
 6      P <- P ∪ { (kırp(I_f, w, s), etiket = 0) }             # aynı görüntüden temiz
 7  eğer I_a varsa ve boyutu I_f ile aynıysa:
 8      P <- P ∪ { (kırp(I_a, c, s), etiket = 0) }             # içerik denetimli negatif
 9  -- bölütleme (s = 512), her epokta görüntü başına iki kırpıntı
10  u <- [0,1) aralığından düzgün rastgele sayı
11  eğer u < 0.7:
12      q <- eğitim referansındaki rastgele bir pozitif piksel + kayma (en çok s/4)
13  değilse:
14      q <- görüntüde rastgele konum
15  C <- C ∪ { (kırp(I_f, q, s), kırp(referans, q, s)) }
16  görüntü s'den küçükse kenar dolgusu uygula; yeniden boyutlandırma yapma
```

### 3.4 Klasik öznitelikler ile SVM ve Random Forest

Her gri düzey yamadan beş gruptan oluşan 56 boyutlu bir öznitelik vektörü çıkarılır (Tablo 10). Bu bölüm dersin uzamsal ve frekans uzayı süzgeçleri, doku analizi ve öznitelik çıkarımı konularının uygulamasıdır [35].

**Blok DCT.** Yama 8×8 bloklara bölünür ve her bloğa ayrık kosinüs dönüşümü uygulanır [36]:

$$F(u,v)=\frac{1}{4}C(u)C(v)\sum_{x=0}^{7}\sum_{y=0}^{7} f(x,y)\cos\frac{(2x+1)u\pi}{16}\cos\frac{(2y+1)v\pi}{16},\qquad C(k)=\begin{cases}1/\sqrt{2}, & k=0\\ 1, & k>0\end{cases}$$

Zikzak sırasındaki ilk 15 AC katsayısının bloklar üzerindeki ortalaması ve standart sapması, DC katsayısının standart sapması ve düşük, orta, yüksek bant enerji oranları öznitelik olarak alınır.

**LBP.** Merkez pikselin gri değeri $g_c$, çevresindeki $P=8$ komşunun değerleri $g_p$ olmak üzere yerel ikili örüntü kodu

$$\mathrm{LBP}_{P,R}=\sum_{p=0}^{P-1}s(g_p-g_c)\,2^{p},\qquad s(z)=\begin{cases}1, & z\geq 0\\ 0, & z<0\end{cases}$$

ile hesaplanır [37]. Yarıçap $R=1$ ve tekdüze örüntü kodlamasıyla 10 kutulu histogram elde edilir.

**GLCM.** Görüntü 32 gri düzeye indirgenir; bir piksel uzaklıkta ve dört yönde eş-oluşum matrisi $p(i,j)$ hesaplanır [38]. Karşıtlık $f_1$, homojenlik $f_2$, enerji $f_3$ ve korelasyon $f_4$ yönler üzerinde ortalanır:

$$f_1=\sum_{i,j}(i-j)^{2}p(i,j),\quad f_2=\sum_{i,j}\frac{p(i,j)}{1+|i-j|},\quad f_3=\sum_{i,j}p(i,j)^{2},\quad f_4=\sum_{i,j}\frac{(i-\mu_i)(j-\mu_j)\,p(i,j)}{\sigma_i\sigma_j}$$

**Laplace artığı.** Görüntü 3×3 Laplace çekirdeğiyle evriştirilir, $r=f*\nabla^{2}$, ve artığın mutlak ortalaması, standart sapması, basıklığı ve çarpıklığı alınır. Bu, yumuşak içerikten arındırılmış yüksek frekans bileşeninin özetidir.

**FFT ışınsal spektrum.** İki boyutlu Fourier dönüşümünün genlik spektrumu $|\mathcal{F}(u,v)|$ merkeze uzaklık $D(u,v)=\sqrt{(u-P/2)^{2}+(v-Q/2)^{2}}$ değerine göre üç banda ayrılır; bant enerji oranları ve logaritmik spektrumun eğimi öznitelik olur.

Table: Tablo 10. Klasik öznitelik grupları (toplam 56 öznitelik).

| Grup | Sayı | İçerik | İlgili ders konusu |
|---|---|---|---|
| Blok DCT | 34 | 15 AC katsayısının ortalaması ve standart sapması (30), DC standart sapması (1), bant enerji oranları (3) | Dönüşüm kodlama, frekans uzayı |
| LBP | 10 | Tekdüze örüntü histogramı | Doku tanımlama |
| GLCM | 4 | Karşıtlık, homojenlik, enerji, korelasyon | Doku tanımlama |
| Laplace artığı | 4 | Mutlak ortalama, standart sapma, basıklık, çarpıklık | Uzamsal keskinleştirme süzgeçleri |
| FFT ışınsal | 4 | Üç bant enerji oranı ve spektrum eğimi | Frekans uzayında süzgeçleme |

Kaynak: `baselines/README.md`, `kod/baselines/features_classical.py`

**Sınıflandırıcılar.** Öznitelikler eğitim kümesinin ortalama ve standart sapmasıyla ölçeklenir. Destek vektör makinesi, sınıfları en geniş payla ayıran hiper düzlemi arar [39]:

$$\min_{w,b,\xi}\ \frac{1}{2}\| w\|^{2}+C\sum_{i}\xi_i,\qquad y_i\left(w^{\top}\phi(x_i)+b\right)\geq 1-\xi_i,\quad \xi_i\geq 0$$

Burada ikinci ifade her örnek için sağlanması gereken kısıttır. Doğrusal olmayan ayrım için ışınsal tabanlı çekirdek $K(x,x')=\exp\left(-\gamma\| x-x'\|^{2}\right)$ kullanılır ve karar fonksiyonu $f(x)=\sum_i \alpha_i y_i K(x_i,x)+b$ olur. $C$ ve $\gamma$ üç katlı çapraz doğrulamayla seçilmiştir ($C=0.5$, $\gamma=0.01$). Rastgele orman, bootstrap örneklerle eğitilen 500 karar ağacının oy çokluğudur [40]; bir düğümün saflığı Gini ölçüsü $G=1-\sum_{c}p_c^{2}$ ile değerlendirilir ve bir özniteliğin önemi, o öznitelikle yapılan bölmelerin ortalama Gini azalmasıdır. Öznitelik çıkarımından değerlendirmeye kadar olan akış Algoritma 2'de verilmiştir.

**Algoritma 2. Klasik öznitelik çıkarımı ile SVM ve Random Forest eğitimi ve çıkarımı**

```
Girdi : yama kümesi {(P_i, y_i)}, sınıflandırıcı türü
Çıktı : eğitilmiş sınıflandırıcı, doğrulama ölçütleri
 1  her yama P_i için:
 2      g      <- gri(P_i)
 3      f_dct  <- blok_dct_istatistikleri(g, blok = 8, ac = 15)        # 34 değer
 4      f_lbp  <- histogram(LBP(g, P = 8, R = 1, tekdüze), 10)         # 10 değer
 5      f_glcm <- yön_ortalaması(GLCM(g, 32 düzey, d = 1))             # 4 değer
 6      f_lap  <- istatistik(evriştir(g, Laplace3x3))                  # 4 değer
 7      f_fft  <- ışınsal_bant_oranları(|FFT2(g)|) ve eğim             # 4 değer
 8      x_i    <- birleştir(f_dct, f_lbp, f_glcm, f_lap, f_fft)        # 56 boyut
 9  (mu, sigma) <- eğitim öznitelikleri üzerinde ortalama ve standart sapma
10  X <- (X - mu) / sigma
11  eğer tür = SVM:
12      (C, gamma) <- 3 katlı çapraz doğrulama ızgarası
13      model <- SVM_RBF(C, gamma).eğit(X_eğitim, y_eğitim)
14  değilse:
15      model <- RastgeleOrman(ağaç = 500).eğit(X_eğitim, y_eğitim)
16      önem  <- model.gini_önemleri
17  -- çıkarım
18  skor <- model.karar(X_doğrulama)
19  raporla: doğruluk, dengeli doğruluk, AUROC, AUPRC, F1
20  ayrıca AUROC'u negatif türüne göre ayır (aynı görüntü, eşleşen özgün)
```

### 3.5 ResNet-18 ile yama sınıflandırma

ResNet, bir katman grubunun doğrudan eşlemeyi değil artık fonksiyonu öğrenmesi düşüncesine dayanır [41]. Girdisi $x$ olan bir temel blok

$$y=\mathrm{ReLU}\left(\mathcal{F}(x,\{W_1,W_2\})+x\right),\qquad \mathcal{F}=\mathrm{BN}\left(W_2*\mathrm{ReLU}\left(\mathrm{BN}(W_1*x)\right)\right)$$

çıktısını üretir. Kısa yol bağlantısı gradyanın derin katmanlara zayıflamadan ulaşmasını sağlar; uzamsal boyut ya da kanal sayısı değiştiğinde kısa yol 1×1 evrişimle eşlenir. Yığın normalizasyonu her kanalı $\hat{x}=\gamma\,(x-\mu_B)/\sqrt{\sigma_B^{2}+\epsilon}+\beta$ biçiminde ölçekler. ResNet-18, 7×7 adım 2 giriş evrişimi ve en büyük havuzlamanın ardından her biri iki temel blok içeren dört evreden (64, 128, 256, 512 kanal), küresel ortalama havuzlamadan ve tam bağlı katmandan oluşur. Bu projede son katman tek çıkışlı bir katmanla değiştirilmiştir; model 11.18 milyon parametrelidir.

Çıkış logiti $z$ sigmoid ile olasılığa çevrilir, $\hat{p}=\sigma(z)=1/(1+e^{-z})$, ve ikili çapraz entropi en küçüklenir:

$$\mathcal{L}_{\mathrm{BCE}}=-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log\hat{p}_i+(1-y_i)\log(1-\hat{p}_i)\right]$$

Ders notundaki "önceden eğitilmiş model ile yeni katman" ve "kendi modeli" karşılaştırması üç ayarla yapılmıştır: gövdesi dondurulmuş, yalnız yeni tam bağlı katmanı eğitilen doğrusal sonda; bütün ağırlıkların güncellendiği tam ince ayar; rastgele başlatılan sıfırdan eğitim. Eğitim AdamW [50] ve kosinüs öğrenme oranı çizelgesiyle 12 epok, yığın boyutu 32 ve karma duyarlıklı hesaplamayla yapılmıştır. Girdi, 256 piksellik doğal çözünürlüklü yamanın 224 piksellik kesitidir; yeniden boyutlandırma ve yatay çevirme uygulanmaz. Üç ayarın ortak eğitim döngüsü Algoritma 3'te gösterilmiştir.

**Algoritma 3. ResNet-18 eğitimi (üç ayar)**

```
Girdi : yama kümeleri D_eğitim, D_doğrulama; ayar ∈ {sonda, ince_ayar, sıfırdan}
Çıktı : en iyi doğrulama AUROC değerine sahip ağırlıklar
 1  eğer ayar = sıfırdan:  model <- resnet18(rastgele başlatma)
 2  değilse:               model <- resnet18(ImageNet ağırlıkları)
 3  model.fc <- Doğrusal(512 -> 1)
 4  eğer ayar = sonda:  model.fc dışındaki bütün parametreleri dondur
 5  eniyileyici <- AdamW(eğitilebilir parametreler); çizelge <- kosinüs
 6  en_iyi <- 0; sabır <- 0
 7  epok = 1 .. 12 için:
 8      her yığın (x, y) ∈ D_eğitim için:
 9          x <- rastgele 224×224 kesit(x); parlaklık ve karşıtlık ±%8
10          z <- model(x)
11          kayıp <- BCE(sigmoid(z), y)
12          geri yayılım; eniyileyici adımı
13      a <- AUROC(sigmoid(model(merkez kesit(x_doğrulama))), y_doğrulama)
14      eğer a > en_iyi:  en_iyi <- a; ağırlıkları kaydet; sabır <- 0
15      değilse:          sabır <- sabır + 1; sabır = 5 ise dur
```

Tam görüntü üzerinde karar için model kayan pencereyle uygulanır (Algoritma 4). Görüntü skoru en yüksek yama olasılığıdır.

**Algoritma 4. ResNet-18 ile tam görüntüde kayan pencere çıkarımı**

```
Girdi : görüntü I (H×W), eğitilmiş model, pencere 224, adım 112, eşik 0.5
Çıktı : yama ısı haritası S, görüntü skoru s, karar
 1  I boyutu 224'ten küçükse kenar dolgusu uygula
 2  her (x, y) ∈ ızgara(H, W, pencere = 224, adım = 112) için:
 3      p <- sigmoid(model(normalize(I[y:y+224, x:x+224])))
 4      p değerini pencerenin konumuyla birlikte S'ye kaydet
 5  s <- en_büyük(bütün yama olasılıkları)
 6  karar <- SAHTE eğer s >= 0.5, değilse GERÇEK
```

**Tam veriyle yeniden eğitim ve karar eşiğinin kalibrasyonu.** Yukarıdaki üç ayar kısıtlı veriyle (400 görüntü, 1.587 yama) eğitilmiştir ve Algoritma 4'teki sabit 0.5 eşiği doğrulanmamış bir karar kuralıdır. Bu nedenle tam ince ayar tarifi hiçbir hiperparametresi değiştirilmeden tam eğitim bölümüne taşınmıştır: 2.748 görüntüden 10.795 yama (5.496 değiştirilmiş, 2.748 eşleşen özgün, 2.551 aynı görüntüden temiz), üç tohum (42–44). İkinci bir koşu kümesinde aynı koordinatlardaki yamalar, yeşil işaretçi kalıntısı onarılmış sahte görüntülerden kesilmiştir (Bölüm 3.9; onarım 2.126 eğitim görüntüsünü etkiler). Yama indeksi bütün koşularda sabittir; tohum yalnız yeni katmanın başlangıç ağırlıklarını, yığın sırasını ve artırımı değiştirir. Görüntü düzeyi karar için bütün yama skorları Algoritma 4'ün penceresiyle üretilmiş (224 piksel, adım 112; doğrulama görüntülerinde görüntü başına medyan 75 yama), görüntü skoru için en yüksek yama ile en yüksek 2, 3 ve 5 yamanın ortalaması aday olarak sınanmıştır. Toplama kuralı ve karar eşiği yalnız doğrulama kümesinde (487 sahte ve 487 eşleşen özgün görüntü) seçilmiştir: toplama kuralı tam veri modelinin üç tohumunun ortalama AUROC değerine göre, eşik ise model başına Youden J istatistiğini en büyükleyen değer olarak. Bu durumda Algoritma 4'ün 6. satırındaki 0.5 değerinin yerini kalibre eşik alır. Resmî test bölümü (812 sahte ve eşi bulunan 89 özgün görüntü) yalnız raporlama için kullanılmıştır. Sonuçlar Bölüm 4.9'dadır.

### 3.6 U-Net ile piksel düzeyinde lokalizasyon

U-Net, daralan bir kodlayıcı yolu ile genişleyen bir kod çözücü yolundan oluşur; aynı çözünürlükteki kodlayıcı özellikleri atlama bağlantılarıyla kod çözücüye taşınır [43]. Bu yapı, 33 piksel yüksekliğindeki alanlar gibi küçük hedeflerde ince uzamsal ayrıntının korunmasını sağlar. Projede kodlayıcı olarak ImageNet ön eğitimli ResNet-18 kullanılmıştır; model 14.328.209 parametrelidir. Şekil 6, 512×512 girdiden olasılık haritasına kadar veri akışını göstermektedir.

![Şekil 6. ResNet-18 kodlayıcılı U-Net mimarisi: kodlayıcı evreleri, atlama bağlantıları, kod çözücü blokları ve olasılık haritasından ikili maskeye geçiş.](reports/figures/UNet_ResNet18_Mimarisi.png)

Kodlayıcının $l$. evresinin çıktısı $e_l$, kod çözücünün $l$. düzeydeki çıktısı $d_l$ olmak üzere her kod çözücü bloğu şu işlemi yapar:

$$d_l=\mathcal{C}\left(\left[\,\mathrm{Up}_{2}(d_{l+1})\ ;\ e_l\,\right]\right),\qquad \mathcal{C}=\left(\mathrm{ReLU}\circ\mathrm{BN}\circ\mathrm{Conv}_{3\times 3}\right)^{2}$$

Burada $\mathrm{Up}_{2}$ en yakın komşu ile iki kat büyütme, köşeli parantez kanal ekseninde birleştirmedir. Girdinin 1/2, 1/4, 1/8, 1/16 ve 1/32 çözünürlüklerinde sırasıyla 64, 64, 128, 256 ve 512 kanallı özellikler üretilir; kod çözücü 256, 128, 64, 32 ve 16 kanallı beş blokla tam çözünürlüğe döner. Son 3×3 evrişim piksel başına bir logit $z_i$ verir ve $\hat{p}_i=\sigma(z_i)$ o pikselin değiştirilmiş olma olasılığıdır.

**Kayıp fonksiyonu.** Pozitif piksellerin oranı çok düşük olduğundan yalnız çapraz entropi, ağı her şeyi negatif saymaya iter. Bu nedenle piksel bazlı çapraz entropiye bölge örtüşmesini doğrudan ölçen Dice terimi eklenmiştir [44]:

$$\mathcal{L}=\mathcal{L}_{\mathrm{BCE}}+\mathcal{L}_{\mathrm{Dice}},\qquad \mathcal{L}_{\mathrm{Dice}}=1-\frac{2\sum_i \hat{p}_i y_i+\epsilon}{\sum_i \hat{p}_i+\sum_i y_i+\epsilon}$$

**Frekans kanallı varyant.** Dersin frekans uzayı konusunun katkısını ölçmek için girdiye iki kanal eklenmiştir: Gauss yüksek geçiren süzgeç çıktısı ve Laplace artığı. Süzgecin aktarım fonksiyonu ve çıktısı

$$H_{\mathrm{HP}}(u,v)=1-\exp\left(-\frac{D^{2}(u,v)}{2D_0^{2}}\right),\qquad g=\mathcal{F}^{-1}\left\{H_{\mathrm{HP}}\cdot\mathcal{F}\{f\}\right\},\qquad D_0=20$$

biçimindedir. Beş kanallı girdide ilk evrişimin RGB kanalları ön eğitimli ağırlıkları korur.

**Birleşik kayıp (Faz 3).** Kutu ve maskeyi birlikte öğrenen çok görevli modeller, iki denetim türünün birleştirilebileceğini göstermiştir [55]. Piksel düzeyi referansın kesinliğini kutu referansının kapsayıcılığıyla birleştirmek için ağa ikinci bir çıkış kanalı eklenmiştir. Birinci kanal piksel-değişim ya da glif referansıyla, ikinci kanal kutu maskesiyle denetlenir; değerlendirmede yalnız birinci kanal kullanılır:

$$\mathcal{L}_{\mathrm{ortak}}=\mathcal{L}\left(\hat{p}^{(1)},y^{\mathrm{piksel}}\right)+0.5\,\mathcal{L}\left(\hat{p}^{(2)},y^{\mathrm{kutu}}\right)$$

**Eğitim.** Küçük veri denemesinde (Faz 1) 15 epok ve yığın boyutu 8; tam veri deneylerinde (Faz 2 ve 3) 10 epok, yığın boyutu 16, AdamW ($10^{-4}$, kosinüs çizelge), karma duyarlık ve 42, 43, 44 tohumları kullanılmıştır. En iyi model, doğrulama kırpıntılarında eğitim referansına göre hesaplanan AUPRC ile seçilir. Değiştirilen tek etken eğitim maskesidir. Yöntem Algoritma 5'te, tam görüntü çıkarımı Algoritma 6'da verilmiştir.

**Algoritma 5. U-Net eğitimi (tek etken: eğitim referansı)**

```
Girdi : eğitim kimlikleri T (2.748), doğrulama kimlikleri V (487),
        referans türü g ∈ {kutu, piksel, glif}, tohum, epok sayısı E
Çıktı : doğrulama kırpıntı AUPRC değeri en yüksek ağırlıklar
 1  rastgele sayı üreteçlerini tohumla
 2  model <- UNet(kodlayıcı = ResNet18(ImageNet), giriş kanalı = 3, sınıf = 1)
 3  eniyileyici <- AdamW(öğrenme oranı = 1e-4); çizelge <- kosinüs(E)
 4  V_sabit <- her doğrulama görüntüsünden 1 pozitif merkezli + 1 rastgele kırpıntı
 5  en_iyi <- 0; sabır <- 0
 6  epok = 1 .. E için:
 7      her görüntü t ∈ T için Algoritma 1 ile 2 kırpıntı (x, m) üret, m = referans g
 8      her yığın (16 kırpıntı) için:
 9          x <- parlaklık ve karşıtlık ±%8; ImageNet normalizasyonu
10          z <- model(x)                                  # bf16 karma duyarlık
11          p <- sigmoid(z)
12          kayıp <- BCE(z, m) + Dice(p, m)
13          geri yayılım; eniyileyici adımı
14      çizelge adımı
15      a <- havuzlanmış AUPRC(sigmoid(model(V_sabit.x)), V_sabit.m)
16      eğer a > en_iyi:  en_iyi <- a; ağırlıkları kaydet; sabır <- 0
17      değilse:          sabır <- sabır + 1; sabır = 4 ise dur
18  birleşik kayıp seçilmişse: sınıf = 2; kayıp <- L(kanal 1, m) + 0.5 · L(kanal 2, kutu)
```

**Algoritma 6. Tam görüntüde kayan pencere çıkarımı ve son işleme (U-Net, DeepLabV3+)**

```
Girdi : görüntü I (H×W), model, pencere 512, adım 448, eşik tau*
Çıktı : 8 bitlik olasılık haritası P, ikili maske, kutular, görüntü skoru
 1  eğer en_büyük(H, W) > 4096:  I' <- I'yi uzun kenar 4096 olacak biçimde küçült
 2  değilse:                     I' <- I
 3  toplam <- 0; sayaç <- 0
 4  her (x, y) ∈ ızgara(I', pencere = 512, adım = 448; son pencere kenara yaslanır) için:
 5      p <- sigmoid(model(normalize(I'[y:y+512, x:x+512])))
 6      toplam[y:y+512, x:x+512] += p; sayaç[y:y+512, x:x+512] += 1
 7  P <- toplam / sayaç                                   # örtüşmede ortalama
 8  küçültme yapıldıysa P'yi çift doğrusal aradeğerleme ile H×W'ye büyüt
 9  P <- yuvarla(255 · P)                                 # 8 bit nicemleme
10  maske  <- P >= tavan(255 · tau*)
11  kutular <- maskenin bağlı bileşenlerinin sınırlayıcı kutuları;
12             kutu skoru = bileşen içindeki en büyük olasılık
13  skor   <- P'nin en yüksek %1 pikselinin ortalaması
14  karar  <- SAHTE eğer skor >= görüntü eşiği (doğrulamada Youden J), değilse GERÇEK
```

### 3.7 DeepLabV3+

Tek mimariyle elde edilen bulguların U-Net kod çözücüsüne özgü olup olmadığını sınamak için aynı ResNet-18 kodlayıcıyı kullanan DeepLabV3+ eğitilmiştir [45]. Model 12.33 milyon parametrelidir. U-Net ayrıntıyı atlama bağlantılarıyla taşırken DeepLabV3+ bağlamı boşluklu (atrous) evrişimle genişletir. Bir boyutlu gösterimle, genişleme oranı $r$ olan boşluklu evrişim

$$y[i]=\sum_{k}x[i+r\cdot k]\,w[k]$$

biçimindedir; çekirdek öğeleri arasına $r-1$ boşluk bırakılır. Boyutu $k$ olan bir çekirdeğin etkin boyutu $k_e=k+(k-1)(r-1)$ olur; parametre sayısı artmadan alıcı alan büyür. Boşluklu uzamsal piramit havuzlama (ASPP) modülü, kodlayıcının son özellik haritası $F$ üzerinde farklı oranlarda paralel dallar çalıştırır ve çıktıları birleştirir:

$$\mathrm{ASPP}(F)=\mathrm{Conv}_{1\times 1}\left(\left[\,\mathrm{Conv}_{1\times 1}(F)\ ;\ \mathrm{Conv}^{r_1}_{3\times 3}(F)\ ;\ \mathrm{Conv}^{r_2}_{3\times 3}(F)\ ;\ \mathrm{Conv}^{r_3}_{3\times 3}(F)\ ;\ \mathrm{Up}\left(\mathrm{GAP}(F)\right)\right]\right)$$

Kod çözücü, ASPP çıktısını dört kat büyütür, 1/4 çözünürlükteki düşük düzeyli kodlayıcı özellikleriyle birleştirir, 3×3 evrişimlerle inceltir ve yeniden dört kat büyüterek tam çözünürlüğe ulaşır (Algoritma 7). Eğitim protokolü U-Net ile aynıdır (kırpıntı örnekleme, kayıp, eniyileyici, epok, model seçimi); çıkarım Algoritma 6 ile yapılır. Üç referans türünün her biri için 42 tohumuyla birer model eğitilmiştir.

**Algoritma 7. DeepLabV3+ ileri geçişi**

```
Girdi : normalleştirilmiş kırpıntı x (3×512×512)
Çıktı : piksel logit haritası z (1×512×512)
 1  (e_1/4, ..., e_son) <- ResNet18_kodlayıcı(x)        # düşük ve yüksek düzey özellikler
 2  dallar <- [ Conv1x1(e_son),
 3              BoşlukluConv3x3(e_son, oran r1),
 4              BoşlukluConv3x3(e_son, oran r2),
 5              BoşlukluConv3x3(e_son, oran r3),
 6              Büyüt(KüreselOrtalamaHavuz(e_son)) ]
 7  a <- Conv1x1(birleştir(dallar))                      # ASPP çıktısı
 8  a <- Büyüt(a, kat = 4)
 9  l <- Conv1x1(e_1/4)                                  # düşük düzey özelliklerin kanal indirgemesi
10  h <- Conv3x3(birleştir(a, l))
11  z <- Büyüt(Conv1x1(h), kat = 4)
12  eğitim ve çıkarım: Algoritma 5 ve Algoritma 6 ile aynı
```

### 3.8 YOLO11 ile kutu tespiti

Resmî referans dikdörtgen bir kutu olduğuna ve her sahte görüntüde tam bir bölge bulunduğuna göre görev, bölütlemeden çok tek sınıflı nesne tespitine benzemektedir. Kutu maskesiyle eğitilen U-Net'in de dikdörtgenimsi tahminler ürettiği Faz 2'de gözlenmiştir (Bölüm 4.2). Bu nedenle kutuyu doğrudan öğrenen bir dedektör, kutu referansının neyi ödüllendirdiğini gösteren bir başvuru modeli olarak eklenmiştir. Kutu temelli manipülasyon tespiti bölge öneren dedektörlerle uzun süredir yapılmaktadır [56]; tek aşamalı dedektörler kutuyu doğrudan regresyonla bulur [46].

**Model.** YOLO11, evrişim ve C3k2 bloklarından oluşan bir omurga, uzamsal piramit havuzlama, dikkat bloğu, özellik piramidi boynu ve çapasız, ayrık bir tespit başlığından oluşur [47]. Başlık üç ölçekte (adım 8, 16 ve 32) her ızgara hücresi için bir sınıf skoru ve kutunun dört kenarına olan uzaklıkların ayrık dağılımını üretir. Küçük (n, 2.60 milyon parametre), orta-küçük (s, 9.43 milyon) ve orta (m, 20.05 milyon) boyutları denenmiştir; ana model YOLO11s'tir.

**Kayıp.** Eğitim kaybı üç terimin ağırlıklı toplamıdır:

$$\mathcal{L}_{\mathrm{YOLO}}=\lambda_{\mathrm{box}}\,\mathcal{L}_{\mathrm{CIoU}}+\lambda_{\mathrm{cls}}\,\mathcal{L}_{\mathrm{BCE}}+\lambda_{\mathrm{dfl}}\,\mathcal{L}_{\mathrm{DFL}}$$

Tahmin kutusu $b$ ile gerçek kutu $b^{g}$ arasındaki örtüşme $\mathrm{IoU}=|b\cap b^{g}|/|b\cup b^{g}|$ ile ölçülür. Tam IoU kaybı buna merkez uzaklığı ve en-boy oranı uyumunu ekler:

$$\mathcal{L}_{\mathrm{CIoU}}=1-\mathrm{IoU}+\frac{\rho^{2}(b,b^{g})}{c^{2}}+\alpha v,\qquad v=\frac{4}{\pi^{2}}\left(\arctan\frac{w^{g}}{h^{g}}-\arctan\frac{w}{h}\right)^{2},\qquad \alpha=\frac{v}{(1-\mathrm{IoU})+v}$$

Burada $\rho$ iki kutunun merkezleri arasındaki uzaklık, $c$ ikisini kapsayan en küçük kutunun köşegenidir. Dağılım odaklı kayıp, sürekli hedef uzaklık $y$ değerini komşu iki ayrık değer $y_l$ ve $y_r$ arasında paylaştırır: $\mathcal{L}_{\mathrm{DFL}}=-\left[(y_r-y)\log S_l+(y-y_l)\log S_r\right]$; $S_l$ ve $S_r$ bu iki değere atanan olasılıklardır. Sınıf terimi ikili çapraz entropidir.

**Karolama.** Medyan alan yüksekliği 33 pikseldir; görüntü 640 piksele küçültülürse hedef, en ince tespit ölçeğinin altına iner. Küçük nesne tespiti dedektörlerin bilinen zayıflığıdır [57]. Bu nedenle görüntüler doğal çözünürlükte 1024×1024 karolara bölünmüş (adım 768), çıkarım da karolu yapılmıştır [48]. Kutunun en az yarısını içeren her karo, kırpılmış kutuyla pozitif etiketlenir; görüntü başına bir rastgele karo boş etiketle negatif olarak eklenir. Karolar kayıpsız PNG olarak yazılır.

**Eğitim.** Aynı bölmeler ve aynı dışlanan çiftler kullanılmıştır. Yeniden örnekleme içeren artırımlar adli izi bozabileceğinden varsayılan profil korumacıdır: mozaik ve ölçek artırımı kapalıdır, yalnız küçük öteleme, yatay çevirme ve hafif doygunluk ile parlaklık değişimi açıktır. AdamW ($10^{-3}$), yığın boyutu 8, 30 epok ve 10 epokluk sabır kullanılmıştır. Planlanan $10^{-2}$ öğrenme oranı ilk duman denemesinde doğrulama kaybını tanımsız yaptığı için düşürülmüş, epok sayısı da süre bütçesi nedeniyle 40'tan 30'a indirilmiştir; bu sapmalar deney günlüğüne işlenmiştir. Model seçimi doğrulama karolarındaki kütüphane uygunluk ölçütüyle (0.1 × mAP50 + 0.9 × mAP50-95) yapılır.

**Bölütleme ile ortak değerlendirme.** Dedektörün her kutusu, güven değeriyle tam görüntü boyutunda bir haritaya boyanır; örtüşen kutularda en büyük güven alınır: $P(x,y)=\max_{j:\,(x,y)\in b_j}c_j$. Bu harita, bölütleme modellerinin olasılık haritasıyla aynı dosya biçimindedir ve aynı değerlendirme kodundan geçer. Böylece iki formülasyon üç referans türü altında aynı ölçütlerle karşılaştırılabilir. Deney matrisi Tablo 11'de, yöntem Algoritma 8 ve Algoritma 9'da verilmiştir.

Table: Tablo 11. YOLO11 deney matrisi (tek etken kuralı; ana model dışındakiler 42 tohumu).

| Koşu | Değişen etken | Değer | Amaç |
|---|---|---|---|
| Ana model | Tohum | 42, 43, 44 | U-Net kutu modelleriyle eşli ana sonuç |
| Tam görüntü 1280 | Girdi rejimi | Görüntünün tamamı 1280 piksele küçültülür | Karolamanın küçük hedeflere etkisi (H5c) |
| Varsayılan artırım | Artırım profili | Kütüphane varsayılanı (mozaik, ölçek) | Korumacı profilin gerekliliği (H5e) |
| Özgün negatif | Negatif örnek | Görüntü başına bir boş etiketli özgün karo | Yanlış alarm etkisi |
| Boyut n ve m | Model boyutu | YOLO11n, YOLO11m | Doğruluk ve maliyet dengesi |
| Yalnız Gemini | Eğitim üreticisi | Ideogram eğitimden çıkarılır | Ayrı tutulan üreticiye aktarım (H5d) |

Kaynak: `phase3_yolo/README.md`, `phase3/results/tables_p3.md`

**Algoritma 8. YOLO11 için karo veri kümesi hazırlama ve eğitim**

```
Girdi : eğitim ve doğrulama kimlikleri, kutu koordinatları, karo 1024, adım 768
Çıktı : en iyi ağırlıklar (best.pt)
 1  her görüntü I ve kutusu B için:
 2      K <- ızgara(I, karo = 1024, adım = 768) pencereleri
 3      her k ∈ K için:
 4          eğer alan(B ∩ k) >= 0.5 · alan(B):
 5              B' <- B'nin k içine kırpılmış ve k'ye göre normalleştirilmiş hâli
 6              (k görüntüsü, etiket "0 xc yc w h") yaz                # pozitif karo
 7      kalan karolardan rastgele 1 tanesini boş etiketle yaz          # negatif karo
 8      (ablasyon) eşleşen özgün görüntüden 1 rastgele karo, boş etiket
 9  model <- YOLO11s (ön eğitimli ağırlıklar), tek sınıf
10  artırım <- korumacı: mozaik = 0, ölçek = 0, öteleme = 0.1, yatay çevirme = 0.5
11  eğit(model, girdi boyutu = 1024, eniyileyici = AdamW, lr0 = 1e-3,
12        yığın = 8, epok = 30, sabır = 10, tohum)
13  her epokta kayıp = L_CIoU + L_BCE + L_DFL (ağırlıklı toplam)
14  doğrulama karolarında uygunluk = 0.1·mAP50 + 0.9·mAP50-95; en iyisini kaydet
```

**Algoritma 9. Karolu çıkarım, kareler arası bastırma ve haritaya boyama**

```
Girdi : görüntü I (H×W), YOLO modeli, karo 1024, adım 768, conf = 0.001, IoU = 0.6
Çıktı : kutu listesi, 8 bitlik güven haritası P
 1  kutular <- boş liste
 2  her (x, y) ∈ ızgara(I, 1024, 768; son karo kenara yaslanır) için:
 3      D <- model(I[y:y+1024, x:x+1024], conf = 0.001)       # karo içi NMS dahil
 4      her tespit (x0, y0, x1, y1, c) ∈ D için:
 5          kutular'a (x0 + x, y0 + y, x1 + x, y1 + y, c) ekle  # tam görüntü koordinatı
 6  -- kareler arası en büyük olmayanı bastırma
 7  kutular'ı c'ye göre azalan sırala; tutulan <- boş liste
 8  kutular boş değilken:
 9      b <- ilk kutu; tutulan'a ekle
10      kutular <- { k ∈ kutular : IoU(b, k) <= 0.6 }
11  -- haritaya boyama
12  P <- sıfır(H, W)
13  her (x0, y0, x1, y1, c) ∈ tutulan için:
14      P[y0:y1, x0:x1] <- en_büyük(P[y0:y1, x0:x1], yuvarla(255 · c))
15  P'yi bölütleme haritalarıyla aynı klasör düzenine kaydet -> ortak değerlendirme
16  (tam görüntü modeli) karolama yerine görüntü 1280 piksele küçültülür
```

### 3.9 Referans (GT) türetimi ve kontrol kümeleri

Resmî maske kutu düzeyindedir. Eğitim bölümündeki sahtelerin özgün kaynakları yayımlandığı için, hangi piksellerin gerçekten değiştiği doğrudan ölçülebilir. Sahte görüntü $I^{f}$ ve özgün görüntü $I^{a}$ için fark haritası

$$D(x,y)=\max_{c\in\{R,G,B\}}\left|I^{f}_{c}(x,y)-I^{a}_{c}(x,y)\right|$$

olarak tanımlanır. Bu haritadan iki referans türetilmiştir.

**Piksel-değişim referansı.** $A=\{D>\varepsilon\}$ kümesi morfolojik işlemlerle temizlenir. Yapı elemanı $S$ için aşınma $A\ominus S$ ve genişleme $A\oplus S$ üzerinden açma ve kapama

$$A\circ S=(A\ominus S)\oplus S,\qquad A\bullet S=(A\oplus S)\ominus S$$

biçimindedir. Referans, $\varepsilon=8$ ile 3×3 açma, alanı 8 pikselden küçük bağlı bileşenlerin atılması ve 3×3 kapama adımlarıyla elde edilir. Açma tek tük gürültü piksellerini siler, kapama harf vuruşlarındaki küçük boşlukları doldurur.

**Glif referansı.** Yalnız değişen karakterlerin mürekkep piksellerini hedefler. Metin pikselleri Sauvola yerel eşiğiyle bulunur [49]:

$$T(x,y)=m(x,y)\left[1+k\left(\frac{s(x,y)}{R}-1\right)\right]$$

Burada $m$ ve $s$ pencere içindeki yerel ortalama ve standart sapma, $R$ standart sapmanın dinamik aralığıdır; pencere 25 piksel ve $k=0.2$ alınmıştır. Sahte ve özgün görüntünün metin piksellerinin birleşimi, $\{D>8\}$ kümesinin 3 piksel genişletilmişiyle kesiştirilir. Koyu zeminli belgelerde kutu içindeki koyu piksel oranı 0.5'i aşarsa kutuplama ters çevrilir.

**Eşik duyarlılığı.** Şekil 7, $\varepsilon$ eşiğinin etkisini göstermektedir. $\varepsilon=0$ iken çiftlerin %47.8'inde kutu dışındaki piksellerin yüzde birinden fazlası değişmiş görünür; bu, görüntünün tümüne yayılan ±1–3 düzeyindeki yeniden kodlama farkıdır ve $\varepsilon=4$ ile %4.7'ye, $\varepsilon=8$ ile %1.3'e iner. Seçilen eşik bu küresel izi dışarıda bırakır, yerel ve gerçek değişimi korur. Referans varyantlarının kutuyla ilişkisi Tablo 12'de verilmiştir: piksel-değişim referansı kutunun medyan %35'ini, glif referansı %29'unu kaplar. Glif referansında kutu dışına taşan piksellerin çoğu, 3 piksellik genişletmenin kutu kenarını aşan dar bandından kaynaklanır.

![Şekil 7. Fark eşiğine duyarlılık: (a) pozitif piksel oranı, (b) kutu dışı değişim içeren çift oranları, (c) üreticiye göre kutu dışı değişim.](phase2/gt/fig_eps_sensitivity_pub.png)

Table: Tablo 12. Türetilen referans varyantlarının 3.336 çift üzerindeki istatistikleri.

| Varyant | Tanım | Kutu kapsama (medyan) | Kutu ile IoU (medyan) | Kutu dışı pikseli olan çift |
|---|---|---|---|---|
| pixel_eps4 | Fark 4'ten büyük | 0.666 | 0.620 | %46.1 |
| pixel_eps8 | Fark 8'den büyük | 0.492 | 0.481 | %33.5 |
| pixel_eps16 | Fark 16'dan büyük | 0.354 | 0.351 | %12.1 |
| pixel_eps8_morph (piksel) | Fark 8'den büyük, açma, bileşen süzme, kapama | 0.351 | 0.350 | %4.8 |
| glyph_v2 (glif) | Sauvola metin pikselleri ile genişletilmiş değişim kesişimi | 0.286 | 0.269 | %66.4 |

Kaynak: `phase2/gt/gt_report.md`

**Yeşil işaretçi kalıntısı ve işaretçisiz varyantlar.** Referans doğrulama panelleri incelenirken sahte görüntülerde kutunun iç kenarı boyunca 1–3 piksel kalınlığında doygun yeşil bir çizgi fark edilmiştir. Bu, veri tabanı makalesinde anlatılan yeşil işaret dikdörtgeninin geri yapıştırılan bölgeye düşen kalıntısıdır [1]. Kalıntı şu renk kuralıyla bulunur:

$$g(x,y)=\left[\,G-\max(R,B)>50\,\right]\wedge\left[\,G>110\,\right],\qquad K=g(I^{f})\wedge\neg\,g(I^{a})\wedge H_{6}(B)$$

Burada $K$ kalıntı piksellerinin kümesi, $H_{6}(B)$ kutu kenarının 6 piksel içini ve dışını kapsayan halkadır. En az 20 kalıntı pikseli olan görüntü işaretçili sayılır. Eşleşen 3.336 çiftin %75.4'ünde kalıntı vardır (Gemini %83.6, Ideogram %1.5); 600 özgün görüntüde yanlış alarm çıkmamıştır. Test sahtelerinin çoğunun eşi olmadığından, test alt küme çözümlemelerinde yalnız kutu halkasındaki doygun yeşil piksel varlığına bakan eşsiz gösterge kullanılmış; buna göre 812 test görüntüsünün 627'si işaretçili, 185'i işaretçisizdir. Bu etiket yalnız alt küme ayrımı içindir; model girdisi değiştirilmez.

İşaretçi çizgisi özgün görüntüden farklı olduğu için piksel-değişim referansının da parçasıdır (etkilenen çiftlerde pozitif piksellerin medyan %10'u). Bu nedenle iki ek ürün üretilmiştir: kalıntı piksellerinin özgün görüntüden geri yazıldığı işaretçisi onarılmış eğitim görüntüleri ve kalıntı piksellerinin çıkarıldığı işaretçisiz referanslar (`pixel_eps8_morph_nm`, `glyph_v2_nm`). Türetim adımları Algoritma 10'da toplanmıştır.

**Algoritma 10. Referans türetimi, işaretçi tespiti ve onarımı**

```
Girdi : sahte I_f, özgün I_a, kutu maskesi B
Çıktı : piksel referansı G_p, glif referansı G_g, işaretçisiz varyantlar, onarılmış görüntü
 1  D <- kanallar üzerinde en_büyük(|I_f - I_a|)
 2  A <- (D > 8)
 3  G_p <- kapama(bileşen_süz(açma(A, 3×3), en küçük alan = 8), 3×3)
 4  T_f <- gri(I_f) < Sauvola(gri(I_f), pencere = 25, k = 0.2)      # metin pikselleri
 5  T_a <- gri(I_a) < Sauvola(gri(I_a), pencere = 25, k = 0.2)
 6  eğer kutu içindeki koyu piksel oranı > 0.5:  T_f, T_a kutuplamasını ters çevir
 7  G_g <- (T_f ∪ T_a) ∩ genişlet(A, 3 piksel)
 8  -- işaretçi
 9  yeşil(I) <- (G - en_büyük(R, B) > 50) ve (G > 110)
10  halka <- genişlet(B, 6) ve değil aşındır(B, 6)
11  kalıntı <- yeşil(I_f) ve değil yeşil(I_a) ve halka
12  işaretçili <- (kalıntı piksel sayısı >= 20)
13  k_gen <- genişlet(kalıntı, 1)
14  I_onarılmış <- I_f; I_onarılmış[k_gen] <- I_a[k_gen]
15  G_p_nm <- G_p ve değil k_gen;  G_g_nm <- G_g ve değil k_gen
16  -- sonda girdileri
17  özgün+işaretçi   <- I_a; [k_gen] pikselleri I_f'den alınır
18  yalnız yeniden çizim <- I_f; genişlet(G_p ∪ kalıntı, 2) pikselleri I_a'ya döndürülür
19  yalnız içerik    <- I_a; genişlet(G_p_nm, 2) pikselleri I_f'den alınır
```

**Kontrol kümeleri ve sondalar.** Modellerin sahtecilik izine mi yoksa üretim sürecinin yan ürünlerine mi tepki verdiğini ayırmak için doğrulama bölümündeki eşli görüntülerden Tablo 13'teki girdiler üretilmiştir. Zararsız geri yapıştırma, özgün görüntünün kutu bölgesini iki kat büyütüp küçülterek yerine koyar; içerik değişmez, yalnız yeniden örnekleme izi kalır. Bu kontrolde fark haritasının en büyük değeri medyan 17'dir; gerçek sahtelerde 180'dir. Şekil 8 altı sonda girdisini tek bir örnek üzerinde göstermektedir.

Table: Tablo 13. Kontrol kümeleri ve sonda girdileri.

| Girdi | Nasıl üretilir | Beklenen doğru karar | Neyi sınar |
|---|---|---|---|
| Sahte | Veri tabanındaki sahte görüntü | Sahte | Standart lokalizasyon |
| Özgün | Eşleşen özgün görüntü | Gerçek | Yanlış alarm |
| Sahte, işaretçi onarılmış | Yalnız kalıntı pikselleri özgünden kopyalanır | Sahte | İşaretçiye bağımlılık |
| Özgün + işaretçi | Özgüne yalnız kalıntı pikselleri aktarılır | Gerçek | İşaretçinin tek başına yeterliliği |
| Yalnız yeniden çizim | Sahtede değişen karakterler ve işaretçi özgüne döndürülür | Gerçek | Üreticinin alanı yeniden çizme dokusu |
| Yalnız içerik | Özgüne yalnız değişen karakterler yapıştırılır | Sahte | İçerik değişiminin tek başına etkisi |
| Zararsız geri yapıştırma | Kutu bölgesi 2 kat büyütülüp küçültülerek yerine konur | Gerçek | Yapıştırma dikişi ve yeniden örnekleme izi |
| Tam görüntü yeniden örnekleme | Görüntünün tamamı 2 kat büyütülüp küçültülür | Gerçek | Küresel yeniden örnekleme izi |
| Sınır, plasebo ve iç bulanıklık | Kutu kenarında 5 piksellik halka, kutu dışında aynı halka, kutu içi | Sahte | Sınır kısayolu (H2) |
| Kutu dışı kırpıntı sınıflandırıcısı | Maskeye değmeyen 224 piksellik kırpıntılarla ResNet-18 | — | Kutu dışında küresel iz (H4) |
| Test zamanı bozulmaları | JPEG kalite 75, 0.75 kat ölçek, Gauss bulanıklık (σ = 1), CLAHE, gri düzey | Sahte | Dayanıklılık (H3) |

Kaynak: `phase2/controls/controls_report.md`, `kod/phase3/probes_p3.py`, `kod/phase2/experiments/probes.py`

![Şekil 8. Sonda girdileri tek bir doğrulama çiftinde (kesikli çizgi resmî kutu): sahte, özgün, işaretçisi onarılmış sahte, özgün + işaretçi, yalnız yeniden çizim, yalnız içerik.](phase3/figures/fig_probe_conditions.png)

**Referansın denetimi.** Türetilen referansların yeniden kodlama gürültüsünden ayrıştığını sınamak için 7 tabakadan 100 çiftlik bir örneklem seçilmiştir. İlk denetim görüntü destekli bir dil modeliyle yapılmış, 100 çiftten 96'sında çözümlenebilir yanıt alınmıştır: piksel-değişim referansı 69 çiftte uygun, 26'sında kısmen uygun, 1'inde hatalı bulunmuştur. Aynı 100 çift üzerinde daha sıkı ölçütlü ikinci denetim yazar tarafından elle yapılmıştır (tek değerlendirici). Bu insan denetimi piksel-değişim referansını 36 çiftte uygun, 64'ünde kısmen uygun bulmuş, hiçbirini hatalı bulmamış; resmî kutuyu 100 çiftin 99'unda doğru saymıştır. İki denetimin uyumu zayıftır (Cohen κ = 0.07; %95 güven aralığı −0.08 ile 0.21; n = 96). İnsan denetiminin notlarına göre kısmen uygun sayılan 64 çiftin tamamında neden fazla kapsamadır: referans, alanın yeniden çizilen arka planını ya da değişmeyen içeriğini de işaretlemektedir (64 notun 46'sı arka planın, 43'ü değişmeyen içeriğin işaretlendiğini söyler; referansın değişen bölgeyi kaçırdığını bildiren not yoktur; `phase2/gt/partial_note_check.json`). İnsan denetimi tek değerlendiricilidir ve değerlendirici çalışmanın yazarıdır; değerlendiriciler arası uyum ölçülememiştir. Konu Bölüm 5'te sınırlama olarak ele alınmıştır.

### 3.10 Değerlendirme ölçütleri ve istatistik

**Piksel ölçütleri.** Bir eşikte doğru pozitif, yanlış pozitif ve yanlış negatif piksel sayıları TP, FP ve FN olmak üzere

$$P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN},\qquad F_1=\frac{2PR}{P+R},\qquad \mathrm{IoU}=\frac{TP}{TP+FP+FN}$$

tanımlanır. Eşikten bağımsız özet olarak kesinlik–duyarlılık eğrisinin altındaki alan (AUPRC, ortalama kesinlik) kullanılır. Olasılık haritaları 8 bit olduğundan her görüntü ve referans için 256 kutulu iki histogram tutulur: pozitif piksellerin histogramı $h^{+}$ ve bütün piksellerin histogramı $h$. Eşik $t$ için

$$TP(t)=\sum_{v\geq t}h^{+}(v),\qquad FP(t)=\sum_{v\geq t}\left[h(v)-h^{+}(v)\right],\qquad \mathrm{AP}=\sum_{k}\left(R_k-R_{k-1}\right)P_k$$

hesaplanır. Böylece AUPRC örnekleme yapılmadan bütün pikseller üzerinde kesin olarak bulunur; havuzlanmış değer histogramların toplamından, görüntü başı AP ise tek görüntünün histogramından elde edilir. Faz 1'de kullanılan alt örneklemeli hesap pozitif oranını yapay olarak yükselttiği için değerleri şişirmiştir; bu raporda Faz 1 değerleri kesin yöntemle yeniden hesaplanmıştır (Bölüm 4.1). ROC eğrisi altındaki alan (AUROC) 1:100 dengesizliğe duyarsız kaldığından tek başına raporlanmamıştır.

**Eşik kalibrasyonu.** Her model ve referans çifti için eşik, doğrulama kümesinde görüntü ortalaması F1'i en büyük yapan değer olarak seçilir ve teste değiştirilmeden taşınır:

$$\tau^{*}=\arg\max_{\tau\in\{0.05,\,0.10,\,\ldots,\,0.95\}}\ \frac{1}{N}\sum_{n=1}^{N}F_1^{(n)}(\tau)$$

Görüntü başına en iyi eşikteki F1 (Oracle-F1) ulaşılabilir üst sınır olarak ayrıca verilir.

**Kutu düzeyi ölçütler.** Bütün model aileleri için COCO biçiminde ortalama kesinlik hesaplanır [51]: IoU eşiği 0.5 için AP50 ve 0.50–0.95 aralığının ortalaması AP50-95. Dedektörde kutular doğrudan kullanılır; bölütleme modellerinde kutular Algoritma 6'daki bağlı bileşenlerden üretilir.

**Görüntü düzeyi ölçütler.** Görüntü skoru, olasılık haritasının en yüksek %1 pikselinin ortalamasıdır. Sahte ve eşleşen özgün doğrulama görüntüleri arasındaki ayrım AUROC ile ölçülür; karar eşiği Youden istatistiği $J=\mathrm{TPR}-\mathrm{FPR}$ en büyük olacak biçimde seçilir.

**Tahmin biçimi.** Tahminin dikdörtgenselliği, eşik üstü bölgenin alanının kendi sınırlayıcı kutusunun alanına oranıdır. Bozulma ve ablasyon etkileri göreli değişimle verilir: temiz girdideki ölçüt $m_0$, koşul altındaki ölçüt $m_c$ için $\mathrm{rel}\Delta=(m_c-m_0)/m_0$.

**Belirsizlik ve çoklu test.** Aynı kaynak belgeden türeyen görüntüler bağımsız olmadığından güven aralıkları kaynak belge grupları üzerinde yeniden örneklemeyle hesaplanır [52]. Tohumlar sabit etken sayılır: önce her görüntü için üç tohumun ortalaması alınır, ardından gruplar yerine koyarak $B=1000$ kez örneklenir ve yüzdelik %95 aralık bulunur. İki koşulun farkı için aynı grup örneklemi iki koşula birlikte uygulanır (eşli bootstrap); iki yönlü değer $p=2\min\left\{\Pr(\Delta^{*}\leq 0),\Pr(\Delta^{*}\geq 0)\right\}$ olur. Her hipotez ailesinin karar karşılaştırmalarına Holm düzeltmesi uygulanır [53]: $m$ testin değerleri küçükten büyüğe $p_{(1)}\leq\ldots\leq p_{(m)}$ sıralanır ve

$$\tilde{p}_{(i)}=\max_{j\leq i}\ \min\left\{1,\ (m-j+1)\,p_{(j)}\right\}$$

alınır. Toplam 58 önceden belirlenmiş karar testi vardır (H1 için 12, H2 için 21, H3 için 14, H4 için 3, ek karşılaştırmalar için 8). Model sıralamalarının referans türüne göre değişimi Kendall sıra uyumu katsayısıyla ölçülür [54]: uyumlu çift sayısı $C$, uyumsuz çift sayısı $U$ ve model sayısı $n$ için $\tau_K=(C-U)/\left(n(n-1)/2\right)$. Değerlendirme adımları Algoritma 11'de özetlenmiştir.

**Algoritma 11. Ortak değerlendirme: histogram tabanlı ölçütler, eşik kalibrasyonu ve grup bootstrap**

```
Girdi : model başına 8 bitlik haritalar P_n, referanslar G_n (kutu, piksel, glif),
        grup kimlikleri k_n, doğrulama ve test kümeleri
Çıktı : AUPRC, AP, F1@tau*, IoU@tau*, güven aralıkları, düzeltilmiş p değerleri
 1  her görüntü n ve her referans türü için:
 2      h_poz[n] <- histogram(P_n[G_n = 1], 256); h_tüm[n] <- histogram(P_n, 256)
 3  -- eşikten bağımsız ölçütler
 4  havuz AUPRC <- kesinlik_duyarlılık_alanı(Σ_n h_poz[n], Σ_n h_tüm[n])
 5  AP[n]       <- kesinlik_duyarlılık_alanı(h_poz[n], h_tüm[n])
 6  -- eşik kalibrasyonu (yalnız doğrulama)
 7  her tau ∈ {0.05, ..., 0.95} için:  F[tau] <- ortalama_n F1(h_poz[n], h_tüm[n], tau)
 8  tau* <- F'yi en büyük yapan tau
 9  -- test: tau* değiştirilmeden kullanılır
10  F1@tau*, IoU@tau* <- test histogramlarından
11  -- belirsizlik
12  her görüntü için üç tohumun ölçütünü ortala
13  b = 1 .. 1000 için:
14      grupları yerine koyarak örnekle; ölçütü (ya da iki koşulun farkını) hesapla
15  güven aralığı <- bootstrap dağılımının %2.5 ve %97.5 yüzdelikleri
16  p <- 2 · en_küçük(oran(fark <= 0), oran(fark >= 0))
17  her hipotez ailesinde p değerlerine Holm düzeltmesi uygula
18  model sıralamaları için referans çiftleri arasında Kendall katsayısını hesapla
```

### 3.11 Açıklanabilirlik

Ölçütler bir modelin ne kadar başarılı olduğunu söyler, neye bakarak karar verdiğini söylemez. Kısayol sorusu doğrudan ikincisiyle ilgili olduğundan uygulamaya birbirinden bağımsız iki açıklanabilirlik aracı eklenmiştir: kararı girdi bölgelerine paylaştıran SHAP çözümlemesi ve ilk evrişim katmanlarının öğrendiği süzgeçleri gösteren katman filtresi görselleştirmesi. Adli lokalizasyonda açıklama sadakatinin ayrıca denetlenmesi gerektiği literatürde de vurgulanmaktadır [58].

**SHAP.** Yöntem [59], işbirlikçi oyun kuramındaki Shapley değerine dayanır. Modelin skaler çıktısı $f$, öznitelik (burada görüntü bölgesi) kümesi $N$ ve $|N|=n$ için $i$. bölgenin katkısı, o bölgenin bütün olası alt kümelere eklenmesiyle oluşan çıktı farklarının ağırlıklı ortalamasıdır:

$$\phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!\,(n-|S|-1)!}{n!}\left[f(S\cup\{i\})-f(S)\right]$$

Bir alt küme $S$ için $f(S)$, $S$ dışındaki bölgeler bir referans görüntünün pikselleriyle değiştirilerek hesaplanır. Katkılar toplamsaldır:

$$f(x)=\phi_0+\sum_{i=1}^{n}\phi_i,\qquad \phi_0=f(x^{\mathrm{ref}})$$

Bütün alt kümeleri gezmek olanaksız olduğundan iki yaklaşık hesap kullanılmıştır. Birincil yöntem bölümleme (Partition) açıklayıcısıdır: görüntü hiyerarşik olarak bölünür ve bölgeler referansla maskelenerek Owen değerleri hesaplanır. Bu yöntem modelden bağımsızdır, uygulamanın kendi ön işleme ve çıkarım fonksiyonlarını olduğu gibi çağırır ve toplamsallığı kesin olarak sağlar. İkincil yöntem beklenen gradyanlardır; piksel düzeyinde daha ince bir doku haritası verir, toplamsallığı yaklaşıktır:

$$\phi_i\approx\mathbb{E}_{x'\sim\mathcal{R},\ \alpha\sim U(0,1)}\left[(x_i-x'_i)\,\frac{\partial f\left(x'+\alpha(x-x')\right)}{\partial x_i}\right]$$

Açıklanan hedef her zaman modelin gerçek çıktısından türetilen skaler bir değerdir: U-Net ve DeepLabV3+ için tam olasılık haritasının en yüksek %1 ortalaması (uygulamanın görüntü skoru), ResNet-18 için en yüksek yama skoru, YOLO11 için en güvenli kutunun güveni. Referans olarak eşleşen özgün görüntü kullanıldığında katkıların toplamı, sahte girdi $x^{f}$ ve özgün girdi $x^{a}$ için $f(x^{f})-f(x^{a})$ olur ve yalnız değiştirilmiş pikseller katkı alabilir; eş yoksa bulanıklaştırılmış görüntü kullanılır.

Tam çözünürlükte piksel düzeyi maskeleme pratik değildir. Bu nedenle maske, uzun kenarı 192 hücre olan bir ızgarada tanımlanır; her değerlendirmede ızgara tam çözünürlüğe taşınır ve görüntü, gösterge 1 olan hücrelerde özgün girdiden, diğerlerinde referanstan kurulur. Böylece tek hedef ve tek referansla görüntünün tamamı açıklanır. Bir sonuç yalnız üç koşul sağlandığında "tam görüntü SHAP (doğrulandı)" etiketini alır: harita boyutu görüntü boyutuna eşittir, hedef tam görüntü hedefidir ve $|\phi_0+\sum_i\phi_i-f(x)|<10^{-3}$. Pencere pencere hesaplanıp birleştirilen döşemeli mozaik ayrı bir seçenektir; pencerelerin taban değerleri farklı olduğundan tam görüntü SHAP olarak sunulmaz. Yöntem Algoritma 12'de verilmiştir.

Açıklamaların sadakati üç ölçüyle sınanır. Silme eğrisi altındaki alan (AOPC) [60] için pencere 16×16 bloğa bölünür, bloklar SHAP toplamına göre azalan sırada yüzde onluk adımlarla referansla değiştirilir ve çıktıdaki ortalama düşüş, $\mathrm{AOPC}=\frac{1}{K}\sum_{k=1}^{K}\left[f(x)-f(x_{(k)})\right]$, rastgele sırayla elde edilen düşüşle karşılaştırılır. İşaret etme oyununda en yüksek katkılı bloğun referans kutusuna değip değmediğine bakılır. Bölge kütlesi, pozitif katkı kütlesinin işaretçi piksellerinde ve kutu kenar halkasında toplanan payıdır; bu payın bölgenin alan payına oranı (yoğunlaşma katsayısı) birden çok büyükse karar o bölgeye yaslanıyor demektir.

**Algoritma 12. Tam görüntü SHAP (ızgara maskeli bölümleme açıklayıcısı)**

```
Girdi : görüntü x (H×W), model ailesi, referans türü, değerlendirme bütçesi max_evals
Çıktı : katkı haritası phi (H×W), taban değeri phi_0, doğrulama etiketi
 1  x_ref <- eşleşen özgün görüntü varsa o; yoksa bulanık(x, 32×32)
 2  ızgara <- (gh, gw): uzun kenar 192 hücre olacak biçimde
 3  hedef f(·):
 4      U-Net, DeepLabV3+ : tam görüntü çıkarımı -> en yüksek %1 piksel ortalaması
 5      ResNet-18         : kayan pencere -> en yüksek yama olasılığı
 6      YOLO11            : karolu tespit -> en güvenli kutunun güveni
 7  değerlendir(gösterge g ∈ {0,1}^(gh×gw)):
 8      G <- g'yi en yakın komşu ile H×W'ye büyüt
 9      x_g <- G ? x : x_ref
10      döndür f(x_g)            # pencere, yama ve karo çıktıları önbelleğe alınır
11  phi_ızgara <- BölümlemeAçıklayıcı(değerlendir, hiyerarşik bölme, bütçe = max_evals)
12  phi <- her hücre değerini hücrenin piksellerine eşit dağıt (toplam korunur)
13  phi_0 <- f(x_ref)
14  hata <- |phi_0 + toplam(phi) - f(x)|
15  etiket <- "doğrulandı" eğer boyut(phi) = (H, W) ve hata < 1e-3, değilse uyarı
16  göster: ısı haritası, bindirme, pozitif ve negatif katkı, bölge kütlesi, katkı dağılımı
```

**Katman filtreleri.** İkinci araç eğitim ya da üretim içermez; yalnız kayıtlı ağırlıkları ve modelin kendi ileri geçişini kullanır. Bir evrişim katmanının ağırlık tensörü $W\in\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times k\times k}$ doğrudan görselleştirilir. Katmanın girdisi $x$ için $f$. süzgecin yanıtı

$$a_f=\sum_{c=1}^{C_{\mathrm{in}}}W_{f,c}\star x_c+b_f$$

ile, ardından yığın normalizasyonu ve doğrusal olmayan etkinleştirme (ResNet gövdesinde ReLU, YOLO'da SiLU) ile elde edilir. "İlk katmanlar" sabit bir sayı değildir: ileri geçiş sırasındaki evrişim katmanları, çıktı çözünürlüğü girdinin en az dörtte biri olduğu sürece alınır ve bu sınır aşılınca ileri geçiş durdurulur. Her katmanda kayıtlı çekirdek, katmanın gerçek girdisine bağımsız olarak yeniden uygulanır ve modelin kendi çıktısıyla karşılaştırılır; analizden önce ve sonra ağırlıkların SHA-256 özeti denetlenir. Görüntünün tamamı doğal çözünürlükte, 1024 piksel çekirdek ve 64 piksel kenar paylı döşemelerle işlenir; pay ilk katmanların alıcı alanından büyük olduğundan sonuç tek geçişle aynıdır. Adımlar Algoritma 13'te özetlenmiştir.

**Algoritma 13. Katman filtreleri: çekirdek ağırlıkları ve tam görüntü etkinleşme haritaları**

```
Girdi : model dosyası, görüntü x (H×W)
Çıktı : katman listesi, çekirdek görselleri, etkinleşme haritaları, doğrulama farkları
 1  mimari <- ağırlık anahtarlarından otomatik tespit (U-Net, DeepLabV3+, ResNet-18, YOLO11)
 2  özet_önce <- SHA256(ağırlıklar)
 3  x_n <- modelin kendi ön işlemesi(x)                  # ImageNet, RGB+frekans ya da /255
 4  döşemeler <- 1024 piksel çekirdek + 64 piksel pay ile böl(x_n)
 5  her döşeme d için:
 6      ileri kancalarla modelin ileri geçişini çalıştır
 7      her Conv2d katmanı L için (birikimli adım <= 4 iken):
 8          W, b <- L'nin kayıtlı ağırlıkları
 9          a_model <- kancadan alınan evrişim çıktısı
10          a_bağımsız <- conv2d(L'nin girdisi, W, b, aynı adım ve dolgu)
11          fark[L] <- en_büyük(|a_model - a_bağımsız|) / ölçek
12          BN ve etkinleştirme çıktıları zincir kesintisizse kaydet
13      birikimli adım 4'ü aşınca ileri geçişi durdur
14      yalnız çekirdek bölgenin çıktılarını tam haritaya yaz
15  özet_sonra <- SHA256(ağırlıklar); özet_önce = özet_sonra olmalı
16  göster: çekirdek ızgarası, süzgeç–harita çiftleri, etkinleşme ızgarası,
17          özgün görüntüyle bindirme, kanal katkıları, en güçlü yanıt veren süzgeçler
```

### 3.12 Uygulama: mimari, sekmeler ve ekran görüntüleri

Bütün hat, Gradio ile yazılmış tek bir etkileşimli uygulamada çalıştırılabilir. Uygulama aynı kaynak koddan iki ortamda çalışır: yerelde `python kod/app/app.py` komutuyla ve Hugging Face Space üzerinde canlı olarak. Mimari Şekil 9'da gösterilmiştir.

![Şekil 9. Uygulama mimarisi: sunum katmanı (dokuz sekme), uygulama servisleri, alan çekirdeği, çıkarım katmanı, açıklanabilirlik paketleri, veri ve yapılandırma ile dağıtım zinciri; numaralar veri akışının adımlarıdır.](reports/figures/fig_uygulama_mimarisi.png)

Giriş noktası `kod/app/app.py` dosyasıdır. Uygulamanın iş mantığı `kod/app/appcore/` paketinde katmanlara ayrılmıştır. Sunum katmanı arayüz düzenini ve görsel temayı (`ui`, `theme`) içerir. Uygulama servisleri tek örnek akışını (`simulation`), toplu testi (`batch`), sekme olay işleyicilerini (`handlers`) ve açıklanabilirlik köprüsünü (`xai_bridge`) barındırır. Alan çekirdeği ön işleme işlemlerinden (`ops`), görüntü ve frekans yardımcılarından (`imaging`), girdi senaryoları ile bozulmalardan (`scenarios`) ve ölçütlerden (`metrics`) oluşur. Çıkarım katmanı model kaydını ve GPU çıkarımını (`models`), en altta veri ve yapılandırma katmanı ise yolları ve veri tabanı erişimini (`config`, `dataset`) yönetir. SHAP çözümlemesi `kod/app/xai/` paketinde, katman filtreleri `kod/app/xai_filters.py` dosyasındadır. Space paketi (`hf_space/`) elle tutulmaz; `kod/tools/build_space.py` betiğiyle bu kaynaklardan türetilir ve `kod/hf_space/deploy_hf.py` ile dağıtılır.

Arayüz dokuz sekmeden oluşur (Tablo 14). İlk iki sekme projenin bulgularını doğrudan sınamaya, 2–5 arası sekmeler ders konularını adım adım uygulamaya, son sekmeler sonuçları ve açıklamaları göstermeye yöneliktir.

Table: Tablo 14. Arayüz sekmeleri ve işlevleri.

| Sekme | İşlev |
|---|---|
| 0 · Simülasyon | Veri tabanından örnek seçimi (bölüm, kaynak, üretici, alan oranı, işaretçi, megapiksel süzgeçleri); girdi senaryosu ve test zamanı bozulması; seçilen modellerle yedi adımlı canlı günlük, adım adım galeri, yakın plan, model başına sonuç tablosu, otomatik yorum ve indirilebilir rapor |
| 1 · Toplu test | Aynı süzgeçlerle tohumlu N rastgele örnek; yalnız sahte ya da sahte ve özgün eşi; model başına ortalama F1, IoU, AP (iki referans), doğru ve yanlış alarm oranları, görüntü AUROC ve kayıtlı sonuçlarla yeniden üretim farkı |
| 2 · Girdi | Sahte görüntü, isteğe bağlı maske ve özgün eş yükleme ya da veri tabanından örnek seçme |
| 3 · Ön işleme (aşamalı pipeline) | 13 kategoride 61 işlem; önizle, hatta ekle, geri al; hazır ayarlar; önce ve sonra görüntüsü ile histogram; 56 klasik öznitelik tablosu |
| 4 · U-Net lokalizasyonu | Kayıtlı bütün lokalizasyon modelleri; tam görüntü çıkarım protokolü; kalibre eşik; Otsu ve morfolojik son işleme; kutu ya da piksel-değişim referansıyla ölçüm; özgün girdi ile ön işlenmiş girdinin karşılaştırması |
| 5 · ResNet-18 patch sınıflandırma | 224 piksellik kayan pencereyle yama ısı haritası, görüntü skoru, referansla kesişen ve temiz yamaların skorları |
| 6 · Yeşil işaretçi sondası | Renk kuralıyla kalıntı tespiti ve onarımı |
| 7 · Sonuçlar ve model kartı | Özet tablolar, görüntü düzeyi eşikler, ölçüt tanımları |
| 8 · Açıklanabilirlik (XAI) | SHAP Açıklamaları (tahmin, tam görüntü SHAP, model karşılaştırması, dışa aktarım) ve Katman Filtreleri (çekirdek ağırlıkları, etkinleşme haritaları, özgün görüntüyle karşılaştırma) |

Kaynak: `kod/app/README_APP.md`

Ön işleme sekmesindeki 13 kategori dersin konu başlıklarını izler: renk işleme, yoğunluk dönüşümleri, histogram işlemleri, eşikleme, gürültü ekleme ve giderme, keskinleştirme, kenar bulma, morfoloji, geometrik dönüşümler, normalizasyon, frekans uzayı süzgeçleri, klasik bölütleme ve dayanıklılık bozulmaları. Her işlem modelin girdisine uygulanıp çıktıya etkisi ölçülebilir; bu, Bölüm 3.3'teki "hangi ön işleme neden yapılmadı" sorusunun etkileşimli karşılığıdır.

Simülasyon sekmesindeki girdi senaryoları Tablo 13'teki ilk altı satırla aynıdır: sahte belge, özgün eş, işaretçisi onarılmış sahte, özgün ve yalnız işaretçi, yalnız yeniden çizim ve yalnız içerik. Bozulma seçenekleri JPEG (kalite 75 ve 50), yeniden ölçekleme (0.75 ve 0.5 kat), Gauss bulanıklığı, Gauss gürültüsü, CLAHE ve gri düzeydir. Ölçütler değerlendirme hattıyla aynı tanımları kullanır: olasılık haritası 8 bite nicemlenir, AP 256 kutulu histogramla bütün piksellerden hesaplanır, eşikler doğrulamada kalibre edilen değerlerdir. Simülasyonun akışı Algoritma 14'te verilmiştir.

**Algoritma 14. Uygulamada tek örnek simülasyonu (yedi adım)**

```
Girdi : örnek kimliği, senaryo, bozulma, seçilen modeller, isteğe bağlı ön işleme hattı
Çıktı : galeri, model başına sonuç tablosu, görüntü düzeyi karar, rapor dosyaları
 1  Veri yükleme     : sahte görüntü, kutu maskesi, üstveri, varsa özgün eş
 2  Referans analizi : D = |sahte - özgün|; piksel-değişim referansı; kutunun değişen oranı
 3  İşaretçi taraması: renk kuralı ile kalıntı pikselleri ve sayısı
 4  Girdi hazırlama  : senaryoya göre model girdisi (Tablo 13); bozulma; ön işleme hattı
 5  Çıkarım          : her model için tam görüntü olasılık haritası (GPU varsa GPU)
 6  Değerlendirme    : F1, IoU, AP, Oracle-F1 (kutu ve piksel referansı), kutu kapsama,
                       görüntü skoru ve eşiği, karar, doğru ya da yanlış işareti
 7  Rapor            : otomatik yorum; Markdown, JSON, CSV ve PNG çıktıları
```

**Dağıtım.** Canlı uygulama Hugging Face Space üzerinde, istek anında GPU ayıran ZeroGPU donanımıyla çalışır. Çıkarım fonksiyonları GPU dekoratörüyle işaretlenmiştir; yerelde bu dekoratör etkisizdir. Veri tabanı lisansı gereği ham veri Space deposuna konmamıştır: görüntüler, maskeler, model ağırlıkları ve denetim tabloları (12.208 dosya, 10.26 GiB) özel bir depolama kovasına yüklenmiş ve Space'e salt okunur olarak bağlanmıştır. Space deposunda yalnız kod, sonuç tabloları ve küçük üstveri dosyaları bulunur.

Arayüzün güncel ekran görüntüleri aşağıdadır. Şekil 10 bir eğitim örneğinde simülasyonu göstermektedir: kutu U-Net kutu referansıyla 0.984, aynı tahmin piksel-değişim referansıyla 0.276 F1 almakta, otomatik yorum bu farkı ve kutu kenarındaki 71 piksellik işaretçi kalıntısını bildirmektedir. Şekil 11 toplu testi, Şekil 12 bir hazır ön işleme hattının (gri düzey, Gauss süzgeci, Sauvola ikilileştirme, morfolojik açma) uygulanmasını, Şekil 13 U-Net sekmesinde olasılık haritasını, ikili maskeyi ve ölçütleri, Şekil 14 sonuç tablolarını, Şekil 15 doğrulanmış tam görüntü SHAP çıktısını, Şekil 16 ise ilk evrişim katmanının çekirdek ağırlıklarını göstermektedir.

![Şekil 10. Simülasyon sekmesi (örnek training:000000004): örnek süzgeçleri, senaryo ve model seçimi, model kararları, otomatik yorum, sonuç tablosu ve yedi adımlı günlük.](app/ekran_0_simulasyon.png)

![Şekil 11. Toplu test sekmesi (sekiz örnek, sahte ve özgün eşi): model bazında özet, iki referansla F1 karşılaştırması, görüntü düzeyi skorlar ve örnek bazında sonuçlar.](app/ekran_1_toplu_test.png)

![Şekil 12. Ön işleme sekmesi: belge ikilileştirme hazır hattı, işlem öncesi ve sonrası görüntü ile gri düzey histogramları.](app/ekran_3_on_isleme.png)

![Şekil 13. U-Net lokalizasyonu sekmesi: olasılık haritası, bindirme, eşik sonrası ikili maske ve ölçüt tablosu (kutu modeli, eşik 0.45).](app/ekran_4_unet.png)

![Şekil 14. Sonuçlar ve model kartı sekmesi: Faz 3 sentez tabloları (doğrulama ve resmî test).](app/ekran_7_sonuclar.png)

![Şekil 15. Açıklanabilirlik sekmesi, SHAP Açıklamaları: tahmin tablosu, tam görüntü SHAP sonuçları, doğrulama satırı ve özgün görüntü ile bindirme kaydırıcısı.](app/ekran_8_xai_shap.png)

![Şekil 16. Açıklanabilirlik sekmesi, Katman Filtreleri: U-Net kutu modelinin ilk katmanındaki 64 çekirdek, seçili süzgecin kanal bazında ağırlık matrisi ve doğrulama özeti.](app/ekran_8_xai_filtre.png)

### 3.13 Donanım ve yazılım ortamı

Bütün eğitim ve değerlendirmeler tek bir NVIDIA RTX 5080 (16 GB) ekran kartında, Windows üzerinde yapılmıştır. Yazılım sürümleri Tablo 15'te verilmiştir. Faz 2 deneyleri sırasında bölütleme kütüphanesinin 0.5.0 sürümü kuruluydu; Faz 3 için ortam 0.6.0 sürümüyle yeniden kurulmuş ve Faz 2 ağırlıklarının yeni ortamda doğrulama kümesinde aynı haritayı ürettiği denetlenmiştir (ortalama mutlak fark 0, en büyük F1 farkı 0).

Table: Tablo 15. Yazılım ortamı.

| Bileşen | Yerel ortam | Hugging Face Space |
|---|---|---|
| Python | 3.11 | 3.12 |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) | 2.11.0 |
| torchvision | 0.26.0+cu128 | 0.26.0 |
| segmentation-models-pytorch | 0.6.0 | 0.6.0 |
| ultralytics | 8.4.208 | 8.4.208 |
| Gradio | 6.31.0 | 6.31.0 |
| shap | 0.51.0 | 0.54.1 |
| scikit-learn, scikit-image | 1.9.1, 0.26.0 | sürüm sabitlenmemiş |
| OpenCV | opencv-python-headless 5.0.0.93 | sürüm sabitlenmemiş |

Kaynak: `requirements.txt`, `kod/hf_space/space_src/requirements.txt`, `kod/hf_space/space_src/README.md`, `phase3/results/repro_check_env.json`

Hesaplama maliyeti düşüktür: küçük veri denemesinin toplam GPU eğitim süresi 23.5 dakika, Faz 2'nin 12 U-Net eğitimi 124 dakikadır (koşu başına yaklaşık 10–11 dakika, en yüksek GPU belleği 4.66 GiB). YOLO11 koşuları 32–87 dakika sürer. Faz 1–3 hattının baştan sona yeniden üretimi tek GPU'da yaklaşık 14 GPU saatidir. Ek tohum kuyruğu (dört U-Net eğitimi, değerlendirme, sondalar ve v2 testi; `kod/phase3/q4_seeds_mr_joint.ps1`) buna 2 saat 10 dakika ekler.

### 3.14 Kod düzeni ve yeniden üretilebilirlik

Bütün kaynak kod tek çatı altında, `kod/` klasöründedir; veri, model ağırlıkları ve çıktılar proje kökünde kalır. Kural basittir: `kod/<alt klasör>/` altındaki bir betik, çıktısını proje kökündeki aynı adlı `<alt klasör>/` altına yazar. Kök, `kod/_paths.py` tarafından bulunur ve `VIZE_ROOT` ortam değişkeniyle değiştirilebilir.

```
kod/
  _paths.py                 proje kökü ve çıktı klasörü çözümleme
  run_everything.ps1        Faz 1–2 hattı (indirme -> denetim -> eğitim -> istatistik)
  dataset_audit/            audit_dataset.py
  baselines/                Faz 1: features_classical.py, train_classical.py,
                            train_resnet.py, train_unet.py, evaluate.py, run_all.ps1
  phase2/gt/                make_gt.py, make_verification_grid.py, score_human_verification.py,
                            partial_note_check.py
  phase2/controls/          make_controls.py
  phase2/literature_baselines/  run_trufor.py, run_adcd.py, evaluate_zero_shot.py
  phase2/experiments/       train_unet_p2.py, infer_fullimage.py, evaluate_p2.py, probes.py,
                            h4_global_trace.py, probe_green_marker.py, run_phase2.ps1
  phase2/stats/             stats_analysis.py
  phase3/                   train_unet_p3.py, make_marker_repaired.py, probes_p3.py, p3ext.py,
                            dataset_curation.py, phase1_exact_auprc.py, bench_speed.py,
                            yolo_controls.py, synthesize_p3.py, q0–q3 kuyruk betikleri
  phase3_yolo/              convert_to_yolo.py, train_yolo.py, infer_to_maps.py,
                            detection_metrics.py, run_phase3.ps1
  phase3_v2/                v2_eval.py, v2_report.py, v2_marker_split.py,
                            v2_marker_paired_check.py, run_v2.ps1
  phase4_resnet/            p4common.py, build_patches_p4.py, train_resnet_p4.py, eval_patches_p4.py,
                            infer_images_p4.py, calibrate_p4.py, run_phase4.ps1
  app/                      app.py, appcore/, xai/, xai_filters.py, README_APP.md
  hf_space/                 deploy_hf.py, live_test.py, space_src/
  scripts/                  xai_benchmark.py
  tests/                    test_app.py, test_app_p3.py, test_xai.py, test_xai_filters.py,
                            ui_test.py, run_all_tests.ps1
  tools/                    download_dataset.py, build_space.py, xai_filter_report.py,
                            xai_fullimage_bench.py, md2docx.py
```

Tablo 16, raporda kullanılan her ana çıktının hangi betikle üretildiğini göstermektedir. Ham veri, türetilmiş maskeler, onarılmış görüntüler, model ağırlıkları ve tahmin haritaları lisans ve boyut nedeniyle depoya dahil değildir; hepsi aşağıdaki betiklerle yeniden üretilir.

Table: Tablo 16. Betikler ve ürettikleri çıktılar.

| Betik | Ürettiği çıktı |
|---|---|
| `kod/tools/download_dataset.py` | `data/AIForge-Doc-v1/` |
| `kod/dataset_audit/audit_dataset.py` | `dataset_audit/audit_report.md`, `splits.json`, denetim şekilleri |
| `kod/baselines/run_all.ps1` | `baselines/results/results_tables.md`, `baselines/figures/`, Faz 1 ağırlıkları |
| `kod/phase2/gt/make_gt.py` | `phase2/gt/` altındaki referans varyantları, `gt_report.md` |
| `kod/phase2/gt/score_human_verification.py`, `partial_note_check.py` | İkinci referans denetiminin özeti (`human_verification_summary.json`, `.md`) ve kısmi kararların not sınıflaması (`partial_note_check.json`) |
| `kod/phase2/controls/make_controls.py` | `phase2/controls/` kontrol kümeleri, `controls_report.md` |
| `kod/phase2/literature_baselines/evaluate_zero_shot.py` | `phase2/literature_baselines/zero_shot_results.csv` |
| `kod/phase2/experiments/run_phase2.ps1` | `models/phase2/`, `phase2/experiments/results/summary_*_agg.csv`, `tau.json`, şekiller |
| `kod/phase2/experiments/probe_green_marker.py` | İşaretçi sondası sonuçları, `fig_green_marker.png` |
| `kod/phase2/stats/stats_analysis.py` | `phase2/stats/hypothesis_decisions.md`, `stats_summary.csv` |
| `kod/phase3/make_marker_repaired.py` | Onarılmış görüntüler, işaretçisiz referanslar, `phase3/results/marker_residue.csv` |
| `kod/phase3/train_unet_p3.py` | DeepLabV3+, birleşik kayıplı ve işaretçisiz U-Net ağırlıkları |
| `kod/phase3/q4_seeds_mr_joint.ps1`, `kod/phase3/seed_extra_stats.py` | Ek tohumlar (43, 44): işaretçisiz kutu ve birleşik kayıplı piksel U-Net; tohum bazında AUPRC ve eşli AP farkları (`phase3/results/extra_seed_stats.md`) |
| `kod/phase3_yolo/convert_to_yolo.py`, `kod/phase3_yolo/train_yolo.py` | `phase3_yolo/yolo_data/`, `phase3_yolo/runs/` |
| `kod/phase3_yolo/infer_to_maps.py`, `kod/phase3_yolo/detection_metrics.py` | Kutu listeleri, güven haritaları, kutu AP tabloları |
| `kod/phase3/probes_p3.py` | `phase3/results/probes_p3_summary_*.csv` |
| `kod/phase3/dataset_curation.py` | `phase3/results/dataset_curation.md` |
| `kod/phase3/phase1_exact_auprc.py` | `phase3/results/phase1_exact_auprc.csv` |
| `kod/phase3/bench_speed.py` | `phase3/results/speed_bench.csv` |
| `kod/phase3/synthesize_p3.py` | `phase3/results/tables_p3.md`, `phase3/figures/` |
| `kod/phase3/yolo_controls.py` | `phase3/results/T7b_yolo_controls.csv` |
| `kod/phase3_v2/run_v2.ps1` | `phase3_v2/results/v2_tables.md`, `v2_numbers.json` |
| `kod/phase3_v2/v2_marker_paired_check.py` | `phase3_v2/results/v2_marker_paired_check.json` |
| `kod/phase4_resnet/run_phase4.ps1` | `models/phase4/`, `phase4_resnet/results/tables_p4.md`, `resnet_image_tau.json` |
| `kod/scripts/xai_benchmark.py` | `docs/xai_test_report.md`, `docs/xai_benchmark.csv` |
| `kod/tools/xai_filter_report.py` | `reports/xai_filters/RAPOR.md` ve şekilleri |
| `kod/tools/xai_fullimage_bench.py` | `reports/xai_fullimage/RAPOR_OLCUM.md` |
| `kod/tests/run_all_tests.ps1` | `tests/run_all_*.log`, `tests/run_all_summary.txt` |
| `kod/hf_space/live_test.py` | `reports/canli_test_v4deploy_log.txt`, `reports/canli_test_v4deploy_faz4_log.txt` |
| `kod/tools/build_space.py` | `hf_space/` Space paketi |

Kaynak: `kod/run_everything.ps1`, `kod/tests/run_all_tests.ps1`, `README.md`

### 3.15 Kullanılan ve kullanılmayan yöntemler

Tablo 17, değerlendirilen yöntem ailelerini ve her biri için verilen kararı gerekçesiyle özetlemektedir.

Table: Tablo 17. Kullanılan ve kullanılmayan yöntemler.

| Yöntem ailesi | Durum | Gerekçe |
|---|---|---|
| Klasik öznitelikler ile SVM ve Random Forest | Kullanıldı (Faz 1) | Ders yöntemlerinin doğrudan uygulaması; derin modeller için alt sınır |
| ResNet-18 (sonda, ince ayar, sıfırdan) | Kullanıldı | Ön eğitimli model ile kendi modelinin karşılaştırması; kutu dışı iz sondası; tam veriyle yeniden eğitim ve karar eşiği kalibrasyonu (Faz 4) |
| U-Net (ResNet-18 kodlayıcı) | Kullanıldı (ana model) | Küçük alanlar için atlama bağlantıları; düşük eğitim maliyeti |
| DeepLabV3+ | Kullanıldı (Faz 3) | Bulguların kod çözücü mimarisinden bağımsızlığını sınamak |
| YOLO11 kutu dedektörü | Kullanıldı (Faz 3) | Referansın geometrisine birebir uyan formülasyon; kutu ölçütünün ne ölçtüğünü göstermek |
| Birleşik piksel ve kutu kaybı | Kullanıldı (Faz 3; piksel ve glif modelleri üç tohum) | İki denetim türünü birleştirmenin etkisi |
| TruFor, ADCD-Net | Yalnız sıfır-atış başvuru | Eşit koşul değildir; ayrı raporlanmıştır |
| TruFor ince ayarı | Yapılamadı | Yayımlanan kodda eğitim betiği yok |
| DTD, CAT-Net, FFDN (RGB ve DCT) | Kullanılmadı | Veri kayıpsız PNG; JPEG ve DCT ipucu bulunmuyor |
| SegFormer, IML-ViT gibi dönüştürücü tabanlı modeller | Kullanılmadı | Referans etkisini az sayıda mimariyle yalıtmak; veri boyutu |
| Çok kipli dil modeli dedektörleri | Yalnız referans denetim aracı | Veri tabanı makalesinde şans düzeyinde sonuç |
| Model topluluğu | Kullanılmadı | Süre bütçesi; gelecek çalışma |

Kaynak: `README.md`, `reports/PHASE3_Sonuclar_ve_Kapanis.md`

Yöntemler birbirinden yalıtık değildir. ImageNet ön eğitimli ResNet-18 ağırlıkları hem sınıflandırıcıda hem iki bölütleme modelinin kodlayıcısında kullanılır; klasik süzgeç çıktıları U-Net'e ek kanal olarak verilir; bölütleme haritasından görüntü düzeyi skor ve kutu türetilir; kutu dedektörünün çıktısı haritaya çevrilerek bölütleme ölçütleriyle puanlanır.

## 4. Sonuçlar

Sonuçlar dört grupta sunulmuştur. Bölüm 4.1, ders notundaki "sonuçlar çok kısıtlı bir veri için elde edilmeli, asıl amaç sistemin ve kodların çalışıp çalışmadığını denetlemektir" koşulunun doğrudan karşılığıdır: bütün model aileleri 400 eğitim ve 100 doğrulama görüntüsünde uçtan uca çalıştırılmıştır. Bölüm 4.2–4.6 aynı kodun tam veriye taşınmasıyla elde edilen sonuçları, Bölüm 4.7–4.8 uygulamanın doğrulanmasını, Bölüm 4.9 ResNet-18 sınıflandırıcısının tam veriyle yeniden eğitimini ve karar eşiğinin kalibrasyonunu içerir. Bütün sayılar proje dosyalarında ölçülmüş değerlerdir; her tablonun altında kaynak dosya, Ek A'da toplu izlenebilirlik tablosu (Tablo 46) verilmiştir.

### 4.1 Kısıtlı veriyle çalışırlık kontrolü

**Kurulum.** Eğitim `smoke_train` (400 sahte görüntü ve eşleşen özgünleri), değerlendirme `smoke_val` (100 görüntü) üzerinde, tek tohumla (42) yapılmıştır. Yama kümesi 1.587 eğitim (800 değiştirilmiş, 398 eşleşen özgün, 389 aynı görüntüden temiz) ve 296 doğrulama yamasından (100, 100, 96) oluşur. Toplam GPU eğitim süresi 23.5 dakikadır ve resmî test bölümü hiçbir adımda okunmamıştır. Değerler küçük bir alt kümeye aittir; nihai başarım iddiası taşımaz.

**Klasik öznitelikler.** Tablo 18, 56 öznitelikle eğitilen iki sınıflandırıcının sonuçlarını vermektedir. Her iki model şans düzeyinin üstündedir ama zayıftır. Random Forest öznitelik önemleri birbirine yakındır (Şekil 17); en önde LBP kutuları, Laplace artığı istatistikleri ve GLCM karşıtlığı yer alır. Veri kayıpsız PNG olduğundan DCT öznitelikleri belirgin bir ayrım gücü taşımamaktadır; bu, JPEG izine dayalı literatür yöntemlerinin bu veri tabanındaki zayıflığıyla tutarlı, beklenen bir olumsuz bulgudur.

Table: Tablo 18. Kısıtlı veri: klasik öznitelikler ile SVM ve Random Forest (296 doğrulama yaması).

| Model | Doğruluk | Dengeli doğruluk | AUROC | AUPRC | F1@0.5 | AUROC (eşleşen özgüne karşı) | AUROC (aynı görüntüden temize karşı) |
|---|---|---|---|---|---|---|---|
| SVM, ışınsal çekirdek | 0.534 | 0.606 | 0.646 | 0.443 | 0.546 | 0.552 | 0.744 |
| Random Forest, 500 ağaç | 0.622 | 0.653 | 0.706 | 0.536 | 0.573 | 0.610 | 0.807 |

Kaynak: `baselines/results/results_tables.md`

![Şekil 17. Random Forest öznitelik önemleri: en önemli 15 klasik öznitelik (renkler öznitelik grubunu gösterir).](baselines/figures/rf_feature_importance.png)

**ResNet-18.** Üç eğitim ayarının sonuçları Tablo 19'da, eğitim eğrileri Şekil 18'de, karışıklık matrisleri Şekil 19'da verilmiştir. Tam ince ayar açık farkla en iyisidir (AUROC 0.899). Gövdesi dondurulmuş model yalnız yeni katmanı öğrenerek 0.726'ya ulaşır; ImageNet özellikleri tek başına bu adli görev için yeterli değildir. Sıfırdan eğitim 1.587 yamayla kararsızdır: doğrulama kaybı dalgalanır ve en iyi epok dördüncüdür. Ders notundaki "önceden eğitilmiş model daha iyi olmuş mu" sorusunun küçük veri yanıtı şudur: evet, ancak gövde de güncellendiğinde.

Table: Tablo 19. Kısıtlı veri: ResNet-18 yama sınıflandırma (296 doğrulama yaması).

| Ayar | En iyi epok / çalışan | Doğruluk | Dengeli doğruluk | AUROC | AUPRC | F1@0.5 | AUROC (eşleşen özgüne karşı) |
|---|---|---|---|---|---|---|---|
| Doğrusal sonda (gövde dondurulmuş) | 10 / 12 | 0.639 | 0.693 | 0.726 | 0.554 | 0.616 | 0.635 |
| Tam ince ayar | 6 / 11 | 0.885 | 0.845 | 0.899 | 0.884 | 0.809 | 0.878 |
| Sıfırdan | 4 / 9 | 0.503 | 0.610 | 0.679 | 0.497 | 0.561 | 0.573 |

Kaynak: `baselines/results/results_tables.md`

![Şekil 18. Kısıtlı veride ResNet-18 eğitim eğrileri: eğitim kaybı, doğrulama kaybı ve doğrulama AUROC değeri (üç ayar).](baselines/figures/training_curves_resnet.png)

![Şekil 19. Kısıtlı veride ResNet-18 karışıklık matrisleri (eşik 0.5): doğrusal sonda, tam ince ayar ve sıfırdan eğitim.](baselines/figures/confusion_resnet.png)

Bütün modellerde içerik denetimli negatif (eşleşen özgün görüntüden aynı koordinatlı yama) daha zordur: ince ayarlı ResNet-18 için AUROC 0.922'den 0.878'e, Random Forest için 0.807'den 0.610'a iner. Modeller kısmen içerik ve konum ipuçlarından yararlanmaktadır; bu denetim tam veri deneylerinde zorunlu tutulmuştur.

**U-Net.** Tablo 20 iki U-Net varyantını karşılaştırmaktadır. RGB girdili model 400 görüntüyle 0.707 piksel F1 ve 0.637 IoU değerine ulaşır; temiz kırpıntılarda yanlış pozitif piksel oranı %0.26'dır. Eğitim eğrileri (Şekil 20) kaybın düzenli azaldığını ve doğrulama F1 değerinin onuncu epoktan sonra doyduğunu göstermektedir.

Table: Tablo 20. Kısıtlı veri: U-Net lokalizasyonu (200 doğrulama kırpıntısı, kutu referansı).

| Varyant | En iyi epok / çalışan | F1@0.5 | IoU@0.5 | Oracle-F1 | AUPRC (alt örneklemeli, ilk rapor) | AUPRC (kesin, bütün pikseller) | Temiz kırpıntıda yanlış pozitif |
|---|---|---|---|---|---|---|---|
| RGB (3 kanal) | 11 / 15 | 0.707 | 0.637 | 0.740 | 0.947 | 0.805 | %0.26 |
| RGB + FFT yüksek geçiren + Laplace (5 kanal) | 12 / 15 | 0.674 | 0.613 | 0.711 | 0.939 | 0.792 | %0.14 |

Kaynak: `baselines/results/results_tables.md`, `phase3/results/phase1_exact_auprc.csv`

Tablodaki iki AUPRC sütunu bir düzeltmeyi belgelemektedir. İlk hesapta kırpıntı başına en çok 20 bin negatif piksel örneklenmiş, bu da pozitif oranını yapay olarak yükselterek 0.947 ve 0.939 değerlerini vermişti. Aynı tahminler 52.428.800 pikselin tamamı üzerinde yeniden puanlandığında (pozitif oranı %3.3) kesin değerler 0.805 ve 0.792'dir. F1, IoU ve AUROC bu etkiden bağımsızdır. Sonraki bütün deneylerde yalnız kesin yöntem kullanılmıştır.

![Şekil 20. Kısıtlı veride U-Net eğitim eğrileri: eğitim kaybı, doğrulama piksel F1 ve doğrulama AUPRC (iki varyant; AUPRC ekseni alt örneklemeli hesaptır).](baselines/figures/training_curves_unet.png)

![Şekil 21. Kısıtlı veride U-Net tahminleri: sahte kırpıntı, kutu maskesi, RGB modelinin ve frekans kanallı modelin olasılık haritaları.](baselines/figures/unet_predictions.png)

Şekil 21 üç gözlemi görünür kılmaktadır. Birincisi, Gemini örneklerinde model kutu bandını bütünüyle, neredeyse kusursuz bir dikdörtgen olarak işaretlemektedir; yani referans dikdörtgenini öğrenmek kolay ve ölçütle ödüllendirilen bir çözümdür. Bu gözlem tam veri deneylerinin çıkış noktasıdır. İkincisi, Ideogram örneklerinde tahminler parçalı ve eksiktir (F1: Gemini 0.800, Ideogram 0.497). Üçüncüsü, alan küçüldükçe başarı düşer (alanı %1'in altındaki örneklerde 0.606, %1–5 diliminde 0.830). Frekans kanalları eklemek RGB modelini geçememiştir; yalnız temiz kırpıntı yanlış pozitifleri azalmıştır.

**Dayanıklılık ve maliyet.** Aynı doğrulama kırpıntıları JPEG kalite 75 ile sıkıştırıldığında ya da 0.75 kat küçültülüp büyütüldüğünde F1 yaklaşık 7 puan düşer, çöküş olmaz (Tablo 21). Kaynak kullanımı Tablo 22'de verilmiştir: en pahalı model olan U-Net bile yaklaşık 10 dakikada eğitilmektedir.

Table: Tablo 21. Kısıtlı veri: test zamanı bozulmaları altında U-Net (yeniden eğitim yok).

| Varyant | Koşul | F1@0.5 | IoU@0.5 | Oracle-F1 | Temiz kırpıntıda yanlış pozitif |
|---|---|---|---|---|---|
| RGB | Temiz PNG | 0.708 | 0.637 | 0.740 | %0.26 |
| RGB | JPEG kalite 75 | 0.636 | 0.565 | 0.680 | %0.18 |
| RGB | 0.75 kat küçült ve büyüt | 0.638 | 0.560 | 0.676 | %1.24 |
| RGB + frekans | Temiz PNG | 0.674 | 0.613 | 0.711 | %0.14 |
| RGB + frekans | JPEG kalite 75 | 0.604 | 0.540 | 0.640 | %0.13 |
| RGB + frekans | 0.75 kat küçült ve büyüt | 0.633 | 0.566 | 0.674 | %0.42 |

Kaynak: `baselines/results/results_tables.md`

Table: Tablo 22. Kısıtlı veri: kaynak kullanımı.

| Model | Eğitim süresi (s) | En yüksek GPU belleği (GiB) | Parametre | Çıkarım (ms / örnek) |
|---|---|---|---|---|
| SVM | 7 | — (CPU) | — | 27.9 (öznitelik çıkarımı) |
| Random Forest | 6 | — (CPU) | — | 27.9 (öznitelik çıkarımı) |
| ResNet-18, doğrusal sonda | 64 | 0.33 | 11.18 milyon | 8.2 |
| ResNet-18, tam ince ayar | 76 | 0.59 | 11.18 milyon | 7.3 |
| ResNet-18, sıfırdan | 65 | 0.59 | 11.18 milyon | 7.5 |
| U-Net RGB | 603 | 1.70 | 14.33 milyon | 15.8 |
| U-Net RGB + frekans | 601 | 1.75 | 14.33 milyon | 18.8 |

Kaynak: `baselines/results/results_tables.md`

**Çalışırlık sonucu.** Veri katmanı, üç model ailesi, değerlendirme ve şekil üretimi küçük veride hatasız çalışmıştır. Derin modeller şans düzeyinin belirgin üstündedir; ön eğitim ve atlama bağlantıları beklenen yararı sağlamaktadır. Dedektör ve ikinci bölütleme mimarisi aynı veri ve değerlendirme katmanını kullandığından ayrı bir küçük veri koşusu yapılmamış; YOLO11 için hattın çalıştığı, verinin yüzde otuzuyla dört epokluk bir duman denemesinde doğrulama mAP50 değerinin 0.43'ten 0.65'e düzenli yükselmesiyle görülmüştür.

### 4.2 Tam veri sonuçları ve H1–H4

Aynı kod 2.748 eğitim, 487 doğrulama ve 812 resmî test görüntüsüne taşınmış; üç referans türünün her biri için üç tohumla dokuz U-Net eğitilmiştir. Bütün koşular planlanan 10 epoku tamamlamış, en iyi epok 7–10 arasında kalmıştır (Şekil 22). Eğitim kaybı kutu referansında 0.24'e inerken piksel ve glif referanslarında 0.43–0.53 düzeyinde kalır: piksel düzeyi hedef daha zordur.

![Şekil 22. Tam veride 12 U-Net koşusunun eğitim eğrileri: eğitim kaybı, doğrulama kırpıntı AUPRC ve F1 (referans türüne ve tohuma göre).](phase2/experiments/figures/fig_training_curves.png)

**Resmî test.** Tablo 23'e göre kutu referansıyla eğitilen U-Net resmî testte 0.950 ± 0.011 AUPRC ve 0.846 ± 0.015 F1 değerine ulaşmaktadır. Sızıntısız 740 görüntülük alt kümede sonuç aynıdır; görüntü başı AP farkı 0.0012'dir (%95 güven aralığı −0.0029 ile 0.0048). Sıfır-atış başvuru modelleri bu düzeyin çok altındadır; TruFor için ölçülen F1 (0.403) ve IoU (0.328), veri tabanı makalesinde bildirilen değerlere (0.434 ve 0.358) yakındır. Bu karşılaştırma eşit koşullu değildir: U-Net veri tabanında eğitilmiş, başvuru modelleri eğitilmemiştir.

Table: Tablo 23. Resmî test sonuçları (kutu referansı; U-Net satırları üç tohum ortalaması ± standart sapma).

| Model | Alt küme | AUPRC | F1 (U-Net: kalibre eşik; sıfır-atış: 0.5) | IoU (aynı eşikte) | Oracle-F1 |
|---|---|---|---|---|---|
| U-Net, kutu referansıyla eğitim | Tam 812 | 0.950 ± 0.011 | 0.846 ± 0.015 | 0.789 ± 0.018 | 0.879 ± 0.011 |
| U-Net, kutu referansıyla eğitim | Sızıntısız 740 | 0.949 ± 0.012 | 0.845 ± 0.015 | 0.788 ± 0.018 | 0.878 ± 0.012 |
| U-Net, piksel-değişim referansıyla eğitim | Tam 812 | 0.638 ± 0.015 | 0.616 ± 0.013 | 0.480 ± 0.015 | 0.624 ± 0.015 |
| U-Net, glif referansıyla eğitim | Tam 812 | 0.403 ± 0.011 | 0.405 ± 0.007 | 0.266 ± 0.006 | 0.410 ± 0.008 |
| TruFor, sıfır-atış | Tam 812 | 0.080 | 0.403 | 0.328 | 0.456 |
| ADCD-Net, sıfır-atış | Tam 812 | 0.130 | 0.182 | 0.149 | 0.328 |

Kaynak: `phase2/experiments/results/summary_test_agg.csv`, `phase2/experiments/README.md`, `phase2/literature_baselines/zero_shot_results.csv`

**H1: referans türü sıralamayı belirler.** Tablo 24 ve Şekil 23 aynı tahminlerin üç referansla puanlanmasını göstermektedir. Kutu modeli kendi ölçütünde 0.923 AUPRC alırken, aynı tahminler piksel-değişim referansıyla 0.374'e, glif referansıyla 0.257'ye düşer. Her referans kendi modelini birinci sıraya koyar ve kutu ölçütündeki sıralama glif ölçütündeki sıralamanın tam tersidir. Kutu modeli için F1 farkı 0.398'dir (%95 güven aralığı 0.377–0.418); bu, 0.05'lik yanlışlama sınırının çok üstündedir. Dokuz modelin kutu ve piksel ölçütlerindeki sıralamaları arasındaki Kendall katsayısı −0.22'dir (−0.28 ile 0.06). Resmî testte de aynı örüntü görülür: kutu modeli kutu ölçütünde 0.950, eşi bulunan 89 test çiftinde piksel-değişim ölçütünde 0.427 alır. Piksel modelinin görüntü başı AP değeri ise iki ölçütte hemen hemen aynıdır (fark 0.012; −0.006 ile 0.028). Ölçüt şişmesi kutu modeline özgüdür: 0.92'lik başarının büyük bölümü, dikdörtgenin içindeki değişmemiş piksellerin doğru sayılmasından gelmektedir. H1 doğrulanmıştır.

Table: Tablo 24. Aynı tahminler, üç referans: doğrulama kümesi (487 görüntü), üç tohum ortalaması ± standart sapma; her hücrede AUPRC ve kalibre eşikte F1.

| Eğitim referansı | Kutu ile puanlama | Piksel-değişim ile puanlama | Glif ile puanlama |
|---|---|---|---|
| Kutu | 0.923 ± 0.015 ve 0.842 ± 0.023 | 0.374 ± 0.008 ve 0.447 ± 0.011 | 0.257 ± 0.007 ve 0.387 ± 0.009 |
| Piksel-değişim | 0.579 ± 0.014 ve 0.600 ± 0.014 | 0.610 ± 0.012 ve 0.577 ± 0.007 | 0.511 ± 0.010 ve 0.532 ± 0.008 |
| Glif | 0.351 ± 0.015 ve 0.401 ± 0.007 | 0.379 ± 0.012 ve 0.457 ± 0.006 | 0.676 ± 0.019 ve 0.674 ± 0.011 |

Kaynak: `phase2/experiments/README.md`, `phase3/results/tables_p3.md`

![Şekil 23. Aynı tahminlerin üç referansla puanlanması (doğrulama, üç tohum): havuzlanmış piksel AUPRC, kalibre eşikte F1 ve görüntü başı AP.](phase2/experiments/figures/fig_dual_gt_metrics.png)

![Şekil 24. Sekiz doğrulama örneği: sahte görüntü, üç referans (kutu, piksel-değişim, glif) ve üç modelin tahminleri.](phase2/experiments/figures/fig_examples_predictions.png)

Şekil 24 farkı nitel olarak göstermektedir: kutu modeli dolu dikdörtgenler, piksel modeli değişen karakterlerin lekelerini, glif modeli harf vuruşlarını üretir.

**H2: kutu modeli neye yaslanıyor?** Tahmin dikdörtgenselliğinin medyanı kutu modelinde 0.87, piksel modelinde 0.32, glif modelinde 0.05'tir. Kutu modeli alanın içindeki karakterlerin bulanıklaştırılmasına duyarsızdır (AUPRC değişimi +%1.2); kutu kenarındaki 5 piksellik halkanın bulanıklaştırılması ise küçük ama sistematik bir düşüş yaratır (AUPRC −%2.0; görüntü başı AP −%1.5, güven aralığı −%2.0 ile −%1.1); aynı halka kutu dışına uygulandığında etki sıfırdır (Şekil 25). Yani kutu modeli alanın içeriğinden çok kapsamını kullanmaktadır. Ancak önceden yazılan karşılaştırmalı beklenti tutmamıştır: sınır bulanıklığına duyarlılık kutu modelinde piksel ve glif modellerinden küçüktür (fark +%1.4; +%0.5 ile +%2.4). Dikkat çekici bir yan bulgu, piksel modelinde alanın içini bulanıklaştırmanın kutu ölçütündeki puanı yükseltmesidir (+%10.9) ve piksel ölçütündeki puanı düşürmesidir (−%8.3): kutu ölçütü içeriğin silinmesini ödüllendirir.

![Şekil 25. Sınır ablasyonu: kutu kenarı bulanıklığı, plasebo halka ve kutu içi bulanıklık altında göreli AUPRC ve AP değişimi (üç tohum).](phase2/experiments/figures/fig_ablation_delta.png)

Zararsız kontrollerin sonuçları Tablo 25'te verilmiştir. Kutu ve piksel modelleri sahteyi eşleşen özgünden aynı ayırt edicilikle ayırır (AUROC 0.892); içeriği değiştirmeyen geri yapıştırma ve yeniden örnekleme bu değeri 1.6–3.0 puan düşürür. Kutu dedektöründe (tohum 42; görüntü skoru en yüksek kutu güveni) aynı kontroller AUROC değerini 0.995'ten 0.992 ve 0.994'e indirir, yani en çok 0.3 puan (`phase3/results/T7b_yolo_controls.csv`). Sabit 0.5 eşiğinde kutu modeli özgün görüntülerin %26'sını, piksel modeli %12'sini işaretler; kalibre eşikte yanlış pozitif piksel oranları ise benzerdir. Fark büyük ölçüde kalibrasyon farkıdır. Bu değerler üç tohumun ortalamasıdır; yalnız tek bir tohuma ait daha yüksek bir oran (%44) önceki taslakta yanlışlıkla genel sonuç gibi verilmişti. Sonuç olarak H2 kısmen desteklenmiştir: dikdörtgen biçimi ve içerik duyarsızlığı doğrulanmış, yapıştırma dikişi mekanizması ise yanlışlanmıştır. Asıl sınır ipucunun bir renk çizgisi olduğu Bölüm 4.3'te gösterilmektedir.

Table: Tablo 25. Zararsız kontroller (doğrulama, üç tohum ortalaması): sahte görüntüyü kontrol görüntüsünden ayırma AUROC değeri, 0.5 eşiğinde işaretlenen özgün görüntü oranı ve kalibre eşikte yanlış pozitif piksel oranı.

| Model | AUROC, özgüne karşı | AUROC, zararsız geri yapıştırmaya karşı | AUROC, tam görüntü yeniden örneklemeye karşı | 0.5 eşiğinde işaretlenen özgün oranı | Kalibre eşikte yanlış pozitif piksel oranı |
|---|---|---|---|---|---|
| U-Net kutu | 0.892 | 0.872 | 0.866 | 0.260 | 0.0016 |
| U-Net piksel-değişim | 0.892 | 0.876 | 0.863 | 0.116 | 0.0020 |
| U-Net glif | 0.799 | 0.779 | 0.769 | 0.229 | 0.0038 |

Kaynak: `phase3/results/tables_p3.md` (T7), `phase2/stats/hypothesis_decisions.md`

**H3 ve H4.** Piksel ve glif denetiminin görülmemiş üreticiye ve bozulmalara daha iyi aktarılacağı beklentisi (H3) desteklenmemiştir; ayrıntılar Bölüm 4.5'tedir. Kutu dışında öğrenilebilir bir küresel iz bulunduğu beklentisi (H4) için maskeye değmeyen 224 piksellik kırpıntılarla bir ResNet-18 eğitilmiştir: sahteyi eşleşen özgünden ayırma AUROC değeri 0.516'dır (0.513–0.519); aynı düzenek kutu merkezli kırpıntılarda 0.980 verir. Düzenek öğrenebilmekte, ancak kutu dışında kullanılabilir bir iz bulunmamaktadır; 457 çiftin 260'ında kutu dışı kırpıntılar piksel düzeyinde özdeştir. Görüntü düzeyi tespit, alanın kendisinden gelmek zorundadır. Dört hipotezin kararları Tablo 26'da özetlenmiştir.

Table: Tablo 26. H1–H4 için gözlenen değerler ve kararlar (kaynak belge grubu bootstrap, %95 güven aralığı).

| Hipotez | Gözlenen | Karar |
|---|---|---|
| H1 | Kutu modeli F1 farkı (kutu − piksel ölçütü) 0.398 [0.377, 0.418]; Kendall katsayısı −0.22 [−0.28, 0.06] | Doğrulandı |
| H2 | Sınır bulanıklığı göreli AP: kutu modeli −%1.5 [−2.0, −1.1], piksel modeli −%3.0 [−3.8, −2.0]; fark +%1.4 [+0.5, +2.4] (beklenenin tersi); dikdörtgensellik 0.87 | Kısmen |
| H3 | Ayrı tutulan üreticide piksel − kutu AP farkı 0.025 [−0.039, 0.088]; bozulmalarda tutarlı üstünlük yok | Yanlışlandı |
| H4 | Kutu dışı kırpıntı AUROC 0.516 [0.513, 0.519]; simetrik yeniden örnekleme 0.498 [0.492, 0.503]; olumlu denetim 0.980 [0.972, 0.988] | Yanlışlandı |

Kaynak: `phase2/stats/hypothesis_decisions.md`

### 4.3 Yeşil işaretçi artefaktı ve işaretçi-onarımlı eğitim

**Kalıntı bir kısayoldur.** Bölüm 3.9'da tanımlanan sondalar, kalıntı taşıyan doğrulama çiftlerinde bütün model ailelerine uygulanmıştır (Tablo 27). Standart kutu modelinde yalnız kalıntı pikselleri özgün görüntüden geri kopyalandığında, yani hiçbir karakter değişmeden, görüntü başı AP 0.988'den 0.766'ya iner (fark −0.222; −0.256 ile −0.190). Tersine, özgün görüntüye yalnız kalıntı pikselleri aktarıldığında model kutunun %80'ini kalibre eşiğin üstünde işaretler (AP 0.905); aynı özgün görüntülerde işaretçi yokken AP 0.176, kapsama 0.04'tür. Piksel ve glif modelleri de ipucunu kullanır, çünkü çizgi piksel-değişim referansının parçasıdır. DeepLabV3+ aynı bağımlılığı gösterir (−0.173; üç tohum).

İlk keşifsel sondada (Faz 2; 368 çift, dokuz model; Şekil 26) öğrenme içermeyen bir kural da sınanmıştır: görüntüdeki yeşil çizgi parçalarını bulup sınırlayıcı kutusunu tahmin eden kural yalnız 0.054 F1 verir. Kalıntı çoğunlukla alanın bir ya da iki kenarını işaretler; kutunun tamamı çizgiden okunamaz. Ağlar çizgiyi komşu metin alanının görünümüyle birleştirerek dikdörtgeni tamamlamaktadır; bu sığ bir piksel kuralı değil, öğrenilmiş bir kısayoldur. Önceden yazılan sınır ablasyonunun bu ipucunu kaçırmasının nedeni de bellidir: Gauss bulanıklığı 1–3 piksellik doygun çizgiyi yayar ama rengini silmez.

Table: Tablo 27. İşaretçi sondaları: kalıntı taşıyan 380 doğrulama çifti, kutu referansına göre görüntü başı AP ve kalibre eşikte kutu kapsama oranı.

| Model | Tohum | AP, sahte | Kapsama, sahte | AP, onarılmış sahte | AP farkı (onarım) [%95 GA] | Kapsama, özgün + işaretçi | AP, özgün + işaretçi |
|---|---|---|---|---|---|---|---|
| U-Net kutu | 3 | 0.988 | 0.939 | 0.766 | −0.222 [−0.256, −0.190] | 0.795 | 0.905 |
| U-Net piksel-değişim | 3 | 0.747 | 0.538 | 0.500 | −0.247 [−0.279, −0.218] | 0.420 | 0.635 |
| U-Net glif | 3 | 0.522 | 0.325 | 0.398 | −0.123 [−0.146, −0.103] | 0.283 | 0.461 |
| DeepLabV3+ kutu | 3 | 0.982 | 0.921 | 0.810 | −0.173 [−0.203, −0.146] | 0.785 | 0.931 |
| U-Net birleşik kayıp (piksel) | 3 | 0.795 | 0.558 | 0.600 | −0.194 [−0.226, −0.164] | 0.457 | 0.700 |
| YOLO11s | 3 | 0.955 | 0.957 | 0.897 | −0.058 [−0.068, −0.049] | 0.553 | 0.808 |
| U-Net kutu, işaretçisiz eğitim | 3 | 0.939 | 0.891 | 0.956 | +0.017 [+0.001, +0.033] | 0.442 | 0.617 |
| U-Net piksel, işaretçisiz eğitim | 3 | 0.647 | 0.419 | 0.700 | +0.054 [+0.037, +0.072] | 0.102 | 0.262 |
| U-Net glif, işaretçisiz eğitim | 3 | 0.502 | 0.301 | 0.511 | +0.009 [+0.002, +0.016] | 0.167 | 0.338 |
| U-Net kutu, yalnız Gemini | 3 | 0.990 | 0.961 | 0.715 | −0.275 [−0.313, −0.238] | 0.816 | 0.910 |
| U-Net kutu, yalnız Gemini, işaretçisiz eğitim | 3 | 0.941 | 0.874 | 0.950 | +0.009 [−0.004, +0.023] | 0.434 | 0.636 |

Kaynak: `phase3/results/tables_p3.md` (T3)

![Şekil 26. İlk işaretçi sondası (Faz 2, 368 doğrulama çifti, üç tohum): dört girdi koşulunda görüntü başı AP, kutu kapsama ve görüntü skoru.](phase2/experiments/figures/fig_green_marker.png)

**Resmî testte etki.** Kalıntının etkisi resmî testte model girdisine dokunmadan da görülür (Tablo 28). Kutu U-Net'in görüntü başı AP değeri işaretçili 627 görüntüde 0.972, işaretçisiz 185 görüntüde 0.690'dır. İşaretçisiz Gemini görüntülerinde 0.722 olması, farkın yalnız bir üretici farkı olmadığını göstermektedir.

Table: Tablo 28. Resmî testte işaretçili ve işaretçisiz görüntülerde görüntü başı AP (kutu referansı, %95 güven aralığı).

| Model | Tohum | İşaretçili (627 görüntü) | İşaretçisiz (185 görüntü) | İşaretçisiz Gemini (102 görüntü) |
|---|---|---|---|---|
| U-Net kutu | 3 | 0.972 [0.966, 0.977] | 0.690 [0.640, 0.739] | 0.722 [0.657, 0.788] |
| DeepLabV3+ kutu | 3 | 0.962 [0.954, 0.968] | 0.702 [0.654, 0.746] | 0.751 [0.692, 0.807] |
| U-Net piksel-değişim | 3 | 0.639 [0.624, 0.652] | 0.503 [0.455, 0.550] | 0.559 [0.504, 0.622] |
| U-Net kutu, işaretçisiz eğitim | 3 | 0.914 [0.899, 0.927] | 0.711 [0.663, 0.758] | 0.762 [0.703, 0.821] |
| YOLO11s | 3 | 0.954 [0.951, 0.957] | 0.878 [0.857, 0.895] | 0.886 [0.861, 0.907] |

Kaynak: `phase3/results/tables_p3.md` (T3b)

**İşaretçisiz yeniden eğitim.** Üç referans türü için U-Net, kalıntısı onarılmış eğitim görüntüleri ve işaretçisiz referanslarla yeniden eğitilmiştir: her biri üç tohumla (42, 43, 44). Sonuç iki yönlüdür. Bağımlılık ortadan kalkar: onarılmış kutu modeli test anında kalıntı silindiğinde hiçbir şey kaybetmez (+0.017 [+0.001, +0.033]) ve yalnız kalıntı eklenmiş özgün görüntüde kutunun %44'ünü işaretler (önce %80; piksel modelinde %42 yerine %10). Kutu ölçütündeki başarı ise büyük ölçüde korunur: doğrulama AUPRC 0.912 ± 0.006 (önce 0.923 ± 0.015), resmî test AUPRC 0.940 ± 0.003 (önce 0.950 ± 0.011), kalibre eşikte F1 0.799 (önce 0.846). Aynı görüntüler eşlenerek bakıldığında yeniden eğitimin bedeli görüntü başı AP'de yaklaşık 0.04'tür (doğrulama −0.038 [−0.054, −0.021], test −0.040 [−0.052, −0.029]). İşaretçili test görüntülerinde AP 0.972'den 0.914'e inerken işaretçisiz Gemini görüntülerinde benzer kalır (0.762 ve 0.722; güven aralıkları örtüşür). Dolayısıyla kalıntı, sunulduğunda kullanılan bir kısayoldur; yüksek kutu puanı için zorunlu değildir.

Kutu modelinin sonucu üç tohumda tutarlıdır (tohum bazında doğrulama AUPRC 0.916, 0.916, 0.905). 43 ve 44 tohumları, öngörüleri ve yanlışlama ölçütleri koşudan önce yazılıp açık depoya gönderilen bir planla eklenmiştir (`reports/inceleme/deney_plani_ek_tohum_2026-11-10.md`); altı öngörünün altısı da tutmuştur (`phase3/results/extra_seed_stats.md`). İşaretçisiz piksel-değişim ve glif modellerinin 43 ve 44 tohumları da aynı biçimde, 21 öngörülü ikinci bir planla eklenmiştir (`reports/inceleme/deney_plani_kalan_tohumlar_2026-11-11.md`). Bu modeller de kalıntı silindiğinde AP kaybetmez (+0.054 [+0.037, +0.072] ve +0.009 [+0.002, +0.016]), ama bedelleri daha büyüktür: kendi referanslarındaki doğrulama AUPRC değerleri 0.507 ± 0.045 ve 0.644 ± 0.014'tür (standart modellerde 0.610 ve 0.676). Piksel modelinin 43 tohumu belirgin biçimde zayıftır (0.458); öngörülen aralığın (0.51–0.58) az altında kalan tek sonuç budur, yanlışlama eşiği (0.49) aşılmamıştır.

**Yeniden çizim ve içerik ayrıştırması.** Üreticiler alanın tamamını yeniden çizdiğinden kutu içinde pek çok piksel eşiğin altında kalan küçük farklarla değişir. İki bileşeni ayırmak için yalnız yeniden çizimi ya da yalnız içerik değişimini taşıyan girdiler bütün doğrulama çiftlerinde sınanmıştır (Tablo 29, Şekil 27). Standart kutu modelinde kutu içi ortalama olasılık sahte girdide 0.86 iken yalnız yeniden çizimde 0.24'e, yalnız içerikte 0.35'e düşer; hiçbir bileşen tek başına gerçek sahteye verilen tepkiyi üretmez. Standart modeller yeniden çizim, değişen karakterler ve işaretçinin birlikteliğine tepki vermektedir. İşaretçi eğitimden çıkarıldığında modeller içeriğe kayar: yalnız içerik girdisinde AUROC kutu modelinde 0.733'ten 0.886'ya (üç tohum), piksel modelinde 0.681'den 0.851'e yükselir (üç tohum).

Table: Tablo 29. Yeniden çizim ve içerik ayrıştırması (bütün doğrulama çiftleri): kutu içi ortalama olasılık ve sahte türü girdiyi özgünden ayırma AUROC değeri.

| Model | Tohum | Olasılık, sahte | Olasılık, yalnız yeniden çizim | Olasılık, yalnız içerik | AUROC, sahte | AUROC, yalnız yeniden çizim | AUROC, yalnız içerik |
|---|---|---|---|---|---|---|---|
| U-Net kutu | 3 | 0.864 | 0.237 | 0.348 | 0.971 | 0.637 | 0.733 |
| U-Net piksel-değişim | 3 | 0.406 | 0.045 | 0.150 | 0.942 | 0.554 | 0.681 |
| DeepLabV3+ kutu | 3 | 0.868 | 0.279 | 0.408 | 0.974 | 0.656 | 0.762 |
| YOLO11s | 3 | 0.867 | 0.448 | 0.390 | 0.998 | 0.890 | 0.884 |
| U-Net kutu, işaretçisiz eğitim | 3 | 0.823 | 0.345 | 0.583 | 0.957 | 0.724 | 0.886 |
| U-Net piksel, işaretçisiz eğitim | 3 | 0.316 | 0.091 | 0.281 | 0.894 | 0.655 | 0.851 |

Kaynak: `phase3/results/tables_p3.md` (T3)

![Şekil 27. Altı sonda koşulunda kutu referansına göre görüntü başı AP: standart, işaretçisiz eğitilmiş ve kutu dedektörü modelleri.](phase3/figures/fig_marker_render.png)

### 4.4 İkinci mimari ve kutu formülasyonu

**DeepLabV3+ H1'i yineler.** Tablo 30 ve Şekil 28, Faz 3'te eğitilen bütün model ailelerinin doğrulama kümesinde üç referansla puanlanmasını göstermektedir. DeepLabV3+ kutu modeli kendi ölçütünde 0.929 ± 0.009, piksel-değişim ölçütünde 0.382 ± 0.002 alır; her referans üç tohumun her birinde yine kendi modelini birinci sıraya koyar. Sayılar U-Net ile neredeyse aynıdır. Etki, U-Net kod çözücüsünün değil denetim türünün bir özelliğidir.

Table: Tablo 30. Faz 3 model aileleri ve varyantları: doğrulama kümesinde üç referansa göre havuzlanmış piksel AUPRC (ana U-Net değerleri için bkz. Tablo 24).

| Model | Eğitim referansı | Tohum | Kutu ile | Piksel-değişim ile | Glif ile |
|---|---|---|---|---|---|
| DeepLabV3+ | Kutu | 3 | 0.929 ± 0.009 | 0.382 ± 0.002 | 0.260 ± 0.005 |
| DeepLabV3+ | Piksel-değişim | 3 | 0.537 ± 0.027 | 0.596 ± 0.022 | 0.437 ± 0.011 |
| DeepLabV3+ | Glif | 3 | 0.357 ± 0.011 | 0.409 ± 0.007 | 0.558 ± 0.010 |
| YOLO11s, karolu | Kutu | 3 | 0.965 ± 0.003 | 0.368 ± 0.015 | 0.221 ± 0.003 |
| U-Net, işaretçisiz eğitim | Kutu | 3 | 0.912 ± 0.006 | 0.373 ± 0.013 | 0.265 ± 0.001 |
| U-Net, işaretçisiz eğitim | Piksel-değişim (işaretçisiz) | 3 | 0.480 ± 0.053 | 0.507 ± 0.045 | 0.415 ± 0.044 |
| U-Net, işaretçisiz eğitim | Glif (işaretçisiz) | 3 | 0.336 ± 0.026 | 0.353 ± 0.017 | 0.644 ± 0.014 |
| U-Net, birleşik kayıp | Piksel-değişim + kutu yardımcı | 3 | 0.668 ± 0.019 | 0.633 ± 0.013 | 0.523 ± 0.009 |
| U-Net, birleşik kayıp | Glif + kutu yardımcı | 3 | 0.395 ± 0.073 | 0.386 ± 0.051 | 0.641 ± 0.063 |
| U-Net, 20 epok | Kutu | 3 | 0.912 ± 0.031 | 0.371 ± 0.003 | 0.261 ± 0.004 |
| U-Net, 20 epok | Piksel-değişim | 3 | 0.597 ± 0.064 | 0.591 ± 0.015 | 0.485 ± 0.018 |
| U-Net, 20 epok | Glif | 3 | 0.346 ± 0.063 | 0.342 ± 0.061 | 0.598 ± 0.082 |

Kaynak: `phase3/results/tables_p3.md` (T1)

![Şekil 28. Üç model ailesinde eğitim referansı (yatay eksen) ve puanlama referansı (renk) için doğrulama AUPRC değerleri: her aile üç tohum; şekil 42 tohumlu ilk koşulardan üretilmiştir.](phase3/figures/fig_dual_gt_families.png)

**Birleşik kayıp ve eğitim süresi.** Piksel-değişim maskesine kutu yardımcı kanalı eklemek, piksel modelini iyileştiren tek varyanttır ve bunu asıl olarak kutu ölçütünde yapar (üç tohum): doğrulamada kutu ölçütü 0.579 ± 0.014'ten 0.668 ± 0.019'a, resmî testte kutu AUPRC 0.638 ± 0.015'ten 0.702 ± 0.020'ye çıkar (eşli görüntü başı AP farkı +0.051 [+0.043, +0.058] ve +0.046 [+0.040, +0.052]). Piksel-değişim ölçütündeki kazanç küçüktür ve doğrulamayla sınırlıdır: 0.610 ± 0.012'den 0.633 ± 0.013'e (görüntü başı AP +0.011 [+0.005, +0.017]); eşi bulunan 89 test görüntüsünde fark yoktur (0.652 ± 0.003 ve 0.654 ± 0.016; +0.004 [−0.008, +0.015]). Birleşik model işaretçiye bağımlı kalır (onarım farkı −0.194; Tablo 27). Tek tohumla görülen kazanım böylece kutu ölçütünde doğrulanmış, piksel-değişim ölçütünde ise daraltılmıştır (`phase3/results/extra_seed_stats.md`). Glif modeline kutu yardımcı kanalı eklemek ise bir şey kazandırmaz (glif ölçütünde 0.641 ± 0.063; standart 0.676 ± 0.019). Epok sayısını 20'ye çıkarmak hiçbir referansta yarar sağlamaz (üç tohum): kutu 0.912 ± 0.031 (10 epokta 0.923), piksel-değişim 0.591 ± 0.015 (0.610), glif 0.598 ± 0.082 (0.676). Glif modeli 20 epokta tohumdan tohuma belirgin biçimde oynar. "10 epok az olabilir" kaygısı desteklenmemiştir.

**Kutu dedektörü.** YOLO11s eğitimi düzenli yakınsar (Şekil 29); doğrulama karolarında mAP50 0.981'dir (Şekil 30). Karolu YOLO11s, kutu ölçütünde çalışmadaki en güçlü modeldir (Tablo 31): üç tohumda resmî test AUPRC 0.969 ± 0.006, kalibre eşikte F1 0.946 ve IoU 0.918'dir (U-Net: 0.950, 0.846, 0.789). Buna karşılık piksel-değişim ölçütünde doğrulama AUPRC değeri 0.368'dir; kutu U-Net'in 0.374'üyle aynı düzeydedir ve piksel modelinin 0.610'unun çok altındadır. Bu sonuç H1'in formülasyondan bağımsız teyididir: kutu ölçütü alan tespitini ölçmektedir ve bu işi en iyi yapan model, hangi piksellerin değiştiğini kutu U-Net'ten daha iyi bilmemektedir.

![Şekil 29. YOLO11s ana modelinin (42 tohumu) eğitim süreci: eğitim ve doğrulama kayıpları (kutu, sınıf, dağılım odaklı), kesinlik, duyarlılık, mAP50 ve mAP50-95.](phase3_yolo/runs/aiforge-tiles-yolo11s-img1024-conserv-s42/results.png)

![Şekil 30. YOLO11s ana modelinin doğrulama karolarındaki kesinlik–duyarlılık eğrisi (mAP50 0.981).](phase3_yolo/runs/aiforge-tiles-yolo11s-img1024-conserv-s42/BoxPR_curve.png)

Table: Tablo 31. Resmî test (812 görüntü, kutu referansı): piksel ölçütleri, kutu düzeyi ortalama kesinlik ve tam görüntü çıkarım süresi.

| Model | Tohum | AUPRC | F1 (kalibre eşik) | IoU (kalibre eşik) | AP50 | AP50-95 | Süre (ms / görüntü) |
|---|---|---|---|---|---|---|---|
| U-Net kutu | 3 | 0.950 ± 0.011 | 0.846 ± 0.015 | 0.789 ± 0.018 | 0.712 ± 0.279 | 0.453 ± 0.224 | 81 |
| DeepLabV3+ kutu | 3 | 0.951 ± 0.001 | 0.830 ± 0.003 | 0.761 ± 0.005 | 0.849 ± 0.024 | 0.589 ± 0.010 | 122 |
| U-Net kutu, işaretçisiz eğitim | 3 | 0.940 ± 0.003 | 0.799 ± 0.016 | 0.727 ± 0.015 | 0.804 ± 0.005 | 0.486 ± 0.007 | ölçülmedi |
| U-Net piksel, birleşik kayıp | 3 | 0.702 ± 0.020 | 0.641 ± 0.012 | 0.506 ± 0.015 | 0.125 ± 0.020 | 0.064 ± 0.010 | 83 |
| YOLO11s, karolu | 3 | 0.969 ± 0.006 | 0.946 ± 0.007 | 0.918 ± 0.006 | 0.880 ± 0.002 | 0.780 ± 0.003 | 98 |
| YOLO11n, karolu | 3 | 0.966 ± 0.003 | 0.928 ± 0.003 | 0.894 ± 0.005 | 0.871 ± 0.002 | 0.733 ± 0.004 | 81 |
| YOLO11m, karolu | 3 | 0.966 ± 0.008 | 0.948 ± 0.002 | 0.920 ± 0.001 | 0.882 ± 0.015 | 0.785 ± 0.014 | 76 |
| YOLO11s, tam görüntü 1280 | 3 | 0.936 ± 0.005 | 0.906 ± 0.004 | 0.872 ± 0.005 | 0.961 ± 0.003 | 0.879 ± 0.004 | 303 |
| YOLO11s, varsayılan artırım | 3 | 0.973 ± 0.002 | 0.954 ± 0.003 | 0.926 ± 0.003 | 0.867 ± 0.004 | 0.769 ± 0.005 | ölçülmedi |
| YOLO11s, özgün negatif karo | 3 | 0.969 ± 0.002 | 0.938 ± 0.006 | 0.907 ± 0.008 | 0.874 ± 0.003 | 0.770 ± 0.005 | ölçülmedi |

Kaynak: `phase3/results/tables_p3.md` (T2, T5b), `phase3/results/speed_bench.csv`

U-Net'in kutu düzeyi AP değerindeki büyük standart sapma tek bir tohumdan kaynaklanır: 42 ve 43 tohumlarında AP50 0.873 iken, kalibre eşiği 0.95 çıkan 44 tohumunda tahmin bölgesi parçalanır ve değer 0.391'e düşer. Dedektörde böyle bir kırılganlık yoktur; işaretçisiz eğitilen kutu U-Net'te de görülmez (üç tohumda AP50 0.804 ± 0.005).

**Ablasyonlar ve H5 kararları.** Tek etkenli ablasyonların sonuçları (her biri üç tohum) Tablo 31 ve 32'de, maliyet ve doğruluk ilişkisi Şekil 31'de verilmiştir. Beklentilerin yalnız bir kısmı tutmuştur.

- **H5a (doğruluk ve hız):** Doğruluk kısmı desteklenmiştir. Hız kısmı reddedilmiştir: tam görüntü çıkarımı karolu dedektörlerde 76–98 ms, U-Net'te 81 ms, DeepLabV3+'ta 122 ms, tam görüntü dedektöründe 303 ms sürer (ortalama 3.1 megapiksel). Karo başına ön ve son işleme maliyeti baskındır.
- **H5b (piksel ölçütünde düşüş):** Desteklenmiştir (0.368).
- **H5c (karolama):** Karışıktır. Görüntünün tamamını 1280 piksele küçültmek test AUPRC değerini 0.969'dan 0.936'ya, işaretçisiz görüntülerdeki AP değerini 0.878'den 0.750'ye indirir; ancak kutu AP50 değerini 0.880'den 0.961'e yükseltir (alanı %1'in altındaki hedeflerde 0.864'ten 0.933'e). Olası neden, karo sınırında bölünen alanların parçalı kutular vermesidir; karosuz model de görüntü başına yaklaşık dört kutu ürettiğinden mekanizma kesinleşmemiştir.
- **H5d (ayrı tutulan üretici):** Desteklenmiştir (üç tohum; Bölüm 4.5).
- **H5e (artırım):** Fark yoktur. Kütüphanenin varsayılan artırımı korumacı profil kadar iyidir (test AUPRC 0.973 ve 0.969); mozaik ve ölçek artırımının adli izi bozacağı beklentisi doğrulanmamıştır.

Model boyutunun etkisi yoktur (n, s ve m için test AUPRC 0.966, 0.969 ve 0.966); görüntü başına boş etiketli bir özgün karo eklemek sonucu değiştirmez (0.969). Sondalar dedektörün farklı bir ipucu kullandığını düşündürmektedir (üç tohum): kalıntının silinmesi dedektöre yalnız 0.058 AP kaybettirir (U-Net 0.222), buna karşılık hiçbir karakterin değişmediği "yalnız yeniden çizim" girdisi özgün görüntüden 0.890 AUROC ile ayrılır (U-Net 0.637). Dedektör, üreticinin yeniden çizdiği bir alanın nasıl göründüğünü öğrenmiştir; bu ipucu iki üreticide de ortaktır ama hangi piksellerin değiştiği hakkında bilgi vermez.

Table: Tablo 32. YOLO11 varyantları (her biri 42–44 tohumları; ortalama ± standart sapma): eğitim maliyeti, kutu düzeyi AP ve doğrulama kümesinde piksel AUPRC.

| Varyant | En iyi epok (30 üzerinden) | Eğitim (dk, ortalama) | Parametre (milyon) | Doğrulama karo mAP50 | Test AP50 | Test AP50, alan %1 altı | Doğrulama AUPRC: kutu ve piksel-değişim |
|---|---|---|---|---|---|---|---|
| YOLO11s, karolu (ana model) | 23–26 | 56.6 | 9.43 | 0.986 ± 0.004 | 0.880 ± 0.002 | 0.864 ± 0.008 | 0.965 ve 0.368 |
| YOLO11s, tam görüntü 1280 | 27–30 | 28.5 | 9.43 | 0.963 ± 0.002 | 0.961 ± 0.003 | 0.933 ± 0.006 | 0.923 ve 0.341 |
| YOLO11s, varsayılan artırım | 25–28 | 36.6 | 9.43 | 0.987 ± 0.001 | 0.867 ± 0.004 | 0.846 ± 0.007 | 0.968 ve 0.367 |
| YOLO11s, özgün negatif karo | 22–28 | 49.6 | 9.43 | 0.985 ± 0.004 | 0.874 ± 0.003 | 0.859 ± 0.004 | 0.964 ve 0.382 |
| YOLO11n, karolu | 28–30 | 30.2 | 2.60 | 0.966 ± 0.007 | 0.871 ± 0.002 | 0.844 ± 0.007 | 0.955 ve 0.371 |
| YOLO11m, karolu | 21–23 | 75.3 | 20.05 | 0.974 ± 0.008 | 0.881 ± 0.015 | 0.877 ± 0.006 | 0.952 ve 0.373 |
| YOLO11s, yalnız Gemini | 26–30 | 39.2 | 9.43 | 0.990 ± 0.002 | 0.857 ± 0.002 | 0.814 ± 0.007 | 0.958 ve 0.359 |

Kaynak: `phase3/results/T5_yolo_ablation.csv` (koşu bazında; 43 ve 44 tohumlarının eğitim süreleri 42 tohumundan kısadır, çünkü aynı makinede başka iş yoktu)

![Şekil 31. Maliyet ve doğruluk: resmî testte kutu AP50, tam görüntü çıkarım süresi (RTX 5080) ve parametre sayısı; her model için 42 tohumlu tek koşunun değerleridir (üç tohum ortalamaları Tablo 31'dedir); turuncu çubuklar kutu dedektörü, mavi çubuklar bölütleme modelleridir.](reports/figures/fig_maliyet_ap.png)

### 4.5 Genelleme ve dayanıklılık

**Ayrı tutulan üretici.** Modeller yalnız Gemini düzenlemeleriyle eğitilmiş ve doğrulama ile test bölümlerindeki Ideogram düzenlemelerinde sınanmıştır (kutu referansı için 126, piksel-değişim referansı için 41 görüntü). Karşılaştırma tabanı, Ideogram ile aynı kaynak veri tabanlarından gelen Gemini görüntüleridir. Tablo 33'e göre yalnız Gemini ile eğitilen bütün U-Net modelleri Ideogram üzerinde belirgin düşer: kutu modeli 0.909'dan 0.412'ye, piksel modeli 0.567'den 0.264'e iner. Piksel modelinin göreli düşüşü daha küçüktür ama mutlak Ideogram puanı daha düşüktür; kendi ölçütünde de kutu modelinden iyi değildir (0.279 ve 0.259). Piksel-değişim denetimi tek başına üretici aktarımı sağlamamaktadır; ilk kez tek tohumla varılan H3 reddi üç tohumla doğrulanmıştır.

Kutu dedektörü belirgin biçimde daha iyi aktarılır: bütün üreticilerle eğitildiğinde Ideogram üzerinde 0.872 (U-Net 0.629), yalnız Gemini ile eğitildiğinde 0.725 AP verir (U-Net 0.412); açık 0.50 yerine 0.21'dir (üç tohum).

**İşaretçi mi, üretici mi?** Ideogram sahteleri hemen hiç işaretçi taşımadığından (%2.1; Gemini %85.9) bu açığın bir bölümü üretici farkı değil işaretçi farkı olabilirdi. İkisini ayırmak için yalnız Gemini modelleri işaretçisi onarılmış görüntülerle yeniden eğitilmiştir (üç referans, üç tohum; öngörüler koşudan önce depoya gönderildi). Onarılmış yalnız-Gemini kutu modeli kalıntıya artık bağımlı değildir (onarım farkı +0.009 [−0.004, +0.023]; standart yalnız-Gemini modelinde −0.275), ama Ideogram'da yine çöker: 0.463 [0.400, 0.528], eşlenmiş Gemini'de 0.874. Aynı görüntüler eşlenerek bakıldığında Ideogram'da 0.051 [0.004, 0.096] kazanır, Gemini'de 0.035 [0.023, 0.047] kaybeder; piksel ve glif modelleri de benzer davranır (Ideogram'da +0.053 ve +0.037). Kutu modelindeki açığın (0.497 → 0.412) yaklaşık altıda biri işaretçiye, geri kalanı üreticiye özgü izlere bağlanabilir. Bütün üreticilerle eğitilen modelde de onarım Ideogram puanını değiştirmez (0.629 ve 0.633).

Table: Tablo 33. Ayrı tutulan üretici: Ideogram ve kaynak eşlemeli Gemini görüntülerinde görüntü başı AP (%95 güven aralığı).

| Model ve eğitim | Eğitim referansı | Puanlama | Tohum | Ideogram | Gemini (eşlemeli) | Açık |
|---|---|---|---|---|---|---|
| U-Net, bütün üreticiler | Kutu | Kutu | 3 | 0.629 [0.569, 0.692] | 0.912 [0.896, 0.928] | 0.283 |
| U-Net, yalnız Gemini | Kutu | Kutu | 3 | 0.412 [0.341, 0.481] | 0.909 [0.892, 0.927] | 0.497 |
| U-Net, yalnız Gemini | Piksel-değişim | Kutu | 3 | 0.264 [0.213, 0.313] | 0.567 [0.550, 0.583] | 0.303 |
| U-Net, yalnız Gemini | Kutu | Piksel-değişim | 3 | 0.259 [0.168, 0.350] | 0.349 [0.322, 0.377] | 0.090 |
| U-Net, yalnız Gemini | Piksel-değişim | Piksel-değişim | 3 | 0.279 [0.182, 0.382] | 0.577 [0.546, 0.608] | 0.297 |
| YOLO11s, bütün üreticiler | Kutu | Kutu | 3 | 0.872 [0.846, 0.895] | 0.945 [0.938, 0.951] | 0.073 |
| U-Net, bütün üreticiler, işaretçisiz eğitim | Kutu | Kutu | 3 | 0.633 [0.573, 0.695] | 0.898 [0.881, 0.915] | 0.265 |
| U-Net, yalnız Gemini, işaretçisiz eğitim | Kutu | Kutu | 3 | 0.463 [0.400, 0.528] | 0.874 [0.856, 0.893] | 0.412 |
| U-Net, yalnız Gemini, işaretçisiz eğitim | Piksel-değişim (işaretçisiz) | Kutu | 3 | 0.317 [0.267, 0.366] | 0.472 [0.453, 0.491] | 0.155 |
| U-Net, yalnız Gemini, işaretçisiz eğitim | Glif (işaretçisiz) | Kutu | 3 | 0.203 [0.168, 0.240] | 0.315 [0.303, 0.327] | 0.112 |
| YOLO11s, yalnız Gemini | Kutu | Kutu | 3 | 0.725 [0.675, 0.778] | 0.938 [0.930, 0.946] | 0.214 |

Kaynak: `phase3/results/tables_p3.md` (T4)

**Test zamanı bozulmaları.** Bölüm 3.3'te model girdisine uygulanmayan işlemlerin etkisi, eğitilmiş modellere test anında uygulanarak ölçülmüştür (Tablo 34, Şekil 32). Üç sonuç öne çıkar. Birincisi, piksel ve glif modelleri bulanıklık ve yeniden ölçeklemeye kutu modelinden daha kırılgandır (bulanıklıkta −%59 ve −%21): ince harf kenarlarına dayanan denetim, yüksek frekans bilgisine daha bağımlı modeller üretir. Birleşik kayıp bu kırılganlığı azaltır (−%46; üç tohum). İkincisi, kutu dedektörü bulanıklık, CLAHE ve yeniden ölçeklemeye neredeyse duyarsızdır (en çok %1.3) ama JPEG sıkıştırmasına kırılgandır (−%45; üç tohum). Üçüncüsü ve en önemlisi, girdinin gri düzeye çevrilmesi sınanan bütün modellerde AUPRC değerinin %50–67'sini götürür; işaretçisiz eğitilen modeller de buna dahildir. Renk, yeşil çizginin ötesinde kanıt taşımaktadır; "işaretçiden kurtulmak için gri düzeye çevirmek" bir çözüm değildir. CLAHE ise en çok %3 değişim yaratır.

Table: Tablo 34. Test zamanı bozulmaları: her modelin kendi eğitim referansına göre havuzlanmış AUPRC değerindeki göreli değişim (doğrulama; üç tohumlu satırlarda ± standart sapma).

| Model | Eğitim referansı | JPEG kalite 75 | 0.75 kat ölçek | Gauss bulanıklık | CLAHE | Gri düzey |
|---|---|---|---|---|---|---|
| U-Net (3 tohum) | Kutu | −%16.6 ± 1.4 | −%12.2 ± 4.3 | −%20.7 ± 6.7 | ölçülmedi | ölçülmedi |
| U-Net (3 tohum) | Piksel-değişim | −%22.5 ± 2.3 | −%40.3 ± 1.5 | −%59.1 ± 4.5 | ölçülmedi | ölçülmedi |
| U-Net (3 tohum) | Glif | −%18.0 ± 0.8 | −%42.7 ± 5.4 | −%59.0 ± 6.8 | ölçülmedi | ölçülmedi |
| DeepLabV3+ (3 tohum) | Kutu | −%13.5 ± 2.0 | −%8.3 ± 1.3 | −%14.4 ± 2.4 | −%0.6 ± 1.3 | −%52.7 ± 5.7 |
| DeepLabV3+ (3 tohum) | Piksel-değişim | −%20.6 ± 2.7 | −%35.1 ± 8.6 | −%49.5 ± 10.3 | +%1.5 ± 2.0 | −%63.6 ± 4.7 |
| U-Net, birleşik kayıp (3 tohum) | Piksel-değişim | −%20.7 ± 2.0 | −%30.8 ± 2.0 | −%46.0 ± 2.9 | −%0.8 ± 0.8 | −%66.3 ± 3.3 |
| U-Net, işaretçisiz eğitim (3 tohum) | Kutu | −%32.6 ± 1.8 | −%14.0 ± 2.7 | −%25.9 ± 5.0 | −%2.1 ± 0.8 | −%59.6 ± 2.6 |
| U-Net, işaretçisiz eğitim (3 tohum) | Piksel-değişim | −%50.4 ± 1.8 | −%50.2 ± 3.1 | −%73.6 ± 4.5 | −%0.3 ± 0.4 | −%50.3 ± 10.1 |
| U-Net, işaretçisiz eğitim (3 tohum) | Glif | −%36.6 ± 4.2 | −%36.9 ± 6.9 | −%51.4 ± 6.0 | −%0.9 ± 2.1 | −%66.5 ± 2.6 |
| YOLO11s (3 tohum) | Kutu | −%45.2 ± 4.0 | −%0.6 ± 0.2 | −%1.3 ± 0.1 | −%1.0 ± 0.2 | −%57.5 ± 3.0 |

Kaynak: `phase3/results/tables_p3.md` (T6)

![Şekil 32. Faz 2 dayanıklılık deneyi (üç tohum): JPEG, yeniden ölçekleme ve bulanıklık altında AUPRC değişimi; solda kutu referansı, sağda piksel-değişim referansıyla puanlama.](phase2/experiments/figures/fig_robustness.png)

Standart U-Net modellerinde gri düzeyin etkisi kutu referansına göre görüntü başı AP üzerinden ayrıca ölçülmüştür: kutu modeli 0.388, piksel modeli 0.370 AP kaybeder (üç tohum). İşaretçisiz eğitilen modeller JPEG sıkıştırmasına standart eşlerinden daha kırılgandır (−%33 ile −%50); işaretçi yokken modeller daha ince ve sıkıştırmayla daha kolay silinen izlere dayanmaktadır.

**AIForge-Doc v2 harici testi.** v1 üzerinde eğitilen bütün modeller, yeniden eğitim ve yeniden kalibrasyon olmadan (eşikler v1 doğrulamasından) v2 birincil kümesinin 567 çiftine uygulanmıştır. v2 farklı bir üreticiyle (GPT-Image-2) üretilmiştir ve işaretçi kalıntısı taşımaz. Aynı düzenleme tarifinin v1 ve v2 görüntüleri karşılaştırılarak üretici değişiminin etkisi işaretçi kaybının etkisinden ayrılmıştır: v1 görüntüsü kalıntı taşıyan 397 çift ve taşımayan 170 çift ayrı raporlanmıştır (Tablo 35).

Table: Tablo 35. AIForge-Doc v2 birincil kümesi (567 çift), v1 üzerinde eğitilmiş modeller: havuzlanmış piksel AUPRC, görüntü AUROC ve aynı tarifin v1 test görüntüsünden v2 görüntüsüne görüntü başı AP değişimi.

| Model (v1 üzerinde eğitildi) | Tohum | Kutu AUPRC | Piksel-değişim AUPRC | Görüntü AUROC | AP değişimi, v1 işaretçili | AP değişimi, v1 işaretçisiz |
|---|---|---|---|---|---|---|
| U-Net kutu | 3 | 0.520 ± 0.018 | 0.272 | 0.717 | −0.490 | −0.169 |
| U-Net piksel-değişim | 3 | 0.227 | 0.246 | 0.693 | −0.398 | −0.172 |
| U-Net glif | 3 | 0.147 | 0.192 | 0.688 | −0.171 | −0.112 |
| DeepLabV3+ kutu | 3 | 0.532 | 0.282 | 0.725 | −0.371 | −0.137 |
| U-Net kutu, işaretçisiz eğitim | 3 | 0.711 ± 0.007 | 0.391 | 0.840 | −0.188 | −0.008 |
| U-Net piksel, işaretçisiz eğitim | 3 | 0.346 | 0.422 | 0.816 | −0.103 | −0.077 |
| U-Net glif, işaretçisiz eğitim | 3 | 0.205 | 0.274 | 0.754 | −0.058 | −0.061 |
| U-Net kutu, yalnız Gemini | 1 | 0.449 | 0.248 | 0.742 | −0.540 | −0.189 |
| YOLO11s, karolu | 3 | 0.887 ± 0.006 | 0.354 | 0.919 | −0.149 | −0.033 |
| YOLO11s, tam görüntü 1280 | 3 | 0.700 | 0.331 | 0.781 | −0.209 | −0.070 |
| YOLO11s, yalnız Gemini | 3 | 0.819 | 0.334 | 0.864 | −0.225 | −0.046 |
| YOLO11s, özgün negatif karo | 3 | 0.857 | 0.338 | 0.932 | −0.196 | −0.056 |

Kaynak: `phase3_v2/results/v2_tables.md`, `phase3_v2/results/T_v2_marker_split_agg.csv`, `reports/PHASE3_Sonuclar_ve_Kapanis.md`

v1 testinde 0.950 alan kutu U-Net v2 üzerinde 0.520'ye, görüntü AUROC değeri 0.717'ye düşer; karolu dedektör 0.887 ve 0.919 değerlerini korur. Kutu U-Net, v1 görüntüsü kalıntı taşıyan tariflerde 0.49, taşımayanlarda 0.17 AP kaybeder. İşaretçisiz eğitilen kutu U-Net ise kalıntısız tariflerde ölçülebilir bir kayıp göstermez (üç tohum ortalaması −0.008, %95 güven aralığı [−0.069, +0.053]; tohum bazında +0.003, −0.026, −0.002) ve toplamda 0.711 ± 0.007'ye ulaşır; v1 üzerinde standart modelin gerisindeyken v2 üzerinde belirgin biçimde önündedir. İşaretçisiz piksel ve glif modelleri de v2 üzerindeki en iyi piksel ve glif lokalizatörleridir (piksel-değişim 0.422 ± 0.015, standart model 0.246; glif 0.432 ± 0.015, standart model 0.304; üç tohum). Bu sonuç, işaretçi bulgusunu veri tabanının dışından doğrulamaktadır: işaretçiyle eğitilen modeller en çok, tam da işaretçiyi taşıyan tariflerde kaybeder. Referans etkisi v2 üzerinde de sürer: bütün modeller piksel-değişim ölçütünde 0.13–0.42 aralığında kalır.

v2'nin 3.066 çiftinin tamamında dedektörün görüntü AUROC değeri 0.912'dir; kaynak belgesi v1 eğitiminde görülmüş çiftlerde 0.911, görülmemiş birincil kümede 0.915'tir, yani belge örtüşmesinin etkisi yok denecek kadar azdır. Veri tabanı yazarlarının sıfır-atış TruFor için bildirdiği değer 0.599, DocTamper için 0.585'tir; dedektör v1 üzerinde eğitildiği için bu bir başvuru değeridir, eşit koşullu karşılaştırma değildir.

### 4.6 Açıklanabilirlik bulguları

**SHAP sadakati ve işaretçi yoğunlaşması.** Dört çekirdek model, sekiz girdi (üç doğru pozitif, üç yanlış negatif, iki doğru negatif) üzerinde bölümleme açıklayıcısıyla sınanmıştır (Tablo 36). Toplamsallık hatası bütün koşularda sıfırdır: katkıların toplamı model çıktısı ile referans çıktısı arasındaki farka eşittir. Silme deneyinde SHAP sırasıyla yapılan silme, rastgele sıraya göre çıktıyı belirgin biçimde daha hızlı düşürür (kutu U-Net için 0.445 ve 0.191); açıklamalar modelin gerçekten kullandığı bölgeleri göstermektedir. En önemli bulgu bölge kütlesindedir: pozitif katkı kütlesi yeşil işaretçi piksellerinde, bu piksellerin alan payının 4.3 ile 15.1 katı kadar yoğunlaşır; kutu kenar halkasında yoğunlaşma 4.1–6.2 kattır. Bölüm 4.3'te girdi müdahalesiyle ölçülen kısayol, burada modelin kendi karar ayrıştırmasında da görülmektedir. Bu ölçüm CPU üzerinde ve düşük değerlendirme bütçesiyle (200) yapılmıştır; haritalar kaba bölümlemelidir ve sekiz girdi küçük bir örneklemdir.

Table: Tablo 36. SHAP sadakat ölçümleri (bölümleme açıklayıcısı, pencere hedefi, 200 değerlendirme, sekiz girdi).

| Model | En büyük toplamsallık hatası | AOPC, SHAP sırası | AOPC, rastgele sıra | SHAP rastgeleden iyi | İşaret etme oyunu | İşaretçi kütlesi (yoğunlaşma) | Kenar halkası kütlesi (yoğunlaşma) |
|---|---|---|---|---|---|---|---|
| U-Net kutu | 0.000000 | 0.445 | 0.191 | %100 | 0.50 | 0.012 (4.3 kat) | 0.132 (5.6 kat) |
| U-Net piksel-değişim | 0.000000 | 0.321 | 0.101 | %88 | 0.50 | 0.025 (8.6 kat) | 0.164 (6.2 kat) |
| U-Net glif | 0.000000 | 0.349 | 0.167 | %100 | 0.50 | 0.022 (7.5 kat) | 0.131 (5.4 kat) |
| ResNet-18, tam ince ayar | 0.000000 | 0.623 | 0.225 | %100 | 0.50 | 0.056 (15.1 kat) | 0.339 (4.1 kat) |

Kaynak: `docs/xai_test_report.md`, `docs/xai_benchmark.csv`

Aynı pencerede üç tohumun açıklamaları karşılaştırıldığında en yüksek %5'lik pozitif bölgelerin örtüşmesi (IoU) kutu modellerinde 0.610, glif modellerinde 0.867'dir (iki görüntü); tohumlar benzer bölgelere bakmakta, ancak sıralama düzeyinde uyum sınırlı kalmaktadır (Spearman 0.344 ve 0.300).

**Tam görüntü SHAP.** Pencere hedefleri belgenin yalnız bir bölümünü açıklıyordu ve U-Net'te açıklanan pencere skoru uygulamanın tam görüntü skorundan 0.33'e kadar farklı olabiliyordu. Tam görüntü hedefleriyle bu fark giderilmiştir. Dört görüntü (0.28–4.48 megapiksel) ve altı model üzerindeki ölçümlerin tamamında harita boyutu görüntü boyutuna eşittir, tam görüntü sonuçlarında toplamsallık hatası en çok $1.1\times10^{-4}$, açıklanan değer ile uygulamanın tahmin tablosundaki değer arasındaki fark en çok 0.0012'dir. Tablo 37, 864×1296 piksellik bir test görüntüsü için model başına ölçümleri vermektedir. Bu görüntü kutu U-Net için bir yanlış negatiftir (skor 0.118; karar eşiği 0.167); iki dedektörün en güvenli kutusu aynı görüntüde 0.33 ve 0.66 güven almaktadır. 500 değerlendirmelik bütçeyle süre üç kanallı U-Net için 18–53 saniye, DeepLabV3+ için 10–26, ResNet-18 için 4–51, YOLO11 için 11–30 saniyedir; frekans kanallı U-Net, kanalları her değerlendirmede yeniden hesapladığından en yavaş modeldir (1.12 megapikselde 106 saniye).

Table: Tablo 37. Tam görüntü SHAP ölçümleri: `testing:000000038` (864×1296 piksel), 500 değerlendirme, RTX 5080.

| Model | Hedef | Açıklanan değer | Uygulama skoru | Taban değeri | Toplamsallık hatası | Süre (s) | Kutu içi pozitif kütle payı |
|---|---|---|---|---|---|---|---|
| U-Net kutu | Tam görüntü skoru | 0.1175 | 0.1176 | 0.0007 | 1.1e-04 | 18.6 | 0.124 |
| U-Net RGB + frekans (Faz 1) | Tam görüntü skoru | 0.0440 | 0.0440 | 0.0507 | 4.9e-17 | 106.0 | 0.002 |
| ResNet-18, tam ince ayar | En yüksek yama | 0.8245 | 0.8233 | 0.0268 | 3.3e-16 | 15.5 | 0.146 |
| DeepLabV3+ kutu | Tam görüntü skoru | 0.4167 | 0.4155 | 0.0056 | 7.9e-05 | 19.9 | 0.007 |
| YOLO11s, tam görüntü 1280 | Kutu güveni | 0.3293 | 0.3290 | 0.0000 | 1.7e-16 | 12.3 | 0.197 |
| YOLO11m, karolu | Kutu güveni | 0.6602 | 0.6600 | 0.0000 | 1.1e-15 | 22.3 | 0.261 |

Kaynak: `reports/xai_fullimage/RAPOR_OLCUM.md`, `reports/xai_fullimage/RAPOR.md`

**Katman filtreleri.** Kayıtlı 50 modelin tamamı (ek tohum ağırlıkları dahil) filtre denetimlerinden geçmiştir: kayıtlı çekirdek modül ağırlığına eşittir, bağımsız evrişim modelin kendi evrişim çıktısını verir (CPU üzerinde en kötü göreli fark $10^{-6}$ düzeyinde, YOLO modellerinde sıfır), kanal katkılarının toplamı evrişim çıktısına eşittir ve ağırlık özetleri değişmemiştir. Döşemeli tam görüntü hesabı tek geçişle birebir aynı sonucu verir. Görselleştirmeden üç gözlem çıkmaktadır.

1. Bütün bölütleme ve sınıflandırma modelleri aynı ResNet-18 gövdesini paylaşır: ilk katman 64 adet 7×7 süzgeçtir. Şekil 33'teki çekirdekler, ImageNet ön eğitiminden gelen yönlü kenar, çizgi ve renk karşıtlığı örüntülerini göstermektedir; bunlar dersin kenar bulma ve frekans süzgeçleri konularının öğrenilmiş karşılıklarıdır. YOLO11s gövdesi farklıdır: ilk katmanı 32 adet 3×3 süzgeçtir.
2. Belge görüntüsünde en güçlü yanıtlar metin kenarlarında oluşur (Şekil 34). Etkinleştirme sonrasında bazı haritalar tümüyle sıfırdır; bunlar renk karşıtlığı süzgeçleridir ve gri tonlu bir fiş üzerinde yanıt vermemeleri gerçek model davranışıdır.
3. Referans kutusunun içindeki yanıtın dışındakine oranı en yüksek olan süzgeç kutu U-Net için 58 numaralı süzgeçtir (Şekil 35). Bu süzgecin kanal katkıları arasında en büyük genlik yeşil kanaldadır; ilk katman renk kanallarını ayrı ayrı ağırlıklandırmaktadır. Bu gözlem, gri düzeye dönüşümün bilgi kaybettirmesiyle (Bölüm 4.5) uyumludur.

![Şekil 33. U-Net kutu modelinin ilk evrişim katmanındaki 64 süzgecin ağırlıkları (7×7, RGB bileşik gösterim).](reports/xai_filters/figs/PHASE2_unet_bbox_s42__1_kernel_grid.png)

![Şekil 34. Aynı katmanın bir test görüntüsündeki 64 etkinleşme haritası (etkinleştirme sonrası, harita başına ölçekleme).](reports/xai_filters/figs/PHASE2_unet_bbox_s42__3_aktivasyon_grid.png)

![Şekil 35. 58 numaralı süzgeç için özgün görüntüyle karşılaştırma: kanal katkıları, evrişim, yığın normalizasyonu ve etkinleştirme aşamaları, bindirme.](reports/xai_filters/figs/PHASE2_unet_bbox_s42__5_karsilastirma_f58.png)

### 4.7 Uygulama testleri ve canlı dağıtım

Uygulamanın, raporlanan sonuçları değiştirmeden yeniden ürettiğini göstermek için otomatik testler, tarayıcı testi ve canlı Space testi yapılmıştır (Tablo 38). Yerel testler `kod/tests/run_all_tests.ps1` ile tek komutla çalışır; aşağıdaki değerler 11 Kasım 2026 tarihli son koşunun kayıtlarıdır (ek tohum ağırlıkları eklendikten sonra). Otomatik testler Faz 4 modelleri uygulamaya eklendikten sonra yinelenmiştir (Bölüm 4.9); tarayıcı testi bu eklemeden önceki sürüme aittir; canlı Space testi, Faz 4 sürümü 11 Kasım 2026'da dağıtıldıktan sonra yinelenmiştir.

Table: Tablo 38. Uygulama testlerinin özeti.

| Test | Kapsam | Sonuç |
|---|---|---|
| `kod/tests/test_app.py` | Ön işleme işlemleri, hazır ayarlar, örnek süzgeçleri, senaryolar, bozulmalar, simülasyon çıktıları, hatalı girdiler, toplu test, kayıtlı sonuçlarla yeniden üretim (35 örnek, 36 model yüklü, GPU) | 289 kontrol, 0 hata, 144 s |
| `kod/tests/test_app_p3.py` | Faz 3 modellerinin listelenmesi ve yüklenmesi, yeni senaryolar ve bozulmalar, uygulama içi harita ile değerlendirme hattının kaydettiği haritanın karşılaştırması (3 görüntü × 9 model, GPU) | 0 hata, 50 s |
| `kod/tests/test_xai_filters.py` | Sentetik çekirdek denetimi, evrişimsiz model reddi, 50 kayıtlı model, YOLO ilk katman dışa aktarımı, GPU ve CPU uyumu, görsel üretimi, döşemeli tam görüntü | 91 kontrol, 0 hata, 134.0 s |
| `kod/tests/test_xai.py` | SHAP adaptörleri, toplamsallık, ağırlık bütünlüğü, kayıt ve karşılaştırma (15 test) | GPU: 15 geçti (49.5 s); CPU: 15 geçti |
| `kod/tools/capture_screens.py` | Gerçek tarayıcıda (Chromium) güncel arayüzün uçtan uca yürütülmesi: örnek seçimi, simülasyon, sekmelere aktarım, hazır ön işleme hattı, U-Net, 8 örnekli toplu test, sonuçlar, SHAP ve katman filtreleri | 8 adımın tümü hatasız (11 Kasım 2026, Faz 4 modellerini içeren güncel arayüz); Şekil 10–16 bu koşunun çıktılarıdır. Görüntülerde kullanıcının ev dizini `~` olarak kısaltılmıştır |
| `kod/hf_space/live_test.py` | Canlı Space üzerinde 6 senaryo × 6 model simülasyon ve 24 çiftli toplu test (ZeroGPU); ek koşuda Faz 4 sınıflandırıcılarıyla 6 senaryo × 8 model | Tamamlandı (11 Kasım 2026, Faz 4 sürümü): 393.4 s; ek koşu 380.9 s |

Kaynak: `tests/run_all_summary.txt`, `tests/run_all_app.log`, `tests/run_all_app_p3.log`, `tests/run_all_xai_filters.log`, `tests/run_all_xai.log`, `tests/run_all_xai_cpu.log`, `app/ekran_*.png`, `reports/canli_test_v4deploy_log.txt`, `reports/canli_test_v4deploy_faz4_log.txt`

**Yeniden üretim.** Yerel testte kayıtlı sonucu bulunan dört görüntüde uygulamanın hesapladığı F1 ile değerlendirme hattının kayıtlı değeri arasındaki ortalama mutlak fark kutu, piksel-değişim ve glif modelleri için sırasıyla 0.0001, 0.0002 ve 0.0005'tir (en büyük 0.0007). Faz 3 modellerinde 27 karşılaştırmanın 25'inde görüntü skoru farkı en çok 0.0005'tir; kalan iki durum YOLO modellerine aittir (0.011 ve 0.043) ve yarı duyarlıklı hesap ile bastırma eşiğinin sınırındaki kutulardan kaynaklanır; haritada değişen piksel oranı en çok %0.12'dir.

**Sayısal duyarlık ve test toleransı.** SHAP paketinin 15 testi hem CPU hem GPU üzerinde geçmektedir. İki kesinlik testi, açıklama adaptörünün hesapladığı $f(x)$ değerini uygulamanın kendi skoruyla karşılaştırır. CPU üzerinde çıkarım tek duyarlıklı (fp32) yapılır ve iki değer $10^{-5}$ içinde eşleşir. GPU üzerinde ise uygulama çıkarımı yarı duyarlıklı (bf16) karma hesapla yapar; bu biçim yaklaşık üç anlamlı basamak taşır ve aynı pencerenin farklı parti bileşimleriyle hesaplanması yuvarlamayı değiştirir. Ölçülen farklar beş kanallı U-Net pencere skorunda $3.0\times10^{-5}$, ResNet-18 en yüksek yama skorunda $2.4\times10^{-3}$'tür. Bu nedenle testin eşleşme toleransı aygıta göre tanımlanmıştır (CPU $10^{-5}$, GPU $5\times10^{-3}$; `kod/tests/test_xai.py`). Açıklamaların toplamsallığı ($E[f] + \sum \text{SHAP} = f(x)$) bu farktan etkilenmez; ayrı bir testle doğrulanmaktadır.

**Kod düzenlemesinin etkisi.** Testler, kodun `kod/` çatısı altına taşınmasından önce, taşınmasından sonra ve `app.py`'nin `appcore/` modüllerine bölünmesinden sonra ayrı ayrı koşulmuş; üç durumda da aynı kontrol sayıları ve aynı sonuçlar elde edilmiştir. Taşıma öncesi ve sonrası alınan yol dökümleri birebir aynıdır; sonuç tabloları ve model ağırlıkları değişmemiştir. Faz 4 modellerinin eklenmesiyle yüklenen model sayısı 34'ten 36'ya, kontrol sayısı 287'den 289'a çıkmıştır.

**Canlı dağıtım.** Uygulama `infinayazilim/ai515` Space'ine ZeroGPU donanımıyla dağıtılmış ve dışarıdan, programlı istemciyle sınanmıştır. Faz 4 sürümü (tam veriyle eğitilen ResNet-18 modelleri ve kalibre görüntü eşikleri) 11 Kasım 2026'da dağıtılmış, canlı test aynı gün iki koşuyla yinelenmiştir: önceki dağıtımlarla aynı model kümesi (393 saniye) ve Faz 4 sınıflandırıcılarının eklendiği küme (381 saniye). Tablo 39, test bölümünden rastgele seçilen 24 eşli örnekte (24 sahte ve 24 özgün) dengeli tespit sonuçlarını vermektedir. Uygulamanın canlı ortamda hesapladığı F1 ve AP değerleri, aynı görüntülerde değerlendirme hattının kayıtlı sonuçlarıyla birebir aynıdır (ortalama mutlak fark 0.000); bölütleme ve dedektör satırları önceki dağıtımdaki değerlerle de aynıdır. Örneklem küçük olduğundan mutlak değerler resmî test sonuçlarından sapabilir; ancak ana örüntü burada da görülür: kutu U-Net kutu ölçütünde 0.925, aynı tahminler piksel-değişim ölçütünde 0.528 F1 alır.

ResNet-18 kararları bu sürümde kalibre eşikle verilmektedir (Bölüm 4.9). Kısıtlı veri modelinin yanlış alarm oranı, önceki dağıtımdaki 0.375'ten (sabit 0.5 eşiği) 0.083'e inmiştir. Tam veri modeli 24 özgün görüntünün hiçbirini işaretlemez ve sahtelerin 0.917'sini yakalar (AUROC 0.998). İşaretçisiz eğitilen tam veri modelinde doğru tespit oranı 0.583'tür; eşi bulunan 89 test sahtesinin tamamında değerlendirme hattının bu model için verdiği oran 0.719'dur ve 24 örneklik fark örnekleme aralığındadır, ancak eşiğin doygunluğa yakın olması (0.965) sınırdaki skorları (0.953–0.957) kaçırmaktadır. Tek örnekli senaryo testinde işaretçi bulgusu canlı ortamda da görülür: yalnız kalıntı eklenmiş özgün görüntüyü kısıtlı veri ve tam veri modelleri sahte sayarken (skor 0.952 ve 1.000) işaretçisiz eğitilen model özgün olarak sınıflandırır (0.529); aynı örnekte bu model sahte görüntünün kendisini de kaçırmıştır (skor 0.648).

Table: Tablo 39. Canlı Space toplu testi (11 Kasım 2026, Faz 4 sürümü): test bölümünden 24 sahte ve 24 özgün görüntü; ResNet-18 satırlarında kalibre eşik.

| Model | F1, kutu | AP, kutu | F1, piksel-değişim | AP, piksel-değişim | Doğru tespit oranı | Yanlış alarm oranı | Görüntü AUROC |
|---|---|---|---|---|---|---|---|
| ResNet-18, kısıtlı veri (Faz 1) | — | — | — | — | 0.917 | 0.083 | 0.990 |
| ResNet-18, tam veri (tohum 42) | — | — | — | — | 0.917 | 0.000 | 0.998 |
| ResNet-18, tam veri, işaretçisiz eğitim (tohum 42) | — | — | — | — | 0.583 | 0.000 | 0.967 |
| U-Net kutu (tohum 42) | 0.925 | 0.964 | 0.528 | 0.410 | 1.000 | 0.042 | 0.988 |
| U-Net piksel-değişim (tohum 42) | 0.739 | 0.730 | 0.635 | 0.609 | 1.000 | 0.083 | 1.000 |
| YOLO11s, karolu (tohum 42) | 0.948 | 0.939 | 0.534 | 0.394 | 1.000 | 0.042 | 1.000 |

Kaynak: `reports/canli_test_v4deploy_log.txt`, `reports/canli_test_v4deploy_faz4_log.txt` (önceki dağıtımlar: `reports/canli_test_v3deploy_log.txt`, `reports/deploy_v3.txt`, `reports/canli_test_v2deploy_log.txt`)

### 4.8 Tek örnek üzerinden uçtan uca gösterim

Canlı testte rastgele seçilen bir test örneği (`testing:000000728`) altı girdi senaryosunun tamamından ve altı modelden geçirilmiştir. Bu tek örnek, raporun üç ana bulgusunu birlikte göstermektedir. Tablo 40 her senaryoda modellerin görüntü düzeyi kararını ve skorunu, Tablo 41 sahte girdi için lokalizasyon ölçütlerini vermektedir.

Table: Tablo 40. `testing:000000728` örneğinde altı senaryo: görüntü düzeyi karar, doğruluğu ve görüntü skoru.

| Senaryo (doğru karar) | ResNet-18 | U-Net kutu | U-Net piksel-değişim | U-Net glif | YOLO11s | DeepLabV3+ kutu |
|---|---|---|---|---|---|---|
| Sahte belge (sahte) | sahte, doğru (1.000) | sahte, doğru (1.000) | sahte, doğru (0.787) | sahte, doğru (0.421) | sahte, doğru (0.969) | sahte, doğru (1.000) |
| Özgün eş (gerçek) | sahte, yanlış (0.665) | gerçek, doğru (0.017) | gerçek, doğru (0.004) | gerçek, doğru (0.007) | gerçek, doğru (0.000) | gerçek, doğru (0.030) |
| Sahte, işaretçi onarılmış (sahte) | sahte, doğru (0.665) | sahte, doğru (1.000) | sahte, doğru (0.176) | sahte, doğru (0.335) | sahte, doğru (0.969) | sahte, doğru (1.000) |
| Özgün + yalnız işaretçi (gerçek) | sahte, yanlış (1.000) | sahte, yanlış (0.998) | sahte, yanlış (0.507) | sahte, yanlış (0.344) | sahte, yanlış (0.603) | sahte, yanlış (0.998) |
| Yalnız yeniden çizim (gerçek) | sahte, yanlış (0.665) | sahte, yanlış (0.662) | gerçek, doğru (0.004) | gerçek, doğru (0.013) | sahte, yanlış (0.878) | sahte, yanlış (0.962) |
| Yalnız içerik (sahte) | sahte, doğru (0.665) | sahte, doğru (0.985) | sahte, doğru (0.095) | sahte, doğru (0.249) | sahte, doğru (0.722) | sahte, doğru (0.996) |

Kaynak: `reports/canli_test_v2deploy_log.txt`

Table: Tablo 41. `testing:000000728` örneğinde sahte girdi için lokalizasyon ölçütleri.

| Model | Görüntü skoru | F1, kutu referansı | F1, piksel-değişim referansı | Kutu kapsama |
|---|---|---|---|---|
| U-Net kutu | 1.000 | 0.978 | 0.478 | 0.976 |
| U-Net piksel-değişim | 0.787 | 0.614 | 0.723 | 0.449 |
| U-Net glif | 0.421 | 0.329 | 0.638 | 0.200 |
| YOLO11s, karolu | 0.969 | 0.994 | 0.520 | 0.989 |
| DeepLabV3+ kutu | 1.000 | 0.982 | 0.504 | 0.990 |

Kaynak: `reports/canli_test_v2deploy_log.txt`

Örnekten üç okuma yapılabilir. Birincisi, referans etkisi tek görüntüde de görünür: kutu U-Net, DeepLabV3+ ve YOLO11s kutu referansıyla 0.98–0.99 F1 alırken aynı tahminler piksel-değişim referansıyla 0.48–0.52'ye iner; piksel modeli ise ters yönde davranır (0.614 ve 0.723). İkincisi, işaretçi kısayolu doğrudan izlenir: hiçbir karakteri değişmemiş özgün görüntüye yalnız yeşil kalıntı pikselleri eklendiğinde altı modelin tamamı görüntüyü sahte olarak işaretler ve kutu U-Net kutunun %74'ünü boyar. Üçüncüsü, modeller farklı ipuçlarına dayanır: karakterler özgün hâline döndürülüp yalnız üreticinin yeniden çizim dokusu bırakıldığında kutu modelleri ve dedektör yine sahte derken piksel ve glif modelleri doğru biçimde gerçek kararı verir. Kutu denetimi "bu alan yeniden çizilmiş" bilgisini, piksel düzeyi denetim "bu karakterler değişmiş" bilgisini öğretmektedir. Bu örnekte işaretçinin onarılması kutu modelini etkilememiş (F1 0.973), piksel modelinin skorunu ise 0.787'den 0.176'ya indirmiştir; bağımlılığın örnekten örneğe değiştiği ve genel sonucun Bölüm 4.3'teki küme düzeyi ölçümlerden okunması gerektiği unutulmamalıdır. ZeroGPU üzerinde her senaryo çağrısı, GPU ayırma ve altı modelin tam çözünürlükte çalışması dahil yaklaşık 65–69 saniye sürmüştür.

### 4.9 ResNet-18: tam veri, karar eşiği kalibrasyonu ve işaretçi onarımı

Bölüm 4.7'deki canlı testte ResNet-18 sahtelerin tamamını yakalarken özgün görüntülerin %37.5'ini de sahte saymıştır. Bunun üç olası nedeni ayrı ayrı ölçülmüştür: uygulamadaki model 400 görüntülük kısıtlı veriyle eğitilmiştir; görüntü kararı doğrulanmamış sabit bir eşikle (en yüksek yama skoru 0.5 ve üzeri) verilmektedir; model yeşil işaretçi kısayoluna yaslanıyor olabilir. Bölüm 3.5'te tanımlanan iki koşu kümesi (tam veri; tam veri ve işaretçi onarımı; üçer tohum) kısıtlı veri modeliyle aynı kümelerde karşılaştırılmıştır. Hipotezler ve karar ölçütleri koşulardan önce yazılmıştır (`kod/phase4_resnet/README.md`); koşu başına eğitim 65–72 saniye, bütün hat yaklaşık 20 dakika sürmüştür.

**Yama düzeyi.** Tablo 42, üç modeli aynı 1.431 doğrulama yamasında karşılaştırmaktadır. Tarif değişmeden yalnız veri miktarının artması AUROC değerini 0.926'dan 0.983'e çıkarmıştır. İşaretçisi onarılmış doğrulama yamalarında ise tam veri modelinin AUROC değeri 0.900'e, 0.5 eşiğindeki doğru tespit oranı 0.582'ye iner; işaretçisiz eğitilen model aynı yamalarda 0.980 ve 0.910 değerlerini korur.

Table: Tablo 42. ResNet-18 yama düzeyi sonuçları (487 doğrulama görüntüsünden 1.431 yama; eşik 0.5): ham doğrulama yamaları ve işaretçi kalıntısı onarılmış yamalar. Üç tohumlu satırlarda ortalama ± standart sapma.

| Model | Tohum | AUROC | Doğru tespit oranı | Yanlış alarm oranı | AUROC, Ideogram | AUROC, onarılmış yamalar | Doğru tespit oranı, onarılmış yamalar |
|---|---|---|---|---|---|---|---|
| Kısıtlı veri (400 görüntü) | 1 | 0.926 | 0.784 | 0.051 | 0.837 | 0.735 | 0.306 |
| Tam veri (2.748 görüntü) | 3 | 0.983 ± 0.003 | 0.912 ± 0.022 | 0.014 ± 0.006 | 0.949 ± 0.016 | 0.900 ± 0.009 | 0.582 ± 0.046 |
| Tam veri, işaretçisiz eğitim | 3 | 0.982 ± 0.001 | 0.906 ± 0.007 | 0.030 ± 0.005 | 0.946 ± 0.009 | 0.980 ± 0.001 | 0.910 ± 0.008 |

Kaynak: `phase4_resnet/results/patch_level_val.csv`, `reports/PHASE4_ResNet_Sonuclar.md`

**Görüntü düzeyi ve karar kuralı.** Toplama kuralı olarak en yüksek yama seçilmiştir: tam veri modelinin üç tohumunda ortalama doğrulama AUROC değeri en yüksek yama için 0.9655, en yüksek 2, 3 ve 5 yamanın ortalaması için sırasıyla 0.962, 0.958 ve 0.951'dir. Tablo 43 eski kuralı ve kalibre eşiği yan yana vermektedir. Tam veri modelinde kalibrasyon doğrulama yanlış alarm oranını 0.242'den 0.021'e indirir; doğru tespit oranı 0.951'den 0.886'ya düşer ve dengeli doğruluk 0.855'ten 0.933'e çıkar. Yalnız raporlanan resmî testte kalibre kural 0.978 AUROC, 0.901 doğru tespit ve 0.022 yanlış alarm oranı verir. Kalibre eşikler 0.94 ile 0.995 arasındadır: görüntü başına onlarca yamanın en yükseği alındığından özgün görüntülerde de skor yükselir ve 0.5 bu yüzden yanlış eşiktir. Tam veri modelinin test AUROC değeri için kaynak belge grubu üzerinden bootstrap %95 güven aralıkları tohum sırasıyla [0.960, 0.989], [0.963, 0.986] ve [0.971, 0.988]'dir.

Table: Tablo 43. ResNet-18 görüntü düzeyi karar (doğrulama: 487 sahte ve 487 eşleşen özgün görüntü; resmî test: 812 sahte ve eşi bulunan 89 özgün görüntü). Eşik yalnız doğrulamada seçilmiştir; üç tohumlu satırlarda ortalama ± standart sapma.

| Model | Karar kuralı | AUROC, doğrulama | Doğru tespit, doğrulama | Yanlış alarm, doğrulama | AUROC, test | Doğru tespit, test | Yanlış alarm, test |
|---|---|---|---|---|---|---|---|
| Kısıtlı veri | En yüksek yama 0.5 ve üzeri | 0.893 | 0.916 | 0.456 | 0.919 | 0.930 | 0.404 |
| Kısıtlı veri | Kalibre eşik (0.938) | 0.893 | 0.791 | 0.101 | 0.919 | 0.805 | 0.079 |
| Tam veri | En yüksek yama 0.5 ve üzeri | 0.965 ± 0.002 | 0.951 ± 0.005 | 0.242 ± 0.048 | 0.978 ± 0.003 | 0.963 ± 0.007 | 0.109 ± 0.034 |
| Tam veri | Kalibre eşik | 0.965 ± 0.002 | 0.886 ± 0.003 | 0.021 ± 0.011 | 0.978 ± 0.003 | 0.901 ± 0.014 | 0.022 ± 0.011 |
| Tam veri, işaretçisiz eğitim | En yüksek yama 0.5 ve üzeri | 0.948 ± 0.002 | 0.958 ± 0.008 | 0.374 ± 0.016 | 0.958 ± 0.004 | 0.967 ± 0.004 | 0.345 ± 0.034 |
| Tam veri, işaretçisiz eğitim | Kalibre eşik | 0.948 ± 0.002 | 0.840 ± 0.018 | 0.061 ± 0.017 | 0.958 ± 0.004 | 0.834 ± 0.011 | 0.064 ± 0.017 |

Kaynak: `phase4_resnet/results/tables_p4.md`, `phase4_resnet/results/resnet_image_tau.json`

**İşaretçi bağımlılığı.** Tablo 44, kalıntı taşıyan 368 doğrulama çiftinde kalibre eşikteki kararları vermektedir. Tam veri modeli sahtelerin 0.995'ini yakalar; aynı görüntülerde yalnız kalıntı pikselleri özgün görüntüden geri yazıldığında bu oran 0.471'e iner, hiçbir karakteri değişmemiş özgün görüntüye yalnız kalıntı eklendiğinde ise görüntülerin 0.955'i sahte sayılır. Model büyük ölçüde bir işaretçi dedektörüdür. İşaretçisiz eğitim bu bağımlılığı kaldırır (0.880 ve 0.899); yalnız kalıntı eklenmiş özgün görüntülerdeki yanlış alarm 0.458'e iner ama yüksek kalır. Bu sonda görüntüsü gerçekten bir yapıştırma içerdiğinden değerin sıfıra inmesi beklenmez. Bedel, doğrulama AUROC değerinin 0.965'ten 0.948'e inmesi ve yanlış alarm oranının 0.021'den 0.061'e çıkmasıdır. Resmî testteki kırılım aynı yönü gösterir (Tablo 45): kalıntı taşımayan 185 sahtede doğru tespit oranı kısıtlı veri modelinde 0.465, tam veri modelinde 0.618, işaretçisiz eğitilen modelde 0.735'tir; Ideogram düzenlemelerinde sıra aynıdır (0.518, 0.624 ve 0.737). Bölüm 4.3–4.5'te bölütleme modelleri ve kutu dedektörü için bulunan örüntü, görüntü düzeyinde karar veren üçüncü bir model ailesinde yinelenmektedir.

Table: Tablo 44. ResNet-18 işaretçi sondaları (kalıntı taşıyan 368 doğrulama çifti, kalibre eşik): doğru tespit ve yanlış alarm oranları. Üç tohumlu satırlarda ortalama ± standart sapma.

| Model | Sahte, doğru tespit | Sahte, kalıntısı onarılmış, doğru tespit | Özgün, yanlış alarm | Özgün + yalnız kalıntı, yanlış alarm |
|---|---|---|---|---|
| Kısıtlı veri | 0.910 | 0.207 | 0.073 | 0.848 |
| Tam veri | 0.995 ± 0.005 | 0.471 ± 0.028 | 0.016 ± 0.017 | 0.955 ± 0.019 |
| Tam veri, işaretçisiz eğitim | 0.880 ± 0.007 | 0.899 ± 0.005 | 0.049 ± 0.010 | 0.458 ± 0.004 |

Kaynak: `phase4_resnet/results/marker_probes.csv`, `phase4_resnet/results/tables_p4.md`

Table: Tablo 45. ResNet-18 resmî test kırılımı (kalibre kural): alt kümelere göre doğru tespit oranı. Üç tohumlu sütunlarda ortalama ± standart sapma.

| Alt küme (görüntü sayısı) | Kısıtlı veri | Tam veri | Tam veri, işaretçisiz eğitim |
|---|---|---|---|
| Kalıntı taşıyan (627) | 0.906 | 0.985 ± 0.004 | 0.863 ± 0.008 |
| Kalıntı taşımayan (185) | 0.465 | 0.618 ± 0.047 | 0.735 ± 0.029 |
| Gemini (727) | 0.839 | 0.934 ± 0.012 | 0.845 ± 0.011 |
| Ideogram (85) | 0.518 | 0.624 ± 0.042 | 0.737 ± 0.024 |
| Alan oranı %1'in altında (429) | 0.772 | 0.849 ± 0.021 | 0.822 ± 0.009 |
| Sızıntısız test (740) | 0.800 | 0.898 ± 0.013 | 0.827 ± 0.013 |

Kaynak: `phase4_resnet/results/breakdown_test.csv`, `phase4_resnet/results/tables_p4.md`

**Hipotez kararları.** Veri miktarının yama ayrımını belirgin iyileştireceği beklentisi (H6a; doğrulama AUROC en az 0.95) desteklenmiştir (0.983 ± 0.003). Kalibre karar kuralının yanlış alarmı düşüreceği beklentisi (H6b) desteklenmiştir; en yüksek birkaç yamanın ortalamasının tek en yüksek yamadan daha iyi olacağı önerisi ise desteklenmemiştir. İşaretçi onarımlı eğitimin kısayolu azaltacağı beklentisi (H6c) kısmen desteklenmiştir: kalıntısı onarılmış sahtelerdeki kayıp ortadan kalkmış, yalnız kalıntı eklenmiş özgün görüntülerdeki yanlış alarm ise yarıya inmekle birlikte yüksek kalmıştır.

**Karşılaştırma, sınırlar ve uygulama durumu.** Aynı doğrulama protokolünde uygulamanın kayıtlı görüntü AUROC değerleri kutu U-Net için tohuma göre 0.838–0.921, YOLO11s için 0.961'dir (`kod/app/meta/image_tau.json`). Tam veri ResNet-18 (0.965) bu düzeydedir, ancak yalnız kaba bir yama ısı haritası üretir ve lokalizasyon ölçütü yoktur. Test bölümünde eşi bulunan özgün görüntü sayısı 89 olduğundan test yanlış alarm oranının aralığı geniştir (Wilson aralığı yaklaşık %0.2–9). Eşikler doygunluğa yakındır ve tohuma duyarlıdır (0.940–0.995). Dayanıklılık ve ayrı tutulan üretici deneyleri bu model için yapılmamıştır. Yerel uygulamaya tam veri modeli ve işaretçisiz eğitilen model (tohum 42) eklenmiş, varsayılan ResNet-18 tam veri modeli olmuş ve görüntü kararı kalibre eşikle verilir hâle gelmiştir (`kod/app/meta/resnet_image_tau.json`). Üç model ve 40 görüntüde uygulamanın skoru ile değerlendirme hattının skoru arasındaki en büyük fark 0.0014'tür ve kararların tamamı aynıdır (`phase4_resnet/results/app_repro_check.json`). Bu sürüm 10 Kasım 2026'da kod deposuna, 11 Kasım 2026'da canlı Space'e gönderilmiştir; Bölüm 4.7'deki canlı test bu sürümle yinelenmiş, Şekil 10–16'daki ekran görüntüleri de 11 Kasım 2026'da bu sürümle yeniden alınmıştır.

## 5. Tartışma ve Sonuç

**Kutu etiketi neyi ödüllendiriyor?** Birbirinden bağımsız üç kanıt aynı yöne işaret etmektedir. Aynı tahminler, gerçekten değişen piksellere göre puanlandığında 0.55 AUPRC kaybeder. Kutu modeli alanın içinin bulanıklaştırılmasına duyarsızdır, kenarının bulanıklaştırılmasına değil. Aynı etiketlerle eğitilen bir kutu dedektörü kutu ölçütünde bütün bölütleme modellerini geçerken piksel-değişim ölçütünde kutu U-Net ile aynı düzeyde kalır. Lokalizasyonu düzenlenen alanın kutusuna göre puanlayan bir kıyas kümesi, piksel düzeyinde kurcalama lokalizasyonunu değil alan tespitini ölçmektedir. Alan tespiti ön eleme için yararlı bir yetenektir ve kutu dedektörü bunu en yüksek puanla sağlar; ancak piksel düzeyi lokalizasyon olarak raporlanmamalıdır.

**İşaretçi bir kısayoldur, hikâyenin tamamı değildir.** Yalnız yeşil kalıntı çizgisini özgün bir görüntüye yapıştırmak standart kutu modelinin alanı bulması için yeterlidir ve modelin test puanı kalıntı taşıyan görüntülerde 0.97, taşımayanlarda 0.69'dur. Buna karşın işaretçisi onarılmış görüntülerle yeniden eğitilen model kutu puanının çoğunu korur. Kalıntı varken kullanılmakta, yokken yerini başka ipuçları almaktadır. Bu, veri tabanında bildirilen başarıların işaretçili görüntülerde zorluk açısından olduğundan iyi göründüğü, ama yalnız işaretçiden ibaret olmadığı anlamına gelir. Kalıntı Gemini sahtelerinin çoğunda bulunup Ideogram sahtelerinde hemen hiç bulunmadığından, v1 üzerindeki üreticiler arası aktarım sonuçları üretici ile işaretçiyi karıştırmaktadır. İşaretçi taşımayan v2 üzerindeki harici test bunu dışarıdan doğrular: işaretçiyle eğitilen modeller en çok işaretçiyi taşıyan tariflerde kaybeder, işaretçisiz eğitilen kutu modeli diğerlerinde hiçbir şey kaybetmez. Aynı bağımlılık, görüntü düzeyinde karar veren ResNet-18 sınıflandırıcısında da ölçülmüştür (Bölüm 4.9): tam veriyle eğitilen model, kalıntı onarıldığında sahtelerin yarısından fazlasını kaçırmaktadır.

**Denetim türü ve model davranışı.** Piksel-değişim denetimi en sadık maskeleri ve sabit eşikte en az yanlış alarmı verir, ancak bulanıklık ve yeniden ölçeklemeye daha kırılgandır ve üretici açığını kapatmaz. Kutu yardımcı kanalı eklemek bu kırılganlığı azaltan tek varyanttır (üç tohum; bulanıklık kaybı −%59 yerine −%46). Gri düzeye dönüşüm bütün modellere büyük zarar verir; ilk katman süzgeçleri ve SHAP bölge kütlesi, modellerin renk bilgisini ve kutu kenarını yoğun biçimde kullandığını göstermektedir.

**Ders açısından kazanımlar.** Klasik öznitelikler şans üstü ama zayıf kalmış, DCT özniteliklerinin kayıpsız PNG verisinde ayırt edici olmadığı görülmüştür. Ön eğitimli ResNet-18 yalnız gövdesi de güncellendiğinde belirgin üstünlük sağlamıştır. Aynı model tam veriyle eğitilip karar eşiği doğrulama kümesinde seçildiğinde görüntü düzeyi yanlış alarm oranı 0.242'den 0.021'e inmiş; sabit 0.5 eşiğinin, görüntü başına onlarca yamanın en yükseğini alan bir sınıflandırıcı için yanlış seçim olduğu görülmüştür. U-Net'in atlama bağlantıları 33 piksellik alanlarda etkili olmuş, DeepLabV3+ benzer başarıyı daha yavaş çıkarımla vermiş, YOLO11 ise referansın geometrisine en uygun formülasyon olarak kutu ölçütünde öne geçmiştir. Frekans kanalları eklemek RGB girdiyi geçememiştir. Ders kapsamındaki ön işleme adımlarının bu görevde girdiyi iyileştirmek yerine çoğunlukla adli izi zayıflattığı, test zamanı bozulma deneyleriyle sayısal olarak gösterilmiştir.

**Kıyas kümeleri için öneriler.** Sonuçlardan dört öneri çıkmaktadır: her eğitim sahtesiyle birlikte özgün kaynağın yayımlanması ve kutu ölçütünün yanında bir piksel-değişim ölçütünün raporlanması; üretim hattına ait işaretlerin yayımlanan görüntülerden temizlenmesi ve basit bir renk testiyle denetlenmesi; kutu düzeyi AP değerinin piksel ölçütlerinden ayrı, bir kutu dedektörü başvuru modeliyle birlikte verilmesi; sızıntısız test alt kümelerinin ve grup farkında güven aralıklarının raporlanması.

**Sınırlamalar.** Çalışmanın sınırları açık biçimde şunlardır.

1. Piksel-değişim referansı otomatik türetilmiştir. 100 çiftlik insan denetimi tek değerlendiricilidir ve değerlendirici çalışmanın yazarıdır; bağımsız ikinci bir değerlendirici ve değerlendiriciler arası uyum ölçümü yoktur. İnsan denetimi ile dil modeli destekli ilk denetim birbirleriyle zayıf uyumludur. İnsan denetimine göre referans değişimi kaçırmamakta, ancak alanın yeniden çizilen arka planını ve değişmeyen içeriğini de kapsayarak fazla işaretlemektedir.
2. TruFor'un ince ayarı yapılamamıştır; yayımlanan kodda eğitim betiği bulunmamaktadır. Bu nedenle veri tabanında eğitilen modeller yalnız sıfır-atış başvuru modelleriyle karşılaştırılmıştır. Mimari bağımsızlık DeepLabV3+ ve YOLO11 ile gösterilmiştir.
3. Faz 3'te eğitilen bütün modeller üç tohumludur (42–44); tek etkenli deneylerin 43 ve 44 tohumları, öngörüleri koşudan önce açık depoya gönderilen iki planla eklenmiştir. Tek tohumlu kalanlar dedektörün zararsız kontrolleri ve hız ölçümüdür. Bazı varyantlar tohuma duyarlıdır (20 epoklu glif modeli ± 0.082, birleşik kayıplı glif modeli ± 0.063, işaretçisiz piksel modeli ± 0.045); bu satırlar ortalama ± standart sapma ile okunmalıdır.
4. Ayrı tutulan üretici deneyi tek bir üreticinin 126 Ideogram görüntüsüne dayanır. İşaretçi ile üreticinin etkisi yalnız Gemini ile işaretçisiz eğitimle ayrıştırılmıştır (Bölüm 4.5): kutu modelindeki açığın yaklaşık altıda biri işaretçiye, geri kalanı üreticiye özgü izlere bağlanır. Bu ayrım başka üreticilerle sınanmamıştır.
5. Görüntü düzeyi tespit, test bölümünün özgün görüntüleri eşli olmadığından yalnız doğrulama çiftlerinde ölçülmüştür. Uygulamadaki görüntü düzeyi eşikler tek tohumla seçilmiştir. ResNet-18 eşiği Faz 4'te doğrulama kümesinde kalibre edilmiştir; ancak eşikler doygunluğa yakındır (0.94–0.995) ve tohuma duyarlıdır, test bölümünde eşi bulunan özgün görüntü sayısı yalnız 89'dur. Kalibre sürüm 11 Kasım 2026'da canlı Space'e dağıtılmıştır.
6. Karolu dedektörde kutu AP değeri tam görüntü dedektörünün altındadır; karolar arası kutu birleştirme uygulanmamıştır. Dedektörün hız üstünlüğü beklentisi doğrulanmamıştır.
7. v2 hem üreticiyi hem geri yapıştırma yöntemini değiştirdiğinden iki etki yalnız v1 kalıntı ayrımı üzerinden ayrıştırılabilmiştir. Veri tabanı tek bir belge ailesini (çoğunlukla fiş) ve iki dili ağırlıklı olarak kapsar.
8. SHAP sadakat ölçümleri sekiz girdide ve düşük bütçeyle yapılmıştır; nitel destek olarak değerlendirilmelidir.

**Dönem sonu planı.** Final aşamasında bu rapor bir konferans bildirisine dönüştürülecektir. İngilizce bildiri taslağı IEEE konferans biçiminde hazırdır (`paper/`, on sayfa) ve Faz 4 sonuçları dahil bu rapordaki bütün ana sonuçları içermektedir. Planlanan güçlendirmelerden ikisi tamamlanmıştır: bütün tek etkenli deneylere iki ek tohum (Bölüm 4.3–4.5) ve işaretçi ile üreticiyi ayırmak için yalnız Gemini üzerinde işaretçisiz eğitim (Bölüm 4.5). Gönderimden önce kalanlar şunlardır: insan denetimine ikinci bir bağımsız değerlendirici eklenmesi; karolu dedektörde karolar arası kutu birleştirme. Hedef, IEEE kapsamındaki bir sinyal ve görüntü işleme ya da adli bilişim konferansıdır; son başvuru tarihleri çağrı metinlerinden doğrulanacaktır.

**Sonuç.** Yapay zekâ ile iç-boyama yapılmış finansal belgelerde resmî kutu maskeleri, değişen pikselleri bulan lokalizatörleri değil, düzenlenen alanı bulan dedektörleri eğitmekte ve ödüllendirmektedir. Yalnız eğitim maskesini değiştirmek model sıralamasını tersine çevirir; etki iki bölütleme mimarisinde ve bir kutu dedektöründe yinelenir. Gemini sahtelerinin çoğunda kalan düzenleme işaretçisi, mevcut olduğunda kısayol olarak kullanılır; onsuz yeniden eğitim kutu puanının çoğunu korur ve işaretçisiz v2 kümesine çok daha iyi aktarılır. Ders açısından proje, klasik görüntü işlemeden derin bölütleme ve tespite uzanan bütün yöntemleri tek bir veri tabanında, kısıtlı veriden tam ölçeğe, ölçülebilir ve yeniden üretilebilir biçimde bir araya getirmiştir.

## 6. Kaynaklar

1. J. Wu, Y. Zhou, M. Xu, Z. Liang, S. Ren, J. Xue, M. Yang, S. Chen, J. Huan. AIForge-Doc: A Benchmark for Detecting AI-Forged Tampering in Financial and Form Documents. arXiv:2602.20569, 2026.
2. H. Mareen, D. Karageorgiou, G. Van Wallendael, P. Lambert, S. Papadopoulos. TGIF: Text-Guided Inpainting Forgery Dataset. IEEE International Workshop on Information Forensics and Security (WIFS), s. 1–6, 2024.
3. Z. Zhao, W. Xia, E. Wei, Y. Zhang, J. Mo, T. Zhang, Y. Dai, Z. Chen, Y. Tao, S. Ren. DOCFORGE-BENCH: A Comprehensive 0-shot Benchmark for Document Forgery Detection and Analysis. arXiv:2603.01433, 2026.
4. L. Maier-Hein, A. Reinke, P. Godau, M. D. Tizabi, F. Buettner ve ark. Metrics reloaded: recommendations for image analysis validation. Nature Methods, 21(2), 195–212, 2024.
5. F. Guillaro, D. Cozzolino, A. Sud, N. Dufour, L. Verdoliva. TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), s. 20606–20615, 2023.
6. C. Qu, C. Liu, Y. Liu, X. Chen, D. Peng, F. Guo, L. Jin. Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), s. 5937–5946, 2023.
7. J. Wu, Y. Zhou, D. T. Ng, X. Shen, K. Zewde, A. Raj, T. Duong, S. Ren. When the Forger Is the Judge: GPT-Image-2 Cannot Recognize Its Own Faked Documents. arXiv:2604.25213, 2026.
8. A. Raj, Y. Zhang, K. Zewde, T. Duong, J. Gan, X. Shen, Y. Zhou, H. Guo, S. Zhang, S. Ren. ChatGPT Images 2.5 on Forgery Tasks: Testing Advertised Improvements Against Known Answers. arXiv:2609.13617, 2026.
9. B. Helali. Beyond Zero-Shot on AIForge-Doc v2: A Trained Detector and a Process-Twin Control. Zenodo, yazılım ve sonuç arşivi, sürüm 1.0, doi:10.5281/zenodo.23917285, 2026 (hakemsiz).
10. leongyiquan. DocDet: Detecting AI-generated content in document images. GitHub deposu, https://github.com/leongyiquan/DocDet, 2026 (erişim: 9 Kasım 2026; hakemsiz).
11. Z. Chen, S. Chen, T. Yao, K. Sun, S. Ding, X. Lin, L. Cao, R. Ji. Enhancing Tampered Text Detection Through Frequency Feature Fusion and Decomposition. Computer Vision – ECCV 2024, LNCS 15091, s. 200–217, Springer, 2024.
12. K. Wong, J. Zhou, H. Wu, Y.-W. Si, J. Zhou. ADCD-Net: Robust Document Image Forgery Localization via Adaptive DCT Feature and Hierarchical Content Disentanglement. IEEE/CVF International Conference on Computer Vision (ICCV), s. 19280–19289, 2025.
13. M.-J. Kwon, S.-H. Nam, I.-J. Yu, H.-K. Lee, C. Kim. Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization. International Journal of Computer Vision, 130(8), 1875–1895, 2022.
14. C. Dong, X. Chen, R. Hu, J. Cao, X. Li. MVSS-Net: Multi-View Multi-Scale Supervised Networks for Image Manipulation Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3), 3539–3553, 2023.
15. X. Liu, Y. Liu, J. Chen, X. Liu. PSCC-Net: Progressive Spatio-Channel Correlation Network for Image Manipulation Detection and Localization. IEEE Transactions on Circuits and Systems for Video Technology, 32(11), 7505–7517, 2022.
16. X. Ma, B. Du, Z. Jiang, X. Du, A. Y. Al Hammadi, J. Zhou. IML-ViT: Benchmarking Image Manipulation Localization by Vision Transformer. arXiv:2307.14863, 2023.
17. X. Zhu, X. Ma, L. Su, Z. Jiang, B. Du, X. Wang, Z. Lei, W. Feng, C.-M. Pun, J.-Z. Zhou. Mesoscopic Insights: Orchestrating Multi-Scale & Hybrid Architecture for Image Manipulation Localization. AAAI Conference on Artificial Intelligence, 39(10), 11022–11030, 2025.
18. M.-J. Kwon, W. Lee, S.-H. Nam, M. Son, C. Kim. SAFIRE: Segment Any Forged Image Region. AAAI Conference on Artificial Intelligence, 39(4), 4437–4445, 2025.
19. H. Mareen, D. Karageorgiou, P. Giakoumoglou, P. Lambert, S. Papadopoulos, G. Van Wallendael. TGIF2: Extended Text-Guided Inpainting Forgery Dataset and Benchmark. Journal on Information Security, 2026(1), makale no. 13, 2026.
20. E. Nebioglu, E. Bilgiç, A. Popescu. AI-Generated Image Detectors Exploit Global Artifacts: Evidence from Inpainting Exchange. arXiv:2602.00192 (v2), 2026.
21. P. Giakoumoglou, S. Papadopoulos. DiffusionPrint: Learning Generative Fingerprints for Diffusion-Based Inpainting Localization. arXiv:2604.12443, 2026.
22. S. Pandolfini, L. Pellegrini, M. Ferrara, D. Maltoni. Detecting Localized Deepfakes: How Well Do Synthetic Image Detectors Handle Inpainting? arXiv:2512.16688, 2025.
23. L. Cai, H. Wang, J. Ji, Y. Zhoumen, S. Chen, T. Yao, X. Sun. Zooming In on Fakes: A Novel Dataset for Localized AI-Generated Image Detection with Forgery Amplification Approach. AAAI Conference on Artificial Intelligence, 40(4), 2534–2542, 2026.
24. C. Qu, Y. Zhong, J. Liu, X. Zhu, B. Yu, L. Jin. TextShield-R1: Reinforced Reasoning for Tampered Text Detection. AAAI Conference on Artificial Intelligence, 40(10), 8621–8629, 2026.
25. F. Zeng, C. Miao, J. Huang, Z. Tan, S. Gong, X. Yu, Y. Wang, W. Yao, J. T. Zhou, J. Li, Y. Yan. DocShield: Towards AI Document Safety via Evidence-Grounded Agentic Reasoning. arXiv:2604.02694, 2026.
26. F. Zeng, J. Li, C. Miao, Y. Li, Z. Cai, C. Kong, H. Yin, J. T. Zhou, R. He, A. Rocha, A. Kot. GenText-Forensics: Challenge on Explainable Forensics and Adversarial Generation for Text-Centric Images. ACM International Conference on Multimedia (MM), s. 17457–17463, 2026.
27. R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, F. A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11), 665–673, 2020.
28. K. Ronfleux-Corail, G. Bernard, M. Coustaty, N. Sidère. DocQT: Improving Document Forgery Localization Robustness via Diverse JPEG Quantization Tables. Document Analysis and Recognition – ICDAR 2026, LNCS, s. 109–126, Springer, 2026.
29. S. Smeu, D.-A. Boldisor, D. Oneata, E. Oneata. Circumventing shortcuts in audio-visual deepfake detection datasets with unsupervised learning. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), s. 18815–18825, 2025.
30. A. Yermakov, J. Cech, J. Matas, M. Fritz. Deepfake Detection that Generalizes Across Benchmarks. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), s. 773–783, 2026.
31. C. Liu, T. Zhu, W. Zhou, W. Zhao. Frequency Bias Matters: Diving Into Robust and Generalized Deep Image Forgery Detection. IEEE Transactions on Dependable and Secure Computing, 23(2), 3636–3651, 2026.
32. X. Shang, Y. Tang, J. Cui, A. Elhagry, S. K. Al Khatib, S. M. Bsharat, J. Liu, X. Zhao, J.-H. Xue, H. Li, S. Khan, Z. Shen. From Masks to Pixels and Meaning: A New Taxonomy, Benchmark, and Metrics for VLM Image Tampering. Computer Vision – ECCV 2026, LNCS, s. 581–598, Springer, 2026.
33. J. Dai, K. He, J. Sun. BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation. IEEE International Conference on Computer Vision (ICCV), s. 1635–1643, 2015.
34. Z. Tian, C. Shen, X. Wang, H. Chen. BoxInst: High-Performance Instance Segmentation with Box Annotations. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), s. 5439–5448, 2021.
35. R. C. Gonzalez, R. E. Woods. Digital Image Processing, 3. baskı. Pearson (dersin zorunlu kaynağı).
36. N. Ahmed, T. Natarajan, K. R. Rao. Discrete Cosine Transform. IEEE Transactions on Computers, C-23(1), 90–93, 1974.
37. T. Ojala, M. Pietikäinen, T. Mäenpää. Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(7), 971–987, 2002.
38. R. M. Haralick, K. Shanmugam, I. Dinstein. Textural Features for Image Classification. IEEE Transactions on Systems, Man, and Cybernetics, SMC-3(6), 610–621, 1973.
39. C. Cortes, V. Vapnik. Support-vector networks. Machine Learning, 20(3), 273–297, 1995.
40. L. Breiman. Random Forests. Machine Learning, 45(1), 5–32, 2001.
41. K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), s. 770–778, 2016.
42. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei. ImageNet: A large-scale hierarchical image database. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), s. 248–255, 2009.
43. O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, LNCS, s. 234–241, Springer, 2015.
44. F. Milletari, N. Navab, S.-A. Ahmadi. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. Fourth International Conference on 3D Vision (3DV), s. 565–571, 2016.
45. L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, H. Adam. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. Computer Vision – ECCV 2018, LNCS, s. 833–851, Springer, 2018.
46. J. Redmon, S. Divvala, R. Girshick, A. Farhadi. You Only Look Once: Unified, Real-Time Object Detection. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), s. 779–788, 2016.
47. G. Jocher, J. Qiu. Ultralytics YOLO11. Yazılım, sürüm 11.0.0, AGPL-3.0, https://github.com/ultralytics/ultralytics, 2024.
48. F. C. Akyon, S. O. Altinuc, A. Temizel. Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection. IEEE International Conference on Image Processing (ICIP), s. 966–970, 2022.
49. J. Sauvola, M. Pietikäinen. Adaptive document image binarization. Pattern Recognition, 33(2), 225–236, 2000.
50. I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. International Conference on Learning Representations (ICLR), 2019.
51. T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, C. L. Zitnick. Microsoft COCO: Common Objects in Context. Computer Vision – ECCV 2014, LNCS, s. 740–755, Springer, 2014.
52. B. Efron. Bootstrap Methods: Another Look at the Jackknife. The Annals of Statistics, 7(1), 1–26, 1979.
53. S. Holm. A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics, 6(2), 65–70, 1979.
54. M. G. Kendall. A New Measure of Rank Correlation. Biometrika, 30(1–2), 81–93, 1938.
55. K. He, G. Gkioxari, P. Dollár, R. Girshick. Mask R-CNN. IEEE International Conference on Computer Vision (ICCV), s. 2980–2988, 2017.
56. P. Zhou, X. Han, V. I. Morariu, L. S. Davis. Learning Rich Features for Image Manipulation Detection. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), s. 1053–1061, 2018.
57. G. Cheng, X. Yuan, X. Yao, K. Yan, Q. Zeng, X. Xie, J. Han. Towards Large-Scale Small Object Detection: Survey and Benchmarks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(11), 13467–13488, 2023.
58. S. Aoudi, S. A. P., N. Al Barghuthi, O. Alamir. Beyond Accuracy: Auditing Image Forgery Localization via Scene-Disjoint Evaluation and XAI Faithfulness. IEEE Access, 14, 53101–53115, 2026.
59. S. M. Lundberg, S.-I. Lee. A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems (NeurIPS) 30, s. 4765–4774, 2017.
60. W. Samek, A. Binder, G. Montavon, S. Lapuschkin, K.-R. Müller. Evaluating the Visualization of What a Deep Neural Network Has Learned. IEEE Transactions on Neural Networks and Learning Systems, 28(11), 2660–2673, 2017.

## Ek A. Sayı–kaynak izlenebilirlik tablosu

Table: Tablo 46. Rapordaki ana sayılar ve alındıkları dosyalar.

| Sayı | Değer | Kaynak dosya |
|---|---|---|
| Sahte görüntü sayısı ve bölme | 4.061; eğitim 3.249, test 812; eğitim 2.762, doğrulama 487; kısıtlı 400 ve 100 | `dataset_audit/audit_report.md` |
| Değiştirilen alan oranı (medyan) | %0.975 | `dataset_audit/audit_report.md` |
| Kutu piksellerinin değişen oranı (medyan) | %49.2 | `dataset_audit/audit_report.md` |
| Kutu dışı değişim içeren çift oranı | %33.5 | `dataset_audit/audit_report.md` |
| Kaynak belge sızıntısı | 72 görüntü (%8.9); sızıntısız 740 | `dataset_audit/audit_report.md` |
| Üretici × işaretçi Cramér V | 0.608 | `phase3/results/dataset_curation.md` |
| Kısıtlı veri: SVM ve RF AUROC | 0.646 ve 0.706 | `baselines/results/results_tables.md` |
| Kısıtlı veri: ResNet-18 AUROC (sonda, ince ayar, sıfırdan) | 0.726, 0.899, 0.679 | `baselines/results/results_tables.md` |
| Kısıtlı veri: U-Net F1 ve IoU | 0.707 ve 0.637 | `baselines/results/results_tables.md` |
| Kısıtlı veri: U-Net kesin AUPRC | 0.805 (RGB), 0.792 (RGB + frekans) | `phase3/results/phase1_exact_auprc.csv` |
| Resmî test, U-Net kutu | AUPRC 0.950 ± 0.011; F1 0.846 ± 0.015; IoU 0.789 ± 0.018 | `phase2/experiments/results/summary_test_agg.csv` |
| Sıfır-atış TruFor ve ADCD-Net F1 | 0.403 ve 0.182 | `phase2/literature_baselines/zero_shot_results.csv` |
| H1: aynı tahminler, kutu ve piksel-değişim AUPRC | 0.923 ve 0.374 | `phase2/experiments/README.md` |
| H1: F1 farkı ve Kendall katsayısı | 0.398 [0.377, 0.418]; −0.22 | `phase2/stats/hypothesis_decisions.md` |
| H2: dikdörtgensellik (kutu, piksel, glif) | 0.87, 0.32, 0.05 | `phase2/experiments/README.md` |
| 0.5 eşiğinde işaretlenen özgün oranı | %26 (kutu), %12 (piksel); üç tohum | `phase3/results/tables_p3.md` (T7) |
| Görüntü düzeyi AUROC (kutu, piksel, glif) | 0.892, 0.892, 0.799 | `phase2/stats/hypothesis_decisions.md` |
| H4: kutu dışı kırpıntı AUROC | 0.516 [0.513, 0.519] | `phase2/stats/hypothesis_decisions.md` |
| İşaretçi kalıntısı yaygınlığı | Çiftlerin %75.4'ü; Gemini %83.6, Ideogram %1.5 | `phase2/gt/vlm_verification_summary.json`, `reports/PHASE2_Metodoloji_ve_Sonuclar.md` |
| İşaretçi onarımı AP farkı (U-Net kutu) | −0.222 [−0.256, −0.190] | `phase3/results/tables_p3.md` (T3) |
| Özgün + işaretçi AP (U-Net kutu) | 0.905 | `phase3/results/tables_p3.md` (T3) |
| İşaretçisiz eğitimde onarım farkı (üç tohum) | +0.017 [+0.001, +0.033] | `phase3/results/tables_p3.md` (T3) |
| İşaretçisiz kutu U-Net, doğrulama ve test AUPRC (üç tohum) | 0.912 ± 0.006 ve 0.940 ± 0.003 | `phase3/results/tables_p3.md` (T1, T2) |
| Birleşik kayıplı piksel U-Net, doğrulama AUPRC (kutu ve piksel-değişim; üç tohum) | 0.668 ± 0.019 ve 0.633 ± 0.013 | `phase3/results/tables_p3.md` (T1) |
| Ek tohum öngörüleri (P1–P6) | Altısı da tuttu | `reports/inceleme/deney_plani_ek_tohum_2026-11-10.md`, `phase3/results/extra_seed_stats.md` |
| Kalan tohumlar ve yeni deney öngörüleri (K1–K21) | Yirmi biri de yanlışlama ölçütünü geçti; K2 öngörülen aralığın az altında | `reports/inceleme/deney_plani_kalan_tohumlar_2026-11-11.md` (§8, §9) |
| Test AP, işaretçili ve işaretçisiz (U-Net kutu) | 0.972 ve 0.690 | `phase3/results/tables_p3.md` (T3b) |
| DeepLabV3+ kutu modeli, kutu ve piksel-değişim AUPRC | 0.939 ve 0.382 | `phase3/results/tables_p3.md` (T1) |
| YOLO11s test AUPRC, F1, IoU | 0.969 ± 0.006; 0.946; 0.918 | `phase3/results/tables_p3.md` (T2) |
| YOLO11s doğrulama piksel-değişim AUPRC | 0.368 ± 0.015 | `phase3/results/tables_p3.md` (T1) |
| U-Net AP50 tohum ayrıntısı | 0.873 (42 ve 43), 0.391 (44) | `paper/sec/results.tex` |
| Çıkarım süreleri | U-Net 81 ms; DeepLabV3+ 122 ms; YOLO11s 98 ms; tam görüntü 303 ms | `phase3/results/speed_bench.csv` |
| Ayrı tutulan üretici (yalnız Gemini, Ideogram AP; üç tohum) | U-Net 0.412; işaretçisiz eğitimle 0.463; YOLO11s 0.725 | `phase3/results/tables_p3.md` (T4) |
| Yalnız Gemini, işaretçisiz − standart (eşli görüntü başı AP, kutu modeli) | Ideogram +0.051 [+0.004, +0.096]; Gemini −0.035 [−0.047, −0.023] | `reports/inceleme/deney_plani_kalan_tohumlar_2026-11-11.md` (§9) |
| Gri düzey etkisi | AUPRC değerinde −%47 ile −%67 | `phase3/results/tables_p3.md` (T6), `reports/PHASE3_Sonuclar_ve_Kapanis.md` |
| v2 harici test (kutu AUPRC) | U-Net 0.520 ± 0.018; YOLO11s 0.887 ± 0.006; işaretçisiz U-Net 0.711 ± 0.007 | `phase3_v2/results/v2_tables.md` |
| v2, v1 kalıntısız tariflerde AP değişimi (işaretçisiz kutu U-Net; üç tohum) | −0.008 [−0.069, +0.053] | `phase3_v2/results/T_v2_marker_split_agg.csv` |
| Referans denetimleri | İlk (dil modeli): 96 yanıt, 69 uygun, 26 kısmi, 1 hatalı; ikinci (insan, yazar): 36 uygun, 64 kısmi; κ 0.07 | `phase2/gt/vlm_verification_summary.json`, `phase2/gt/human_verification_summary.md`, `phase2/gt/partial_note_check.json` |
| SHAP işaretçi yoğunlaşması | 4.3–15.1 kat | `docs/xai_test_report.md` |
| Tam görüntü SHAP toplamsallık hatası | En çok 1.1e-04 | `reports/xai_fullimage/RAPOR_OLCUM.md` |
| Katman filtresi testleri | 91 kontrol, 0 hata; 50 model | `tests/run_all_xai_filters.log`, `reports/xai_filters/test_results.json` |
| Uygulama testleri | 289 kontrol, 0 hata; 36 model | `tests/run_all_app.log` |
| Canlı Space toplu testi (U-Net kutu) | F1 0.925; AP 0.964; yeniden üretim farkı 0.000 | `reports/canli_test_v4deploy_log.txt` |
| Canlı Space, ResNet-18 tam veri (kalibre eşik; 24 çift): doğru tespit, yanlış alarm, AUROC | 0.917; 0.000; 0.998 | `reports/canli_test_v4deploy_faz4_log.txt` |
| Dedektörün zararsız kontrolleri (görüntü AUROC, en yüksek kutu güveni; tohum 42) | Özgün 0.995; geri yapıştırma 0.992; yeniden örnekleme 0.994 | `phase3/results/T7b_yolo_controls.csv` |
| v2 işaretçi kalıntısı (v1 ile aynı eşli kural) | Kutu dışında farklı piksel: 3.066 çiftin 0'ında; kural 1 çiftte, kutu içinde | `phase3_v2/results/v2_marker_paired_check.json` |
| ResNet-18 tam veri: doğrulama yama AUROC | 0.983 ± 0.003 (kısıtlı veri modeli 0.926) | `phase4_resnet/results/patch_level_val.csv` |
| ResNet-18 tam veri, kalibre kural: test AUROC, doğru tespit, yanlış alarm | 0.978 ± 0.003; 0.901 ± 0.014; 0.022 ± 0.011 | `phase4_resnet/results/tables_p4.md` |
| ResNet-18 işaretçi sondası: onarılmış sahtede doğru tespit (tam veri, işaretçisiz eğitim) | 0.471 ve 0.899 | `phase4_resnet/results/marker_probes.csv` |
| Yazılım sürümleri | torch 2.11.0+cu128; segmentation-models-pytorch 0.6.0; ultralytics 8.4.208; Gradio 6.31.0 | `requirements.txt`, `kod/hf_space/space_src/README.md` |

Kaynak: tabloda her satır için ayrı ayrı verilmiştir.
