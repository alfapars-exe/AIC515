# Proje Konumlandırma Notu: Amaç, Literatüre Katkı, Yenilik, Ar-Ge Niteliği, Ön İşleme ve Yöntem Karşılaştırması

**Ders:** AIC515 Görüntü İşleme (Prof. Dr. Onur Osman, Topkapı Üniversitesi) · **Veri seti:** AIForge-Doc v1 · **Tarih:** 27 Ekim 2026
**Dayandığı dosyalar:** `reports/Bildiri_Tam_Metin.md` (İngilizce tam metin), `reports/PHASE2_Metodoloji_ve_Sonuclar.md`, `reports/Vize_Raporu_Taslak.md`, `literature/literature_matrix.csv`, `phase2/experiments/common/p2data.py`, `dataset_audit/audit_stats.json`. Tüm sayılar bu dosyalardan aktarılmıştır; yeni deney yapılmamıştır.

---

## 0. Kısa yanıtlar

| Soru | Kısa yanıt | Bölüm |
|---|---|---|
| Amaç segmentasyon mu, sınıflandırma mı? | **Birincil görev piksel düzeyinde ikili segmentasyondur** (tampering localization: hangi pikseller değiştirildi?). Görüntü düzeyi sınıflandırma (belge sahte mi?) ayrı bir model eğitilmeden segmentasyon haritasından türetilir. Pilot aşamada ayrıca patch sınıflandırması yapılmıştır. | §1 |
| Literatüre ne ekliyor, bir boşluğu dolduruyor mu? | Evet, ancak sınırları belli bir boşluğu: AIForge-Doc üzerinde *kutu maskesinin modele ne öğrettiği* ve *raporlanan skorun neyi ölçtüğü* kontrollü deneyle ilk kez sınanıyor; üretim hattındaki işaretçi kalıntısı ölçülüyor ve modellerin onu kullandığı gösteriliyor. Eşzamanlı bir çalışma (v3) benzer türde bir kısayolu başka bir turda bildirdi. | §2 |
| Yenilikçi yönü nedir? | Yeni bir mimari değil; **metodolojik ve ampirik yenilik**: aynı model–veri–bütçe ile yalnız eğitim maskesinin değiştirildiği çift-GT deneyi, eşleşmiş görüntülerden türetilmiş piksel/glif GT'leri, işaretçi aktarım/onarım sondası, kısayol denetim protokolü. | §3 |
| Ar-Ge niteliği var mı? | Evet. OECD Frascati (2015) beş ölçütünü karşılıyor; türü **uygulamalı araştırma**. Sonuç önceden bilinmiyordu: iki hipotez yanlışlandı, bir mekanizma yanlış çıktı. | §4 |
| Yöntemsel değişiklik var mı, karşılaştırıldı mı? | 12 yöntemsel değişiklik; 10'u için doğrudan karşılaştırma var. Mimari, kayıp fonksiyonu, epoch sayısı ve veri artırımı ablate edilmedi. | §5 |
| Ön işleme var mı? | Var, ama bilinçli olarak **minimum ve izi koruyan** türden: kayıpsız okuma, doğal çözünürlükte kırpma, ImageNet normalizasyonu, hafif fotometrik jitter, kayan pencere. Görüntü iyileştirme (gürültü giderme, kontrast germe, yeniden boyutlandırma) yapılmadı. | §6.3–6.4 |
| Görüntülere bakarak ön işleme yapmak anlamlı mı, gerekli mi? | *İyileştirme* amaçlı ön işleme gerekli değildir ve bu veri setinde zararlıdır; kendi bozulma deneylerimiz bunu gösteriyor. *Veri temizliği* amaçlı tek bir ön işleme ise **gereklidir**: yeşil işaretçi kalıntısının nötralize edilmesi. Görüntüleri incelemek artefakt keşfi için değerlidir (işaretçi böyle bulundu), fakat her adım algoritmik tanımlanmalı ve tüm görüntülere, sahte ve authentic için simetrik uygulanmalıdır. | §6.5 |
| Hangi modeller ve yöntemler kullanıldı, birlikte kullanım var mı? | SVM, Random Forest, ResNet-18 (3 eğitim rejimi), U-Net/ResNet-18 (3 GT × 3 seed + Gemini-only), TruFor ve ADCD-Net (zero-shot), kural tabanlı sezgisel, morfolojik son işleme, VLM (yalnız denetim aracı). Hibrit kullanımlar: sınıflandırma ağının segmentasyon kodlayıcısı olarak aktarılması, klasik filtre kanallarıyla erken füzyon, segmentasyondan sınıflandırma türetme, BCE+Dice kayıp füzyonu. Ensemble yok. | §7 |
| Kullanılan ve kullanılmayan yöntemler karşılaştırılabilir mi? | Evet; tablo §8'de. Kullanılmayanların bu veri setindeki davranışına dair satırlar **hipotezdir**, ölçülmemiştir. | §8 |

---

## 1. Projenin amacı: segmentasyon mu, sınıflandırma mı? Görüntüden nasıl yapılıyor?

### 1.1 İki düzeyli problem
Bir belge sahteciliği sistemi iki soruya yanıt verebilir:

1. **Görüntü düzeyi sınıflandırma:** "Bu belge görüntüsü değiştirilmiş mi?" Çıktı tek bir skordur, $s(I)\in[0,1]$.
2. **Piksel düzeyi segmentasyon (tampering localization):** "Belgenin hangi pikselleri değiştirilmiş?" Çıktı, görüntüyle aynı boyutta bir olasılık haritasıdır, $P(x,y)\in[0,1]$; eşiklendiğinde ikili maske $\hat M(x,y)\in\{0,1\}$ elde edilir.

Bu projenin **ana görevi ikincisidir**. Bunun üç gerekçesi vardır:

- **Adli kullanım bunu gerektirir.** Bir fişin sahte olduğunu söylemek yetmez; hangi alanın (ör. toplam tutar) değiştirildiğinin gösterilmesi gerekir.
- **Veri seti bunu destekler.** Her sahte görüntü için bir maske ve eğitim tarafında eşleşmiş authentic kaynak vardır. Kaynak görüntü, piksel düzeyinde ikinci bir GT türetmemizi mümkün kıldı.
- **Resmi test setinde görüntü düzeyi sınıflandırma temiz değerlendirilemez.** Denetimde, test klasöründeki authentic görüntülerin 812 test sahtesinin yalnızca 86'sının kaynağı olduğu, bir kısmının da eğitim authentic'leriyle bayt düzeyinde aynı olduğu görüldü. Bu nedenle negatif sınıf olarak kullanılamaz.

Sınıflandırma iki biçimde yine de yapıldı. **Pilot (vize) aşamasında** 256 px patch'ler üzerinde "sahte alan içeren / temiz" sınıflandırması klasik öznitelikler + SVM/RF ve ResNet-18 ile yapıldı. Amaç dersin sınıflandırma yöntemlerini sınamak ve "önceden eğitilmiş model + yeni katman mı, ince ayar mı, sıfırdan mı?" sorusunu yanıtlamaktı. **Ana çalışmada** görüntü düzeyi skor, segmentasyon haritasından türetildi (§1.3).

### 1.2 Görüntüden maskeye: işlem hattı
```
girdi: I  (RGB PNG, doğal çözünürlük; genişlik 150–6016 px)
1  x <- (I/255 - μ_ImageNet) / σ_ImageNet                      # normalizasyon
2  if max(H,W) > 4096: I <- downscale(I, 4096)                  # yalnız çok büyük görüntüler
3  for each 512×512 pencere w (adım 448, 64 px örtüşme):
4      z_w <- UNet_ResNet18(x[w])                               # piksel başına logit
5      P[w] += sigmoid(z_w);  C[w] += 1
6  P <- P / C                                                   # örtüşen bölgelerde ortalama
7  if küçültüldüyse: P <- upsample_bilinear(P, H, W)
8  M_hat <- 1[P >= τ*]                                          # τ*: doğrulama setinde F1'i en büyükleyen eşik
9  s(I) <- mean(P'nin en yüksek %1'lik pikselleri)              # görüntü düzeyi skor
çıktı: P (olasılık haritası), M_hat (maske), s(I) (skor)
```

### 1.3 Matematiksel altyapı (özet)
- **Model:** U-Net kodlayıcısı ResNet-18'dir. Residual blok $y=\mathcal F(x,\{W_i\})+x$ biçimindedir. Kod çözücü, atlama bağlantılarıyla aynı çözünürlükteki kodlayıcı özelliklerini birleştirir: $d_l=\mathrm{Up}(d_{l+1})\oplus e_l$. Çıkış $\hat p=\sigma(z)$'dir.
- **Kayıp:** $\mathcal L=\mathcal L_{BCE}+\mathcal L_{Dice}$, burada $\mathcal L_{Dice}=1-\frac{2\sum_i\hat p_iy_i+\epsilon}{\sum_i\hat p_i+\sum_iy_i+\epsilon}$. Kurcalanmış pikseller görüntünün medyan %0.98'i olduğundan Dice terimi baskın negatif sınıfa duyarsızdır.
- **Görüntü skoru:** $s(I)=\frac{1}{|T|}\sum_{(x,y)\in T}P(x,y)$; $T$, $P$'nin en yüksek %1'lik piksellerinin kümesidir. Görüntü düzeyi AUROC bu skorla hesaplanır (doğrulamada bbox ve piksel modeli için 0.892).
- **"Doğru piksel" ne demek? Üç GT tanımı.** Fark haritası $D(x,y)=\max_c|F_c(x,y)-A_c(x,y)|$ ($F$ sahte, $A$ authentic) üzerinden:
  - **bbox (resmi):** düzenlenen alanın sınırlayıcı kutusu (tüm maskeler tek dikdörtgen).
  - **pixel (piksel-değişim):** $\{D>8\}$ → 3×3 açma → 8 px'ten küçük bileşenlerin atılması → 3×3 kapama.
  - **glyph:** (Sauvola metin pikselleri, pencere 25, $k=0.2$) ∩ dilate($\{D>8\}$, 7×7); kutu başına vuruş polaritesi düzeltmesiyle (`glyph_v2`).

  Model aynı kalsa bile "neyi lokalize ettiği" bu tanıma bağlıdır. Projenin merkezi sorusu budur.

---

## 2. Literatüre katkı ve araştırma boşluğu

### 2.1 Bu veri setiyle yapılmış çalışmalar (27 Ekim 2026 itibarıyla)

| Çalışma | Ne yaptı | Yöntem | Raporlanan sonuç | Açık bıraktığı nokta |
|---|---|---|---|---|
| Wu ve ark. 2026, AIForge-Doc v1 (arXiv:2602.20569) | Veri setini tanıttı (4,061 sahte; Gemini 2.5 Flash Image, Ideogram v2 Edit) | Zero-shot TruFor, DocTamper/DTD, GPT-4o yargıç | TruFor AUC 0.751; DocTamper AUC 0.563, IoU 0.020; GPT-4o AUC 0.509 | Eğitim yok. Kutu maskesi "pixel-precise" olarak sunuluyor. Üretim hattında çizilen yeşil işaretçinin çıktıda kalıp kalmadığı denetlenmemiş. |
| Wu ve ark. 2026, v2 (arXiv:2604.25213) | GPT-Image-2 ile 3,066 eşleşmiş sahte | Zero-shot TruFor, DocTamper; insan 2AFC; öz-yargıç | İnsan 0.501; TruFor 0.599; DocTamper 0.585 | Zero-shot. 3 px yeşil kontur işaretçisi tasarım olarak açıklanıyor; çıktıdaki kalıntı ölçülmüyor. |
| Raj ve ark. 2026, v3 (arXiv:2609.13617) — *eşzamanlı, literatür kesimimizden sonra* | ChatGPT Images 2.5 değerlendirmesi; v3'te işaretçi yerine alfa maske, kırpıntının tamamı geri yapıştırılıyor | Zero-shot TruFor, DocTamper, Community Forensics | TruFor haritaları yapıştırılan kırpıntı dikdörtgenine karşı da alan kutusu kadar iyi skorluyor (piksel AUC 0.921 vs 0.918); yazarlar v3'te bir splice localizer'ın düzenlemeyi lokalize etmeden yüksek skor alabileceğini belirtiyor | v1, eğitilmiş modeller, kutu-piksel GT farkı ve işaretçi kalıntısı incelenmiyor. Bulgu bizimkiyle **aynı türde** (hat kaynaklı kısayol), ancak farklı tur ve mekanizma. |
| Ren ve ark. 2026 (arXiv:2609.23953) | Ajan tabanlı PDF sahteciliği | — | — | AIForge-Doc'a yalnızca atıf yapıyor; bu veriyle deney yok. |
| GitHub: Joya15/Multi-Generator-Deepfake-Document-Detection-and-Tamper-Localization (**hakemsiz, gri literatür**) | v1 ve v2 üzerinde SegFormer-Lite eğitimi | Çapraz-veri | v2→v1 piksel F1 0.044; v1→v2 piksel F1 0.0001 | Doğrulanmadı; kontrollü GT karşılaştırması ve artefakt denetimi yok. |

**Düzeltme notu.** Vize taslağında yer alan "AIForge-Doc üzerinde denetimli eğitim yapan ilk çalışmalardan biri" ifadesi artık savunulamaz, çünkü hakemsiz bir depo v1 üzerinde eğitim yapmıştır. Savunulabilir iddia daha dardır: *v1 üzerinde eğitim maskesinin geometrisini kontrollü biçimde değiştiren ve işaretçi kalıntısını ölçen bir çalışmaya rastlanmamıştır* (38 çalışmalık matris + 27 Ekim 2026 web/arXiv güncellemesi). Bu taramanın da sınırlı olduğu not edilmelidir.

### 2.2 Araştırma boşluğu: literatürde ne var, ne eksik, bu çalışma ne yapıyor?

| # | Literatürde var olan | Eksik olan (boşluk) | Bu çalışmanın katkısı | Kanıt |
|---|---|---|---|---|
| B1 | AIForge-Doc'ta zero-shot dedektör sonuçları | Denetimli eğitim ve kontrollü karşılaştırma | 3 GT × 3 seed U-Net + Gemini-only koşular, resmi test değerlendirmesi | Test (bbox GT) AUPRC 0.950, F1@τ* 0.846 |
| B2 | "Pixel-precise" kutu maskeleri | Kutu maskesinin modele ne öğrettiği ve skorun neyi ölçtüğü | Eşleşmiş görüntülerden piksel/glif GT türetimi ve çift-GT değerlendirme (H1) | Kutu pikselinin medyan %49'u değişmiş. bbox modeli bbox GT'de AUPRC 0.923, piksel GT'de 0.374, glif GT'de 0.257; Kendall τ −0.22 |
| B3 | Genel kısayol öğrenme kuramı (Geirhos ve ark., 2020); v3'te dikiş kısayolu (eşzamanlı) | v1 üretim hattı artefaktlarının *eğitilmiş* modellerce kullanılıp kullanılmadığı | Benign kontroller, sınır/plasebo/iç bulanıklık, rectangle-fit, temiz-crop global iz, **işaretçi aktarım/onarım** | Authentic + işaretçi → AP 0.906; işaretçi onarımı → ΔAP −0.225 |
| B4 | F1@0.5 ve tek koşu nokta tahminleri | Eşik kalibrasyonu ve belirsizlik | Eşikten bağımsız AUPRC/AP, doğrulamada kalibre τ*, grup bootstrap (B=1000), Holm düzeltmesi (463 kontrast) | bbox modeli 0.5 eşiğinde authentic görüntülerin %44'ünü işaretliyor (piksel modeli %10); görüntü AUROC ikisinde de 0.892 |
| B5 | Resmi train/test ayrımı | Kaynak belge sızıntısı | pHash gruplarıyla grup-farkında doğrulama ayrımı; sızıntı ölçümü | Test sahtelerinin %8.9'u eğitimle kaynak paylaşıyor; temiz 740 görüntüde skorlar aynı |

**Boşluk dolduruluyor mu?** B1–B5 v1 için doldurulmaktadır. İki sınır vardır: (i) eşzamanlı v3 çalışması B3'ün genel fikrini (hat kaynaklı lokalizasyon kısayolu) başka bir tur ve zero-shot model için ortaya koymuştur; (ii) işaretçisiz yeniden eğitim yapılmadığından "işaretçi olmadan model gerçekte ne öğrenebilir?" sorusu açıktır (RQ-A).

---

## 3. Yenilikçi yönler (ve her birinin sınırı)

| # | Yenilik | Neden yeni | Kanıt | Sınır |
|---|---|---|---|---|
| Y1 | **Çift-GT kontrollü deney**: aynı mimari, veri, artırım ve bütçe; değişen tek şey eğitim maskesi | Belge sahteciliği literatüründe GT geometrisi sabit kabul edilir; etkisi izole edilmemiştir | Her GT kendi modelini birinci sıraya koyuyor; sıralama GT'ye göre tersine dönüyor | Tek mimari (ResNet-18 U-Net), 10 epoch |
| Y2 | **Eşleşmiş authentic'ten türetilmiş piksel-değişim ve glif GT'leri** (3,336 çift, 7 varyant, ε-duyarlılık) | Veri setinin authentic kaynakları ilk kez GT üretimi için kullanılıyor | VLM denetimi: pixel_8_morph 96 panelin 69'unda uygun, 26'sında kısmen uygun, 1'inde yanlış | İnsan doğrulaması yapılmadı; VLM ≠ insan |
| Y3 | **Yeşil işaretçi kalıntısının ölçülmesi ve nedensel sonda** | İşaretçinin *çizildiği* v1 makalesinde (§3.3) yazılıdır; yeni olan, yayımlanan sahtelerin %75.4'ünde kalıntının bulunduğu ve tüm model ailelerinin onu kullandığıdır | Authentic + işaretçi: bbox modeli AP 0.906, kutu kapsama 0.80. Onarım: AP 0.988 → 0.763. Öğrenmesiz kural F1 0.054 | Keşifsel, ön-kayıtlı değil; doğrulama setinde |
| Y4 | **Kısayol denetim protokolü** (plasebo/iç/sınır ablasyonu, rectangle-fit, benign paste-back, temiz-crop global iz, ipucu aktarımı) | Kutu-etiketli adli veri setleri için yeniden kullanılabilir bir kontrol listesi | H2 kısmen doğrulandı, H4 yanlışlandı; ipucu aktarımı sınır bulanıklığının kaçırdığı ipucunu yakaladı | Diğer veri setlerinde sınanmadı |
| Y5 | **Ön-kayıt ve negatif sonuçların raporlanması** | Alanda olumsuz sonuçlar nadiren yayımlanır | H3 (transfer/dayanıklılık) ve H4 (global iz) yanlışlandı; H2'nin ön-kayıtlı mekanizması yanlış çıktı | — |

**Yeniliğin türü:** artımsal (incremental) metodolojik yenilik ve yeni ampirik bulgu. Yeni bir ağ mimarisi önerilmemektedir. Makalede bu açıkça yazılmalı; "ilk", "yeni yöntem" gibi ifadeler kullanılmamalıdır.

---

## 4. Ar-Ge niteliği

OECD *Frascati Manual 2015* (doi:10.1787/9789264239012-en), bir faaliyetin Ar-Ge sayılması için beş ölçüt tanımlar. Projenin her ölçütle eşleşmesi:

| Frascati ölçütü | Projede karşılığı |
|---|---|
| **Yenilik (novel)** — yeni bilgi üretmek | Kutu GT'nin ölçülen başarıyı nasıl belirlediği; işaretçi kalıntısının yaygınlığı ve modellerce kullanılması; piksel GT'nin dayanıklılık kazandırmadığı. Bunlar önceden bilinmiyordu. |
| **Yaratıcılık (creative)** — özgün kavram ve yöntem | Eşleşmiş görüntülerden GT türetimi; ipucu aktarım/onarım testi (şüpheli ipucunu temiz görüntüye taşıyıp modeli izleme). |
| **Belirsizlik (uncertain)** — sonucun önceden kestirilememesi | Belirsizlik gerçekleşti: H3 ve H4 yanlışlandı; H2'nin öngörülen mekanizması (yeniden örnekleme dikişi) yanlış çıktı, gerçek ipucu bir renk çizgisiydi. |
| **Sistematiklik (systematic)** — planlı, bütçeli, kayıtlı | İki fazlı ön-kayıtlı plan; sabit bütçe (10 epoch, 12 koşu, 124 dk GPU); grup bootstrap + Holm; tüm kararlar `phase2/stats/hypothesis_decisions.md`'de. |
| **Aktarılabilirlik/tekrarlanabilirlik** | Kod, ağırlıklar, GT maskeleri, `run_everything.ps1` (tek 16 GB GPU'da yaklaşık 6 saat), `requirements.txt`, teslim paketi. |

**Ar-Ge türü:** *Uygulamalı araştırma* (applied research): belirli bir pratik amaca, yani güvenilir belge sahteciliği lokalizasyonu ve değerlendirmesine yönelik yeni bilgi üretimi. Ürün ya da süreç geliştirme (deneysel geliştirme) değildir.

**Ar-Ge olmayan kısım:** Hazır bir U-Net'i veri setinde eğitip skor raporlamak tek başına rutin mühendisliktir. Ar-Ge niteliğini taşıyan, *skorun neyi ölçtüğüne dair belirsizliğin* sistematik deneylerle giderilmesidir. Bu ayrım, TÜBİTAK tipi bir öneride "teknolojik/bilimsel belirsizlik" başlığına şöyle yazılabilir: *Kutu etiketli, üretim hattı artefaktı içeren bir veriyle eğitilen lokalizasyon modelinin gerçek düzenleme kanıtını mı, yoksa etiket geometrisini ve hat artefaktını mı öğrendiği bilinmemektedir.*

---

## 5. Yöntemsel değişiklikler ve karşılaştırmaları

| # | Değişiklik | Varsayılan / literatürdeki uygulama | Karşılaştırma yapıldı mı? | Sonuç |
|---|---|---|---|---|
| D1 | Eğitim GT'si: bbox → piksel-değişim → glif | v1: alan kutusu | **Evet**, 3×3 matris (eğitim GT × değerlendirme GT) | Sıralama GT'ye göre tersine dönüyor (τ −0.22) |
| D2 | Metrik: F1@0.5 yerine AUPRC/AP + kalibre F1@τ* | AUC, IoU, F1@0.5 | **Evet** | 0.5 eşiğinde bbox modeli authentic'lerin %44'ünü işaretliyor, piksel modeli %10 |
| D3 | Grup-farkında doğrulama ayrımı ve sızıntı ölçümü | Resmi split | **Evet**, sızıntılı 812 / temiz 740 | Fark yok (AUPRC 0.950) |
| D4 | 3 seed + grup bootstrap + Holm | Tek koşu | Çerçeve değişikliği (karşılaştırma değil) | 463 kontrast raporlandı |
| D5 | Ön eğitim rejimi | — | **Evet** (pilot): linear probe / ince ayar / sıfırdan | AUROC 0.726 / 0.899 / 0.679 |
| D6 | Klasik öznitelik + ML ile derin öğrenme | — | **Evet** (pilot) | SVM 0.646, RF 0.706, ResNet ince ayar 0.899 |
| D7 | Girdi: RGB ile RGB + FFT yüksek-geçiren + Laplacian | DTD/FFDN/ADCD-Net: DCT akışı | **Evet** (pilot) | F1 0.707 / 0.674; AUPRC 0.947 / 0.939 (frekans kanalı kazanç sağlamadı) |
| D8 | Doğal çözünürlükte kırpma + kayan pencere | Sabit boyuta küçültme (ör. 512 veya 1024 px) | **Dolaylı**: test zamanı 0.75× yeniden ölçekleme | bbox ΔAP −0.063, piksel −0.071, glif −0.105 (kendi GT'lerinde) |
| D9 | Benign kontroller (paste-back, tam görüntü yeniden örnekleme, PNG yeniden kodlama) | Yok | **Evet** | Benign paste-back yanlış pozitif artışı +0.50‰; dikişin katkısı küçük |
| D10 | Kısayol sondaları (sınır/plasebo/iç bulanıklık, rectangle-fit, işaretçi, H4) | Yok | **Evet** | §3, Y3–Y4 |
| D11 | Görülmemiş üretici testi (yalnız Gemini ile eğitim) | — | **Evet**, tek seed | bbox açığı 0.513, piksel açığı 0.303; ortak piksel GT'de fark +0.025 [−0.039, 0.088] |
| D12 | GT doğrulaması: insan yerine VLM destekli denetim | İnsan doğrulaması | **Kısmen**: VLM ile nesnel işaretçi dedektörü karşılaştırıldı | Yeşil çizgi: 37 doğru tespit, 16 kaçırma, 3 yanlış alarm |

**Karşılaştırması yapılmayanlar:** mimari (yalnız U-Net/ResNet-18), kayıp fonksiyonu (yalnız BCE+Dice), epoch sayısı (10, önceden sabitlendi), kırpma boyutu ve örnekleme oranı (512 px, %70/%30), veri artırımı. Bunlar mutlak skorları değiştirebilir; çalışmanın sorusu GT etkisi olduğu için sabit tutuldular.

---

## 6. Görüntü ön işleme (preprocessing)

### 6.1 Görüntülere bakınca ne görüyoruz?

![Şekil 1. Kaynak ve üretici başına örnekler (kırmızı: resmi kutu maskesi, mavi: metadata bbox).](dataset_audit/fig_examples_overlay.png)

![Şekil 2. Bir doğrulama paneli: A sahte, B authentic, C |A−B|×4, D resmi kutu, E piksel-değişim maskesi, F glif maskesi.](phase2/gt/vlm_verification/000000034.png)

Örneklerden ve denetim istatistiklerinden çıkan gözlemler:

- **Üç farklı edinim türü bir arada.** CORD ve WildReceipt fotoğraflanmış fişlerdir: perspektif, kıvrım, düzensiz aydınlatma; el, ahşap ya da kumaş arka plan. SROIE yüksek kontrastlı taramalardır. XFUND render edilmiş/taranmış formlardır (ör. 2480×3508).
- **Ölçek çok değişken.** Genişlik 150–6016 px (medyan 750), yükseklik 168–7016 px (medyan 1296), 0.03–34.8 MP. Kurcalanmış alan görüntünün medyan %0.98'i (p05 %0.04), medyan 5,820 pikseldir. XFUND örneklerinde alan görüntünün %0.02–0.06'sı kadardır.
- **Format:** Görüntü ve maske dosyalarının tamamı (12,183 dosya) PNG'dir. Sahte görüntülerde JPEG sıkıştırma ızgarası beklenmez.
- **Kaynakta var olan bulanıklıklar:** CORD örneklerinde, görünüşe göre kişisel bilgiyi gizlemek için uygulanmış bulanık bölgeler vardır (Şekil 1, sol üst). Bunlar sahtecilik değildir, ama yerel yeniden örneklemeye benzer bir iz taşırlar.
- **Hat artefaktları:** kutunun iç kenarında 1–3 px doygun yeşil çizgi (sahtelerin %75.4'ü); sahtelerin %33.5'inde kutu dışında da piksel değişimi; %48.6'sında görüntü geneline yayılmış ±1–3 gri seviyelik iz (ε ≥ 4'te kaybolur).

### 6.2 Adli bilişimde ön işleme neden farklıdır?
Klasik görüntü işlemede ön işleme, sonraki tanıma adımı için görüntü kalitesini artırır: gürültü giderme, kontrast iyileştirme, geometrik düzeltme. Adli bilişimde aranan sinyal ise tam da bu düşük seviyeli istatistiklerdir: gürültü artığı, yeniden örnekleme izi, sıkıştırma izi, glif kenarlarındaki anti-aliasing. İyileştirme amaçlı her işlem bu sinyali değiştirir. Sahte ve authentic görüntülere farklı uygulanırsa da yeni bir kısayol yaratır. Bu nedenle ilke şudur: **ön işleme izi korumalı, simetrik olmalı ve her adımın gerekçesi yazılmalıdır.**

### 6.3 Projede yapılan ön işlemler

| Adım | Nerede | Gerekçe |
|---|---|---|
| Kayıpsız PNG okuma, BGR→RGB | Tüm modeller | Bilgi kaybı yok |
| **Yeniden boyutlandırma yok**; 512×512 doğal çözünürlük kırpıntıları; küçük görüntülerde sıfır dolgu | U-Net eğitimi/doğrulaması | Küçük alanları korumak, yeniden örnekleme izi eklememek |
| ImageNet ortalama/std normalizasyonu | ResNet-18, U-Net | Ön eğitimli kodlayıcının girdi dağılımı |
| Fotometrik jitter: kontrast α∈[0.92, 1.08], parlaklık β∈[−0.08, 0.08] (yalnız eğitim) | U-Net (GPU'da) | Aydınlatma çeşitliliği. Geometrik artırım yok (metin yönü) |
| Pozitif merkezli örnekleme: kırpıntıların %70'i rastgele bir GT pikseline ±128 px jitter ile, %30'u rastgele konumda | U-Net eğitimi | Sınıf dengesizliği (pozitif ≈ %1) |
| Kayan pencere 512/448 + örtüşme ortalaması; en uzun kenarı 4096 px'i aşan görüntülerde küçültme (15 doğrulama, 29 test görüntüsü) | Çıkarım | GPU belleği |
| Gri seviyeye dönüşüm; LBP, GLCM, Laplacian artığı, FFT radyal spektrum, 8×8 blok-DCT öznitelikleri | Pilot klasik hat | Dersin 3–4. ve 9–11. hafta yöntemleri |
| FFT Gaussian yüksek-geçiren ($D_0\approx20$) + 3×3 Laplacian artığı kanalları | Pilot U-Net (RGB+freq) | Uzamsal/frekans filtrelemenin katkısını sınamak |
| Fark haritası, ε eşiği, morfolojik açma/kapama, bağlı bileşen filtresi, Sauvola ikilileştirme | **GT türetimi** (model girdisi değil) | Piksel/glif GT |
| JPEG Q100 yeniden kodlama + 512 px uzun kenar (ADCD-Net); ≤ 2 MP (TruFor) | Zero-shot referanslar | Modellerin kendi protokol gereksinimleri |
| Simetrik yeniden kodlama, tam görüntü yeniden örnekleme | Kontroller ve H4 | Global izin sahtecilik kanıtı olmadığını göstermek |

### 6.4 Bilinçli olarak yapılmayanlar ve kanıtı

Aşağıdaki bozulmalar test zamanında uygulanarak ölçüldü (doğrulama, her model kendi GT'sinde, 3 seed). Bunlar, ilgili ön işlemin modele zarar verip vermeyeceğinin dolaylı kanıtıdır.

| İşlem | Neden yapılmadı | Ölçülen etki (ΔAUPRC / ΔAP) |
|---|---|---|
| Sabit boyuta yeniden boyutlandırma | XFUND'da alan görüntünün %0.02'si kadar; 512 px'e küçültmede birkaç piksele iner. Ayrıca yeniden örnekleme izi ekler | 0.75× ölçekleme: bbox −0.112 / −0.063; piksel −0.245 / −0.071; glif −0.289 / −0.105 |
| Gürültü giderme / bulanıklaştırma | Gürültü artığını ve glif kenarlarını siler | Gauss σ=1: bbox −0.190 / −0.095; piksel −0.361 / −0.111; glif −0.398 / −0.157 |
| JPEG'e dönüştürme | Yeni sıkıştırma izi ekler; PNG verisinde DCT ipucu yoktur | JPEG Q75: bbox −0.154 / −0.126; piksel −0.137 / −0.105; glif −0.122 / −0.103 |
| Histogram eşitleme, CLAHE, kontrast germe | Yerel istatistikleri değiştirir; aydınlatma çeşitliliği jitter ile modellendi | **Test edilmedi** |
| Model girdisinde ikilileştirme veya gri seviye | Renk ve gri ton geçişleri kanıt taşıyabilir. Öte yandan işaretçi yeşil olduğu için gri seviye bir savunma da olabilir | **Test edilmedi** (iki yönlü hipotez) |
| Perspektif/eğiklik düzeltme, belge kırpma | Yeniden örnekleme gerektirir, GT koordinatlarını bozar | Uygulanmadı |
| Yatay çevirme, döndürme | Metin yönü | Uygulanmadı |

Pilotta, frekans kanallarının eklenmesi U-Net'i iyileştirmedi (F1 0.707 → 0.674). ADCD-Net'in PNG verisinde çalışabilmesi için gereken JPEG Q100 dönüşümü sonrasında test F1@0.5 değeri 0.182 oldu; bu dönüşümün düşüşe katkısı ayrıca ölçülmedi.

### 6.5 Görüntülere bakarak ön işleme yapmak anlamlı mı? Yapılması gerekir mi?

**İki ayrı anlamı ayırmak gerekir:**

1. **Görüntüyü "daha iyi görünür" kılmak için ön işleme (iyileştirme):** Bu veri setinde **gerekli değildir ve zararlıdır**. Kanıt §6.4'teki bozulma tablosudur: bulanıklık, ölçekleme ve sıkıştırma tüm modellerde AUPRC'yi 0.11–0.40 düşürüyor. Piksel ve glif modelleri, ince glif kenarlarına dayandıkları için daha da duyarlıdır.
2. **Veri setindeki istenmeyen ipuçlarını temizlemek için ön işleme (veri temizliği):** Bu **gereklidir**. Görüntüleri incelemek tam olarak bu tür ipuçlarını bulmak için değerlidir; yeşil işaretçi bu yolla bulundu. Ancak elle, görüntüye göre değişen bir işlem tekrarlanabilir değildir. Her adım algoritmik tanımlanmalı ve tüm görüntülere, sahte ve authentic için aynı biçimde uygulanmalıdır.

**Karar tablosu**

| Öncelik | Ön işleme | Neden | Durum |
|---|---|---|---|
| **Zorunlu** | İşaretçi kalıntısının nötralize edilmesi: eğitim görüntülerinde işaretçi piksellerinin authentic'ten kopyalanarak onarılması; piksel/glif GT'den işaretçi piksellerinin çıkarılması; testte kalıntılı ve kalıntısız alt kümelerin ayrı raporlanması | İşaretçi tek başına kutuyu tetikliyor (AP 0.906); onarım performansın dörtte birini götürüyor (ΔAP −0.225) | Sonda yapıldı; **yeniden eğitim yapılmadı** (bir sonraki deney, RQ-A) |
| **Zorunlu** | Sahte ve authentic için simetrik işlem zinciri | Görüntü düzeyi sınıflandırmada kodlama geçmişi farkı sahte bir ipucu olur | H4 kontrollerinde uygulandı |
| **Zorunlu** | İzi koruyan ölçek yönetimi (doğal çözünürlük + kayan pencere) | Küçük alanlar | Uygulandı |
| Önerilir, ablasyonla sınanmalı | Kroma bastırma / gri seviye girdi | İşaretçiye karşı savunma; ancak renk bilgisi kaybolur | Hipotez |
| Önerilir, ablasyonla sınanmalı | Gürültü artığı kanalları (SRM, BayarConv) | PNG'de bile gürültü tutarsızlığı kanıt olabilir (MVSS-Net, TruFor ailesi) | Hipotez; pilotta FFT/Laplacian kazanç vermedi |
| Önerilir, ablasyonla sınanmalı | Belge bölgesi maskesi (arka plandaki el/ahşap/kumaş üzerinde tahminin bastırılması; yeniden örnekleme olmadan, son işleme olarak) | Arka plan yanlış pozitiflerini azaltabilir | Hipotez |
| Yalnız karşılaştırma için | CLAHE / aydınlatma normalizasyonu | Fotoğraflanmış fişlerde aydınlatma farkı | Varsayılan yapılmamalı |
| Yapılmamalı | Yeniden boyutlandırma, gürültü giderme, JPEG dönüşümü, perspektif düzeltme | §6.4 | — |

---

## 7. Kullanılan modeller/yöntemler ve birlikte kullanımı

### 7.1 Kullanılanlar

| Yöntem | Görev | Rol | Veri | Ana sonuç |
|---|---|---|---|---|
| Klasik öznitelik (LBP, GLCM, Laplacian, FFT, DCT) + SVM | Patch sınıflandırma | Klasik referans | Pilot 400/100 | AUROC 0.646 |
| Aynı öznitelikler + Random Forest | Patch sınıflandırma | Klasik referans | Pilot | AUROC 0.706 |
| ResNet-18 linear probe (dondurulmuş gövde + yeni FC) | Patch sınıflandırma | "Önceden eğitilmiş + yeni katman" | Pilot | AUROC 0.726 |
| ResNet-18 tam ince ayar | Patch sınıflandırma | Transfer öğrenme | Pilot | AUROC 0.899 |
| ResNet-18 sıfırdan | Patch sınıflandırma | "Kendi modelimiz" | Pilot | AUROC 0.679 |
| U-Net (ResNet-18 kodlayıcı), RGB | Segmentasyon | Ana model | Pilot; tam veri 3 GT × 3 seed + Gemini-only | Pilot F1 0.707 / AUPRC 0.947. Tam veri bbox GT test AUPRC 0.950 |
| U-Net, RGB + FFT + Laplacian (5 kanal) | Segmentasyon | Klasik-derin füzyon | Pilot | F1 0.674 / AUPRC 0.939 |
| TruFor (resmi ağırlıklar) | Segmentasyon, zero-shot | Literatür referansı | Resmi test | F1@0.5 0.403, IoU 0.328 |
| ADCD-Net (MIT lisansı) | Segmentasyon, zero-shot | Literatür referansı | Resmi test | F1@0.5 0.182, IoU 0.149 |
| ResNet-18 temiz-crop sınıflandırıcısı | H4 sondası | Global iz testi | Tam veri | AUROC 0.516; pozitif kontrol 0.980 |
| Kural tabanlı yeşil çizgi → kutu | İşaretçi sondası | Öğrenmesiz alt sınır | Resmi test | Ortalama F1 0.054 |
| Morfolojik rectangle-fit | Son işleme sondası | H2 | Doğrulama | — |
| VLM (görüntü + istem) | GT denetimi | Araç, dedektör değil | 100 çift | 96 ayrıştırılmış panel |

TruFor ve ADCD-Net AIForge-Doc üzerinde eğitilmemiştir. Bizim modellerimizle karşılaştırmaları eşit koşullarda değildir ve her yerde ayrı tabloda verilmiştir.

### 7.2 Birlikte (hibrit) kullanım

| Hibrit | Nasıl | Sonuç |
|---|---|---|
| Sınıflandırma ağı → segmentasyon kodlayıcısı | ImageNet'te eğitilmiş ResNet-18 ağırlıkları U-Net kodlayıcısına aktarıldı (transfer öğrenme köprüsü) | Pilotta ön eğitim gerekli çıktı (ince ayar 0.899, sıfırdan 0.679) |
| Klasik filtre + derin model (erken füzyon) | FFT yüksek-geçiren ve Laplacian çıktıları U-Net'e ek kanal olarak verildi | Kazanç yok |
| Segmentasyon → sınıflandırma | Tek model iki görev: $s(I)$ = en yüksek %1 piksel ortalaması | Görüntü AUROC 0.892 |
| Kayıp füzyonu | BCE + Dice | Küçük pozitif bölgede kararlı eğitim |
| Derin model + morfolojik son işleme | Tahmine dikdörtgen uydurma | bbox modelini bozmuyor (zaten dikdörtgen üretiyor) |
| Kural tabanlı dedektör + derin model | İşaretçi dedektörü, modelin neye baktığını açıklamak için | Kural tek başına F1 0.054; model aynı ipucuyla AP 0.906 |
| Literatürdeki hazır hibritler | TruFor: RGB + Noiseprint++; ADCD-Net: RGB + DCT | Zero-shot referans |

**Yapılmayan birlikte kullanımlar:** model topluluğu (ör. TruFor + U-Net), çok görevli ağ (ayrı sınıflandırma başı + segmentasyon başı), birleşik GT kaybı (piksel + kutu; RQ-B), TruFor'un AIForge-Doc üzerinde ince ayarı.

---

## 8. Kullanılan ve kullanılmayan yöntemlerin karşılaştırması

"Beklenen davranış" sütunu **hipotezdir, ölçülmemiştir.**

| Yöntem ailesi | Örnek | Kullanıldı mı? | Gerekçe | Bu veri setinde beklenen davranış | Kaynak |
|---|---|---|---|---|---|
| Klasik öznitelik + ML | SVM, RF | Evet (pilot) | Ders yöntemleri, referans tabanı | Ölçüldü: 0.646 / 0.706 | Cortes & Vapnik 1995; Breiman 2001 |
| CNN sınıflandırıcı | ResNet-18 | Evet (pilot, H4) | Ders (hafta 12), transfer öğrenme sorusu | Ölçüldü | He ve ark. 2016 |
| Kodlayıcı–kod çözücü segmentasyon | U-Net | Evet (ana) | Küçük alan için atlama bağlantıları; düşük maliyet | Ölçüldü | Ronneberger ve ark. 2015 |
| Transformer segmentasyon | SegFormer, IML-ViT | Hayır | Tek mimariyle GT etkisini izole etmek; 2.7k görüntü; IML-ViT 1024 px sınırı | Mutlak skor değişebilir; GT etkisinin (H1) mimariden bağımsız olması beklenir | Ma ve ark. 2023 (IML-ViT) |
| RGB + DCT çift akış | DTD, CAT-Net, FFDN, ADCD-Net, DiffNet | Yalnız ADCD-Net (zero-shot) | Veri PNG; JPEG ızgarası/nicemleme tablosu yok | DCT akışı ipucu bulamaz; FFDN yazarları PNG'de frekans ipuçlarının zararlı olabileceğini belirtir | Qu ve ark. 2023; Kwon ve ark. 2022; Chen ve ark. 2024; Wong ve ark. 2025 |
| Gürültü artığı tabanlı | TruFor, MVSS-Net (BayarConv) | TruFor (zero-shot) | Genel amaçlı referans | Noiseprint++ kamera önseli render/tarama belgelerde zayıf | Guillaro ve ark. 2023; Dong ve ark. 2023 |
| Nesne tespiti | YOLO, Faster R-CNN | Hayır | Görev piksel düzeyinde; kutu tespiti doğrudan kutu GT'yi öğrenir | Yüksek kutu-AP ama aynı kutu/işaretçi kısayolu | Ders hafta 14 |
| Örnek segmentasyonu | Mask R-CNN | Hayır | Görüntü başına tek alan; örnek ayrımı gereksiz | — | — |
| MLLM/VLM yargıç | GPT-4o | Dedektör olarak hayır; denetim aracı olarak evet | Literatürde şansa yakın | v1: AUC 0.509; v2 öz-yargıç 0.532 | Wu ve ark. 2026a, 2026b |
| Yeniden yapılandırma tabanlı | AEROBLADE | Hayır | Yerel düzenlemede görüntünün çoğu değişmez | v3 yazarları da aynı gerekçeyle dışlıyor | Raj ve ark. 2026 |
| Model topluluğu | TruFor + U-Net | Hayır | Bütçe; GT sorusu | Kalibrasyon iyileşebilir | — |

**Hangi karşılaştırmalar deneysel olarak mevcut?** Klasik ile derin, üç ön eğitim rejimi, RGB ile RGB+frekans, üç GT, eğitilmiş U-Net ile zero-shot TruFor/ADCD-Net. **Yeni deney gerektirenler:** SegFormer/IML-ViT, YOLO, DTD ince ayarı, model toplulukları. Bir U-Net koşusu 9.5–11 dk sürdüğünden, benzer ölçekte bir alternatif mimarinin 3 seed ile bir gün içinde denenebileceği tahmin edilmektedir (ölçülmedi).

---

## 9. Vize raporu şablonuna uygunluk

| Şablon maddesi | Vize raporundaki yeri | Durum |
|---|---|---|
| 1. Proje Adı | §1 | Var |
| 2a. Veri tabanıyla yapılmış çalışmalar, yöntemler, sonuçlar | §2.2 | Var; 27 Ekim güncellemesiyle v3 eşzamanlı çalışması ve hakemsiz depo eklendi; "ilk çalışma" iddiası daraltıldı |
| 2b. Kullanılan modellerin sunulduğu çalışmalar | §2.4 | Var (ResNet, U-Net, ImageNet, LBP, GLCM, DCT, SVM, RF, Dice, frekans filtreleme) |
| 2c. (Ek) Araştırma boşluğu ve katkı | §2.5 | **Yeni eklendi** |
| 3a. Veri tabanı bilgileri | §3.1 | Var |
| 3b. Modellerin açıklaması, pseudo kod, matematik | §3.3–3.5 | Var (her model için formül ve pseudo kod) |
| 3c. (Ek) Ön işleme ve gerekçesi | §3.2.1 | **Yeni eklendi** |
| 3d. (Ek) Kullanılan/kullanılmayan yöntemler | §3.8 | **Yeni eklendi** |
| 4. Kısıtlı veriyle sonuçlar (sistemin çalıştığının gösterilmesi) | §4 (smoke 400/100) | Var; §4.b ve §4.c tam veri ve işaretçi bulgusu |

---

## Kaynaklar (bu notta ek olarak kullanılanlar)
- OECD (2015). *Frascati Manual 2015: Guidelines for Collecting and Reporting Data on Research and Experimental Development.* OECD Publishing, Paris. doi:10.1787/9789264239012-en.
- Raj A., Zhang Y., Zewde K., Duong T., Gan J., Shen X., Zhou Y., Guo H., Zhang S., Ren S. (2026). *ChatGPT Images 2.5 on Forgery Tasks: Testing Advertised Improvements Against Known Answers.* arXiv:2609.13617.
- Ren S. ve ark. (2026). *Agents That Edit Documents: Measuring Agentic PDF Forgery Against a Non-Agentic Control.* arXiv:2609.23953.
- Joya15 (t.y.; erişim 27 Ekim 2026). *Multi-Generator-Deepfake-Document-Detection-and-Tamper-Localization* [GitHub deposu, hakemsiz]. https://github.com/Joya15/Multi-Generator-Deepfake-Document-Detection-and-Tamper-Localization
- Diğer tüm kaynaklar: `literature/references.bib` ve `reports/Bildiri_Tam_Metin.md` kaynakçası (51 kayıt).
