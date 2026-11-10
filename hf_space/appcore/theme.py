"""appcore.theme — arayüz teması: CSS, sol panel açıklamaları (TIPS) ve TIP_JS."""
from __future__ import annotations
from ._common import *  # noqa: F401,F403 — ortak içe aktarmalar
from .xai_bridge import (  # noqa: F401
    XAI, XAI_OK
)

# ================================================================== ARAYÜZ
# ---- ortak görsel ölçüleri: tüm sekmelerde yalnız bu dört boy kullanılır
H_SM = 180   # yardımcı girdiler (maske, authentic eş, küçük yükleme alanları)
H_MD = 340   # yan yana gösterilen paneller
H_LG = 480   # tek başına gösterilen ana panel, galeriler
H_XL = 600   # XAI inceleme panelleri (yakınlaştırılabilir)

# ---- ortak renk açıklaması (appcore.metrics.overlay / prob_heat ve simülasyon panelleriyle aynı renkler)
LEGEND_MD = ("**Renkler** · <span style='color:#e02020'>■</span> model tahmini · "
             "<span style='color:#00a0ff'>■</span> resmî GT (alan kutusu) · "
             "<span style='color:#e0b000'>■</span> piksel-değişim GT · "
             "<span style='color:#28dc28'>■</span> tahminin sınırlayıcı kutusu · "
             "<span style='color:#d020d0'>■</span> yeşil işaretçi pikselleri · "
             "olasılık haritası: mavi (düşük) → kırmızı (yüksek)")

CSS = """
.small-md p {font-size: 0.9em}
.legend-md p {font-size: 0.88em; color: #4b5563; margin: 2px 0}
.status-md p {font-size: 0.92em; color: #374151; margin: 0}
#verdict {border-left: 4px solid #4C72B0; padding-left: 10px}
footer {display:none !important}
#aic-tip {position:fixed; z-index:10000; max-width:340px; padding:8px 11px; border-radius:6px; font-size:13px; line-height:1.45; background:#1f2937; color:#f9fafb; box-shadow:0 4px 14px rgba(0,0,0,.25); pointer-events:none; opacity:0; transition:opacity .12s; left:-9999px; top:0}
#aic-tip.show {opacity:1}
#aic-tip b {color:#fde68a}
#tip-f_split [data-testid="block-info"]::after, #tip-f_source [data-testid="block-info"]::after, #tip-f_tool [data-testid="block-info"]::after, #tip-f_area [data-testid="block-info"]::after, #tip-f_marker [data-testid="block-info"]::after, #tip-f_mp [data-testid="block-info"]::after, #tip-f_pair [data-testid="block-info"]::after, #tip-b_filter [data-testid="block-info"]::after, #tip-b_rand [data-testid="block-info"]::after, #tip-dd_key [data-testid="block-info"]::after, #tip-rd_input [data-testid="block-info"]::after, #tip-dd_deg [data-testid="block-info"]::after, #tip-ck_pipe [data-testid="block-info"]::after, #tip-cg_unet [data-testid="block-info"]::after, #tip-cg_res [data-testid="block-info"]::after, #tip-sl_stride_sim [data-testid="block-info"]::after, #tip-b_sim [data-testid="block-info"]::after, #tip-b_to_tabs [data-testid="block-info"]::after, #tip-t_split [data-testid="block-info"]::after, #tip-t_source [data-testid="block-info"]::after, #tip-t_tool [data-testid="block-info"]::after, #tip-t_area [data-testid="block-info"]::after, #tip-t_marker [data-testid="block-info"]::after, #tip-t_mp [data-testid="block-info"]::after, #tip-t_n [data-testid="block-info"]::after, #tip-t_seed [data-testid="block-info"]::after, #tip-t_mode [data-testid="block-info"]::after, #tip-t_deg [data-testid="block-info"]::after, #tip-t_unet [data-testid="block-info"]::after, #tip-t_res [data-testid="block-info"]::after, #tip-b_batch [data-testid="block-info"]::after, #tip-in_img [data-testid="block-info"]::after, #tip-in_mask [data-testid="block-info"]::after, #tip-in_auth [data-testid="block-info"]::after, #tip-btn_load [data-testid="block-info"]::after, #tip-dd_sample [data-testid="block-info"]::after, #tip-btn_sample [data-testid="block-info"]::after, #tip-tx_draft [data-testid="block-info"]::after, #tip-dd_draft [data-testid="block-info"]::after, #tip-dd_cat [data-testid="block-info"]::after, #tip-dd_op [data-testid="block-info"]::after, #tip-sl1 [data-testid="block-info"]::after, #tip-sl2 [data-testid="block-info"]::after, #tip-sl3 [data-testid="block-info"]::after, #tip-dd_mode [data-testid="block-info"]::after, #tip-btn_prev [data-testid="block-info"]::after, #tip-btn_apply [data-testid="block-info"]::after, #tip-btn_undo [data-testid="block-info"]::after, #tip-btn_reset [data-testid="block-info"]::after, #tip-dd_preset [data-testid="block-info"]::after, #tip-btn_preset [data-testid="block-info"]::after, #tip-dd_unet [data-testid="block-info"]::after, #tip-rd_src [data-testid="block-info"]::after, #tip-rd_gt [data-testid="block-info"]::after, #tip-sl_thr [data-testid="block-info"]::after, #tip-ck_otsu [data-testid="block-info"]::after, #tip-sl_morph [data-testid="block-info"]::after, #tip-btn_unet [data-testid="block-info"]::after, #tip-btn_cmp [data-testid="block-info"]::after, #tip-dd_rn [data-testid="block-info"]::after, #tip-rd_src2 [data-testid="block-info"]::after, #tip-sl_stride [data-testid="block-info"]::after, #tip-btn_rn [data-testid="block-info"]::after, #tip-btn_green [data-testid="block-info"]::after, #tip-x_src [data-testid="block-info"]::after, #tip-x_up_img [data-testid="block-info"]::after, #tip-x_up_auth [data-testid="block-info"]::after, #tip-x_up_gt [data-testid="block-info"]::after, #tip-x_unet [data-testid="block-info"]::after, #tip-x_res [data-testid="block-info"]::after, #tip-b_xpred [data-testid="block-info"]::after, #tip-x_target [data-testid="block-info"]::after, #tip-x_k [data-testid="block-info"]::after, #tip-x_method [data-testid="block-info"]::after, #tip-x_ref [data-testid="block-info"]::after, #tip-x_evals [data-testid="block-info"]::after, #tip-b_xshap [data-testid="block-info"]::after, #tip-b_xcmp [data-testid="block-info"]::after, #tip-x_show [data-testid="block-info"]::after {content:" ⓘ"; opacity:.55; font-size:.9em}
"""


# ---------------------------------------------------------------- sol panel açıklamaları (fareyle üzerine gelince)
TIPS = {
    "tip-f_split": 'Görüntülerin veri setinin hangi bölümünden seçileceği.<br><b>test</b>: modellerin eğitimde hiç görmediği görüntüler (adil değerlendirme).<br><b>eğitim</b>: modellerin öğrendiği görüntüler; sonuçlar iyimser çıkar.<br><b>hepsi</b>: iki bölüm birlikte.',
    "tip-f_source": 'Sahte belgenin türetildiği orijinal belge koleksiyonu (ör. <b>cord</b> fişleri). Belge türüne ve düzenine göre filtrelemeye yarar.',
    "tip-f_tool": 'Belgedeki alanı değiştirmek için kullanılan üretken yapay zekâ (inpainting) modeli, ör. <b>Gemini</b>, <b>Ideogram</b>. Her üretici farklı izler bırakır; modellerin bir üreticiden diğerine genelleşmesini incelemek için kullanılır.',
    "tip-f_area": 'Değiştirilen bölgenin (resmî GT kutusu) görüntü alanına oranı.<br><b>&lt;1%</b>: çok küçük hedef, tespiti en zor rejim.<br><b>1-5%</b>, <b>5-20%</b>, <b>&gt;20%</b>: hedef büyüdükçe lokalizasyon kolaylaşır.',
    "tip-f_marker": "Bazı sahte görüntülerde (çoğunlukla Gemini) düzenlenen alanın çevresinde ~3 px'lik yeşil çerçeve kalıntısı kalmıştır. Bir model sahteciliği değil bu çerçeveyi öğrenebilir (kısayol).<br><b>var</b> / <b>yok</b> ile bu örnekleri ayırabilirsiniz.",
    "tip-f_mp": 'Görüntü boyutu üst sınırı (genişlik × yükseklik / 10⁶). Modeller natif çözünürlükte çalıştığı için büyük görüntülerde çıkarım uzar; hızlı denemeler için düşük tutun.',
    "tip-f_pair": "Yalnızca değiştirilmemiş orijinali (<i>authentic eşi</i>) aynı boyutta bulunan örnekleri listeler. Piksel-değişim GT'si, authentic negatif kontrol ve yeşil işaretçi senaryoları bu eşe ihtiyaç duyar.",
    "tip-b_filter": 'Yukarıdaki filtreleri uygular ve <b>Örnek</b> listesini günceller.',
    "tip-b_rand": 'Filtreye uyan örnekler arasından rastgele birini seçer.',
    "tip-dd_key": 'Çalıştırılacak görüntü. Biçim: <b>bölüm:kimlik</b> · kaynak · üretici · alan oranı · MP (+ <i>eşli</i>, <i>işaretçi</i> etiketleri). Listeden seçebilir ya da kimliği yazarak arayabilirsiniz.',
    "tip-rd_input": 'Modele hangi görüntünün verileceği:<br><b>Sahte belge</b>: görüntü olduğu gibi (beklenen karar SAHTE).<br><b>Authentic eş</b>: değiştirilmemiş orijinal (beklenen GERÇEK); yanlış alarm kontrolü.<br><b>Sahte, yeşil işaretçi onarılmış</b>: işaretçi pikselleri orijinalden geri kopyalanır, içerik sahte kalır (beklenen SAHTE); model işaretçiye mi dayanıyor?<br><b>Authentic + yalnız yeşil işaretçi</b>: orijinale yalnızca işaretçi eklenir, hiçbir karakter değişmez (beklenen GERÇEK); SAHTE derse model kısayol öğrenmiştir.<br>Son üç senaryo authentic eşi olan örnek gerektirir.',
    "tip-dd_deg": 'Modelden önce girdiye uygulanan bozulma; gerçek hayattaki işlemlere dayanıklılığı ölçer (modeller ham PNG ile eğitildi).<br><b>JPEG Q75 / Q50</b>: kayıplı sıkıştırma<br><b>0.75× / 0.5×</b>: küçültüp eski boyuta geri büyütme<br><b>Gauss bulanıklık σ=1</b>, <b>Gauss gürültüsü σ=5</b>',
    "tip-ck_pipe": 'İşaretliyse <b>Ön işleme</b> sekmesinde kurduğunuz işlem zinciri modelden önce girdiye uygulanır; ön işlemenin model başarısına etkisini görmek içindir.',
    "tip-cg_unet": 'Piksel düzeyinde sahte bölge olasılık haritası üreten segmentasyon modelleri. Adlar hangi GT ile eğitildiğini gösterir (bbox, piksel-değişim, glyph). Tam görüntüde 512 px pencere / 448 px adımla çalışır. Birden fazlasını seçip karşılaştırabilirsiniz.',
    "tip-cg_res": 'Görüntüyü 224×224 px parçalara (patch) bölüp her parçayı sahte/gerçek olarak sınıflandıran model. Çıktısı kaba bir ısı haritası ve en yüksek patch skoruna dayanan görüntü kararıdır.',
    "tip-sl_stride_sim": "224 px'lik ResNet penceresinin her adımda kaç piksel kaydırılacağı. Küçük adım daha çok örtüşme ve daha ince ısı haritası verir ama yavaştır (adım yarıya inince patch sayısı ~4 katına çıkar). 224 = örtüşmesiz.",
    "tip-b_sim": 'Seçilen örnek, senaryo ve modellerle hattı adım adım çalıştırır: GT analizi → yeşil işaretçi taraması → girdi senaryosu ve bozulma → modeller → iki GT (bbox ve piksel-değişim) ile metrikler → görüntü-düzeyi karar.',
    "tip-b_to_tabs": 'Bu örneği (sahte görüntü, GT maskesi, authentic eş) Girdi, Ön işleme, U-Net ve ResNet-18 sekmelerine yükler.',
    "tip-t_split": 'Görüntülerin veri setinin hangi bölümünden seçileceği.<br><b>test</b>: modellerin eğitimde hiç görmediği görüntüler (adil değerlendirme).<br><b>eğitim</b>: modellerin öğrendiği görüntüler; sonuçlar iyimser çıkar.<br><b>hepsi</b>: iki bölüm birlikte.',
    "tip-t_source": 'Sahte belgenin türetildiği orijinal belge koleksiyonu (ör. <b>cord</b> fişleri). Belge türüne ve düzenine göre filtrelemeye yarar.',
    "tip-t_tool": 'Belgedeki alanı değiştirmek için kullanılan üretken yapay zekâ (inpainting) modeli, ör. <b>Gemini</b>, <b>Ideogram</b>. Her üretici farklı izler bırakır; modellerin bir üreticiden diğerine genelleşmesini incelemek için kullanılır.',
    "tip-t_area": 'Değiştirilen bölgenin (resmî GT kutusu) görüntü alanına oranı.<br><b>&lt;1%</b>: çok küçük hedef, tespiti en zor rejim.<br><b>1-5%</b>, <b>5-20%</b>, <b>&gt;20%</b>: hedef büyüdükçe lokalizasyon kolaylaşır.',
    "tip-t_marker": "Bazı sahte görüntülerde (çoğunlukla Gemini) düzenlenen alanın çevresinde ~3 px'lik yeşil çerçeve kalıntısı kalmıştır. Bir model sahteciliği değil bu çerçeveyi öğrenebilir (kısayol).<br><b>var</b> / <b>yok</b> ile bu örnekleri ayırabilirsiniz.",
    "tip-t_mp": 'Görüntü boyutu üst sınırı (genişlik × yükseklik / 10⁶). Modeller natif çözünürlükte çalıştığı için büyük görüntülerde çıkarım uzar; hızlı denemeler için düşük tutun.',
    "tip-t_n": 'Filtreye uyan örnekler arasından rastgele kaç tanesinin çalıştırılacağı. Süre N ile doğrusal artar.',
    "tip-t_seed": 'Rastgele örnek seçiminin tohumu. Aynı tohum ve aynı filtreler aynı örnekleri verir (tekrar üretilebilirlik).',
    "tip-t_mode": '<b>Yalnız sahte görüntüler</b>: yalnızca lokalizasyon metrikleri (F1, IoU, AP).<br><b>Sahte + authentic eşi</b>: her sahte görüntünün orijinali de çalıştırılır; böylece görüntü-düzeyi tespit (TPR/FPR/AUROC) hesaplanır. Yalnız authentic eşi olan örnekler seçilir.',
    "tip-t_deg": 'Modelden önce girdiye uygulanan bozulma; gerçek hayattaki işlemlere dayanıklılığı ölçer (modeller ham PNG ile eğitildi).<br><b>JPEG Q75 / Q50</b>: kayıplı sıkıştırma<br><b>0.75× / 0.5×</b>: küçültüp eski boyuta geri büyütme<br><b>Gauss bulanıklık σ=1</b>, <b>Gauss gürültüsü σ=5</b>',
    "tip-t_unet": 'Piksel düzeyinde sahte bölge olasılık haritası üreten segmentasyon modelleri. Adlar hangi GT ile eğitildiğini gösterir (bbox, piksel-değişim, glyph). Tam görüntüde 512 px pencere / 448 px adımla çalışır. Birden fazlasını seçip karşılaştırabilirsiniz.',
    "tip-t_res": 'Görüntüyü 224×224 px parçalara (patch) bölüp her parçayı sahte/gerçek olarak sınıflandıran model. Çıktısı kaba bir ısı haritası ve en yüksek patch skoruna dayanan görüntü kararıdır.',
    "tip-b_batch": 'Filtreye uyan N rastgele örnekte simülasyon hattını koşturur; model başına lokalizasyon (bbox ve piksel-değişim GT), görüntü-düzeyi tespit (TPR/FPR/AUROC) ve boru hattının kayıtlı sonuçlarıyla yeniden-üretim farkı raporlanır.',
    "tip-in_img": 'İncelenecek belge görüntüsü (PNG/JPG). Kayıpsız PNG önerilir; JPEG sıkıştırması sahtecilik izlerini zayıflatır.',
    "tip-in_mask": 'Değiştirilen bölgenin ikili maskesi (beyaz 255 = sahte, siyah 0 = gerçek), sahte görüntüyle aynı boyutta. Verilirse model sonuçları bu GT ile ölçülür; verilmezse yalnızca tahmin gösterilir.',
    "tip-in_auth": "Aynı belgenin değiştirilmemiş hali, aynı boyutta. Verilirse iki görüntünün farkından <b>piksel-değişim GT'si</b> türetilir; yeşil işaretçi analizi de bu eşle yapılır.",
    "tip-btn_load": "Yüklediğiniz görüntüleri (ve varsa maske ile authentic eşi) diğer sekmelerin girdisi yapar. Ön işleme pipeline'ı sıfırlanır.",
    "tip-dd_sample": 'Veri setinden doğrudan örnek seçimi (bölüm:kimlik). Filtrelerle daha hızlı seçim için <b>Simülasyon</b> sekmesini kullanıp örneği oradan aktarabilirsiniz.',
    "tip-btn_sample": "Seçilen örneği (sahte görüntü, GT maskesi ve varsa authentic eşi) yükler. Ön işleme pipeline'ı sıfırlanır.",
    "tip-tx_draft": 'Taslağa vereceğiniz ad (boş bırakılırsa tarih kullanılır).',
    "tip-btn_dsave": "O an yüklü olan girdiyi <b>taslak</b> olarak diske kaydeder: sahte görüntü, GT maskesi, authentic eş ve <b>Ön işleme</b> sekmesindeki pipeline adımları. Henüz <b>Yüklenenleri kullan</b>'a basılmadıysa yukarıdaki yükleme alanları kaydedilir.",
    "tip-dd_draft": 'Daha önce kaydedilen taslaklar (en yeni üstte): ad · tarih · boyut · içerik.',
    "tip-btn_dload": "Seçili taslağı tüm sekmelere girdi olarak yükler; kayıtlı pipeline adımları yeniden uygulanır.",
    "tip-btn_ddel": 'Seçili taslağı kalıcı olarak siler.',
    "tip-dd_cat": 'Ders haftalarına göre gruplanmış ön işleme kategorileri (renk, histogram, eşikleme, gürültü, morfoloji, frekans domeni…).',
    "tip-dd_op": 'Seçili kategorideki işlem. Seçince parametreleri ve kısa açıklaması altta görünür.',
    "tip-sl1": 'Seçili işlemin parametresi; anlamı etikette ve işlem açıklamasında yazar. Değiştirip <b>Önizle</b> ile etkisini görün.',
    "tip-sl2": 'Seçili işlemin parametresi; anlamı etikette ve işlem açıklamasında yazar. Değiştirip <b>Önizle</b> ile etkisini görün.',
    "tip-sl3": 'Seçili işlemin parametresi; anlamı etikette ve işlem açıklamasında yazar. Değiştirip <b>Önizle</b> ile etkisini görün.',
    "tip-dd_mode": 'Seçili işlemin çalışma modu (ör. çekirdek şekli, alçak/yüksek geçiren).',
    "tip-btn_prev": "İşlemi mevcut pipeline çıktısına uygulayıp sonucu <b>Sonra</b> panelinde gösterir; pipeline'a eklemez.",
    "tip-btn_apply": "İşlemi pipeline'a kalıcı adım olarak ekler; sonraki işlemler bu çıktının üzerine uygulanır. Pipeline çıktısı U-Net, ResNet-18 ve Simülasyon sekmelerinde model girdisi olarak seçilebilir.",
    "tip-btn_undo": "Pipeline'daki son adımı kaldırır.",
    "tip-btn_reset": 'Tüm adımları siler ve orijinal görüntüye döner.',
    "tip-dd_preset": 'Hazır işlem zincirleri (ör. belge ikilileştirme, frekans analizi, kenar haritası).',
    "tip-btn_preset": "Seçili preseti <b>orijinal</b> görüntüye uygular; mevcut pipeline'ın yerine geçer.",
    "tip-dd_unet": "Kullanılacak U-Net ağırlığı. Ad; eğitimde kullanılan GT'yi (bbox / piksel-değişim / glyph), giriş kanallarını (RGB ya da RGB + frekans) ve tohumu (s42/43/44) gösterir.",
    "tip-rd_src": 'Modele <b>orijinal görüntünün</b> mü yoksa <b>Ön işleme</b> sekmesinde kurduğunuz <b>pipeline çıktısının</b> mı verileceği.',
    "tip-rd_gt": 'Tahminin hangi referansla ölçüleceği.<br><b>resmî bbox</b>: veri setinin dikdörtgen etiketi.<br><b>piksel-değişim</b>: sahte ile authentic eş arasındaki fark eşiklenip (eps 8) morfolojiyle temizlenerek türetilen, yalnız gerçekten değişen pikselleri içeren GT. Authentic eşi gerekir.',
    "tip-sl_thr": 'Olasılık haritasını ikili maskeye çeviren eşik. Model ya da GT türü değişince, o model için doğrulama kümesinde seçilmiş en iyi eşik (τ*) otomatik gelir; elle değiştirebilirsiniz.',
    "tip-ck_otsu": 'Sabit eşik yerine, olasılık haritasının histogramından Otsu yöntemiyle otomatik eşik hesaplanır.',
    "tip-sl_morph": 'İkili maskeye önce açma (küçük gürültüleri siler), sonra kapama (boşlukları doldurur) uygulanır; değer çekirdek boyutudur (px). 0 = kapalı.',
    "tip-btn_unet": 'Seçili modeli çalıştırır: olasılık haritası, bindirme, ikili maske ve ölçütler (F1, IoU, AP…).',
    "tip-btn_cmp": "Aynı modeli hem orijinal görüntüde hem pipeline çıktısında çalıştırıp ölçütleri yan yana koyar; ön işlemenin etkisini gösterir. Pipeline'da en az bir adım olmalı.",
    "tip-dd_rn": "ResNet-18 ağırlığı. <b>ft</b>: ImageNet'ten tam ince ayar · <b>probe</b>: yalnız son katman eğitildi · <b>scratch</b>: sıfırdan eğitim. <b>h4</b> modelleri Faz 2 H4 deneylerine aittir. <b>tam veri (Faz 4)</b>: aynı tarif, 2748 eğitim görüntüsü; <b>işaretçisiz eğitim</b>: yeşil işaretçi kalıntısı onarılmış görüntülerle eğitildi. Görüntü kararı: maks patch ≥ val'de kalibre edilmiş eşik.",
    "tip-rd_src2": 'Modele <b>orijinal görüntünün</b> mü yoksa <b>Ön işleme</b> sekmesinde kurduğunuz <b>pipeline çıktısının</b> mı verileceği.',
    "tip-sl_stride": "224 px'lik pencerenin her adımda kaç piksel kaydırılacağı. Küçük adım daha ince ısı haritası verir ama yavaştır (adım yarıya inince patch sayısı ~4 katına çıkar). 224 = örtüşmesiz.",
    "tip-btn_rn": "Görüntüyü 224 px patch'lerle tarar; patch skoru ısı haritası ve ölçütleri üretir. Görüntü kararı: en yüksek patch skoru ≥ 0.5 ise SAHTE.",
    "tip-btn_green": 'Mevcut görüntüde yeşil işaretçi piksellerini (G − max(R,B) > 50 ve G > 110) arar ve magenta ile gösterir. Authentic eşi varsa orijinalde de bulunan yeşiller çıkarılır.',
    "tip-x_src": "Açıklanacak görüntünün kaynağı.<br><b>Girdi sekmesi</b>: yüklenen ya da aktarılan orijinal görüntü.<br><b>Pipeline çıktısı</b>: Ön işleme sekmesindeki zincirin çıktısı.<br><b>Son simülasyon girdisi</b>: sekme 0'da modele verilen görüntü (senaryo + bozulma dahil).<br><b>Yeni yükleme</b>: buradan yeni görüntü (+ isteğe bağlı authentic eş ve GT).",
    "tip-x_up_img": 'Açıklanacak belge görüntüsü (PNG önerilir).',
    "tip-x_up_auth": 'Aynı belgenin değiştirilmemiş hali (aynı boyut). Verilirse SHAP referansı olarak kullanılır: ΣSHAP = f(sahte) − f(authentic).',
    "tip-x_up_gt": 'Değiştirilen bölgenin maskesi (beyaz = sahte). GT kutusu hedefi ve bölge kütlesi grafiği için kullanılır.',
    "tip-x_unet": 'Açıklanacak U-Net modelleri. Her biri kendi gerçek çıktısından türetilen hedefle ayrı açıklanır.',
    "tip-x_res": 'Açıklanacak ResNet-18 modelleri (224 px patch sınıflandırıcı).',
    "tip-b_xpred": 'Seçili modelleri uygulamanın kendi çıkarımıyla (değiştirilmeden) çalıştırır: görüntü skoru, eşik, karar ve tespit bölgeleri. SHAP hedefleri bu tahminlerden türetilir.',
    "tip-x_target": "Hangi skaler çıktının açıklanacağı:<br><b>Tam görüntü · karar</b> (önerilen): modelin uygulamadaki tam görüntü karar skoru (U-Net: tüm olasılık haritasının top-%1 ortalaması · ResNet: tüm ızgaranın maks patch skoru · YOLO11: en güvenli kutu). Açıklama orijinal görüntünün TAMAMINI tek hedef ve tek referansla kapsar; E[f]+ΣSHAP=f(x) doğrulanır.<br><b>Tam görüntü · tespitler</b>: ilk k bölge / patch / kutu, her biri tam görüntüde.<br><b>Tam görüntü · GT</b>: GT kutusu içi ortalama (ResNet: kesişen patch'lerin maks skoru · YOLO: GT ile IoU≥0.5 kutu güveni).<br><b>Döşemeli mozaik</b>: 512 px örtüşen pencerelerin yerel açıklamaları orijinal koordinatlarda birleşir — tabanlar farklı olduğundan tam görüntü SHAP DEĞİLDİR.<br><b>Pencere</b> hedefleri: önceki 512 px pencere / 224 patch açıklamaları (değişmedi).<br><b>Ortak pencere</b>: karşılaştırma (③).",
    "tip-x_k": 'Tespitler hedefinde model başına açıklanacak en fazla bölge / patch sayısı.',
    "tip-x_method": "<b>Partition</b>: modelden bağımsız, hiyerarşik Owen değerleri; referansla maskeleme yapar, toplamsallık kesindir (ΣSHAP = f(x) − f(referans)).<br><b>Gradient</b>: beklenen gradyanlar (fp32); piksel düzeyinde ince doku izi, 5 kanallı U-Net'te RGB / FFT-HP / Laplacian grup katkısı. Toplamsallık yaklaşıktır.",
    "tip-x_ref": "Gizlenen piksellerin neyle doldurulacağı (SHAP'ın karşı-olgusal referansı).<br><b>authentic eş</b>: orijinal belge — en anlamlı referans; ΣSHAP = f(sahte) − f(orijinal).<br><b>blur(32,32)</b>: görüntünün bulanık hali.<br><b>inpaint_telea</b>: çevreden doldurma (yalnız Partition).",
    "tip-x_evals": "Partition açıklayıcısının model değerlendirme bütçesi. Yüksek değer daha ince bölümleme ve daha ayrıntılı harita verir ama süre doğrusal artar (ZeroGPU'da çağrı başına ≤ 120 s).",
    "tip-b_xshap": 'Seçilen hedefleri SHAP ile açıklar. Tahminden bağımsızdır; aynı görüntü, model, hedef, yöntem, referans ve bütçe önbellekteyse yeniden hesaplanmaz. Sonuçlar PNG · JSON · NPZ · MD olarak indirilebilir.',
    "tip-b_xcmp": 'Tüm seçili modelleri aynı pencerede, aynı referans ve bütçeyle açıklar; top-%5 pozitif bölge IoU/Dice, 16×16 blok Spearman korelasyonu ve ortak / modele özgü bölge haritası üretir. Farklı hedef türleri yalnız normalize ölçütlerle karşılaştırılır.',
    "tip-x_show": 'Ayrıntılı panelleri gösterilecek açıklama (model · hedef).',
}

CSS += (XAI.viewer.ZOOM_CSS if XAI_OK else "")  # büyük, en-boy korumalı, yakınlaştırılabilir XAI görüntüleri

TIP_JS = """
() => {
  const tips = """ + json.dumps(TIPS, ensure_ascii=False) + """;
  if (window.__aicTips) return; window.__aicTips = true;
  const box = document.createElement('div'); box.id = 'aic-tip'; document.body.appendChild(box);
  let cur = null;
  const find = (el) => { while (el && el !== document.body) { if (el.id && tips[el.id]) return el; el = el.parentElement; } return null; };
  const place = (t) => {
    const r = t.getBoundingClientRect(), bw = box.offsetWidth, bh = box.offsetHeight, vw = innerWidth, vh = innerHeight;
    let x = r.right + 12, y = r.top;
    if (x + bw > vw - 8) { x = Math.max(8, Math.min(r.left, vw - bw - 8)); y = r.bottom + 8; if (y + bh > vh - 8) y = r.top - bh - 8; }
    y = Math.max(8, Math.min(y, vh - bh - 8));
    box.style.left = x + 'px'; box.style.top = y + 'px';
  };
  document.addEventListener('mouseover', (e) => {
    const t = find(e.target);
    if (t === cur) return;
    cur = t;
    if (!t) { box.classList.remove('show'); return; }
    box.innerHTML = tips[t.id]; box.classList.add('show'); place(t);
  });
  document.addEventListener('scroll', () => { if (cur) place(cur); }, true);
  document.addEventListener('mousedown', () => { cur = null; box.classList.remove('show'); });
}
"""
