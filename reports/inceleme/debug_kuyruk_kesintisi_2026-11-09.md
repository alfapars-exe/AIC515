# Hata incelemesi — 9 Kasım 2026 gece kuyruk kesintisi

**Tarih:** 10 Kasım 2026 · **Yöntem:** Lab Rat `debug` (önce ucuz yoklamalar, sonra hipotez; kanıt yoksa neden ilan edilmez)
**Olay:** Faz 3 kuyruklarının bütün süreçleri 9 Kasım 23:30 dolayında sonlandı; makine yeniden başlamadı. `phase3/logs/queue_notes.txt` ve Faz 3 kapanış belgesi (§4) nedeni "belirsiz" olarak kaydetmişti. Kuyruk 01:14'te yeniden başlatıldı; `.done` işaretleri sayesinde yalnız yarım kalan iki adım yeniden koşuldu.

## 1. Yoklamalar

| Yoklama | Sonuç |
|---|---|
| `phase3/logs/queue.log` | 22:45:49 `START p_probes_unet_s4344`, 23:23:39 `START y_E1_geminionly`; ikisi için de `DONE` ya da hata satırı yok. Sonraki satır 01:14:03 `RESTART q2/q3a/q3b after interruption`. İki ayrı kuyruk aynı anda susmuş. |
| Windows Sistem günlüğü, 22:30–01:20 | Güç, uyku, yeniden başlatma ya da oturum kapatma olayı yok. |
| Windows Sistem günlüğü, 23:30:39 | `nvlddmkm` olay 153: "Error occurred on GPUID: 100". 8–10 Kasım aralığındaki tek `nvlddmkm` olayı. |
| Windows Uygulama günlüğü, 23:30:40–23:30:42 | Claude masaüstü uygulamasının paketi güncellendi: `Claude_4.23118.0.0` paketi kaldırıldı, `Claude_4.23118.1.0` kuruldu; "Claude VM Service stopped" (23:30:40) ve "starting" (23:30:42). 8–10 Kasım aralığında hizmetin durdurulduğu tek an. |
| Günlüklerde CUDA hatası araması | `phase3/logs`, `phase3_yolo/logs`, `phase3_yolo/runs` altında "CUDA error", "out of memory" ya da benzeri bir kayıt yok (yarım kalan adımların günlükleri yeniden koşuda üzerine yazılmış olabilir). |

## 2. Hipotez

**Hipotez (en çok desteklenen):** Kuyruklar Claude masaüstü uygulamasının alt süreçleri olarak çalışıyordu ve uygulamanın 23:30:40'taki otomatik güncellemesi süreç ağacını sonlandırdı.

Destekleyen kanıt: güncelleme ile kesinti aynı saniyeye düşüyor; iki bağımsız kuyruk aynı anda ve hata satırı yazmadan duruyor (tek bir eğitim sürecindeki GPU hatası kuyruk betiğine bir hata satırı yazdırır ve öteki kuyruğu durdurmazdı); makine yeniden başlamamış.

Dışlanamayan seçenek: bir saniye önceki GPU sürücü hatası bağımsız bir arıza olabilir ve iki GPU işini birlikte düşürmüş olabilir. Ancak bu, kuyruk betiklerinin de hata satırı yazmadan durmasını açıklamaz. Sürücü olayı, GPU işlerinin aniden sonlandırılmasının sonucu da olabilir; hangisinin önce geldiği bu kayıtlardan çıkarılamıyor.

## 3. Duman testi

Yapılamadı: bir uygulama güncellemesi istenerek tetiklenemez ve o gecenin kuyruklarının hangi üst süreçten başlatıldığı kayıtlı değil. Dolaylı kontrol: 10 Kasım'daki koşular (Faz 3'ün kalan adımları, v2 testi, Faz 4, test paketi; toplam birkaç GPU saati) sırasında ne paket güncellemesi ne de `nvlddmkm` olayı var ve hiçbiri kesilmedi.

## 4. Hüküm

**Büyük olasılıkla neden: Claude masaüstü uygulamasının otomatik güncellemesi. Kesin değil.** Kesinleştirmek için eksik olan tek bilgi, kuyrukların üst süreci. Sonuçlar etkilenmedi: yarım kalan iki adım baştan koşuldu, tamamlanan adımlar korunmuştu.

## 5. Öneriler

1. Saatler süren kuyrukları uygulamanın süreç ağacının dışında başlatmak: Görev Zamanlayıcı ile ya da kullanıcının kendi açtığı bir PowerShell penceresinden. Uygulama içinden başlatılacaksa bile ayrık bir süreçle (`Start-Process`), çıktısı dosyaya yönlendirilerek.
2. Kuyruk başlangıcında üst süreç zincirini `queue.log` dosyasına yazmak; bir sonraki kesintide neden tek yoklamayla görülür.
3. `.done` işaretli, kaldığı yerden süren kuyruk düzeni korunmalı; bu olayda kaybı 1 saat 44 dakikalık duruşla sınırlayan o oldu.
4. `nvlddmkm` olay 153 yinelenirse (uygulama güncellemesi olmadan) GPU tarafı ayrıca incelenmeli: sıcaklık, güç, sürücü sürümü.
