"""appcore — AIC515 AIForge-Doc uygulamasının çekirdek paketi.

Katmanlar (içe aktarma sırası):
  config      yollar (ROOT, DATA, MODELS, META ...), ZeroGPU algılama ve gpu() dekoratörü
  imaging     görüntü/frekans yardımcıları, yeşil işaretçi dedektörü, piksel-değişim GT türetimi
  ops         ön işleme operasyon kaydı (OPS, 13 kategori) ve hazır hatlar (PRESETS)
  dataset     veri seti dizini, filtreleme ve örnek yükleme
  models      model kaydı, checkpoint yükleme, U-Net / DeepLabV3+ / YOLO / ResNet çıkarımı, gpu_infer
  metrics     piksel ve görüntü düzeyi ölçütler, son işleme, bindirme, eşikler (tau.json, image_tau.json)
  scenarios   girdi senaryoları, bozulmalar (JPEG, ölçekleme, bulanıklık, CLAHE, gri), kırpma ve montaj
  simulation  tek örnek simülasyonu: simulate, insights, rapor dosyaları
  batch       toplu test: batch_test, aggregate, batch_figure
  handlers    diğer sekmelerin olay işleyicileri (girdi, taslaklar, ön işleme, U-Net, ResNet, işaretçi, sonuçlar)
  xai_bridge  XAI köprüsü: SHAP kurulumu, tahmin, açıklama, model karşılaştırma, dışa aktarım (xai/ paketi)
  theme       arayüz teması: CSS, sol panel açıklamaları (TIPS) ve TIP_JS
  ui          Gradio arayüzü: açılış (model ön yükleme, XAI kurulumu), sekmeler ve olay bağlantıları; `demo` nesnesi

Giriş noktası app.py bu modülleri sırayla yükler ve adlarını `app.<ad>` olarak yeniden dışa aktarır."""
MODULES = ('_common', 'config', 'imaging', 'ops', 'dataset', 'models', 'metrics', 'scenarios', 'simulation', 'batch', 'handlers', 'xai_bridge', 'theme', 'ui')
