"""capture_screens.py — Çalışan uygulamadan rapor için güncel ekran görüntüleri alır (Playwright, Chromium).

Önce uygulamayı başlatın:   $env:APP_NO_BROWSER="1"; $env:APP_PORT="7861"; python kod/app/app.py
Sonra:                      python kod/tools/capture_screens.py [http://127.0.0.1:7861/] [--key training:000000004]

Çıktılar: <proje>/app/ekran_<sekme>_<ad>.png (rapor ve sunum bu adlarla atıf yapar).
Her adım gerçek arayüz üzerinden yürütülür. Bir işlem bitmeden ya da hata verdiğinde görüntü alınmaz:
Gradio'nun durum göstergesi (status-tracker) boşa dönene kadar beklenir; gösterge "Error" yazıyorsa betik durur.
"""
import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.append(str(next(p for p in Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT  # noqa: E402

from playwright.async_api import async_playwright  # noqa: E402

OUT = ROOT / "app"
W, H = 1680, 1050
_TRACK = "[...document.querySelectorAll('[data-testid=\"status-tracker\"]')].filter(e => !e.classList.contains('hide'))"
BUSY_JS = f"{_TRACK}.some(e => !/Error/.test(e.innerText))"
ERROR_JS = f"{_TRACK}.filter(e => /Error/.test(e.innerText)).length"


async def idle(pg, what, min_ms=2500, timeout=900000):
    """Tetiklenen işlem (ve zincirlenmiş olaylar) bitene kadar bekler; hata varsa durur."""
    await pg.wait_for_timeout(min_ms)
    await pg.wait_for_function(f"!({BUSY_JS})", timeout=timeout)
    await pg.wait_for_timeout(1500)
    await pg.wait_for_function(f"!({BUSY_JS})", timeout=timeout)   # .then() ile zincirlenen ikinci olay
    if await pg.evaluate(ERROR_JS):
        raise RuntimeError(f"{what}: arayüz hata verdi — ekran görüntüsü alınmadı (sunucu günlüğüne bakın)")


async def shot(pg, name, max_h=2600):
    """Tam sayfa görüntüsü; çok uzun sayfalar max_h pikselde kesilir (rapor sayfasına sığsın)."""
    await pg.mouse.move(2, 2)                      # açıklama balonlarını kapat
    await pg.wait_for_timeout(1200)
    h = await pg.evaluate("document.documentElement.scrollHeight")
    await pg.evaluate("window.scrollTo(0, 0)")
    hh = max(H, min(h, max_h))
    await pg.screenshot(path=str(OUT / name), full_page=True, clip={"x": 0, "y": 0, "width": W, "height": hh})
    print(f"  ✓ {name}  ({W}×{hh})")


async def tab(pg, name, exact=True):
    await pg.get_by_role("tab", name=name, exact=exact).first.click()
    await pg.wait_for_timeout(900)


async def main(url, key):
    OUT.mkdir(exist_ok=True)
    async with async_playwright() as p:
        br = await p.chromium.launch()
        pg = await br.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        pg.set_default_timeout(240000)
        await pg.goto(url)
        await pg.wait_for_selector("text=Simülasyonu başlat")
        await idle(pg, "açılış", 3000)

        # ---- 0 · Simülasyon: sabit örnek, sahte belge senaryosu
        dd = pg.get_by_label("Örnek (bölüm:kimlik)")
        await dd.click(); await dd.fill(key); await pg.keyboard.press("Enter")
        await idle(pg, "örnek seçimi")
        await pg.get_by_role("button", name="▶ Simülasyonu başlat").click()
        await idle(pg, "simülasyon", 6000)
        await shot(pg, "ekran_0_simulasyon.png", max_h=1900)
        await tab(pg, "Yakın plan")
        await shot(pg, "ekran_0_simulasyon_yakin_plan.png", max_h=1500)
        await tab(pg, "Sonuç tablosu")

        # ---- 3 · Ön işleme: örneği aktar, hazır hattı uygula
        await pg.get_by_role("button", name="Bu örneği ön işleme / model sekmelerine aktar").click()
        await idle(pg, "sekmelere aktarım")
        await tab(pg, "3 · Ön işleme (aşamalı pipeline)")
        await pg.get_by_label("Hazır pipeline (preset)").click(); await pg.wait_for_timeout(600)
        await pg.get_by_role("option", name="Belge ikilileştirme (gri → Gauss → Sauvola → açma)").click(); await pg.wait_for_timeout(600)
        await pg.get_by_role("button", name="Preseti uygula").click()
        await idle(pg, "hazır hat")
        await shot(pg, "ekran_3_on_isleme.png", max_h=1700)
        await pg.get_by_role("button", name="Sıfırla").click()
        await idle(pg, "sıfırlama")

        # ---- 4 · U-Net lokalizasyonu
        await tab(pg, "4 · U-Net lokalizasyonu")
        await pg.get_by_role("button", name="U-Net çalıştır").click()
        await idle(pg, "U-Net", 3500)
        await shot(pg, "ekran_4_unet.png", max_h=1500)

        # ---- 1 · Toplu test (küçük N)
        await tab(pg, "1 · Toplu test")
        sl = pg.get_by_label("Örnek sayısı N", exact=False).first
        await sl.fill("8"); await pg.keyboard.press("Enter"); await pg.wait_for_timeout(600)
        await pg.get_by_role("button", name="▶ Toplu testi başlat").click()
        await pg.get_by_text(re.compile(r"Tamamlandı — \d+ örnek")).wait_for(timeout=900000)
        await idle(pg, "toplu test")
        await shot(pg, "ekran_1_toplu_test.png", max_h=1900)

        # ---- 7 · Sonuçlar ve model kartı
        await tab(pg, "7 · Sonuçlar ve model kartı")
        await shot(pg, "ekran_7_sonuclar.png", max_h=1600)

        # ---- 8 · Açıklanabilirlik › SHAP açıklamaları
        await tab(pg, re.compile(r"^8 · "), exact=False)
        await tab(pg, re.compile(r"SHAP"), exact=False)
        await pg.get_by_role("button", name="① Tahmin yap").click()
        await idle(pg, "SHAP tahmin", 4000)
        await pg.get_by_role("button", name="② SHAP analizini başlat").click()
        await idle(pg, "SHAP analizi", 8000)
        await shot(pg, "ekran_8_xai_shap.png", max_h=2600)

        # ---- 8 · Açıklanabilirlik › Katman filtreleri
        await tab(pg, re.compile(r"Katman Filtreleri"), exact=False)
        await pg.get_by_role("button", name="▶ Filtreleri çıkar ve görüntüye uygula").click()
        await idle(pg, "katman filtreleri", 6000)
        await shot(pg, "ekran_8_xai_filtre.png", max_h=2400)
        await br.close()
    print("Bitti →", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:7861/")
    ap.add_argument("--key", default="training:000000004", help="simülasyonda kullanılacak örnek (bölüm:kimlik)")
    a = ap.parse_args()
    asyncio.run(main(a.url, a.key))
