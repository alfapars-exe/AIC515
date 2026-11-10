"""Playwright UI testi: Simülasyon + Toplu test + aktarım. Kullanım: python3 tests/ui_test.py http://127.0.0.1:7871/ out_prefix"""
import asyncio, sys, time, re
from playwright.async_api import async_playwright

import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7871/"
OUT = sys.argv[2] if len(sys.argv) > 2 else str(_out_dir(__file__) / "ui")
ERR = []


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(viewport={"width": 1600, "height": 1100}, proxy={"server": "http://127.0.0.1:1", "bypass": "127.0.0.1,localhost"})
        pg = await ctx.new_page()
        pg.on("pageerror", lambda e: ERR.append(f"pageerror: {e}"))
        await pg.goto(URL, wait_until="load", timeout=120000)
        await pg.wait_for_selector("text=Simülasyonu başlat", timeout=120000)
        # 1) filtre: yalnız eşli → rastgele
        await pg.get_by_label("Yalnız authentic eşi olanlar").check()
        await pg.get_by_role("button", name="🎲 Rastgele seç").click()
        await pg.wait_for_selector("text=rastgele seçildi", timeout=60000)
        await pg.wait_for_timeout(2500)
        await pg.screenshot(path=f"{OUT}_1_secim.png")
        # 2) senaryo: kısayol testi, simülasyon
        await pg.get_by_label("Sahte belge (forged)").check()
        t = time.time()
        await pg.get_by_role("button", name="▶ Simülasyonu başlat").click()
        await pg.wait_for_selector("text=7/7 Rapor hazır", timeout=300000)
        await pg.wait_for_timeout(2500)
        print(f"simülasyon UI süresi {time.time() - t:.1f}s")
        await pg.screenshot(path=f"{OUT}_2_simulasyon.png", full_page=True)
        txt = await pg.inner_text("body")
        for must in ("Gerçek etiket", "Yorum", "Kutu davranışı", "U-Net · bbox-GT (s42)"):
            if must not in txt:
                ERR.append(f"eksik metin: {must}")
        # 3) kısayol senaryosu
        await pg.get_by_label("Authentic + yalnız yeşil işaretçi (kısayol testi)").check()
        await pg.get_by_role("button", name="▶ Simülasyonu başlat").click()
        await pg.wait_for_timeout(1500)
        await pg.wait_for_selector("text=7/7 Rapor hazır", timeout=300000)
        await pg.wait_for_timeout(2000)
        txt = await pg.inner_text("body")
        if "Kısayol testi" not in txt:
            ERR.append("kısayol yorumu yok")
        await pg.screenshot(path=f"{OUT}_3_kisayol.png", full_page=True)
        # 4) aktar → U-Net sekmesi
        await pg.get_by_role("button", name="Bu örneği ön işleme / model sekmelerine aktar").click()
        await pg.wait_for_timeout(2000)
        await pg.get_by_role("tab", name="3 · Ön işleme (aşamalı pipeline)").click()
        await pg.get_by_label("Hazır pipeline (preset)").click(); await pg.wait_for_timeout(500)
        await pg.get_by_role("option", name="Belge ikilileştirme (gri → Gauss → Sauvola → açma)").click(); await pg.wait_for_timeout(500)
        await pg.get_by_role("button", name="Preseti uygula").click()
        await pg.wait_for_selector("text=Preset uygulandı", timeout=60000)
        await pg.wait_for_timeout(1500)
        await pg.screenshot(path=f"{OUT}_4_onisleme.png", full_page=True)
        # 5) toplu test
        await pg.get_by_role("tab", name="1 · Toplu test").click()
        sl = pg.get_by_label("Örnek sayısı N", exact=False).first
        await sl.fill("6"); await sl.press("Enter")
        t = time.time()
        await pg.get_by_role("button", name="▶ Toplu testi başlat").click()
        await pg.get_by_text(re.compile(r"Tamamlandı — \d+ örnek")).wait_for(timeout=900000)
        await pg.wait_for_timeout(2500)
        print(f"toplu test UI süresi {time.time() - t:.1f}s")
        await pg.screenshot(path=f"{OUT}_5_toplu.png", full_page=True)
        # 6) sonuçlar sekmesi
        await pg.get_by_role("tab", name="7 · Sonuçlar ve model kartı").click()
        await pg.wait_for_timeout(1000)
        txt = await pg.inner_text("body")
        if "Youden" not in txt:
            ERR.append("eşik tablosu yok")
        await b.close()

asyncio.run(main())
print("UI HATALARI:", ERR if ERR else "yok")
sys.exit(1 if ERR else 0)
