#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_human_form.py — 100 çiftlik insan GT doğrulaması için kör Excel formu üretir.
Girdi : verification_sheet.csv (VLM kararları İÇERMEZ)
Çıktı : verification_human.xlsx (Form + Yönerge sayfaları, açılır listeli)
Kullanım: python make_human_form.py
"""
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "verification_human.xlsx")
if os.path.exists(OUT):
    raise SystemExit("verification_human.xlsx zaten var; üzerine yazılmadı (doldurulmuş form kaybolmasın).")

d = pd.read_csv(os.path.join(HERE, "verification_sheet.csv"), dtype=str).fillna("")
info = ["new_id", "page", "tool", "source", "area_bin", "auto_flags"]
fields = [  # (sütun, izinli değerler, başlık açıklaması)
    ("bbox_verdict", "OK,PARTIAL,WRONG", "Kırmızı kutu değişen içeriği kapsıyor mu?"),
    ("eps8_verdict", "OK,PARTIAL,WRONG", "pixel_eps8 maskesi değişen içeriği doğru işaretliyor mu?"),
    ("glyph_v2_verdict", "OK,PARTIAL,WRONG", "glyph_v2 maskesi değişen glifleri doğru işaretliyor mu?"),
    ("outside_change", "no,yes,unclear", "Kutu DIŞINDA gerçek içerik değişimi var mı?"),
    ("green_marker", "no,yes,unsure", "Sahte kırpıntıda ince yeşil çizgi/çerçeve görünüyor mu?"),
    ("changed_text", None, "(isteğe bağlı) Ne değişmiş? ör. 57,000 → 132,438"),
    ("note", None, "(isteğe bağlı) not"),
    ("seconds", None, "(isteğe bağlı) bu çifte harcanan süre, sn"),
]

wb = Workbook()
ws = wb.active
ws.title = "Form"
hdr = info + [f[0] for f in fields]
ws.append(hdr)
for _, r in d.iterrows():
    ws.append([r[c] for c in info] + [""] * len(fields))
bold = Font(bold=True)
fill = PatternFill("solid", fgColor="FFF2CC")
for j, h in enumerate(hdr, 1):
    c = ws.cell(row=1, column=j); c.font = bold
    if j > len(info):
        c.fill = fill
        ws.cell(row=1, column=j).comment = None
n = len(d) + 1
for j, (name, allowed, desc) in enumerate(fields, len(info) + 1):
    col = ws.cell(row=1, column=j).column_letter
    ws.column_dimensions[col].width = max(14, len(name) + 2)
    if allowed:
        dv = DataValidation(type="list", formula1='"%s"' % allowed, allow_blank=True, showErrorMessage=True,
                            errorTitle="Geçersiz değer", error="Listeden seçin: " + allowed)
        dv.prompt = desc; dv.promptTitle = name; dv.showInputMessage = True
        ws.add_data_validation(dv); dv.add("%s2:%s%d" % (col, col, n))
for col, w in zip("ABCDEF", [12, 6, 13, 12, 9, 34]):
    ws.column_dimensions[col].width = w
ws.freeze_panes = "G2"

g = wb.create_sheet("Yönerge")
lines = [
 "İNSAN GT DOĞRULAMASI — YÖNERGE (AIForge-Doc v1, 100 training çifti, tabakalı örneklem seed=42)",
 "",
 "Malzeme: verification_grid.pdf (20 sayfa × 5 çift; sütunlar: sahte | authentic | bbox GT | pixel_eps8 | glyph | glyph_v2).",
 "Fark görmek zorsa: vlm_verification\\<new_id>.png → C paneli |sahte−authentic|×4 farkı gösterir (bu PNG'lerde model kararı YOK).",
 "KÖR KURAL: verification_sheet_vlm.csv ve vlm_verification_summary.json dosyalarını bitirene kadar AÇMAYIN.",
 "",
 "Önce sahte ile authentic'i karşılaştırıp neyin değiştiğini kendiniz bulun; maskelere SONRA bakın.",
 "",
 "Maske kararları (bbox_verdict, eps8_verdict, glyph_v2_verdict):",
 "  OK      = değişen karakterlerin neredeyse tamamı (~%90+) işaretli; değişmemiş bölgede anlamlı yanlış işaret yok (tek tük nokta gürültüsü sayılmaz).",
 "  PARTIAL = değişimi buluyor ama en az bir karakteri/parçayı kaçırıyor YA DA değişmemiş metin/arka planda belirgin yanlış bölge var.",
 "  WRONG   = değişimin çoğunu kaçırıyor, büyük ölçüde yanlış yeri işaretliyor ya da boş.",
 "  bbox için: kutu değişen içeriği tam kapsıyorsa OK (içinde değişmemiş boşluk olması normal, ceza değil); bir karakter dışarıda kalıyorsa PARTIAL.",
 "",
 "outside_change: kutu dışında gerçek İÇERİK değişimi (rakam/harf/çizgi) varsa yes; yalnız sıkıştırma/renk gürültüsü ise no; karar veremiyorsanız unclear.",
 "green_marker : sahte kırpıntıda kutu çevresinde ince yeşil çizgi/çerçeve kalıntısı görüyorsanız yes.",
 "",
 "Kalibrasyon: ilk 2 sayfayı (10 çift) yapın, kuralları gerekirse netleştirin, sonra o 10'u bir kez gözden geçirip devam edin. Kuralları sonradan değiştirmeyin.",
 "Süre: çift başına ~40–60 sn → toplam ~1–1.5 saat. 2–3 oturuma bölün.",
 "Bitince: python score_human_verification.py  (eksik/geçersiz hücreleri listeler, VLM ile uyumu hesaplar).",
 "",
 "İsteğe bağlı ikinci değerlendirici: dosyayı verification_human_r2.xlsx olarak kopyalayıp aynı yönergeyle (ilk değerlendirmeyi görmeden) doldurtun; betik insan–insan κ'yı da hesaplar.",
]
for L in lines:
    g.append([L])
g.column_dimensions["A"].width = 160
g["A1"].font = Font(bold=True, size=12)
for row in g.iter_rows():
    for c in row:
        c.alignment = Alignment(wrap_text=False)
wb.save(OUT)
print("yazıldı:", OUT, "satır:", len(d))
