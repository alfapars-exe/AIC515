#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""partial_note_check.py — ikinci GT denetiminde piksel-değişim referansı PARTIAL sayılan çiftlerin notlarını sınıflar.
Kullanım: python partial_note_check.py [--form verification_human.xlsx]
Çıktı : partial_note_check.json (+ ekrana özet)

Amaç: bildirideki "in all 64 the auditor's note describes over-coverage" cümlesinin kaynağını dosyaya bağlamak.
Bir not, eps8 maskesinin (a) arka planı / yamayı doldurduğunu ya da (b) değişmeyen içeriği de işaretlediğini söylüyorsa
"fazla kapsama" sayılır. "Eksik kapsama" sözcükleri ayrıca aranır ve geçtikleri notlar elle okunmak üzere listelenir.
Sınıflama anahtar sözcüklerle yapılır; notlar serbest metindir, bu yüzden sonuç notların okunmasıyla birlikte değerlendirilmelidir.
"""
import argparse, json, os
import pandas as pd

import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import out_dir as _out_dir  # çıktı klasörü
HERE = str(_out_dir(__file__))

OVER_BG = r"arka plan|yama|dolduruyor|dolu dikdörtgen|blok"          # (a) arka plan / yama doldurma
OVER_SAME = r"değişmeyen|aynı kalan|korunmuş|değişmemiş"              # (b) değişmeyen içerik
UNDER = r"eksik|kaçır|yakalamıyor|atlıyor|dışarıda bırak|kapsamıyor"  # eksik kapsama sözcükleri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", default=os.path.join(HERE, "verification_human.xlsx"))
    args = ap.parse_args()
    f = pd.read_excel(args.form, sheet_name="Form", dtype=str).fillna("")
    p = f[f["eps8_verdict"].str.strip().str.upper() == "PARTIAL"].copy()
    low = p["note"].str.lower()
    a, b, u = low.str.contains(OVER_BG), low.str.contains(OVER_SAME), low.str.contains(UNDER)
    out = {"form": os.path.basename(args.form), "n_partial": int(len(p)),
           "over_background": int(a.sum()), "over_unchanged_content": int(b.sum()),
           "over_any": int((a | b).sum()), "neither": int((~(a | b)).sum()),
           "under_keyword_notes": [{"new_id": r["new_id"].zfill(9), "source": r["source"], "note": r["note"]} for _, r in p[u].iterrows()],
           "by_source": {s: {"n_partial": int((p["source"] == s).sum()), "over_any": int(((a | b) & (p["source"] == s)).sum())}
                         for s in sorted(p["source"].unique())},
           "patterns": {"over_background": OVER_BG, "over_unchanged_content": OVER_SAME, "under": UNDER},
           "note": "Anahtar sözcük sınıflaması. 'under_keyword_notes' içindeki notlar elle okunmalıdır; "
                   "10 Kasım 2026 okumasında tek eşleşme glyph_v2 maskesine aittir, eps8'e değil."}
    json.dump(out, open(os.path.join(HERE, "partial_note_check.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("PARTIAL: %d · arka plan/yama: %d · değişmeyen içerik: %d · en az biri: %d · hiçbiri: %d · eksik-kapsama sözcüğü geçen not: %d" % (
        out["n_partial"], out["over_background"], out["over_unchanged_content"], out["over_any"], out["neither"], len(out["under_keyword_notes"])))


if __name__ == "__main__":
    main()
