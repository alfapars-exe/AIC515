"""md2docx.py — basit Markdown → DOCX dönüştürücü (başlık, paragraf, liste, tablo, figür). Kullanım: md_to_docx(md_path, docx_path, base_dir)"""
import re, os
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def _add_inline(par, text):
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.S); text = re.sub(r"\$(.+?)\$", r"\1", text)
    text = text.replace("\\*", "\u2217").replace("\\|", "|")
    for t in re.split(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+?\*)", text):
        if not t: continue
        if t.startswith("**") and t.endswith("**"): r = par.add_run(t[2:-2]); r.bold = True
        elif t.startswith("`") and t.endswith("`"): r = par.add_run(t[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9)
        elif t.startswith("*") and t.endswith("*") and len(t) > 2: r = par.add_run(t[1:-1]); r.italic = True
        else: par.add_run(t)

def md_to_docx(md_path, docx_path, base_dir):
    doc = Document(); st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
    for s in doc.sections: s.left_margin = s.right_margin = Inches(0.9); s.top_margin = s.bottom_margin = Inches(0.9)
    lines = open(md_path, encoding="utf-8").read().splitlines(); i = 0; n = len(lines)
    while i < n:
        ln = lines[i]
        if ln.startswith("```"):
            j = i+1; buf=[]
            while j < n and not lines[j].startswith("```"): buf.append(lines[j]); j += 1
            p = doc.add_paragraph(); r = p.add_run("\n".join(buf)); r.font.name="Consolas"; r.font.size=Pt(8.5); p.paragraph_format.left_indent = Inches(0.3); i = j+1; continue
        m = re.match(r"^(#{1,6})\s+(.*)", ln)
        if m:
            lvl = len(m.group(1)); doc.add_heading(m.group(2).strip(), 0 if lvl == 1 else min(lvl-1, 4)); i += 1; continue
        if ln.strip() == "---": i += 1; continue
        mi = re.match(r"^!\[(.*?)\]\((.*?)\)", ln.strip())
        if mi:
            cap, src = mi.group(1), mi.group(2); path = src if os.path.isabs(src) else os.path.join(base_dir, src)
            if os.path.exists(path):
                try:
                    doc.add_picture(path, width=Inches(6.0)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    c = doc.add_paragraph(); r = c.add_run(cap); r.italic = True; r.font.size = Pt(9); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
                except Exception as e: doc.add_paragraph(f"[Figure could not be embedded: {src} — {e}]")
            else: doc.add_paragraph(f"[Figure not found: {src}]")
            i += 1; continue
        if ln.strip().startswith("|") and i+1 < n and re.match(r"^\s*\|?\s*:?-{2,}", lines[i+1]):
            rows=[]; j=i
            while j < n and lines[j].strip().startswith("|"):
                if not re.match(r"^\s*\|?\s*:?-{2,}", lines[j]): rows.append([c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", re.sub(r"^\||(?<!\\)\|$", "", lines[j].strip()))])
                j += 1
            ncol = max(len(r) for r in rows); tbl = doc.add_table(rows=len(rows), cols=ncol); tbl.style = "Table Grid"
            for ri,row in enumerate(rows):
                for ci in range(ncol):
                    cell = tbl.cell(ri,ci); cell.text=""; par = cell.paragraphs[0]; _add_inline(par, row[ci] if ci < len(row) else "")
                    for r in par.runs: r.font.size = Pt(7.5); r.bold = r.bold or ri == 0
            doc.add_paragraph(); i = j; continue
        mb = re.match(r"^(\s*)[-*]\s+(.*)", ln); mn = re.match(r"^(\s*)(\d+)[.)]\s+(.*)", ln)
        if mb or mn:
            indent = len((mb or mn).group(1)); txt = mb.group(2) if mb else mn.group(3)
            style = ("List Bullet" if mb else "List Number") + (" 2" if indent >= 2 else "")
            p = doc.add_paragraph(style=style); _add_inline(p, txt); i += 1; continue
        if ln.strip() == "": i += 1; continue
        buf=[ln]; j=i+1
        while j < n and lines[j].strip() != "" and not re.match(r"^(#{1,6}\s|```|\||!\[|\s*[-*]\s|\s*\d+[.)]\s|---)", lines[j]): buf.append(lines[j]); j += 1
        p = doc.add_paragraph(); _add_inline(p, " ".join(b.strip() for b in buf)); i = j
    doc.save(docx_path); return docx_path
