"""build_report.py — Dönem içi proje raporunu Markdown'dan Word (.docx) ve PDF'e çevirir.

    python kod/tools/build_report.py                      # reports/Donem_Ici_Proje_Raporu.md → .docx + .pdf
    python kod/tools/build_report.py --md reports/X.md    # başka bir rapor
    python kod/tools/build_report.py --no-pdf             # yalnız .docx

Hat: pandoc (pypandoc_binary) → .docx (denklemler Word'ün yerel denklem biçiminde, şekil/tablo başlıkları, içindekiler)
     → python-docx ile biçim düzeltmeleri (tablo kenarlıkları, ortalanmış şekiller, kod bloğu yazı tipi)
     → Microsoft Word (COM) ile içindekiler alanının güncellenmesi ve PDF dışa aktarımı.
Görsel yolları proje köküne göredir (ör. `reports/figures/x.png`); eksik görsel varsa üretim durur.
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.append(str(next(p for p in Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT  # noqa: E402

import pypandoc  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

BODY_FONT, MONO_FONT = "Times New Roman", "Consolas"
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def check_images(md_text: str):
    missing = [p for p in IMG_RE.findall(md_text) if not (ROOT / p).is_file()]
    return sorted(set(missing))


def _set_font(style, name, size=None, bold=None, color=None):
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), name)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = RGBColor(*color)


def make_reference_doc(path: Path):
    """pandoc'un varsayılan şablonundan rapor biçemli bir başvuru belgesi üretir."""
    raw = subprocess.run([pypandoc.get_pandoc_path(), "--print-default-data-file", "reference.docx"], capture_output=True, check=True).stdout
    path.write_bytes(raw)
    d = Document(str(path))
    for s in d.sections:
        s.page_width, s.page_height = Cm(21.0), Cm(29.7)
        s.left_margin = s.right_margin = Cm(2.5); s.top_margin = s.bottom_margin = Cm(2.5)
    st = d.styles
    sizes = {"Normal": 11, "Body Text": 11, "First Paragraph": 11, "Compact": 10, "Title": 18, "Subtitle": 13, "Author": 12, "Date": 11,
             "Heading 1": 15, "Heading 2": 13, "Heading 3": 11.5, "Heading 4": 11, "Table Caption": 10, "Image Caption": 10, "Caption": 10,
             "Captioned Figure": 10, "Block Text": 10.5, "TOC Heading": 14, "Bibliography": 10, "Abstract": 10.5}
    for name, size in sizes.items():
        if name in st:
            head = name.startswith("Heading") or name in ("Title", "TOC Heading")
            _set_font(st[name], BODY_FONT, size, bold=True if head else None, color=(0, 0, 0) if head or "Caption" in name else None)
    for name in ("Body Text", "First Paragraph", "Normal"):
        if name in st:
            pf = st[name].paragraph_format
            pf.space_after = Pt(6); pf.space_before = Pt(0); pf.line_spacing = 1.15
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT if name == "Normal" else WD_ALIGN_PARAGRAPH.JUSTIFY
    for name in ("Table Caption", "Image Caption"):
        if name in st:
            pf = st[name].paragraph_format
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER; pf.space_before = Pt(4); pf.space_after = Pt(8)
            st[name].font.italic = False
    if "Table Caption" in st:
        st["Table Caption"].paragraph_format.keep_with_next = True
    for name in ("Source Code", "Verbatim Char"):
        if name in st:
            _set_font(st[name], MONO_FONT, 8.5)
    if "Source Code" in st:
        pf = st["Source Code"].paragraph_format
        pf.space_after = Pt(6); pf.line_spacing = 1.0; pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        if name in st:
            pf = st[name].paragraph_format
            pf.space_before = Pt(14 if name == "Heading 1" else 10); pf.space_after = Pt(6); pf.keep_with_next = True
    d.save(str(path))


def _borders(tbl):
    tblPr = tbl._tbl.tblPr
    b = tblPr.find(qn("w:tblBorders"))
    if b is not None:
        tblPr.remove(b)
    b = OxmlElement("w:tblBorders")
    for edge, sz in (("top", 8), ("bottom", 8), ("insideH", 4), ("left", 0), ("right", 0), ("insideV", 0)):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single" if sz else "nil"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "404040")
        b.append(e)
    tblPr.append(b)


def move_toc_after_front_matter(d):
    """pandoc içindekiler tablosunu belgenin en başına koyar; künye ve özetten sonraya (ilk numaralı bölümün önüne) alınır."""
    body = d.element.body
    toc = next((el for el in body if el.tag == qn("w:sdt")), None)
    target = next((p._p for p in d.paragraphs if p.style is not None and p.style.name.startswith("Heading") and re.match(r"^1[.\s]", p.text.strip())), None)
    if toc is None or target is None:
        return
    body.remove(toc)
    target.addprevious(toc)
    brk = OxmlElement("w:p"); r = OxmlElement("w:r"); b = OxmlElement("w:br"); b.set(qn("w:type"), "page"); r.append(b); brk.append(r)
    target.addprevious(brk)          # içindekilerden sonra yeni sayfa
    brk2 = OxmlElement("w:p"); r2 = OxmlElement("w:r"); b2 = OxmlElement("w:br"); b2.set(qn("w:type"), "page"); r2.append(b2); brk2.append(r2)
    toc.addprevious(brk2)            # künye + özet ayrı sayfada


def postprocess(docx_path: Path):
    d = Document(str(docx_path))
    move_toc_after_front_matter(d)
    for t in d.tables:
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        _borders(t)
        ncol = len(t.columns)
        size = 9 if ncol <= 6 else 8
        for ri, row in enumerate(t.rows):
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.space_after = Pt(1); p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for r in p.runs:
                        r.font.size = Pt(size)
                        if ri == 0:
                            r.font.bold = True
        hdr = t.rows[0]._tr.get_or_add_trPr()   # başlık satırı sayfa bölünmesinde yinelensin
        if hdr.find(qn("w:tblHeader")) is None:
            hdr.append(OxmlElement("w:tblHeader"))
    for p in d.paragraphs:
        if p.style is not None and p.style.name == "Source Code":   # kod blokları: sola yaslı, eş aralıklı
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.line_spacing = 1.0
            for r in p.runs:
                r.font.name = MONO_FONT; r.font.size = Pt(8.5)
        if p._p.findall(".//" + qn("w:drawing")):
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
    d.save(str(docx_path))


PS_PDF = r'''
param([string]$Docx, [string]$Pdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false; $word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($Docx, $false, $false)
  foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
  $doc.Fields.Update() | Out-Null
  $doc.Save()
  $doc.ExportAsFixedFormat($Pdf, 17)
  "sayfa: " + $doc.ComputeStatistics(2)
  $doc.Close($false)
} finally {
  if ($word.Documents.Count -eq 0) { $word.Quit() }
  [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
'''


def to_pdf(docx_path: Path, pdf_path: Path):
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8-sig") as f:
        f.write(PS_PDF); ps = f.name
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps, "-Docx", str(docx_path), "-Pdf", str(pdf_path)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    Path(ps).unlink(missing_ok=True)
    if r.returncode != 0 or not pdf_path.is_file():
        raise RuntimeError("Word ile PDF üretilemedi:\n" + (r.stderr or r.stdout))
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--md", default=str(ROOT / "reports" / "Donem_Ici_Proje_Raporu.md"))
    ap.add_argument("--out", default=None, help="çıktı .docx (varsayılan: .md ile aynı ad)")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-toc", action="store_true")
    a = ap.parse_args()
    md = Path(a.md) if Path(a.md).is_absolute() else ROOT / a.md
    out = Path(a.out) if a.out else md.with_suffix(".docx")
    text = md.read_text(encoding="utf-8")
    missing = check_images(text)
    if missing:
        print("EKSİK GÖRSEL (üretim durdu):"); [print("  ", m) for m in missing]; sys.exit(2)
    ref = Path(tempfile.gettempdir()) / "ai515_reference.docx"
    make_reference_doc(ref)
    args = ["--standalone", f"--resource-path={ROOT}", f"--reference-doc={ref}", "--wrap=none", "--dpi=200"]
    if not a.no_toc:
        args += ["--toc", "--toc-depth=3", "--metadata=toc-title:İçindekiler"]
    fmt = "markdown+pipe_tables+table_captions+implicit_figures+tex_math_dollars+fenced_code_blocks-yaml_metadata_block-smart"
    try:
        pypandoc.convert_text(text, "docx", format=fmt, outputfile=str(out), extra_args=args)
    except PermissionError:
        out = out.with_name(out.stem + "_yeni.docx")
        pypandoc.convert_text(text, "docx", format=fmt, outputfile=str(out), extra_args=args)
        print("UYARI: hedef .docx açık/kilitli; şu dosyaya yazıldı:", out.name)
    postprocess(out)
    print(f"docx: {out}  ({out.stat().st_size / 1e6:.1f} MB) · şekil {len(IMG_RE.findall(text))} · tablo başlığı {len(re.findall(r'^Table: ', text, flags=re.M))}")
    if not a.no_pdf:
        pdf = out.with_suffix(".pdf")
        info = to_pdf(out, pdf)
        print(f"pdf:  {pdf}  ({pdf.stat().st_size / 1e6:.1f} MB) · {info}")


if __name__ == "__main__":
    main()
