"""check_refs_dblp.py — Bildiride atıf yapılan kaynakların künyesini DBLP ile karşılaştırır (Lab Rat paper-check, 5. denetim).

Her atıf için DBLP yayın aramasında başlık aranır; başlık, yazar soyadları (sıra dahil), yıl ve yayın yeri karşılaştırılır.
DBLP'de bulunmayan kaynaklar için DOI varsa Crossref, arXiv numarası varsa arXiv API'si kullanılır.
Ayrıca ön baskı olarak atıf yapılan bir kaynağın DBLP'de hakemli sürümü varsa bildirilir.
Yalnız başlık ve DOI/arXiv numarası gönderilir; kişisel bilgi gönderilmez.

DBLP'nin arama API'si 2026'da bot denetimi (tarayıcıda çözülen bir doğrulama sayfası) ve sıkı bir istek sınırı arkasındadır:
betikten yapılan sorgu yanıt alamaz; tarayıcı oturumundan 1,3 s arayla yapılan sorgular üçüncü istekten sonra 429 döndürür.
Bu yüzden varsayılan kip DBLP'siz çalışır (--no-dblp ile aynı): DOI'li kayıtlar Crossref, arXiv kayıtları arXiv API'si ile
denetlenir; ön baskıların hakemli sürümü ve DOI'siz kayıtlar Crossref başlık aramasıyla aranır. DBLP kayıtları elle
(tarayıcıdan) alınmışsa --dblp-json ile verilebilir. Dosya biçimi:
{anahtar: {"m": [{"t": başlık, "y": yıl, "v": yer, "ty": C|J|I|P|B, "a": [yazarlar], "d": doi}]}}.

Kullanım:  python kod/tools/check_refs_dblp.py [--dblp-live | --dblp-json dosya.json] [--out ...md] [--all]
Çıktı:     özet konsola, ayrıntılı tablo --out dosyasına. --all: atıf yapılmayan bib kayıtları da denetlenir.
"""
from __future__ import annotations
import argparse
import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.append(str(next(p for p in Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT  # noqa: E402

UA = {"User-Agent": "AIC515-refcheck/1.0 (bibliography verification script)"}
DBLP_CACHE = None   # --dblp-json ile yüklenir
DBLP_MODE = "off"   # off | live | json
_TY = {"C": "Conference and Workshop Papers", "J": "Journal Articles", "I": "Informal and Other Publications",
       "P": "Parts in Books or Collections", "B": "Books and Theses", "E": "Editorship"}


def get(url: str, tries: int = 4, pause: float = 1.3):
    """GET; 429/5xx durumunda bekleyip yeniden dener. (metin, hata) döndürür."""
    err = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                txt = r.read().decode("utf-8", "replace")
            time.sleep(pause)
            return txt, None
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}"
            if e.code == 404:
                return None, err
            wait = float(e.headers.get("Retry-After") or 0) if e.code == 429 else 0
            time.sleep(max(wait, 4 * (k + 1)))
        except Exception as e:  # ağ hatası
            err = type(e).__name__
            time.sleep(4 * (k + 1))
    return None, err


def delatex(s: str) -> str:
    s = re.sub(r"\\[`'^\"~=.uvHtcdbk]\s*\{?\\?([A-Za-z])\}?", r"\1", s)   # aksanlar
    s = s.replace(r"\ss", "ss").replace(r"\o", "o").replace(r"\O", "O").replace(r"\l", "l").replace(r"\L", "L").replace(r"\i", "i")
    s = s.replace(r"\&", "&").replace("~", " ").replace("--", "-")
    s = re.sub(r"\\[A-Za-z]+\s*", "", s)
    return re.sub(r"[{}$]", "", s).strip()


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def surname(name: str) -> str:
    name = delatex(name).strip()
    if "," in name:
        return norm(name.split(",")[0])
    name = re.sub(r"\s+\d{4}$", "", name)          # DBLP ad ayırıcı son eki (0001)
    parts = name.split()
    return norm(parts[-1]) if parts else ""


def parse_bib(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", txt):
        typ, key = m.group(1).lower(), m.group(2)
        i, depth, start = m.end(), 1, m.end()
        while i < len(txt) and depth:
            depth += {"{": 1, "}": -1}.get(txt[i], 0); i += 1
        body = txt[start:i - 1]
        f = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            name, j = fm.group(1).lower(), fm.end()
            if j >= len(body):
                continue
            if body[j] == "{":
                d, k = 1, j + 1
                while k < len(body) and d:
                    d += {"{": 1, "}": -1}.get(body[k], 0); k += 1
                val = body[j + 1:k - 1]
            elif body[j] == '"':
                k = body.index('"', j + 1); val = body[j + 1:k]
            else:
                val = re.match(r"[^,\n]+", body[j:]).group(0)
            f.setdefault(name, " ".join(val.split()))
        head = txt[:m.start()].rstrip("\n").split("\n")
        src = []
        while head and head[-1].lstrip().startswith("%"):
            src.insert(0, head.pop().lstrip("% ").strip())
        out[key] = dict(type=typ, _src=" ".join(src), **f)
    return out


def cited_keys(tex_dir: Path) -> list:
    keys = []
    for p in [tex_dir / "main.tex", *sorted((tex_dir / "sec").glob("*.tex"))]:
        t = re.sub(r"(?<!\\)%.*", "", p.read_text(encoding="utf-8"))
        for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", t):
            for k in m.group(1).split(","):
                k = k.strip()
                if k and k not in keys:
                    keys.append(k)
    return keys


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def dblp_search(title: str, key: str = ""):
    if DBLP_MODE == "off":
        return [], None
    if DBLP_CACHE is not None:
        c = DBLP_CACHE.get(key)
        if c is None:
            return None, "kayıt dosyasında yok"
        if "err" in c:
            return None, f"tarayıcı sorgusu hata: {c['err']}"
        return [dict(title=h["t"], authors=h["a"], year=h["y"], venue=h["v"], type=_TY.get(h["ty"], "Diğer"), doi=h.get("d", ""), url="")
                for h in c.get("m", [])], None
    q = " ".join(w for w in norm(title).split() if len(w) > 1)[:180]
    txt, err = get("https://dblp.org/search/publ/api?" + urllib.parse.urlencode({"q": q, "format": "json", "h": 12}))
    if txt is None:
        return None, err
    hits = json.loads(txt).get("result", {}).get("hits", {}).get("hit", []) or []
    res = []
    for h in hits:
        i = h.get("info", {})
        au = i.get("authors", {}).get("author", [])
        au = [au] if isinstance(au, dict) else au
        ven = i.get("venue", "")
        res.append(dict(title=str(i.get("title", "")).rstrip("."), authors=[a.get("text", "") for a in au], year=str(i.get("year", "")),
                        venue=" / ".join(ven) if isinstance(ven, list) else str(ven), type=i.get("type", ""), doi=str(i.get("doi", "")), url=i.get("url", "")))
    return res, None


def crossref(doi: str):
    txt, err = get("https://api.crossref.org/works/" + urllib.parse.quote(doi), pause=0.6)
    if txt is None:
        return None, err
    m = json.loads(txt).get("message", {})
    yr = (m.get("published-print") or m.get("published-online") or m.get("issued") or {}).get("date-parts", [[None]])[0][0]
    return dict(title=(m.get("title") or [""])[0], authors=[f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author", [])],
                year=str(yr or ""), venue=(m.get("container-title") or [""])[0], type=m.get("type", ""), doi=doi, url="https://doi.org/" + doi), None


def datacite(doi: str):
    txt, err = get("https://api.datacite.org/dois/" + urllib.parse.quote(doi), pause=0.6)
    if txt is None:
        return None, err
    a = json.loads(txt).get("data", {}).get("attributes", {})
    return dict(title=(a.get("titles") or [{}])[0].get("title", ""), authors=[c.get("name", "") for c in a.get("creators", [])], year=str(a.get("publicationYear") or ""),
                venue=str(a.get("publisher") or ""), type=(a.get("types") or {}).get("resourceTypeGeneral", ""), doi=doi, url=""), None


def doi_registered(doi: str) -> bool:
    txt, _ = get("https://doi.org/api/handles/" + urllib.parse.quote(doi), pause=0.4)
    return txt is not None


def crossref_title(title: str):
    """Crossref başlık araması: başlığı eşleşen hakemli kayıtlar (arXiv/SSRN ön baskı DOI'leri hariç)."""
    txt, err = get("https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.title": title, "rows": 6, "select": "title,type,container-title,DOI,issued,published-print,published-online,author"}), pause=0.8)
    if txt is None:
        return None, err
    out = []
    for m in json.loads(txt).get("message", {}).get("items", []):
        t = (m.get("title") or [""])[0]
        if sim(title, t) < 0.93 or m.get("type") in ("posted-content", "dataset", "component", "peer-review", "report"):
            continue
        if str(m.get("DOI", "")).lower().startswith(("10.48550", "10.2139", "10.36227", "10.21203", "10.1101", "10.20944")):
            continue
        yr = (m.get("published-print") or m.get("published-online") or m.get("issued") or {}).get("date-parts", [[None]])[0][0]
        out.append(dict(title=t, authors=[f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author", [])], year=str(yr or ""),
                        venue=(m.get("container-title") or [""])[0], type=m.get("type", ""), doi=m.get("DOI", ""), url=""))
    return out, None


def arxiv(eid: str):
    txt, err = get("http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": eid}), pause=3.1)
    if txt is None:
        return None, err
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    e = ET.fromstring(txt).find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        return None, "kayıt yok"
    jr, pd_ = e.find("x:journal_ref", ns), e.find("x:doi", ns)
    return dict(title=" ".join(e.find("a:title", ns).text.split()), authors=[x.find("a:name", ns).text for x in e.findall("a:author", ns)],
                year=(e.find("a:published", ns).text or "")[:4], venue="arXiv", type="preprint", doi="", url="https://arxiv.org/abs/" + eid,
                version=(e.find("a:id", ns).text or "").rsplit("v", 1)[-1], journal_ref=(jr.text if jr is not None else ""), pub_doi=(pd_.text if pd_ is not None else "")), None


def compare(entry: dict, rec: dict) -> list:
    """Fark listesi (boşsa künye uyumlu)."""
    d = []
    if sim(delatex(entry.get("title", "")), rec["title"]) < 0.93:
        d.append(f"başlık: kayıt «{rec['title']}»")
    ba = [surname(a) for a in re.split(r"\s+and\s+", entry.get("author", "")) if a.strip() and norm(a) != "others"]
    ra = [surname(a) for a in rec["authors"]]
    etal = "others" in norm(entry.get("author", ""))
    if rec["authors"] and ((ba != ra[:len(ba)]) if etal else (ba != ra)):
        if sorted(ba) == sorted(ra):
            d.append("yazar sırası farklı")
        elif len(ba) != len(ra) and not etal:
            d.append(f"yazar sayısı: bib {len(ba)}, kayıt {len(ra)}")
        else:
            diff = [f"{x}≠{y}" for x, y in zip(ba, ra) if x != y][:3]
            d.append("yazar adı: " + ", ".join(diff))
    return d


def check(key: str, e: dict) -> dict:
    title = delatex(e.get("title", ""))
    year = e.get("year", "")
    is_pre = e["type"] == "misc" and ("eprint" in e or "arxiv" in e.get("url", "").lower())
    row = dict(key=key, type=e["type"], year=year, source="", status="", notes=[])
    hits, err = dblp_search(title, key)
    if hits is None:
        row["notes"].append(f"DBLP sorgusu başarısız ({err})")
        hits = []
    good = [h for h in hits if sim(title, h["title"]) >= 0.93]
    formal = [h for h in good if h["type"] in ("Conference and Workshop Papers", "Journal Articles", "Parts in Books or Collections", "Books and Theses")]
    informal = [h for h in good if h not in formal]
    pick = None
    if not is_pre and formal:
        same = [h for h in formal if h["year"] == year]
        pick = (same or formal)[0]
    elif is_pre and informal:
        pick = informal[0]
    elif good:
        pick = good[0]
    if pick is not None:
        row["source"] = "DBLP"
        diffs = compare(e, pick)
        if not is_pre and pick["year"] != year:
            diffs.append(f"yıl: bib {year}, DBLP {pick['year']}")
        if is_pre and pick in informal and year and pick["year"] and abs(int(pick["year"]) - int(year)) > 1:
            diffs.append(f"yıl: bib {year}, DBLP {pick['year']}")
        if not is_pre and pick not in formal:
            row["notes"].append("DBLP'de yalnız ön baskı kaydı var; hakemli sürüm DOI/yayıncı üzerinden doğrulanmalı")
        if is_pre and formal:
            f0 = formal[0]
            row["notes"].append(f"hakemli sürümü var: {f0['venue']} {f0['year']}")
        row["status"] = "uyumlu" if not diffs else "fark"
        row["notes"] = diffs + row["notes"]
        row["record"] = f"{pick['venue']} {pick['year']}".strip()
        if row["status"] == "uyumlu" and not (not is_pre and pick not in formal):
            return row
    # DBLP'de yok ya da hakemli kayıt yok: Crossref / arXiv
    rec, src = None, ""
    if e.get("doi"):
        rec, err = crossref(e["doi"]); src = "Crossref"
        if rec is None and err == "HTTP 404":          # Crossref dışı kayıt ajansı (DataCite: Zenodo vb.)
            rec, err = datacite(e["doi"]); src = "DataCite"
            if rec is None:
                src = "Crossref/DataCite"
                if doi_registered(e["doi"]):
                    row["notes"].append("DOI doi.org'da kayıtlı; kayıt ajansı Crossref ve DataCite dışında")
    elif e.get("eprint"):
        rec, err = arxiv(e["eprint"]); src = "arXiv"
    if rec is not None:
        diffs = compare(e, rec)
        if rec["year"] and year and rec["year"] != year and not (src == "Crossref" and abs(int(rec["year"]) - int(year)) <= 1 and e["type"] == "inproceedings"):
            diffs.append(f"yıl: bib {year}, {src} {rec['year']}")
        elif rec["year"] and year and rec["year"] != year:
            row["notes"].append(f"{src} yılı {rec['year']} (basım yılı; bildiri kitabı yılı {year})")
        if pick is None:
            row["source"], row["status"] = src, ("uyumlu" if not diffs else "fark")
            row["notes"] = diffs + row["notes"]
            row["record"] = f"{rec['venue']} {rec['year']}".strip() + (f" (v{rec['version']})" if rec.get("version") else "")
            if rec.get("journal_ref") or rec.get("pub_doi"):
                row["notes"].append("arXiv kaydında yayın bilgisi: " + " ".join(x for x in (rec.get("journal_ref"), rec.get("pub_doi")) if x))
        else:
            row["source"] = f"DBLP + {src}"
            if not diffs and row["status"] == "uyumlu":
                row["notes"] = [n for n in row["notes"] if not n.startswith("DBLP'de yalnız")] + [f"hakemli sürüm {src} ile doğrulandı"]
            row["notes"] += [f"{src}: {x}" for x in diffs]
            if diffs:
                row["status"] = "fark"
    elif pick is None:
        cr, err2 = (crossref_title(title) if not src else (None, None))
        if cr:
            rec = cr[0]; diffs = compare(e, rec)
            if rec["year"] and year and abs(int(rec["year"]) - int(year)) > 1:
                diffs.append(f"yıl: bib {year}, Crossref {rec['year']}")
            row["source"], row["status"] = "Crossref (başlık)", ("uyumlu" if not diffs else "fark")
            row["notes"] = diffs + row["notes"] + [f"DOI {rec['doi']} bib kaydında yok"]
            row["record"] = f"{rec['venue']} {rec['year']}".strip()
        else:
            row["source"] = "—"
            row["status"] = "doğrulanamadı"
            row["notes"].append((f"{src} sorgusu başarısız ({err})" if src else "DOI ve arXiv numarası yok; Crossref başlık aramasında kayıt yok (yazılım, web kaynağı ya da DOI vermeyen yayın yeri)"))
    if row["status"] == "doğrulanamadı" and e.get("_src"):
        row["notes"].append("bib notu: " + e["_src"][:260])
    if is_pre and DBLP_MODE == "off" and row["status"] != "doğrulanamadı":
        cr, _ = crossref_title(title)
        if cr:
            row["notes"].append(f"hakemli sürümü olabilir: {cr[0]['venue']} {cr[0]['year']} (DOI {cr[0]['doi']})")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", default=str(ROOT / "paper" / "refs.bib"))
    ap.add_argument("--tex", default=str(ROOT / "paper"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "inceleme" / f"dblp_kontrol_{time.strftime('%Y-%m-%d')}.md"))
    ap.add_argument("--dblp-json", default="", help="tarayıcı oturumundan alınmış DBLP kayıtları (bkz. dosya başı)")
    ap.add_argument("--dblp-live", action="store_true", help="DBLP API'sini doğrudan sorgula (bot denetimi kalkmışsa)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keys", default="", help="yalnız bu anahtarlar (virgülle)")
    a = ap.parse_args()
    global DBLP_CACHE, DBLP_MODE
    if a.dblp_json:
        DBLP_CACHE = json.loads(Path(a.dblp_json).read_text(encoding="utf-8")); DBLP_MODE = "json"
    elif a.dblp_live:
        DBLP_MODE = "live"
    bib = parse_bib(Path(a.bib))
    cited = cited_keys(Path(a.tex))
    keys = (cited + [k for k in bib if k not in cited]) if a.all else cited
    missing = [k for k in keys if k not in bib]
    keys = [k for k in keys if k in bib and (not a.keys or k in a.keys.split(","))][: a.limit or None]
    rows = []
    for n, k in enumerate(keys, 1):
        r = check(k, bib[k]); r["cited"] = k in cited; rows.append(r)
        if r["status"] != "uyumlu" or r["notes"]:
            print(f"[{n}/{len(keys)}] {k}: {r['status']} ({r['source']}) — " + "; ".join(r["notes"]), flush=True)
    cnt = {s: sum(r["status"] == s for r in rows) for s in ("uyumlu", "fark", "doğrulanamadı")}
    src = {}
    for r in rows:
        src[r["source"]] = src.get(r["source"], 0) + 1
    print(f"toplam {len(rows)} · uyumlu {cnt['uyumlu']} · fark {cnt['fark']} · doğrulanamadı {cnt['doğrulanamadı']} · kaynak: {src} · bib'de olmayan anahtar: {missing}")
    intro = {"off": "DBLP'nin arama API'si bot denetimi ve istek sınırı arkasında olduğundan (betik sorgusu yanıtsız; tarayıcıdan üçüncü istekten sonra 429) "
                    "DBLP sorgulanmadı. DOI'li kayıtlar Crossref kaydıyla, arXiv kayıtları arXiv API'siyle karşılaştırıldı: başlık, yazar soyadları (sırasıyla) ve yıl. "
                    "Ön baskı olarak atıf yapılan kaynakların hakemli sürümü ve DOI'siz kayıtlar Crossref başlık aramasıyla arandı.",
             "live": "Her kaynağın başlığı DBLP yayın aramasında arandı; başlık, yazar soyadları (sırasıyla), yıl ve yayın yeri karşılaştırıldı. "
                     "DBLP'de bulunmayan kaynaklar DOI varsa Crossref, arXiv numarası varsa arXiv API'si ile denetlendi.",
             "json": f"DBLP kayıtları bir tarayıcı oturumundan alındı (`{Path(a.dblp_json).name if a.dblp_json else ''}`); başlık, yazar soyadları (sırasıyla), yıl ve yayın yeri "
                     "karşılaştırıldı. DBLP'de bulunmayan kaynaklar DOI varsa Crossref, arXiv numarası varsa arXiv API'si ile denetlendi."}[DBLP_MODE]
    L = [f"# Atıf künyelerinin denetimi: Crossref, arXiv ve DBLP ({time.strftime('%d.%m.%Y')})", "",
         "Betik: `kod/tools/check_refs_dblp.py`. " + intro, "",
         f"**Özet.** {len(rows)} kayıt ({sum(r['cited'] for r in rows)} tanesi bildiride atıf yapılan): {cnt['uyumlu']} uyumlu, {cnt['fark']} fark, "
         f"{cnt['doğrulanamadı']} bu kaynaklarla doğrulanamadı. "
         + "Kaynak dağılımı: " + ", ".join(f"{k} {v}" for k, v in sorted(src.items())) + "."
         + (f" Bib dosyasında bulunmayan anahtar: {', '.join(missing)}." if missing else ""), "",
         "| Anahtar | Bildiride | Tür | Yıl | Denetim kaynağı | Kayıttaki yer ve yıl | Durum | Not |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| `{r['key']}` | {'evet' if r['cited'] else 'hayır'} | {r['type']} | {r['year']} | {r['source']} | {r.get('record', '—')} | {r['status']} | "
                 + ('; '.join(r['notes']) or '—').replace('|', '/') + " |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("yazıldı:", a.out)


if __name__ == "__main__":
    main()
