"""make_arch_figures.py — rapor için iki mimari diyagramı (yalnız matplotlib + numpy).

    python kod/tools/make_arch_figures.py
        → reports/figures/fig_sistem_mimarisi.png     (uçtan uca işlem hattı)
        → reports/figures/fig_uygulama_mimarisi.png   (uygulamanın katmanlı mimarisi ve veri akışı)

Tasarım birimi = şeklin Word'de tam sayfa genişliğine (16 cm = 6,30 inç) yerleştirilmiş hâlindeki inç ve punto.
Şekil S = 1,5 kat büyük çizilip 300 dpi kaydedilir (2835 px genişlik); yani dosyadaki yazı boyutu
tasarım puntosu × 1,5'tir ve 16 cm'e küçültüldüğünde en küçük yazı 7 pt kalır.

Yerleşim ölçülerek yapılır: her metin gerçek yazı tipi genişliğiyle satırlara bölünür, kutu yükseklikleri
içerikten hesaplanır. Kayıttan önce `denetle()` çalışır ve şunları sayar: kutusundan taşan metin, üst üste
binen metin, metnin ya da başka bir kutunun üzerinden geçen ok, çakışan kutu, 7 pt altı yazı.
"""
import re
import sys
from pathlib import Path

sys.path.append(str(next(p for p in Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

OUT = ROOT / "reports" / "figures"

W = 6.30          # tasarım genişliği (inç) = 16 cm
S = 1.5           # çizim ölçeği (dosya = tasarım × S)
DPI = 300
MIN_PT = 7.0      # 16 cm'de izin verilen en küçük yazı

FS_T, FS_ST, FS_B, FS_C, FS_F = 8.0, 7.6, 7.4, 7.0, 7.2   # başlık, alt başlık, gövde, kod, akış listesi
LH, ASC = 1.28, 0.92                                      # satır yüksekliği ve ilk taban çizgisi (em)
TXT, CODE, FLOW = "#22282E", "#4E5A66", "#3A3F45"


class Pal:
    def __init__(self, fill, edge, dark):
        self.fill, self.edge, self.dark = fill, edge, dark


VERI = Pal("#E7EDF3", "#6F8499", "#33475B")    # veri: gri-mavi
MODEL = Pal("#DDE9F6", "#4F81B5", "#1F4B7A")   # model / çıkarım: mavi
DEGER = Pal("#FBEBD9", "#CF8B43", "#8A4E12")   # değerlendirme / alan çekirdeği: turuncu
CIKTI = Pal("#E3F1E1", "#63A05F", "#2E6A2B")   # çıktı / dağıtım: yeşil
UYG = Pal("#ECE5F5", "#876CB3", "#4E3784")     # uygulama / sunum: mor
SERVIS = Pal("#F6E6EE", "#B7718F", "#7D3355")  # uygulama servisleri: gül
XAI = Pal("#DFF1EF", "#4F9C95", "#1F6660")     # açıklanabilirlik: camgöbeği
NOTR = Pal("#F6F6F6", "#A3A3A3", "#3A3F45")    # gruplama / açıklama

_FP = {}


def fp(style, size):
    key = (style, round(size, 3))
    if key not in _FP:
        _FP[key] = FontProperties(family="DejaVu Sans Mono" if style in ("m", "mb") else "DejaVu Sans",
                                  weight="bold" if style in ("b", "mb") else "normal", size=size)
    return _FP[key]


_TOK = re.compile(r"(``.+?``|`.+?`|\*\*.+?\*\*|\[\[.+?\]\])")


def parse(markup, default):
    """'**kalın** `kod` ``kalın kod`` [[normal]]' → sözcük listesi; sözcük = [(metin, stil), ...] (bölünmez).
    '~' bölünmez boşluktur."""
    atoms, pending = [], False
    for seg in _TOK.split(markup):
        if not seg:
            continue
        if seg.startswith("``"):
            st, body = "mb", seg[2:-2]
        elif seg.startswith("`"):
            st, body = "m", seg[1:-1]
        elif seg.startswith("**"):
            st, body = "b", seg[2:-2]
        elif seg.startswith("[["):
            st, body = "r", seg[2:-2]
        else:
            st, body = default, seg
        parts = body.split()
        if not parts:
            pending = pending or bool(body)
            continue
        lead = body[:1].isspace()
        for i, p in enumerate(parts):
            atoms.append((p.replace("~", " "), st, (pending or lead) if i == 0 else True))
        pending = body[-1:].isspace()
    words = []
    for t, st, sp in atoms:
        if sp or not words:
            words.append([(t, st)])
        else:
            words[-1].append((t, st))
    return words


class Box:
    def __init__(self, x, y, w, h, name="", parent=None, container=False):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.name, self.parent, self.container = name, parent, container

    x0 = property(lambda s: s.x)
    x1 = property(lambda s: s.x + s.w)
    y0 = property(lambda s: s.y)
    y1 = property(lambda s: s.y + s.h)
    cx = property(lambda s: s.x + s.w / 2)
    cy = property(lambda s: s.y + s.h / 2)


class Canvas:
    """y = 0 üst kenardır; çizim aşağı doğru (negatif y) ilerler, şekil yüksekliği sonda belirlenir."""

    def __init__(self, name):
        self.name = name
        self.fig = plt.figure(figsize=(W * S, 4.0), dpi=DPI)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")
        self.rend = self.fig.canvas.get_renderer()
        self.boxes, self.texts, self.arrows = [], [], []
        self.ymin = 0.0
        self._tw = {}

    # ---------- ölçüm ----------
    def _raw(self, s, style, size):
        w, _, _ = self.rend.get_text_width_height_descent(s, fp(style, size * S), ismath=False)
        return w / DPI / S

    def tw(self, s, style, size):
        """Metnin ilerleme genişliği (tasarım inç); uçtaki boşluklar da sayılsın diye '|' arasında ölçülür."""
        key = (s, style, size)
        if key not in self._tw:
            self._tw[key] = self._raw("|" + s + "|", style, size) - self._raw("||", style, size)
        return self._tw[key]

    def wrap(self, words, width, size):
        lines, cur, curw = [], [], 0.0
        for wd in words:
            ww = sum(self.tw(t, st, size) for t, st in wd)
            mono = bool(cur) and cur[-1][0][-1][1] in ("m", "mb") and wd[0][1] in ("m", "mb")
            sp = self.tw(" ", "m" if mono else "r", size) if cur else 0.0
            if cur and curw + sp + ww > width:
                lines.append((cur, curw))
                cur, curw, sp = [], 0.0, 0.0
            cur.append((wd, ww, curw + sp))
            curw += sp + ww
        if cur:
            lines.append((cur, curw))
        return lines

    # ---------- içerik yerleşimi ----------
    def layout(self, items, w, pal):
        """items → (çizim işlemleri, yükseklik, en geniş satır). Koordinatlar sol üst köşeye göre, y aşağı pozitif."""
        ops, y, maxw, prev = [], 0.0, 0.0, None
        for it in items:
            kind, arg = it[0], it[1]
            opt = it[2] if len(it) > 2 else {}
            if kind == "gap":
                y += arg
                continue
            if prev is not None:
                y += 0.04 if prev in ("h", "hs") else 0.05 if kind == "f" else 0.02
            prev = kind
            if kind == "cols":
                fr = opt.get("fr") or [1.0 / len(arg)] * len(arg)
                gx = opt.get("gap", 0.14)
                avail, dx, hm = w - gx * (len(arg) - 1), 0.0, 0.0
                for col, f in zip(arg, fr):
                    o, h, _ = self.layout(col, avail * f, pal)
                    ops += [[op[0], op[1] + dx, op[2] + y] + op[3:] for op in o]
                    hm = max(hm, h)
                    dx += avail * f + gx
                y += hm
                maxw = w
                continue
            size, default, color, indent = {
                "h": (FS_T, "b", pal.dark, 0.0), "hs": (FS_ST, "b", pal.dark, 0.0),
                "p": (FS_B, "r", TXT, 0.0), "pc": (FS_B, "r", TXT, 0.0), "b": (FS_B, "r", TXT, 0.095),
                "c": (FS_C, "m", CODE, 0.0), "f": (FS_F, "r", TXT, 0.215)}[kind]
            color = opt.get("color", color)
            lh = LH * size / 72.0
            if "num" in opt:
                indent = 0.215
                ops.append(["nb", 0.082, y + lh / 2, str(opt["num"])])
            if kind == "b":
                ops.append(["t", 0.008, y + ASC * size / 72.0, "•", "r", size, color])
            if kind == "f":
                ops.append(["xi" if opt.get("label") == "x" else "fb", 0.082, y + lh / 2, str(opt.get("label", ""))])
            for line, lw_ in self.wrap(parse(arg, default), w - indent, size):
                x0 = indent + ((w - indent - lw_) / 2 if kind == "pc" else 0.0)
                for wd, _, wx in line:
                    x = x0 + wx
                    for t, st in wd:
                        ops.append(["t", x, y + ASC * size / 72.0, t, st, size, color])
                        x += self.tw(t, st, size)
                maxw = max(maxw, indent + lw_)
                y += lh
        return ops, y, maxw

    @staticmethod
    def _split(items):
        k = len(items)
        while k > 0 and items[k - 1][0] == "c":
            k -= 1
        return items[:k], items[k:]

    CODE_GAP = 0.045

    def need(self, items, w, padx=0.07, pady=0.06, pal=NOTR):
        main, code = self._split(items)
        hm = self.layout(main, w - 2 * padx, pal)[1]
        hc = self.layout(code, w - 2 * padx, pal)[1] if code else 0.0
        return 2 * pady + hm + ((self.CODE_GAP if main else 0.0) + hc if code else 0.0)

    def natw(self, items, padx=0.07, pal=NOTR):
        return self.layout(items, 50.0, pal)[2] + 2 * padx

    # ---------- çizim ilkel öğeleri ----------
    def text(self, x, y, s, style, size, color, owner=None, ha="left", va="baseline", rot=0, on_arrow=False):
        t = self.ax.text(x, y, s, fontproperties=fp(style, size * S), color=color, ha=ha, va=va, rotation=rot,
                         zorder=6)
        self.texts.append(dict(t=t, owner=owner, on_arrow=on_arrow, size=size, s=s))
        return t

    def badge(self, cx, cy, label, pal=None, on_arrow=False, r=0.078):
        """pal verilirse dolu (aşama numarası), verilmezse beyaz içli (akış adımı) rozet."""
        fc, ec, tc = (pal.dark, pal.dark, "white") if pal else ("white", FLOW, FLOW)
        self.ax.add_patch(Circle((cx, cy), r, fc=fc, ec=ec, lw=0.8 * S, zorder=7))
        t = self.ax.text(cx, cy - 0.003, label, fontproperties=fp("b", MIN_PT * S), color=tc, ha="center",
                         va="center_baseline", zorder=8)
        self.texts.append(dict(t=t, owner=None, on_arrow=on_arrow, size=MIN_PT, s=label))

    def draw_ops(self, ops, x, top, owner, pal):
        for op in ops:
            X, Y = x + op[1], top - op[2]
            if op[0] == "t":
                self.text(X, Y, op[3], op[4], op[5], op[6], owner)
            elif op[0] == "nb":
                self.badge(X, Y, op[3], pal=pal)
            elif op[0] == "fb":
                self.badge(X, Y, op[3])
            elif op[0] == "xi":
                self.arrow([(X - 0.085, Y), (X + 0.09, Y)], color=XAI.dark, ls=(0, (2.2, 1.6)), audit=False)

    def node(self, x, top, w, items, pal, h=None, fill=None, edge=None, lw=0.7, padx=0.07, pady=0.06,
             parent=None, container=False, ls="-", r=0.05, valign="top", name=""):
        main, code = self._split(items)
        om, hm, _ = self.layout(main, w - 2 * padx, pal)
        oc, hc, _ = self.layout(code, w - 2 * padx, pal) if code else ([], 0.0, 0.0)
        need = 2 * pady + hm + ((self.CODE_GAP if main else 0.0) + hc if code else 0.0)
        h = need if h is None else h
        box = Box(x, top - h, w, h, name or (items[0][1] if items else ""), parent, container)
        self.boxes.append(box)
        self.ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                         fc=pal.fill if fill is None else fill, ec=edge or pal.edge,
                                         lw=lw * S, ls=ls, zorder=1 if container else 2))
        off = (h - need) / 2 if valign == "center" else 0.0
        self.draw_ops(om, x + padx, top - pady - off, box, pal)
        if code:
            ctop = top - h + pady + hc if valign == "top" else top - pady - off - hm - (self.CODE_GAP if main else 0)
            self.draw_ops(oc, x + padx, ctop, box, pal)
        self.ymin = min(self.ymin, top - h)
        return box

    def arrow(self, pts, color=FLOW, lw=0.9, both=False, ls="-", ms=8.5, audit=True, name=""):
        style = ("<|-|>" if both else "-|>") + ",head_length=0.55,head_width=0.24"
        kw = dict(mutation_scale=ms * S, lw=lw * S, fc=color, ec=color, joinstyle="miter", capstyle="butt", zorder=4)
        P = np.asarray(pts, float)
        if ls != "-":  # kesikli çizgi ok ucunu bozmasın: gövde kesikli, uç ayrı ve dolu çizilir
            d = P[-1] - P[-2]
            L = float(np.hypot(*d))
            mid = P[-1] - d / L * min(0.085, 0.6 * L)
            self.ax.add_patch(FancyArrowPatch(path=MPath(np.vstack([P[:-1], mid])), arrowstyle="-", ls=ls, **kw))
            P, ls = np.vstack([mid, P[-1]]), "-"
        self.ax.add_patch(FancyArrowPatch(path=MPath(P), arrowstyle=style, ls=ls, **kw))
        if audit:
            self.arrows.append((name or f"ok{len(self.arrows) + 1}", [tuple(p) for p in pts]))

    # ---------- denetim ve kayıt ----------
    def denetle(self):
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()
        tb = []
        for d in self.texts:
            bb = d["t"].get_window_extent(r).transformed(inv)
            tb.append((d, bb.x0, bb.y0, bb.x1, bb.y1))
        y_lo, y_hi = self.ax.get_ylim()
        sorun, m = [], 0.018
        for d, x0, y0, x1, y1 in tb:
            o = d["owner"]
            if o is not None and (x0 < o.x0 + m or x1 > o.x1 - m or y0 < o.y0 + m or y1 > o.y1 - m):
                sorun.append(f"taşan metin: '{d['s']}' ⊄ kutu '{o.name}'")
            if x0 < 0.02 or x1 > W - 0.02 or y0 < y_lo + 0.02 or y1 > y_hi - 0.02:
                sorun.append(f"şekil dışına taşan metin: '{d['s']}'")
            if d["size"] < MIN_PT - 1e-6:
                sorun.append(f"küçük yazı ({d['size']} pt): '{d['s']}'")
        e = 0.006
        for i in range(len(tb)):
            for j in range(i + 1, len(tb)):
                a, b = tb[i], tb[j]
                if a[1] + e < b[3] - e and b[1] + e < a[3] - e and a[2] + e < b[4] - e and b[2] + e < a[4] - e:
                    sorun.append(f"üst üste binen metin: '{a[0]['s']}' × '{b[0]['s']}'")
        for name, pts in self.arrows:
            samp = []
            for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
                n = max(2, int(np.hypot(xb - xa, yb - ya) / 0.008))
                samp += [(xa + (xb - xa) * k / n, ya + (yb - ya) * k / n) for k in range(n + 1)]
            for d, x0, y0, x1, y1 in tb:
                if not d["on_arrow"] and any(x0 - 0.012 < px < x1 + 0.012 and y0 - 0.012 < py < y1 + 0.012
                                             for px, py in samp):
                    sorun.append(f"metnin üzerinden geçen ok: {name} × '{d['s']}'")
            for b in self.boxes:
                if not b.container and any(b.x0 + 0.012 < px < b.x1 - 0.012 and b.y0 + 0.012 < py < b.y1 - 0.012
                                           for px, py in samp):
                    sorun.append(f"kutunun içinden geçen ok: {name} × '{b.name}'")
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                a, b = self.boxes[i], self.boxes[j]
                if a.parent is b.parent and a.x0 < b.x1 - 0.004 and b.x0 < a.x1 - 0.004 \
                        and a.y0 < b.y1 - 0.004 and b.y0 < a.y1 - 0.004:
                    sorun.append(f"çakışan kutular: '{a.name}' × '{b.name}'")
        for b in self.boxes:
            p = b.parent
            if p is not None and (b.x0 < p.x0 + 0.02 or b.x1 > p.x1 - 0.02 or b.y0 < p.y0 + 0.02 or b.y1 > p.y1 - 0.02):
                sorun.append(f"kapsayıcısından taşan kutu: '{b.name}' ⊄ '{p.name}'")
        return sorun, min(d["size"] for d in self.texts)

    def save(self, path, pad=0.06):
        y0, y1 = self.ymin - pad, pad
        self.fig.set_size_inches(W * S, (y1 - y0) * S)
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(y0, y1)
        sorun, minpt = self.denetle()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=DPI, facecolor="white")
        plt.close(self.fig)
        hpx, wpx = plt.imread(path).shape[:2]
        print(f"{path}")
        print(f"  {wpx}×{hpx} px, {path.stat().st_size / 1024:.0f} KB; 16 cm genişlikte yükseklik "
              f"{(y1 - y0) * 2.54:.1f} cm, en küçük yazı {minpt:.1f} pt; {len(self.texts)} metin, "
              f"{len(self.boxes)} kutu, {len(self.arrows)} ok; denetim: {len(sorun)} sorun")
        for s in sorun:
            print("   !", s)
        return sorun


# =====================================================================================================
# Şekil 1 — uçtan uca işlem hattı
# =====================================================================================================
def fig_sistem():
    c = Canvas("sistem")
    X0 = 0.07
    FW = W - 2 * X0
    G = 0.22                      # kutular arası ok boşluğu

    # 1–3: veri → denetim → GT (soldan sağa)
    ws = (1.94, 1.64, FW - 1.94 - 1.64 - 2 * G)
    its = (
        [("h", "Veri", {"num": 1}),
         ("b", "**AIForge-Doc v1:** 4.061 sahte belge, bbox maskeleri; CC~BY-NC-SA"),
         ("b", "**AIForge-Doc v2:** harici test; 567 sızıntısız çift"),
         ("c", "data/")],
        [("h", "Veri denetimi", {"num": 2}),
         ("b", "14 boyutlu audit"),
         ("b", "grup-farkında train/val bölmesi"),
         ("b", "küçük veri alt kümesi (400/100)"),
         ("c", "kod/dataset_audit/")],
        [("h", "Referans (GT) türetimi ve kontroller", {"num": 3}),
         ("b", "bbox, piksel-değişim (ε~=~4/8/16), glif, işaretçisiz varyantlar"),
         ("b", "benign kontrol kümeleri"),
         ("c", "kod/phase2/gt/"), ("c", "kod/phase2/controls/")],
    )
    h1 = max(c.need(i, w) for i, w in zip(its, ws))
    row, x = [], X0
    for i, w in zip(its, ws):
        row.append(c.node(x, 0.0, w, i, VERI, h=h1))
        x += w + G
    c.arrow([(row[0].x1, row[0].cy), (row[1].x0, row[1].cy)])
    c.arrow([(row[1].x1, row[1].cy), (row[2].x0, row[2].cy)])

    # 4: modeller (üç alt kutu)
    HEAD = 0.255                  # kapsayıcı başlık satırı
    top = -h1 - G
    sg = 0.08
    sw = (FW - 2 * 0.08 - 2 * sg) / 3
    subs = (
        [("hs", "Faz 1 [[· küçük veri, çalışırlık]]"),
         ("b", "56 klasik öznitelik + SVM / RF"),
         ("b", "ResNet-18 (probe / ince ayar / sıfırdan)"),
         ("b", "U-Net (RGB, RGB+frekans)"),
         ("c", "kod/baselines/")],
        [("hs", "Faz 2 [[· tam veri]]"),
         ("b", "U-Net × 3~GT × 3~tohum"),
         ("b", "held-out üretici"),
         ("b", "zero-shot TruFor / ADCD-Net"),
         ("c", "kod/phase2/experiments/"), ("c", "third_party/")],
        [("hs", "Faz 3"),
         ("b", "DeepLabV3+"),
         ("b", "YOLO11 kutu dedektörü"),
         ("b", "işaretçi-onarımlı eğitim"),
         ("b", "birleşik piksel+kutu kaybı"),
         ("b", "20~epoch"),
         ("c", "kod/phase3/"), ("c", "kod/phase3_yolo/")],
    )
    hs = max(c.need(i, sw) for i in subs)
    mod = c.node(X0, top, FW, [("h", "Modeller", {"num": 4})], MODEL, h=HEAD + hs + 0.08, container=True)
    for k, i in enumerate(subs):
        c.node(X0 + 0.08 + k * (sw + sg), top - HEAD, sw, i, MODEL, h=hs, fill="white", lw=0.6, parent=mod)
    c.arrow([(row[2].cx, row[2].y0), (row[2].cx, mod.y1)])

    # model ağırlıkları (ara ürün) → değerlendirme ve uygulama
    EW = 5.60                     # değerlendirme kutusu genişliği; sağında ağırlık oku için kanal kalır
    xch = 5.95
    ga = 0.17
    pill_items = [("pc", "**Model ağırlıkları** · `models/`")]
    pw = c.natw(pill_items, padx=0.12) + 0.02
    ptop = mod.y0 - ga
    pill = c.node(X0 + EW / 2 - pw / 2, ptop, pw, pill_items, MODEL, fill="white", ls=(0, (3, 2)), padx=0.12,
                  pady=0.055, r=0.11, name="Model ağırlıkları")
    c.arrow([(pill.cx, mod.y0), (pill.cx, pill.y1)])

    # 5: değerlendirme
    ev = c.node(X0, pill.y0 - ga, EW, [
        ("h", "Değerlendirme", {"num": 5}),
        ("cols", [[("b", "**Çıkarım:** tam görüntü, kayan pencere"),
                   ("b", "**Ölçütler:** AUPRC, F1@τ*, IoU, görüntü AUROC"),
                   ("b", "**İstatistik:** bootstrap CI, Holm")],
                  [("b", "**Sondalar:** sınır bulanıklığı, render/içerik, yeşil işaretçi"),
                   ("b", "**Dayanıklılık:** JPEG, ölçekleme, bulanıklık, CLAHE, gri")]],
         {"fr": (0.46, 0.54), "gap": 0.16}),
        ("c", "kod/phase2/experiments/ · kod/phase2/stats/ · kod/phase3/ · kod/phase3_v2/"),
    ], DEGER)
    c.arrow([(pill.cx, pill.y0), (pill.cx, ev.y1)])

    # 6: çıktılar
    top = ev.y0 - G
    IP, CG, RG = 0.08, 0.24, 0.20  # iç kenar payı, sütun aralığı, satır aralığı
    wa, wb = 1.95, 1.75
    wc = FW - 2 * IP - wa - wb - 2 * CG
    xa = X0 + IP
    xb = xa + wa + CG
    xc = xb + wb + CG
    t_son = [("hs", "Sonuç tabloları ve şekiller"), ("b", "ölçüt tabloları (CSV, MD)"),
             ("b", "hipotez kararları, şekiller"), ("c", "phase*/results/ · …/figures/")]
    t_xai = [("hs", "Açıklanabilirlik"), ("b", "SHAP"), ("b", "katman filtreleri"),
             ("c", "kod/app/xai/"), ("c", "kod/app/xai_filters.py")]
    t_app = [("hs", "Uygulama (Gradio)"), ("b", "simülasyon, toplu test"), ("b", "ön işleme, model sekmeleri"),
             ("c", "kod/app/")]
    t_rap = [("hs", "Rapor ve bildiri"), ("c", "reports/ · paper/")]
    t_hf = [("hs", "Hugging Face Space (ZeroGPU) + veri/model bucket'ı"),
            ("c", "kod/tools/build_space.py · kod/hf_space/deploy_hf.py")]
    ht = max(c.need(t_son, wa), c.need(t_xai, wb), c.need(t_app, wc))
    hb = max(c.need(t_rap, wa), c.need(t_hf, wb + CG + wc))
    out = c.node(X0, top, FW, [("h", "Çıktılar", {"num": 6})], NOTR, h=HEAD + ht + RG + hb + IP,
                 container=True, ls=(0, (4, 2.5)))
    b_son = c.node(xa, top - HEAD, wa, t_son, CIKTI, h=ht, parent=out)
    b_xai = c.node(xb, top - HEAD, wb, t_xai, UYG, h=ht, parent=out)
    b_app = c.node(xc, top - HEAD, wc, t_app, UYG, h=ht, parent=out)
    b_rap = c.node(xa, top - HEAD - ht - RG, wa, t_rap, CIKTI, h=hb, parent=out)
    b_hf = c.node(xb, top - HEAD - ht - RG, wb + CG + wc, t_hf, UYG, h=hb, parent=out)
    c.arrow([(b_son.cx, ev.y0), (b_son.cx, b_son.y1)])
    c.arrow([(b_son.cx, b_son.y0), (b_rap.cx, b_rap.y1)])
    c.arrow([(b_xai.x1, b_xai.cy), (b_app.x0, b_app.cy)])
    c.arrow([(b_app.cx, b_app.y0), (b_app.cx, b_hf.y1)])
    c.arrow([(pill.x1, pill.cy), (xch, pill.cy), (xch, b_app.y1)], color=MODEL.dark)

    # renk açıklaması
    leg = (("Veri ve hazırlık", VERI), ("Modeller", MODEL), ("Değerlendirme", DEGER), ("Çıktılar", CIKTI),
           ("Uygulama ve açıklanabilirlik", UYG))
    sw_, tg, eg = 0.15, 0.055, 0.26
    tot = sum(sw_ + tg + c.tw(t, "r", MIN_PT) for t, _ in leg) + eg * (len(leg) - 1)
    x, y = (W - tot) / 2, out.y0 - 0.17
    for t, p in leg:
        c.ax.add_patch(FancyBboxPatch((x, y - 0.048), sw_, 0.096, boxstyle="round,pad=0,rounding_size=0.025",
                                      fc=p.fill, ec=p.edge, lw=0.7 * S, zorder=2))
        c.text(x + sw_ + tg, y, t, "r", MIN_PT, TXT, va="center_baseline")
        x += sw_ + tg + c.tw(t, "r", MIN_PT) + eg
    c.ymin = y - 0.085
    return c.save(OUT / "fig_sistem_mimarisi.png")


# =====================================================================================================
# Şekil 2 — uygulamanın katmanlı mimarisi ve veri akışı
# =====================================================================================================
def fig_uygulama():
    c = Canvas("uygulama")
    X0 = 0.07
    FW = W - 2 * X0
    HEAD = 0.255
    IP = 0.08                     # katman içi kenar payı
    G = 0.25                      # katmanlar arası ok boşluğu
    XL, XR = 0.42, 4.60           # 2–5. katmanların sol/sağ kenarı (solda veri hattı için oluk)
    XS = 4.75                     # sağ sütun (akış listesi + dağıtım şeridi)
    SWID = W - X0 - XS
    LWID = XR - XL

    def chips(x, top, width, specs, pal, parent, gap=0.07, pady=0.055, valign="top"):
        """Bir satır beyaz kutu; specs = [(genişlik oranı | None, items)]. None → doğal genişlik + eşit pay."""
        nat = [c.natw(it, padx=0.06) if w is None else None for w, it in specs]
        free = width - gap * (len(specs) - 1)
        if all(n is not None for n in nat):
            extra = (free - sum(nat)) / len(nat)
            ws = [n + extra for n in nat]
        else:
            ws = [free * w for w, _ in specs]
        h = max(c.need(it, w, padx=0.06, pady=pady) for (_, it), w in zip(specs, ws))
        out, xx = [], x
        for (_, it), w in zip(specs, ws):
            out.append(c.node(xx, top, w, it, pal, h=h, fill="white", lw=0.6, padx=0.06, pady=pady,
                              parent=parent, valign=valign))
            xx += w + gap
        return out, h

    def layer(x, top, width, title, rows, pal, gap=0.07, opt=None):
        """Katman şeridi: başlık + bir ya da birkaç kutu satırı. Yükseklik önce kuru çalıştırmayla ölçülür."""
        hs = []
        for specs in rows:
            nat = [c.natw(it, padx=0.06) if w is None else None for w, it in specs]
            free = width - 2 * IP - gap * (len(specs) - 1)
            ws = [n + (free - sum(nat)) / len(nat) for n in nat] if all(n is not None for n in nat) \
                else [free * w for w, _ in specs]
            hs.append(max(c.need(it, w, padx=0.06, pady=0.055) for (_, it), w in zip(specs, ws)))
        h = HEAD + sum(hs) + 0.06 * (len(rows) - 1) + IP
        box = c.node(x, top, width, [("h", title, opt or {})], pal, h=h, container=True)
        y = top - HEAD
        inner = []
        for specs, hh in zip(rows, hs):
            inner.append(chips(x + IP, y, width - 2 * IP, specs, pal, box, gap=gap)[0])
            y -= hh + 0.06
        return box, inner

    # 1: sunum katmanı (tam genişlik)
    tabs = ("Simülasyon", "Toplu~test", "Girdi", "Ön~işleme", "U-Net", "ResNet-18", "Yeşil~işaretçi", "Sonuçlar",
            "XAI")
    ui, _ = layer(X0, 0.0, FW, "Sunum katmanı — Gradio arayüzü [[(]]`appcore/ui.py`[[,]] `theme.py`[[)]]",
                  [[(None, [("pc", f"**{i}**", {"color": UYG.dark}), ("gap", -0.012), ("pc", t)])
                    for i, t in enumerate(tabs)]], UYG, gap=0.05)

    # 2–5: servisler, alan çekirdeği, çıkarım, açıklanabilirlik
    top = ui.y0 - G
    srv, _ = layer(XL, top, LWID, "Uygulama servisleri", [
        [(0.5, [("p", "``simulation`` — tek örnek: senaryo → çıkarım → karar → rapor")]),
         (0.5, [("p", "``batch`` — toplu test, özet tablo/grafik")])],
        [(0.5, [("p", "``handlers`` — sekme işleyicileri, taslaklar")]),
         (0.5, [("p", "``xai_bridge`` — SHAP tahmin / açıklama / karşılaştırma")])]], SERVIS)
    dom, _ = layer(XL, srv.y0 - G, LWID, "Alan çekirdeği", [
        [(0.5, [("p", "``ops`` — ön işleme işlemleri, 13~kategori, hazır hatlar")]),
         (0.5, [("p", "``imaging`` — görüntü/frekans, yeşil işaretçi, piksel-GT")])],
        [(0.5, [("p", "``scenarios`` — girdi senaryoları, bozulmalar")]),
         (0.5, [("p", "``metrics`` — F1, IoU, AP, eşikler~τ, son işleme")])]], DEGER)
    inf, inf_in = layer(XL, dom.y0 - G, LWID, "Çıkarım katmanı", [
        [(0.655, [("p", "``models`` — model kaydı ve önbellek"),
                  ("b", "U-Net / DeepLabV3+: kayan pencere (512~px)"),
                  ("b", "YOLO11: kareli çıkarım + NMS"),
                  ("b", "ResNet-18: patch (224~px)")]),
         (0.345, [("p", "``gpu_infer``"), ("p", "ZeroGPU / CUDA / CPU")])]], MODEL)
    xai, _ = layer(XL, inf.y0 - G, LWID, "Açıklanabilirlik paketleri", [
        [(0.60, [("p", "``xai/`` — registry → targets → adapters → manager → viz/compare → service"),
                 ("p", "tam görüntü SHAP: `fullimage`")]),
         (0.40, [("p", "``xai_filters.py``"),
                 ("p", "ilk evrişim katmanı kernel'leri ve aktivasyon haritaları")])]], XAI)

    # 6: veri ve yapılandırma (oluk dahil sol sütunun tamamı)
    dat, _ = layer(X0, xai.y0 - G, XR - X0, "Veri ve yapılandırma [[(]]`config`[[,]] `dataset`[[)]]", [
        [(None, [("p", "Veri seti: `data/AIForge-Doc-v1`")]),
         (None, [("p", "Denetim tabloları: `dataset_audit/`")])],
        [(None, [("p", "Ağırlıklar: `models/`")]),
         (None, [("p", "Eşikler ve referans tablolar: `kod/app/meta/`")])],
        [(None, [("p", "Sonuç dosyaları: `phase2/experiments/results`, `baselines/results`")])]], VERI)

    # numaralı veri akışı okları (katman aralıklarında; rozet okun solunda)
    def flow(x, ya, yb, label):
        c.arrow([(x, ya), (x, yb)], name=f"akış {label}")
        c.badge(x - 0.14, (ya + yb) / 2, label)

    xd, xu = 1.30, 4.25
    x3 = inf_in[0][1].x0 + 0.30
    flow(xd, ui.y0, srv.y1, "1")
    flow(xd, srv.y0, dom.y1, "2")
    flow(x3, dom.y0, inf.y1, "3")
    flow(xu, inf.y1, dom.y0, "4")
    flow(xu, dom.y1, srv.y0, "5")
    flow(xu, srv.y1, ui.y0, "5")
    xx = inf_in[0][1].cx
    c.arrow([(xx, inf.y0), (xx, xai.y1)], color=XAI.dark, ls=(0, (2.2, 1.6)), name="XAI kolu")
    c.text(xx + 0.07, (inf.y0 + xai.y1) / 2, "XAI kolu", "r", MIN_PT, XAI.dark, va="center_baseline")

    # veri hattı: veri katmanı → çıkarım (ağırlıklar) ve servisler (① veri setinden örnek)
    xt = 0.275
    c.arrow([(xt, dat.y1), (xt, inf.cy), (XL, inf.cy)], color=VERI.dark, name="veri hattı (ağırlıklar)")
    c.arrow([(xt, dat.y1), (xt, srv.cy), (XL, srv.cy)], color=VERI.dark, name="veri hattı (örnek)")
    c.badge(xt, dom.cy, "1", on_arrow=True)
    c.text(xt - 0.135, xai.cy, "veri · ağırlıklar", "r", MIN_PT, VERI.dark, ha="center", va="center", rot=90)

    # sağ sütun, üst: numaralı akış listesi
    leg = c.node(XS, top, SWID, [
        ("h", "Veri akışı"),
        ("f", "girdi (veri setinden örnek / yükleme)", {"label": 1}),
        ("f", "senaryo + bozulma + ön işleme hattı", {"label": 2}),
        ("f", "`gpu_infer`: olasılık haritası / kutular / patch skorları", {"label": 3}),
        ("f", "eşikleme ve ölçütler", {"label": 4}),
        ("f", "görseller ve rapor dosyaları", {"label": 5}),
        ("f", "XAI kolu: 3. adımdan açıklanabilirlik paketlerine", {"label": "x"}),
    ], NOTR, pady=0.075, name="Veri akışı")

    # sağ sütun, alt: dağıtım şeridi (veri katmanının alt kenarına kadar)
    stop = leg.y0 - 0.14
    dep = c.node(XS, stop, SWID, [("h", "Dağıtım")], CIKTI, h=stop - dat.y0, container=True)
    chain = (
        [("pc", "`kod/app/`"), ("pc", "uygulama kaynağı")],
        [("pc", "`tools/build_space.py`"), ("pc", "Space paketini üretir")],
        [("pc", "`hf_space/`"), ("pc", "türetilmiş Space paketi")],
        [("pc", "`deploy_hf.py`"), ("pc", "Hugging Face'e dağıtır")],
        [("pc", "**Hugging Face Space**"), ("pc", "Gradio 6, ZeroGPU")],
        [("pc", "**bucket** (`/data`)"), ("pc", "veri + ağırlıklar")],
    )
    CI = 0.06
    cw = SWID - 2 * CI
    chs = [c.need(it, cw, padx=0.05, pady=0.05) for it in chain]
    free = dep.h - HEAD - IP - sum(chs)
    ag = free / (len(chain) - 1)
    y, cb = stop - HEAD, []
    for it, hh in zip(chain, chs):
        cb.append(c.node(XS + CI, y, cw, it, CIKTI, h=hh, fill="white", lw=0.6, padx=0.05, pady=0.05, parent=dep))
        y -= hh + ag
    for k in range(len(cb) - 1):
        c.arrow([(cb[k].cx, cb[k].y0), (cb[k + 1].cx, cb[k + 1].y1)], color=CIKTI.dark, both=(k == len(cb) - 2),
                name=f"dağıtım {k + 1}")
    c.arrow([(cb[-1].x0, cb[-1].cy), (dat.x1, cb[-1].cy)], color=VERI.dark, ls=(0, (2.2, 1.6)), name="bucket → veri")
    return c.save(OUT / "fig_uygulama_mimarisi.png")


def main():
    for s in (sys.stdout, sys.stderr):  # Windows cp1254 konsolunda Unicode çıktı
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    sorun = fig_sistem() + fig_uygulama()
    return 1 if sorun else 0


if __name__ == "__main__":
    sys.exit(main())
