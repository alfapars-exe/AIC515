"""Açıklama sonucu veri yapısı ve dışa aktarım (JSON · PNG · NPZ(float16) · MD)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import datetime
import json
import numpy as np


@dataclass
class ExplanationResult:
    model: str
    label: str
    family: str
    method: str                      # partition | gradient
    target: dict                     # TargetSpec.to_dict()
    reference: str                   # kullanılan referans (authentic eş / blur(32,32) / inpaint_telea …)
    values: Optional[np.ndarray] = None   # (h, w, C) float16 — SHAP değerleri (C = 3 RGB; Gradient 5 kanallıda 5)
    fx: float = float("nan")         # f(x): hedefin modeldeki gerçek değeri
    base: float = float("nan")       # E[f] / f(referans)
    sum_shap: float = float("nan")
    add_err: float = float("nan")    # |base + ΣSHAP − f(x)|
    n_evals: int = 0
    seconds: float = 0.0
    peak_mem_mb: Optional[float] = None
    status: str = "OK"               # OK | UNSUPPORTED | FAIL
    message: str = ""
    groups: dict = field(default_factory=dict)   # kanal grubu katkıları (5 kanal U-Net, Gradient)
    notes: list = field(default_factory=list)
    cached: bool = False
    key: str = ""

    @property
    def ok(self):
        return self.status == "OK" and self.values is not None

    def shap_map(self):
        """(h, w) — kanallar üzerinden toplam (piksel başına katkı)."""
        return None if self.values is None else self.values.astype(np.float32).sum(axis=-1)

    def summary(self):
        d = {k: v for k, v in asdict(self).items() if k != "values"}
        d["values_shape"] = None if self.values is None else list(self.values.shape)
        return d


def _f(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def result_table_rows(results):
    rows = []
    for r in results:
        rows.append({"Model": r.label, "Hedef": r.target.get("label", r.target.get("kind")), "Yöntem": r.method,
                     "Referans": r.reference, "f(x)": r.fx, "E[f]": r.base, "ΣSHAP": r.sum_shap,
                     "Toplamsallık hatası": r.add_err, "Değerlendirme": r.n_evals, "Süre s": r.seconds,
                     "Tepe bellek MB": r.peak_mem_mb, "Durum": r.status + (" (önbellek)" if r.cached else ""),
                     "Not": "; ".join([r.message] + list(r.notes)).strip("; ")})
    return rows


def write_markdown(results, meta: dict, comparison: Optional[dict] = None, panels_index: Optional[dict] = None):
    L = [f"# SHAP açıklama raporu · {meta.get('title', '')}", "",
         f"- Tarih: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
         f"- Görüntü: {meta.get('image', '?')} · kaynak: {meta.get('source', '?')}",
         f"- Yöntem ayarları: {meta.get('config', {})}", "",
         "## Model bazında sonuçlar", "",
         "| Model | Hedef | Yöntem | Referans | f(x) | E[f] | ΣSHAP | Toplamsallık hatası | Süre s | Durum |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r.label} | {r.target.get('label', '')} | {r.method} | {r.reference} | {_f(r.fx)} | {_f(r.base)} | "
                 f"{_f(r.sum_shap)} | {_f(r.add_err, 5)} | {_f(r.seconds, 1)} | {r.status} |")
    L += ["", "Notlar:", "- Tüm hedefler olasılık ölçeğindedir (sigmoid). SHAP toplamı = f(x) − f(referans).",
          "- GERÇEK sınıfının SHAP haritası SAHTE'nin tam negatifidir (tek çıkışlı ikili model: p(gerçek) = 1 − p(sahte)).",
          "- Pozitif değer: pikselin bu hedefi SAHTE yönüne ittiği; negatif: GERÇEK yönüne ittiği anlamına gelir."]
    for r in results:
        if r.groups:
            L.append(f"- {r.label} kanal grubu katkıları: " + ", ".join(f"{k} {_f(v)}" for k, v in r.groups.items()))
        if r.status != "OK":
            L.append(f"- {r.label}: **{r.status}** — {r.message}")
    if comparison:
        L += ["", "## Model karşılaştırması", "", comparison.get("note", "")]
        for title, tab in (("Top-%5 pozitif bölge IoU", "iou"), ("Top-%5 pozitif bölge Dice", "dice"),
                           ("16×16 blok Spearman", "spearman")):
            M = comparison.get(tab)
            names = comparison.get("names", [])
            if M is None:
                continue
            L += ["", f"### {title}", "", "| | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
            for i, n in enumerate(names):
                L.append(f"| {n} | " + " | ".join(_f(v, 3) for v in M[i]) + " |")
    if panels_index:
        L += ["", "## Paneller", ""] + [f"- `{v}` — {k}" for k, v in panels_index.items()]
    return "\n".join(L) + "\n"


def export_results(results, outdir, meta: dict, panels: Optional[dict] = None, comparison: Optional[dict] = None,
                   extra_images: Optional[dict] = None):
    """panels: {model_label: {panel_adı: RGB uint8}} · extra_images: {dosya_adı: RGB uint8}. Dönüş: dosya yolları."""
    import cv2
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    files, index = [], {}

    def save_png(name, img):
        p = out / name
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img)
        files.append(str(p)); return p.name

    for mi, (lab, pd_) in enumerate((panels or {}).items()):
        for pj, (pname, img) in enumerate(pd_.items()):
            if img is not None:
                index[f"{lab} · {pname}"] = save_png(f"m{mi:02d}_{pj:02d}.png", img)
    for name, img in (extra_images or {}).items():
        if img is not None:
            index[name] = save_png(name, img)
    arrays = {}
    for i, r in enumerate(results):
        if r.values is not None:
            arrays[f"r{i:02d}_values"] = r.values.astype(np.float16)
    if arrays:
        p = out / "shap_values.npz"; np.savez_compressed(p, **arrays); files.append(str(p))
    js = dict(meta=meta, results=[r.summary() for r in results],
              comparison={k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in (comparison or {}).items()
                          if k not in ("masks", "maps", "common_rgb")})
    p = out / "shap_report.json"
    p.write_text(json.dumps(js, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                 encoding="utf-8")
    files.append(str(p))
    p = out / "shap_report.md"; p.write_text(write_markdown(results, meta, comparison, index), encoding="utf-8"); files.append(str(p))
    return files
