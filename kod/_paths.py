"""Proje yolları — tek doğruluk kaynağı.

Düzen: tüm kaynak kod `kod/` çatısı altındadır ve proje kökündeki veri/çıktı klasörlerini
aynı göreli yolla "aynalar":

    kod/phase3_yolo/train_yolo.py   →  çıktıları  <proje>/phase3_yolo/{runs,results}
    kod/phase2/experiments/*.py     →  çıktıları  <proje>/phase2/experiments/{results,preds,logs,figures}

KOD   : bu dosyanın bulunduğu klasör (kod çatısı)
ROOT  : proje kökü (data/, models/, dataset_audit/ ve tüm çıktı klasörleri burada)
out_dir(__file__) : bir betiğin çıktı klasörü = ROOT / <betiğin KOD'a göre klasörü>

Kök, VIZE_ROOT ortam değişkeniyle değiştirilebilir. Kök bulunamazsa sessizce yanlış
klasöre yazmak yerine hata verilir.
"""
import os
from pathlib import Path

KOD = Path(__file__).resolve().parent


def _is_root(p: Path) -> bool:
    return (p / "data").is_dir() and (p / "dataset_audit").is_dir() and (p / "models").is_dir()


def _find_root() -> Path:
    env = os.environ.get("VIZE_ROOT")
    if env:
        p = Path(env).resolve()
        if not ((p / "data").is_dir() or (p / "dataset_audit").is_dir()):
            raise RuntimeError(f"VIZE_ROOT hatalı: {p} (data/ ve dataset_audit/ yok)")
        return p
    for p in (KOD, *KOD.parents):
        if _is_root(p):
            return p
    raise RuntimeError(f"Proje kökü bulunamadı ({KOD} ve üst klasörlerinde data/, models/, dataset_audit/ yok) — VIZE_ROOT ayarlayın")


ROOT = _find_root()


def out_dir(file) -> Path:
    """kod/<alt>/x.py → ROOT/<alt> (betiğin çıktı/girdi veri klasörü)."""
    return ROOT / Path(file).resolve().parent.relative_to(KOD)
