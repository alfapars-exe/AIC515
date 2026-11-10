"""build_space.py — Hugging Face Space paketini (<proje>/hf_space) kaynaklardan üretir.

`hf_space/` elle düzenlenmez; tek doğruluk kaynağı `kod/app/` (uygulama), `kod/baselines/features_classical.py`,
`kod/hf_space/space_src/` (Space README + requirements + packages) ve proje kökündeki sonuç dosyalarıdır.

Kullanım:
    python kod/tools/build_space.py            # hf_space/ klasörünü güncelle
    python kod/tools/build_space.py --check    # yalnız karşılaştır (hiçbir şey yazmaz); fark varsa çıkış kodu 1
"""
import argparse
import filecmp
import shutil
import sys
from pathlib import Path

sys.path.append(str(next(p for p in Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT, KOD  # noqa: E402

SPACE = ROOT / "hf_space"
APP = KOD / "app"
SRC = KOD / "hf_space" / "space_src"   # README.md, requirements.txt, packages.txt (Space'e özgü)
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SKIP_FILES = {"README_APP.md"}


def _walk(src: Path, dst: Path):
    """(kaynak, hedef) dosya çiftleri — klasör yapısı korunur."""
    for p in sorted(src.rglob("*")):
        if p.is_file() and not (set(p.relative_to(src).parts[:-1]) & SKIP_DIRS) and p.name not in SKIP_FILES and p.suffix != ".pyc":
            yield p, dst / p.relative_to(src)


def plan():
    pairs = list(_walk(APP, SPACE))                                              # app.py, xai_filters.py, xai/, appcore/, meta/
    pairs += [(KOD / "baselines" / "features_classical.py", SPACE / "baselines" / "features_classical.py")]
    pairs += list(_walk(SRC, SPACE)) if SRC.is_dir() else []
    pairs += [(ROOT / "baselines" / "results" / "results_tables.md", SPACE / "baselines" / "results" / "results_tables.md")]
    for n in ("summary_test_agg.csv", "tau.json"):
        pairs += [(ROOT / "phase2" / "experiments" / "results" / n, SPACE / "phase2" / "experiments" / "results" / n)]
    pairs += list(_walk(ROOT / "docs", SPACE / "docs"))
    return pairs


def build(check: bool = False, log=print) -> int:
    pairs = plan()
    missing = [s for s, _ in pairs if not s.is_file()]
    if missing:
        raise FileNotFoundError("Space kaynağı eksik: " + ", ".join(str(m) for m in missing))
    changed = [(s, d) for s, d in pairs if not (d.is_file() and filecmp.cmp(s, d, shallow=False))]
    expected = {d.resolve() for _, d in pairs}
    stale = [p for p in SPACE.rglob("*") if p.is_file() and p.resolve() not in expected
             and not (set(p.relative_to(SPACE).parts) & (SKIP_DIRS | {"_sample"})) and p.suffix != ".pyc"] if SPACE.is_dir() else []
    if KOD == ROOT:  # eski (taşıma öncesi) düzen: dağıtım betikleri hf_space/ içinde durur — hiçbir şey silinmez
        stale = []
    for s, d in changed:
        log(f"[space] {'FARK' if check else 'kopya'}: {d.relative_to(ROOT)}  ←  {s.relative_to(ROOT)}")
        if not check:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
    for p in stale:
        log(f"[space] {'FAZLA' if check else 'silindi'}: {p.relative_to(ROOT)} (kaynağı yok)")
        if not check:
            p.unlink()
    log(f"[space] {len(pairs)} dosya · {len(changed)} {'farklı' if check else 'güncellendi'} · {len(stale)} {'fazla' if check else 'kaldırıldı'} → {SPACE}")
    return len(changed) + len(stale)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="yalnız karşılaştır, yazma")
    a = ap.parse_args()
    n = build(check=a.check)
    sys.exit(1 if (a.check and n) else 0)
