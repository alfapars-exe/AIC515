"""
v2_marker_paired_check.py — AIForge-Doc v2'de yeşil işaretçi kalıntısının EŞLİ kuralla sayımı (TR).

v1'deki kalıntı tanımı (kod/phase3/make_marker_repaired.py; bildiri Sec. III):
    residue = gmask(forged) & ~gmask(authentic) & ring(bbox, 6 px)      kalıntı var  <=>  residue >= 20 piksel
v2_eval.py 'prep' aşaması ise yalnız renge bakar (authentic ile karşılaştırmaz; halka = kutu + 4 px, içi dolu):
    green_px_in_ring = gmask(forged) & dilate(bbox, 9x9)                green_residue <=> >= 20 piksel
Bu betik v2'nin bütün çiftlerinde v1 tanımını birebir uygular, iki kuralı yan yana raporlar ve sahte ile özgün
görüntünün kutu dışında farklılaşıp farklılaşmadığını sayar.

Çıktı : phase3_v2/results/v2_marker_paired_check.csv (çift başına) ve v2_marker_paired_check.json (özet)
Kullanım: python kod/phase3_v2/v2_marker_paired_check.py [--threads 4] [--limit N]
Yarıda kesilirse kaldığı yerden sürer (CSV'de bulunan anahtarlar atlanır). Bağımlılık: numpy, pandas, opencv.
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.append(str(next(p for p in Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT, out_dir  # noqa: E402

OUT = out_dir(__file__)                      # <proje>/phase3_v2
RES = OUT / "results"
TAB = OUT / "v2_table.csv"
CSV = RES / "v2_marker_paired_check.csv"
CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))   # scipy.ndimage varsayılan yapı elemanı ile aynı
COLS = ["key", "split", "primary", "n_residue_paired", "n_green_ring_coloronly", "n_diff_outside_box", "note"]


def v2_root() -> Path:
    for c in [ROOT / "data" / "AIForge-Doc-v2"] + [Path(p) for p in glob.glob(str(ROOT / "data" / "AIForge-Doc-v2" / "*"))]:
        if (c / "TrainingSet" / "authentic").is_dir():
            return c
    raise SystemExit("v2 tam paketi (authentic dahil) bulunamadı: data/AIForge-Doc-v2/")


def gmask_bgr(bgr: np.ndarray) -> np.ndarray:
    """Yeşil işaretçi pikseli; p3ext.gmask ile aynı kural (burada kanal sırası BGR)."""
    b, g, r = (bgr[..., k].astype(np.int16) for k in range(3))
    return (g - np.maximum(r, b) > 50) & (g > 110)


def ring_of(mb: np.ndarray) -> np.ndarray:
    """make_marker_repaired.ring_of ile aynı: 6 yinelemeli haç genişletme eksi 6 yinelemeli haç aşındırma (sınır dışı = 0)."""
    u = mb.astype(np.uint8)
    dil = cv2.dilate(u, CROSS, iterations=6, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    ero = cv2.erode(u, CROSS, iterations=6, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (dil > 0) & ~(ero > 0)


def work(root: Path, r) -> dict:
    row = dict(key=r.key, split=r.split, primary=bool(r.primary), n_residue_paired=-1, n_green_ring_coloronly=-1, n_diff_outside_box=-1, note="")
    f = cv2.imread(str(root / r.image), cv2.IMREAD_COLOR)
    a = cv2.imread(str(root / r.image.replace("/images/", "/authentic/")), cv2.IMREAD_COLOR)
    m = cv2.imread(str(root / r.mask), cv2.IMREAD_GRAYSCALE)
    if f is None or m is None:
        row["note"] = "sahte görüntü ya da maske okunamadı"; return row
    mb = m > 127
    ring = ring_of(mb)
    gf = gmask_bgr(f)
    row["n_green_ring_coloronly"] = int((gf & ring).sum())
    if a is None or a.shape != f.shape:
        row["note"] = "özgün eş yok ya da boyutu farklı"; return row
    row["n_residue_paired"] = int((gf & ~gmask_bgr(a) & ring).sum())
    row["n_diff_outside_box"] = int(((f != a).any(axis=2) & ~mb).sum())
    return row


def main():
    for _s in (sys.stdout, sys.stderr):   # Windows'ta çıktı dosyaya ya da boruya yönlendirilince cp1254 '−' karakterinde çöker
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="en çok N yeni çift işle (0 = tümü)")
    ap.add_argument("--max-seconds", type=float, default=0, help="bu süre dolunca kaydedip çık (0 = sınırsız)")
    a = ap.parse_args()
    root = v2_root()
    tab = pd.read_csv(TAB, low_memory=False)
    done = pd.read_csv(CSV) if CSV.is_file() else pd.DataFrame(columns=COLS)
    todo = tab[~tab.key.isin(set(done.key))]
    if a.limit:
        todo = todo.head(a.limit)
    RES.mkdir(parents=True, exist_ok=True)
    t0, rows, items = time.time(), [], list(todo.itertuples())
    with ThreadPoolExecutor(a.threads) as ex:
        for i in range(0, len(items), 64):                       # parça parça: süre sınırı parçalar arasında denetlenir
            rows += list(ex.map(lambda r: work(root, r), items[i:i + 64]))
            if a.max_seconds and time.time() - t0 > a.max_seconds:
                break
    if rows:
        done = pd.concat([done, pd.DataFrame(rows, columns=COLS)], ignore_index=True)
        done.to_csv(CSV, index=False)
    left = len(tab) - len(done)
    print(f"işlendi {len(rows)} · toplam {len(done)}/{len(tab)} · kalan {left} · {time.time() - t0:.0f}s", flush=True)
    if left:
        return
    ok = done[done.n_residue_paired >= 0]
    summ = {
        "n_pairs": int(len(tab)), "n_pairs_with_authentic": int(len(ok)),
        "rule_v1_paired": "gmask(forged) & ~gmask(authentic) & ring(bbox, 6 px haç genişletme − 6 px haç aşındırma); kalıntı <=> >= 20 piksel",
        "paired_residue_ge20": int((ok.n_residue_paired >= 20).sum()),
        "paired_residue_any": int((ok.n_residue_paired > 0).sum()),
        "paired_residue_max_px": int(ok.n_residue_paired.max()),
        "coloronly_ring_ge20": int((done.n_green_ring_coloronly >= 20).sum()),
        "coloronly_ring_any": int((done.n_green_ring_coloronly > 0).sum()),
        "coloronly_ring_ge20_keys": done.loc[done.n_green_ring_coloronly >= 20, "key"].tolist(),
        "v2_eval_green_residue": int(tab.green_residue.sum()) if "green_residue" in tab else None,
        "v2_eval_green_residue_keys": tab.loc[tab.green_residue.astype(bool), "key"].tolist() if "green_residue" in tab else None,
        "pairs_with_any_difference_outside_box": int((ok.n_diff_outside_box > 0).sum()),
        "primary": {"n": int(done.primary.sum()), "paired_residue_ge20": int(((done.n_residue_paired >= 20) & done.primary).sum()),
                    "coloronly_ring_ge20": int(((done.n_green_ring_coloronly >= 20) & done.primary).sum())},
    }
    json.dump(summ, open(RES / "v2_marker_paired_check.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
