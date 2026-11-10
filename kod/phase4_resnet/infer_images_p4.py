"""
infer_images_p4.py — Faz 4: tam görüntü kayan pencere skorları (TR). Uygulamadaki `_predict_resnet` protokolü
(224 pencere, adım 112, bf16) ile her model için görüntü başına TÜM patch skorları kaydedilir; görüntü düzeyi
agregasyon ve eşik seçimi ayrı adımdadır (calibrate_p4.py) — böylece kalibrasyon için çıkarım tekrarlanmaz.

Görüntü kümeleri:
  val_forged       487 val sahtesi                              val_auth        eşleşen authentic (487)
  val_forged_mr    işaretçi kalıntılı val sahteleri, onarılmış  val_auth_marker aynı kimliklerin authentic'i + yalnız işaretçi
  test_forged      812 resmî test sahtesi                       test_auth       eşi doğrulanmış test authentic (89)
Çıktı: phase4_resnet/preds/<stem>/<küme>.npz (scores, offsets, yx, ids), phase4_resnet/results/image_sets_meta.csv
Kullanım: python infer_images_p4.py --models resnet18_ft_full_s42,resnet18_ft_best [--limit 20 --tag dryrun]
"""
from __future__ import annotations
import argparse, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
import p4common as P
D = P.D

SETS = ["val_forged", "val_auth", "val_forged_mr", "val_auth_marker", "test_forged", "test_auth"]


def marker_ids() -> set:
    """Kalıntısı (n_residue >= 20) olan ve onarılmış dosyası bulunan val kimlikleri (phase3/results/marker_residue.csv)."""
    mr = pd.read_csv(P.ROOT / "phase3" / "results" / "marker_residue.csv", dtype={"new_id": str})
    mr["new_id"] = mr.new_id.str.zfill(9)
    v = mr[(mr.split == "val") & (mr.has_residue == 1)]
    return {i for i in v.new_id if (P.REPAIR_DIR / f"{i}.png").exists()}


def items_of(name: str, limit: int = 0):
    """[(new_id, meta_row, yükleyici)] — yükleyici RGB uint8 döndürür."""
    split = "test" if name.startswith("test") else "val"
    tab = D.split_table(split)
    paired = tab[tab.pair_accepted.astype(bool) & tab.pair_same_size.astype(bool)]
    mids = marker_ids() if name in ("val_forged_mr", "val_auth_marker") else None
    if name.endswith("_forged"):
        use = tab; load = lambda i, r: D.read_rgb(D.img_path(i, split))
    elif name.endswith("_auth"):
        use = paired; load = lambda i, r: D.read_rgb(D.auth_path(r["pair_name"], split))
    elif name == "val_forged_mr":
        use = tab[tab.index.isin(mids)]; load = lambda i, r: D.read_rgb(P.REPAIR_DIR / f"{i}.png")
    elif name == "val_auth_marker":
        use = paired[paired.index.isin(mids)]

        def load(i, r):
            f = D.read_rgb(D.img_path(i, split)); a = D.read_rgb(D.auth_path(r["pair_name"], split))
            resd = P.marker_residue(f, a, D.read_mask(D.bbox_mask_path(i, split)))
            out = a.copy(); out[resd] = f[resd]
            return out
    else:
        raise ValueError(name)
    if limit:
        use = use.iloc[:limit]
    return [(i, r, load) for i, r in use.iterrows()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="virgülle ayrılmış checkpoint adları")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--chunk", type=int, default=12)
    args = ap.parse_args()
    P.ensure_dirs()
    device = torch.device("cuda")
    stems = [s for s in args.models.split(",") if s]
    models = [P.load_resnet(s, device) for s in stems]
    sfx = f"_{args.tag}" if args.tag else ""
    meta_rows = []
    for name in args.sets.split(","):
        outs = [P.PRED / s / f"{name}{sfx}.npz" for s in stems]
        its = items_of(name, args.limit)
        t0 = time.time()
        if all(o.exists() for o in outs):
            print(f"skip {name} (hazır)", flush=True)
        else:
            sc = [[] for _ in stems]; yx, ids = [], []
            with ThreadPoolExecutor(6) as ex:
                for c in range(0, len(its), args.chunk):
                    chunk = its[c:c + args.chunk]
                    imgs = list(ex.map(lambda t: t[2](t[0], t[1]), chunk))
                    for (i, r, _), img in zip(chunk, imgs):
                        s, w = P.score_image(models, img, device)
                        for k in range(len(stems)):
                            sc[k].append(s[k])
                        yx.append(w); ids.append(i)
                    if (c // args.chunk) % 10 == 0:
                        print(f"  {name} {min(c + args.chunk, len(its))}/{len(its)} {time.time() - t0:.0f}s", flush=True)
            off = np.concatenate([[0], np.cumsum([len(w) for w in yx])]).astype(np.int64)
            for k, o in enumerate(outs):
                o.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(o, scores=np.concatenate(sc[k]), offsets=off, yx=np.concatenate(yx), ids=np.array(ids))
            print(f"done {name}: {len(ids)} görüntü, {off[-1]} patch, {time.time() - t0:.0f}s", flush=True)
        for i, r, _ in its:
            meta_rows.append(dict(set=name, new_id=i, tool=r["tool"], source=r["source"], area_bin=r["area_bin"], group=r["group"],
                                  W=int(r["W"]), H=int(r["H"]), clean_test=bool(r["clean_test"]) if "clean_test" in r.index else None))
    pd.DataFrame(meta_rows).to_csv(P.RES / f"image_sets_meta{sfx}.csv", index=False)


if __name__ == "__main__":
    main()
