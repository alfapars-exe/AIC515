"""
build_patches_p4.py — Faz 4 parça önbelleği (TR). Train (dışlananlar hariç 2748 görüntü, görüntü başına 2 tampered) ve
val (487 görüntü, 1 tampered) için parça indeksini kurar; ham ve işaretçi-onarımlı parçaları TEK geçişte çıkarır.
Çıktı: phase4_resnet/results/patch_index_{train,val}.csv, phase4_resnet/cache/patches_{split}_{raw,mr}.npy
Kullanım: python build_patches_p4.py [--limit 150 --tag dryrun]
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import p4common as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="duman testi: split başına ilk N görüntü")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    P.ensure_dirs()
    summary = {}
    for split, n_tamp in (("train", 2), ("val", 1)):
        t0 = time.time()
        idx_p, raw_p, mr_p = P.cache_paths(split, args.tag)
        tab = P.split_ids_table(split)
        idx = P.build_index(split, n_tamp, args.limit)
        raw, mr, n_rep = P.extract_both(idx, tab)
        idx.to_csv(idx_p, index=False); np.save(raw_p, raw); np.save(mr_p, mr)
        diff = int((raw != mr).reshape(len(raw), -1).any(1).sum())
        s = dict(n_images=int(idx.new_id.nunique()), n_patches=int(len(idx)), by_type=idx.patch_type.value_counts().to_dict(),
                 n_images_with_repair=int(n_rep), n_patches_changed_by_repair=diff,
                 changed_by_type=idx[(raw != mr).reshape(len(raw), -1).any(1)].patch_type.value_counts().to_dict(),
                 gib=round(raw.nbytes / 2 ** 30, 2), sec=round(time.time() - t0, 1))
        summary[split] = s
        print(split, json.dumps(s, ensure_ascii=False), flush=True)
    sfx = f"_{args.tag}" if args.tag else ""
    json.dump(summary, open(P.RES / f"patch_build_summary{sfx}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
