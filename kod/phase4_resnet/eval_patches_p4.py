"""
eval_patches_p4.py — Faz 4: verilen ResNet-18 ağırlıklarını AYNI val parçalarında (487 görüntü; ham ve işaretçi-onarımlı)
değerlendirir. Faz 1 modeli (400 görüntüyle eğitilmiş resnet18_ft_best) ile tam veri modellerini aynı kümede karşılaştırmak içindir.
Çıktı: phase4_resnet/results/patch_level_val.csv
Kullanım: python eval_patches_p4.py --models resnet18_ft_best,resnet18_ft_full_s42,...
"""
from __future__ import annotations
import argparse
import pandas as pd
import torch
import p4common as P
from train_resnet_p4 import predict, patch_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = torch.device("cuda")
    rows = []
    for variant in ("raw", "mr"):
        idx, X = P.get_patches("val", variant, args.tag)
        y, pt = idx.label.values.astype(int), idx.patch_type.values
        for stem in [s for s in args.models.split(",") if s]:
            m = P.load_resnet(stem, device).to(memory_format=torch.channels_last)
            s = predict(m, X, device)
            rows.append(dict(model=stem, val_variant=variant, **patch_metrics(y, s, pt)))
            for tool in sorted(idx.tool.unique()):
                k = (y == 0) | (idx.tool.values == tool)
                rows[-1][f"auroc_tool_{tool}"] = patch_metrics(y[k], s[k], pt[k])["auroc"]
    df = pd.DataFrame(rows)
    sfx = f"_{args.tag}" if args.tag else ""
    df.to_csv(P.RES / f"patch_level_val{sfx}.csv", index=False)
    print(df.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
