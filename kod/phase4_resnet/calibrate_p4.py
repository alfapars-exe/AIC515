"""
calibrate_p4.py — Faz 4: görüntü düzeyi agregasyon + eşik kalibrasyonu ve raporlama (TR; yalnız CPU).

Kural (koşmadan önce sabitlendi — bkz. README.md §Analiz planı):
  * Aday agregasyonlar: max, top2, top3, top5 (en yüksek k patch skorunun ortalaması).
  * Agregasyon SEÇİMİ: tam veri/ham varyantın (resnet18_ft_full_s42/43/44) val AUROC ortalaması en yüksek olan;
    0.002 içindeki farklarda küçük k tercih edilir. Seçilen agregasyon TÜM modellere uygulanır.
  * Eşik: model başına val'de Youden-J (uygulamadaki U-Net/YOLO image_tau.json kuralıyla aynı).
  * Test (812 sahte + 89 eşli authentic) yalnız raporlama içindir; hiçbir seçim testte yapılmaz.
  * Eski kural (max patch >= 0.5) aynı kümelerde karşılaştırma için raporlanır.
Çıktı: phase4_resnet/results/{image_level_val.csv, image_level_test.csv, marker_probes.csv, breakdown_test.csv,
       resnet_image_tau.json, tables_p4.md}
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import p4common as P

AGGS = {"max": 1, "top2": 2, "top3": 3, "top5": 5}
REF_FAMILY = "resnet18_ft_full_s4"          # agregasyon seçiminde kullanılan koşular (ham, 3 tohum)
OLD_AGG, OLD_THR = "max", 0.5


def load_set(stem, name, sfx=""):
    p = P.PRED / stem / f"{name}{sfx}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    s, off, ids = z["scores"], z["offsets"], [str(i) for i in z["ids"]]
    rows = []
    for k, i in enumerate(ids):
        v = np.sort(s[off[k]:off[k + 1]])[::-1]
        rows.append(dict(new_id=i, n_patches=len(v), **{a: float(v[:n].mean()) for a, n in AGGS.items()}))
    return pd.DataFrame(rows).set_index("new_id")


def youden(pos, neg):
    """Skorlar üzerinde J = TPR - FPR'yi en büyükleyen eşik (karar: skor >= eşik). Eşitlikte en yüksek eşik."""
    cand = np.unique(np.concatenate([pos, neg]))
    tpr = (pos[None, :] >= cand[:, None]).mean(1); fpr = (neg[None, :] >= cand[:, None]).mean(1)
    j = tpr - fpr; k = len(j) - 1 - int(np.argmax(j[::-1]))
    return float(cand[k]), float(tpr[k]), float(fpr[k])


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((c - h) / d), float((c + h) / d))


def auroc(pos, neg):
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def boot_auroc(pos, gpos, neg, gneg, B=1000, seed=0):
    """Kaynak-belge grubu üzerinde küme bootstrap %95 GA (sahte ve authentic aynı grup kimliğini paylaşır)."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; s = np.r_[pos, neg]; g = np.r_[gpos, gneg]
    ug, inv = np.unique(g, return_inverse=True)
    out = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[inv].astype(float)
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        out.append(roc_auc_score(y, s, sample_weight=w))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    meta = pd.read_csv(P.RES / f"image_sets_meta{sfx}.csv", dtype={"new_id": str}); meta["new_id"] = meta.new_id.str.zfill(9)
    M = {n: g.set_index("new_id") for n, g in meta.groupby("set")}
    stems = sorted(d.name for d in P.PRED.iterdir() if d.is_dir() and (d / f"val_forged{sfx}.npz").exists())
    S = {s: {n: load_set(s, n, sfx) for n in ("val_forged", "val_auth", "val_forged_mr", "val_auth_marker", "test_forged", "test_auth")} for s in stems}

    # 1) val: tüm agregasyonlar
    val_rows = []
    for s in stems:
        for a in AGGS:
            pos, neg = S[s]["val_forged"][a].values, S[s]["val_auth"][a].values
            thr, tpr, fpr = youden(pos, neg)
            val_rows.append(dict(model=s, agg=a, val_auroc=auroc(pos, neg), thr=thr, val_tpr=tpr, val_fpr=fpr, youden_j=tpr - fpr,
                                 n_pos=len(pos), n_neg=len(neg)))
    val = pd.DataFrame(val_rows); val.to_csv(P.RES / f"image_level_val{sfx}.csv", index=False)

    # 2) agregasyon seçimi (önceden sabitlenmiş kural)
    ref = val[val.model.str.startswith(REF_FAMILY) & ~val.model.str.contains("_mr")]
    if len(ref) == 0:
        ref = val
    mean_auc = ref.groupby("agg").val_auroc.mean().reindex(list(AGGS))
    best = mean_auc.max(); chosen = next(a for a in AGGS if mean_auc[a] >= best - 0.002)
    print("agregasyon val AUROC ortalaması:", mean_auc.round(4).to_dict(), "→ seçilen:", chosen, flush=True)

    # 3) test + eski kural + kırılımlar + işaretçi sondaları
    mres = pd.read_csv(P.ROOT / "phase3" / "results" / "marker_residue.csv", dtype={"new_id": str}); mres["new_id"] = mres.new_id.str.zfill(9)
    ring = mres[mres.split == "test"].set_index("new_id").ring_green_px
    test_rows, brk_rows, probe_rows, tau = [], [], [], {"_doc": "ResNet-18 görüntü düzeyi karar kuralı: skor = en yüksek k patch skorunun ortalaması (agg), "
                                                              "eşik val 487 sahte + 487 eşleşen authentic üzerinde Youden-J. Üretici: kod/phase4_resnet/calibrate_p4.py",
                                                       "_agg_selection": dict(rule="tam veri/ham 3 tohumun val AUROC ortalaması; 0.002 içinde küçük k", mean_val_auroc=mean_auc.round(4).to_dict(), chosen=chosen)}
    for s in stems:
        v = val[(val.model == s) & (val["agg"] == chosen)].iloc[0]
        thr = float(v.thr)
        tau[s] = dict(agg=chosen, k=AGGS[chosen], thr=round(thr, 6), tpr=round(float(v.val_tpr), 3), fpr=round(float(v.val_fpr), 3),
                      auroc=round(float(v.val_auroc), 3), n_pos=int(v.n_pos), n_neg=int(v.n_neg))
        for rule, a, t in (("kalibre", chosen, thr), ("eski_max_0.5", OLD_AGG, OLD_THR)):
            for split in ("val", "test"):
                P_, N_ = S[s][f"{split}_forged"], S[s][f"{split}_auth"]
                if P_ is None or N_ is None:
                    continue
                pos, neg = P_[a].values, N_[a].values
                tp, fp = int((pos >= t).sum()), int((neg >= t).sum())
                row = dict(model=s, rule=rule, agg=a, thr=t, split=split, n_pos=len(pos), n_neg=len(neg), tpr=tp / len(pos), fpr=fp / len(neg),
                           bal_acc=0.5 * (tp / len(pos) + 1 - fp / len(neg)), auroc=auroc(pos, neg))
                row["tpr_lo"], row["tpr_hi"] = wilson(tp, len(pos)); row["fpr_lo"], row["fpr_hi"] = wilson(fp, len(neg))
                if split == "test":
                    gp = M["test_forged"].loc[P_.index, "group"].values; gn = M["test_auth"].loc[N_.index, "group"].values
                    row["auroc_lo"], row["auroc_hi"] = boot_auroc(pos, gp, neg, gn)
                    # eşli alt küme: yalnız authentic eşi olan 89 sahte (dengeli)
                    pp = P_.loc[P_.index.intersection(N_.index), a].values
                    row["paired_n"] = len(pp)
                    if len(pp):
                        row["paired_tpr"] = float((pp >= t).mean()); row["paired_auroc"] = auroc(pp, neg)
                test_rows.append(row)
        # kırılımlar (test sahteleri, kalibre kural): üretici ve işaretçi alt kümesi
        Pt = S[s]["test_forged"]
        if Pt is not None:
            d = pd.DataFrame(dict(score=Pt[chosen], tool=M["test_forged"].loc[Pt.index, "tool"], marker=(ring.reindex(Pt.index).fillna(0) > 0).map({True: "işaretçili", False: "işaretçisiz"}),
                                  area=M["test_forged"].loc[Pt.index, "area_bin"], clean=M["test_forged"].loc[Pt.index, "clean_test"].map({True: "temiz test", False: "eğitimle kaynak paylaşan"})))
            neg = S[s]["test_auth"][chosen].values
            for col in ("tool", "marker", "area", "clean"):
                for val_, g in d.groupby(col):
                    brk_rows.append(dict(model=s, by=col, value=val_, n=len(g), tpr=float((g.score >= thr).mean()),
                                         auroc_vs_test_auth=auroc(g.score.values, neg) if len(g) > 1 else float("nan")))
        # işaretçi sondaları (val, kalıntılı kimlikler): onarım sahteyi kaçırtıyor mu, yalnız işaretçi alarm verdiriyor mu
        Fm, Am = S[s]["val_forged_mr"], S[s]["val_auth_marker"]
        if Fm is not None and Am is not None:
            ids = Fm.index.intersection(Am.index).intersection(S[s]["val_forged"].index).intersection(S[s]["val_auth"].index)
            if len(ids) == 0:
                continue
            f0 = S[s]["val_forged"].loc[ids, chosen].values; f1 = Fm.loc[ids, chosen].values
            a0 = S[s]["val_auth"].loc[ids, chosen].values; a1 = Am.loc[ids, chosen].values
            probe_rows.append(dict(model=s, n=len(ids), thr=thr,
                                   tpr_forged=float((f0 >= thr).mean()), tpr_forged_repaired=float((f1 >= thr).mean()),
                                   fpr_auth=float((a0 >= thr).mean()), fpr_auth_marker_only=float((a1 >= thr).mean()),
                                   auroc_forged_vs_auth=auroc(f0, a0), auroc_repaired_vs_auth=auroc(f1, a0),
                                   auroc_markeronly_vs_auth=auroc(a1, a0), median_score_drop_on_repair=float(np.median(f0 - f1))))
    test = pd.DataFrame(test_rows); test.to_csv(P.RES / f"image_level_test{sfx}.csv", index=False)
    brk = pd.DataFrame(brk_rows); brk.to_csv(P.RES / f"breakdown_test{sfx}.csv", index=False)
    prb = pd.DataFrame(probe_rows); prb.to_csv(P.RES / f"marker_probes{sfx}.csv", index=False)
    json.dump(tau, open(P.RES / f"resnet_image_tau{sfx}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # 4) özet tablo (varyant ortalaması +- std)
    def fam(m):
        if m.startswith("resnet18_ft_full_mr"): return "Faz 4 tam veri + işaretçi onarımı"
        if m.startswith("resnet18_ft_full"): return "Faz 4 tam veri (ham)"
        return f"Faz 1 ({m})"
    md = [f"# Faz 4 — ResNet-18 görüntü düzeyi sonuçlar (otomatik: calibrate_p4.py)\n", f"Seçilen agregasyon: **{chosen}** · val AUROC ortalamaları (ham, 3 tohum): {mean_auc.round(4).to_dict()}\n"]
    t = test.copy(); t["aile"] = t.model.map(fam)
    for split in ("val", "test"):
        g = t[t.split == split].groupby(["aile", "rule"]).agg(n_model=("model", "nunique"), auroc=("auroc", "mean"), auroc_std=("auroc", "std"),
                                                             tpr=("tpr", "mean"), tpr_std=("tpr", "std"), fpr=("fpr", "mean"), fpr_std=("fpr", "std"),
                                                             bal_acc=("bal_acc", "mean")).reset_index().round(3)
        md += [f"\n## {split} — aile × karar kuralı (ortalama ± std, tohumlar üzerinden)\n", g.to_markdown(index=False)]
    md += ["\n## Model başına (kalibre kural)\n", test[test.rule == "kalibre"].round(3).to_markdown(index=False)]
    if len(prb):
        p = prb.copy(); p["aile"] = p.model.map(fam)
        md += ["\n## İşaretçi sondaları (val, kalıntılı kimlikler)\n", p.groupby("aile").mean(numeric_only=True).round(3).reset_index().to_markdown(index=False),
               "\n### Model başına\n", prb.round(3).to_markdown(index=False)]
    if len(brk):
        b = brk.copy(); b["aile"] = b.model.map(fam)
        md += ["\n## Test kırılımları (kalibre kural; TPR)\n", b.groupby(["aile", "by", "value"]).agg(n=("n", "first"), tpr=("tpr", "mean"), tpr_std=("tpr", "std"), auroc=("auroc_vs_test_auth", "mean")).round(3).reset_index().to_markdown(index=False)]
    md += ["\n## Val — tüm agregasyonlar\n", val.round(4).to_markdown(index=False)]
    (P.RES / f"tables_p4{sfx}.md").write_text("\n".join(md), encoding="utf-8")
    print(test[test.rule == "kalibre"][["model", "split", "auroc", "tpr", "fpr", "thr"]].round(3).to_string(index=False), flush=True)
    if len(prb):
        print(prb.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
