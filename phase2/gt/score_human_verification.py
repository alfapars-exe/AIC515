#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""score_human_verification.py — doldurulmuş verification_human.xlsx'i puanlar.
Kullanım: python score_human_verification.py [--form verification_human.xlsx] [--r2 verification_human_r2.xlsx]
Çıktı : human_verification_summary.json / .md (+ eksik/geçersiz hücre listesi ekrana)
- Karar dağılımları (Wilson %95 GA) + tabaka-ağırlıklı tüm-train tahmini
- İnsan–VLM uyumu: yüzde uyum, Cohen κ (ağırlıksız; 3 düzeyli kararlar için ayrıca doğrusal-ağırlıklı), bootstrap %95 GA
- İnsan–otomatik bayrak çapraz tabloları; isteğe bağlı insan–insan κ
"""
import argparse, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ALLOWED = {"bbox_verdict": ["OK", "PARTIAL", "WRONG"], "eps8_verdict": ["OK", "PARTIAL", "WRONG"],
           "glyph_v2_verdict": ["OK", "PARTIAL", "WRONG"], "outside_change": ["no", "yes", "unclear"],
           "green_marker": ["no", "yes", "unsure"]}
ORD = {"OK": 0, "PARTIAL": 1, "WRONG": 2}


def load(path):
    d = pd.read_excel(path, sheet_name="Form", dtype=str).fillna("")
    d["new_id"] = d["new_id"].str.zfill(9)
    bad = []
    for c, al in ALLOWED.items():
        d[c] = d[c].str.strip()
        low = {a.lower(): a for a in al}
        d[c] = d[c].map(lambda v: low.get(v.lower(), v))
        for i, v in zip(d["new_id"], d[c]):
            if v not in al:
                bad.append((i, c, v))
    return d, bad


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 3
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(p, 4), round(c - h, 4), round(c + h, 4)]


def kappa(a, b, cats, weighted=False):
    idx = {c: i for i, c in enumerate(cats)}; k = len(cats)
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    if n == 0:
        return float("nan")
    w = np.array([[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]) if weighted else 1 - np.eye(k)
    exp = np.outer(m.sum(1), m.sum(0)) / n
    de = (w * exp).sum()
    return float("nan") if de == 0 else float(1 - (w * m).sum() / de)


def agree(a, b, cats, weighted=False, B=1000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.RandomState(seed); ks = []
    for _ in range(B):
        s = rng.randint(0, len(a), len(a)); ks.append(kappa(a[s], b[s], cats, weighted))
    ks = np.array(ks)
    return {"n": int(len(a)), "pct_agree": round(float((a == b).mean()), 4),
            "kappa": round(kappa(a, b, cats, weighted), 4),
            "kappa_ci95": [round(float(np.nanpercentile(ks, 2.5)), 4), round(float(np.nanpercentile(ks, 97.5)), 4)],
            "crosstab{human:{other:n}}": pd.crosstab(pd.Series(a, name="human"), pd.Series(b, name="other")).to_dict(orient="index")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", default=os.path.join(HERE, "verification_human.xlsx"))
    ap.add_argument("--r2", default=os.path.join(HERE, "verification_human_r2.xlsx"))
    args = ap.parse_args()
    d, bad = load(args.form)
    if bad:
        print("EKSİK/GEÇERSİZ %d hücre (ilk 30):" % len(bad))
        for b in bad[:30]:
            print("  ", b)
        print("Önce bunları tamamlayın; puanlama yalnız tam satırlarla yapılır.")
    full = d[[all(r[c] in ALLOWED[c] for c in ALLOWED) for _, r in d.iterrows()]].copy()
    out = {"form": os.path.basename(args.form), "n_rows": int(len(d)), "n_complete": int(len(full))}

    # 1) dağılımlar
    dist = {}
    for c, al in ALLOWED.items():
        dist[c] = {v: {"k": int((full[c] == v).sum()), "p_ci95": wilson(int((full[c] == v).sum()), len(full))} for v in al}
    out["distribution"] = dist

    # 2) tabaka-ağırlıklı tüm-train tahmini (örneklem orantılı ama tabaka başına min 1 → hafif düzeltme)
    sm = json.load(open(os.path.join(HERE, "verification_summary.json"), encoding="utf-8"))
    wmap = {r["stratum"]: r["n_available"] / r["n_sampled"] for r in sm["strata_allocation"] if r["n_sampled"] > 0}
    full["w"] = [wmap.get(str((s, t, a)), np.nan) for s, t, a in zip(full["source"], full["tool"], full["area_bin"])]
    if full["w"].notna().all() and len(full):
        out["weighted_fulltrain"] = {c: {v: round(float(full.loc[full[c] == v, "w"].sum() / full["w"].sum()), 4) for v in al}
                                     for c, al in ALLOWED.items()}

    # 3) insan–VLM
    v = pd.read_csv(os.path.join(HERE, "verification_sheet_vlm.csv"), dtype=str, keep_default_na=False)  # "n/a" metin olarak kalsın
    v["new_id"] = v["new_id"].str.zfill(9)
    m = full.merge(v, on="new_id", how="inner", suffixes=("", "_v"))
    m = m[m["vlm_parse_ok"].str.lower() == "true"]
    vlm = {}
    for hc, vc in [("eps8_verdict", "vlm_eps8_verdict"), ("glyph_v2_verdict", "vlm_glyph_verdict")]:
        a = m[hc].tolist(); b = m[vc].str.upper().tolist()
        vlm[hc] = {"unweighted": agree(a, b, ALLOWED[hc]), "linear_weighted": agree(a, b, ALLOWED[hc], weighted=True)}
    oc = m["vlm_outside_bbox_real_change"].str.lower().replace({"n/a": "no", "": "unclear"})
    vlm["outside_change"] = agree(m["outside_change"].tolist(), oc.tolist(), ALLOWED["outside_change"])
    vlm["green_marker"] = agree(m["green_marker"].tolist(), m["vlm_green_marker_line"].str.lower().tolist(), ALLOWED["green_marker"])
    out["human_vs_vlm"] = vlm
    out["human_vs_vlm_note"] = "VLM glif kararı hangi glif sürümüne bakılarak verildiyse ona göre yorumlanmalı; insan formu glyph_v2'yi değerlendirir."

    # 4) insan–objektif/otomatik
    mo = m[m["green_residue_objective"].isin(["yes", "no"])]
    out["green_human_vs_objective"] = agree(mo["green_marker"].tolist(), mo["green_residue_objective"].tolist(), ALLOWED["green_marker"])
    fl = full["auto_flags"].str.contains("outside_bbox_component")
    out["outside_human_vs_autoflag"] = pd.crosstab(full["outside_change"], fl.map({True: "flag", False: "no_flag"})).to_dict()
    lc = full["auto_flags"].str.contains("low_bbox_coverage_eps8")
    out["eps8_human_vs_lowcoverage_flag"] = pd.crosstab(full["eps8_verdict"], lc.map({True: "flag", False: "no_flag"})).to_dict()

    # 5) isteğe bağlı ikinci değerlendirici
    if os.path.exists(args.r2):
        d2, bad2 = load(args.r2)
        mm = full.merge(d2, on="new_id", suffixes=("", "_r2"))
        mm = mm[[all(r[c + "_r2"] in ALLOWED[c] for c in ALLOWED) for _, r in mm.iterrows()]]
        out["human_vs_human"] = {c: agree(mm[c].tolist(), mm[c + "_r2"].tolist(), ALLOWED[c], weighted=(c in ("eps8_verdict", "glyph_v2_verdict", "bbox_verdict")))
                                 for c in ALLOWED}

    json.dump(out, open(os.path.join(HERE, "human_verification_summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)

    # md özet + bildiri cümlesi
    n = len(full); e = dist["eps8_verdict"]; g = dist["glyph_v2_verdict"]; b = dist["bbox_verdict"]
    ke = vlm["eps8_verdict"]["unweighted"]
    L = ["# İnsan GT doğrulaması — özet", "", "Tam satır: %d / %d" % (n, len(d)), "",
         "| karar | OK | PARTIAL | WRONG |", "|---|---|---|---|"]
    for c in ("bbox_verdict", "eps8_verdict", "glyph_v2_verdict"):
        L.append("| %s | %s |" % (c, " | ".join("%d (%.0f%% [%.0f, %.0f])" % (dist[c][x]["k"], 100 * dist[c][x]["p_ci95"][0], 100 * dist[c][x]["p_ci95"][1], 100 * dist[c][x]["p_ci95"][2]) for x in ALLOWED[c])))
    L += ["", "| insan–VLM | n | uyum | κ [%95 GA] |", "|---|---|---|---|"]
    for c in ("eps8_verdict", "glyph_v2_verdict"):
        u = vlm[c]["unweighted"]; w = vlm[c]["linear_weighted"]
        L.append("| %s | %d | %.2f | %.2f [%.2f, %.2f] (ağırlıklı %.2f) |" % (c, u["n"], u["pct_agree"], u["kappa"], *u["kappa_ci95"], w["kappa"]))
    for c in ("outside_change", "green_marker"):
        u = vlm[c]; L.append("| %s | %d | %.2f | %.2f [%.2f, %.2f] |" % (c, u["n"], u["pct_agree"], u["kappa"], *u["kappa_ci95"]))
    if n:
        L += ["", "**Bildiri cümlesi taslağı:** Tabakalı 100 eğitim çiftinin insan denetiminde piksel-değişim GT'si %d/%d OK, %d PARTIAL, %d WRONG; glif GT'si %d/%d OK; resmî kutu %d/%d OK bulunmuştur. İnsan ile VLM denetimi arasındaki uyum piksel-değişim GT'si için %%%.0f (Cohen κ = %.2f [%.2f, %.2f], n = %d) olmuştur." % (
            e["OK"]["k"], n, e["PARTIAL"]["k"], e["WRONG"]["k"], g["OK"]["k"], n, b["OK"]["k"], n, 100 * ke["pct_agree"], ke["kappa"], *ke["kappa_ci95"], ke["n"])]
    open(os.path.join(HERE, "human_verification_summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
