"""
train_classical.py â€” B deneyi: el-yapÄ±mÄ± Ã¶znitelikler + SVM-RBF / RandomForest (TR).

AkÄ±ÅŸ: smoke_train / smoke_val patch'leri (common.data Ã¶nbelleÄŸi) â†’ features_classical.extract_many â†’
StandardScaler + SVM-RBF (kÃ¼Ã§Ã¼k C Ä±zgarasÄ±, 3-kat CV) ve RandomForest(500) â†’ val metrikleri
(accuracy, balanced accuracy, AUROC, AUPRC, F1), 'tampered vs clean_paired' alt kÃ¼mesinde AUROC,
tool / alan-dilimi kÄ±rÄ±lÄ±mÄ±, RF Ã¶znitelik Ã¶nemleri (CSV + figÃ¼r).
KullanÄ±m: python train_classical.py [--seed 42] [--n-tampered 2]
"""
from __future__ import annotations
import os, sys, json, time, argparse
os.environ.setdefault("USERNAME", "user")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import common.data as D
from common.metrics import classification_metrics, subset_auroc, safe_auroc
from features_classical import extract_many, feature_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-tampered", type=int, default=2)
    ap.add_argument("--rf-trees", type=int, default=500)
    args = ap.parse_args()
    D.ensure_dirs()
    t0 = time.time()
    cache = D.ImageCache()
    idx_tr, P_tr = D.get_or_build_patches("smoke_train", args.n_tampered, args.seed, cache=cache)
    idx_va, P_va = D.get_or_build_patches("smoke_val", 1, args.seed + 1, cache=cache)
    del cache
    print(f"patches: train {P_tr.shape} val {P_va.shape}", flush=True)

    tf = time.time()
    X_tr = extract_many(P_tr); X_va = extract_many(P_va)
    y_tr = idx_tr.label.values; y_va = idx_va.label.values
    feat_time = time.time() - tf
    print(f"features: {X_tr.shape} in {feat_time:.1f}s ({feat_time / (len(X_tr) + len(X_va)) * 1000:.1f} ms/patch)", flush=True)
    names = feature_names()
    np.savez_compressed(D.CACHE_DIR / "classical_features.npz", X_tr=X_tr, X_va=X_va, y_tr=y_tr, y_va=y_va)

    results, rows = {}, []
    cv = StratifiedKFold(3, shuffle=True, random_state=args.seed)

    # --- SVM-RBF
    ts = time.time()
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=False, random_state=args.seed))
    grid = GridSearchCV(svm, {"svc__C": [0.5, 1, 2, 5, 10], "svc__gamma": ["scale", 0.01, 0.05]},
                        scoring="roc_auc", cv=cv, n_jobs=1)
    # Windows sandbox'Ä±nda loky (named pipe) engelli â†’ thread tabanlÄ± paralellik
    grid.fit(X_tr, y_tr)
    svm_best = grid.best_estimator_
    # eÅŸik iÃ§in karar fonksiyonu â†’ [0,1] sigmoid benzeri Ã¶lÃ§ekleme (sÄ±ralama korunur; 0.5 eÅŸik = karar sÄ±nÄ±rÄ±)
    s_va = svm_best.decision_function(X_va)
    svm_score = 1 / (1 + np.exp(-s_va))
    svm_train_time = time.time() - ts
    print("SVM best:", grid.best_params_, "cv auroc %.3f" % grid.best_score_, flush=True)

    # --- RandomForest
    tr = time.time()
    rf = RandomForestClassifier(n_estimators=args.rf_trees, random_state=args.seed, n_jobs=1, min_samples_leaf=2)
    rf.fit(X_tr, y_tr)
    rf_score = rf.predict_proba(X_va)[:, 1]
    rf_train_time = time.time() - tr

    for name, score, extra in [("svm_rbf", svm_score, dict(best_params=grid.best_params_, cv_auroc=float(grid.best_score_),
                                                             train_time_s=svm_train_time)),
                               ("random_forest", rf_score, dict(n_estimators=args.rf_trees, train_time_s=rf_train_time))]:
        m = classification_metrics(y_va, score)
        pt = idx_va.patch_type.values
        m["auroc_tampered_vs_clean_paired"] = subset_auroc(y_va, score, pt, lambda g: g in ("tampered", "clean_paired"))
        m["auroc_tampered_vs_clean_same"] = subset_auroc(y_va, score, pt, lambda g: g in ("tampered", "clean_same"))
        # tool / alan kÄ±rÄ±lÄ±mÄ±: ilgili tool'un tampered patch'leri vs tÃ¼m clean patch'ler
        for tool in sorted(idx_va.tool.unique()):
            keep = (y_va == 0) | (idx_va.tool.values == tool)
            m[f"auroc_tool_{tool}"] = safe_auroc(y_va[keep], score[keep])
        for ab in sorted(idx_va.area_bin.unique()):
            keep = (y_va == 0) | (idx_va.area_bin.values == ab)
            m[f"auroc_area_{ab}"] = safe_auroc(y_va[keep], score[keep])
        m.update(extra)
        results[name] = m
        rows.append(dict(model=name, **{k: v for k, v in m.items() if not isinstance(v, dict)}))
        print(name, {k: round(v, 3) for k, v in m.items() if isinstance(v, float)}, flush=True)

    # --- Ã¶znitelik Ã¶nemleri
    imp = pd.DataFrame({"feature": names, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    imp.to_csv(D.RESULTS_DIR / "rf_feature_importance.csv", index=False)
    results["rf_feature_importance_top15"] = imp.head(15).to_dict(orient="records")
    results["meta"] = dict(n_train=int(len(y_tr)), n_val=int(len(y_va)), n_features=len(names),
                           feature_extract_time_s=feat_time, total_time_s=time.time() - t0, seed=args.seed,
                           train_patch_types=idx_tr.patch_type.value_counts().to_dict(),
                           val_patch_types=idx_va.patch_type.value_counts().to_dict())
    with open(D.RESULTS_DIR / "classical_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    pd.DataFrame(rows).to_csv(D.RESULTS_DIR / "classical_results.csv", index=False)
    # tahminleri sakla (sonradan analiz iÃ§in)
    pred = idx_va[["new_id", "patch_type", "tool", "area_bin", "label"]].copy()
    pred["svm_score"] = svm_score; pred["rf_score"] = rf_score
    pred.to_csv(D.RESULTS_DIR / "classical_val_predictions.csv", index=False)
    print("done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
