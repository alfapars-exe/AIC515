"""
v2_eval.py — AIForge-Doc v2 (GPT-Image-2) HARİCİ TEST (TR). v1'de eğitilmiş modeller, v2'de yeniden eğitim/kalibrasyon YOK.

Aşamalar
  prep   : v2 tablosu + sızıntı bayrakları + GT varyantları (make_gt.process_pair ile v1'le birebir) + yeşil kalıntı
  infer  : tam görüntü olasılık haritaları (U-Net/DLV3+: 512/448 pencere; YOLO: 1024/768 kare + NMS veya tam 1280)
  eval   : 3 GT altında piksel metrikleri (τ* = v1 val'den donmuş, tau.json) + görüntü düzeyi AUROC (sahte vs eşleşik authentic)

Sızıntı kuralı (koşmadan önce sabitlendi): v2, v1 spesifikasyonlarını aynı kaynak belgelerle yeniden üretir. Modeller v1
TrainingSet'te (train+val) eğitildi/seçildi. BİRİNCİL küme = v1 karşılığı TestingSet'te olan VE kaynak belgesi (v1 'group'
ve v2 image_id) v1 TrainingSet'te hiç görünmeyen v2 çiftleri. 'seen' = belge v1 eğitiminde görüldü (yalnız ikincil analiz).

Kullanım (docforensics):
  python v2_eval.py prep
  python v2_eval.py infer --models unet_bbox_s42,unet_pixel_eps8_morph_s42,... --subset primary
  python v2_eval.py infer --yolo aiforge-tiles-yolo11s-img1024-conserv-s42 --subset all
  python v2_eval.py eval  --models ... --subset all
"""
from __future__ import annotations
import os, sys, json, time, argparse, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, cv2

os.environ.setdefault("USERNAME", "user")
VP = Path(os.environ.get("VIZE_ROOT", r"C:\Users\PARS\Desktop\PROJELER\AI 515\Vize Proje"))
for p in (VP / "phase2" / "experiments", VP / "phase3", VP / "phase3_yolo", VP / "phase2" / "gt"):
    sys.path.insert(0, str(p))
from common import p2data as D, p2metrics as M  # noqa: E402

OUT = VP / "phase3_v2"; PRED = OUT / "preds"; GTD = OUT / "gt"; RES = OUT / "results"
TAB = OUT / "v2_table.csv"
GTS = ["bbox", "pixel_eps8_morph", "glyph_v2"]


def v2_root() -> Path:
    for c in [VP / "data" / "AIForge-Doc-v2"] + [Path(p) for p in glob.glob(str(VP / "data" / "AIForge-Doc-v2" / "*"))]:
        if (c / "TrainingSet" / "authentic").is_dir():
            return c
    raise SystemExit("v2 tam paketi (authentic dahil) bulunamadı: data/AIForge-Doc-v2/")


def sub(split): return "TrainingSet" if split == "training" else "TestingSet"
def key(split, nid): return ("tr_" if split == "training" else "te_") + nid


def gmask(rgb):
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return (g - np.maximum(r, b) > 50) & (g > 110)


# ---------------------------------------------------------------- prep
def prep(threads=8):
    import make_gt as MG
    root = v2_root()
    meta = pd.read_json(root / "metadata.jsonl", lines=True, dtype={"spec_id": str, "new_id": str})
    v1 = pd.read_csv(VP / "dataset_audit" / "samples_forged.csv", dtype=str, low_memory=False)
    v1 = v1.set_index("spec_id")
    sp = json.load(open(VP / "dataset_audit" / "splits.json"))
    v1_trainval = set(sp["train"]) | set(sp["val"])
    v1_tr = v1[v1.split == "training"]
    seen_groups = set(v1_tr.loc[v1_tr.new_id.isin(v1_trainval), "group"].dropna())
    meta["key"] = [key(s, n) for s, n in zip(meta.split, meta.new_id)]
    # v1'de 12 spec_id iki kez geçiyor (bazıları iki split'te) → muhafazakâr: herhangi bir kopya training ise training
    agg_split = v1.groupby(level=0)["split"].agg(lambda s: "training" if (s == "training").any() else s.iloc[0])
    agg_groups = v1.groupby(level=0)["group"].agg(lambda s: set(s.dropna()))
    meta["v1_split"] = meta.spec_id.map(agg_split).fillna("none")
    meta["v1_group"] = meta.spec_id.map(lambda x: ";".join(sorted(agg_groups.get(x, set()))))
    grp_seen = meta.spec_id.map(lambda x: bool(agg_groups.get(x, set()) & seen_groups))
    meta["v1_tool"] = meta.v1_assigned_tool.map(D.TOOL_LABEL).fillna(meta.v1_assigned_tool)
    # v2 image_id düzeyi: aynı kaynak görüntünün herhangi bir v2 spesifikasyonu v1 TrainingSet'teyse → görüldü
    img_seen = set(meta.loc[meta.v1_split == "training", "image_id"])
    meta["doc_seen"] = grp_seen | meta.image_id.isin(img_seen) | (meta.v1_split == "training")
    meta["primary"] = (meta.v1_split == "testing") & ~meta.doc_seen

    for v in MG.VARIANTS:
        (GTD / v / "test").mkdir(parents=True, exist_ok=True)

    def work(r):
        rec = dict(split=r.split, new_id=r.new_id, assigned_tool=r.assigned_tool, source_dataset=r.source_dataset,
                   area_bin="", pair_name=r.new_id + ".png")
        try:
            out, _ = MG.process_pair(rec, str(root), str(GTD), write=True)
        except Exception as e:  # noqa: BLE001
            return dict(key=r.key, skipped=f"err:{e}")
        f = D.read_rgb(root / sub(r.split) / "images" / f"{r.new_id}.png")
        m = cv2.imread(str(root / sub(r.split) / "masks" / f"{r.new_id}.png"), cv2.IMREAD_GRAYSCALE) > 127
        gm = gmask(f)
        bb = np.zeros_like(m)
        if "bbox_x1" in out:
            bb[out["bbox_y1"]:out["bbox_y2"] + 1, out["bbox_x1"]:out["bbox_x2"] + 1] = True
        ring = cv2.dilate(bb.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        frac = m.mean()
        keep = {k: out.get(k) for k in ("H", "W", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_px", "D_max",
                                         "D_mean_outside", "mask_rect_frac", "pixel_eps8_n_pos", "pixel_eps8_n_out",
                                         "pixel_eps8_iou_bbox", "pixel_eps8_morph_iou_bbox", "glyph_v2_n_pos", "skipped")}
        keep.update(key=r.key, tamper_ratio=float(frac),
                    area_bin="<1%" if frac < .01 else "1-5%" if frac < .05 else "5-20%" if frac < .2 else ">20%",
                    green_px_in_ring=int((gm & ring).sum()), green_px_total=int(gm.sum()))
        return keep
    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        rows = list(ex.map(work, list(meta.itertuples())))
    st = pd.DataFrame(rows)
    tab = meta.merge(st, on="key", how="left")
    tab["green_residue"] = tab.green_px_in_ring >= 20
    OUT.mkdir(parents=True, exist_ok=True); tab.to_csv(TAB, index=False)
    print(f"prep: {len(tab)} çift, {time.time() - t0:.0f}s; primary={int(tab.primary.sum())} seen={int(tab.doc_seen.sum())} "
          f"skipped={int(tab.skipped.notna().sum())}; v1_split={tab.v1_split.value_counts().to_dict()}; "
          f"yeşil kalıntı={tab.green_residue.mean():.3f}", flush=True)


def load_tab(subset):
    t = pd.read_csv(TAB, dtype={"new_id": str, "spec_id": str})
    t = t[t.skipped.isna()]
    if subset == "primary":
        t = t[t.primary]
    return t.set_index("key")


def gt_path(variant, k):
    nid = k[3:]
    return GTD / variant / ("test" if k.startswith("te_") else "") / f"{nid}.png"


# ---------------------------------------------------------------- infer
def infer(seg, yolos, subset, sources, full=0):
    import torch
    root = v2_root(); tab = load_tab(subset)
    dev = torch.device("cuda")
    for src in sources:
        def path(k):
            r = tab.loc[k]
            return root / sub(r.split) / ("images" if src == "forged" else "authentic") / f"{r.new_id}.png"
        if seg:
            import p3ext as P
            import infer_fullimage as IF
            todo = [k for k in tab.index if not all((PRED / n / src / f"{k}.png").exists() for n in seg)]
            nets = [P.load_model(n, dev) for n in seg]
            t0 = time.time(); pool = ThreadPoolExecutor(4)
            for i, k in enumerate(todo):
                img = D.read_rgb(path(k))
                probs, _ = IF.predict_full_multi(nets, img, dev)
                for n, p in zip(seg, probs):
                    (PRED / n / src).mkdir(parents=True, exist_ok=True)
                    pool.submit(IF.write_png, PRED / n / src / f"{k}.png", IF.to_u8(p))
                if (i + 1) % 100 == 0:
                    print(f"[seg {src}] {i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
            pool.shutdown(wait=True); del nets; torch.cuda.empty_cache()
            print(f"[seg {src}] bitti {len(todo)} görüntü {time.time() - t0:.0f}s", flush=True)
        for run in yolos:
            from ultralytics import YOLO
            import infer_to_maps as YM
            a = argparse.Namespace(tile=1024, stride=768, full=(1280 if "-full-" in run else 0), conf=0.001, iou=0.6, batch=8)
            mdl = YOLO(str(VP / "phase3_yolo" / "runs" / run / "weights" / "best.pt"))
            od = PRED / run / src; od.mkdir(parents=True, exist_ok=True)
            todo = [k for k in tab.index if not (od / f"{k}.png").exists()]
            t0 = time.time()
            for i, k in enumerate(todo):
                img = D.read_rgb(path(k)); H, W = img.shape[:2]
                b, c = YM.detect_full(mdl, img, a)
                cv2.imwrite(str(od / f"{k}.png"), YM.boxes_to_map(b, c, H, W))
                if (i + 1) % 100 == 0:
                    print(f"[{run[:28]} {src}] {i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
            del mdl; torch.cuda.empty_cache()
            print(f"[{run[:28]} {src}] bitti {len(todo)} {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- eval
def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def boot_auc(pos, neg, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(pos); v = []
    for _ in range(B):
        s = rng.integers(0, n, n)  # eşleşik çiftler birlikte yeniden örneklenir
        v.append(auroc(pos[s], neg[s]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def evaluate(models, subset, threads=8):
    root = v2_root(); tab = load_tab(subset)
    taus = json.load(open(VP / "phase2" / "experiments" / "results" / "tau.json"))
    itau = json.load(open(VP / "hf_space" / "meta" / "image_tau.json", encoding="utf-8"))
    RES.mkdir(parents=True, exist_ok=True)
    allrows, summ, imglev = [], [], []
    for mdl in models:
        if not all((PRED / mdl / "forged" / f"{k}.png").exists() for k in tab.index):
            print("eksik tahmin, atlandı:", mdl); continue
        tm = {g: taus.get(mdl, {}).get(g) for g in GTS}

        def work(k):
            p = cv2.imread(str(PRED / mdl / "forged" / f"{k}.png"), cv2.IMREAD_GRAYSCALE)
            r = tab.loc[k]; out = {}
            for g in GTS:
                if g == "bbox":
                    gt = cv2.imread(str(root / sub(r.split) / "masks" / f"{r.new_id}.png"), cv2.IMREAD_GRAYSCALE) > 127
                else:
                    gt = cv2.imread(str(gt_path(g, k)), cv2.IMREAD_GRAYSCALE) > 127
                m, h = M.per_image_metrics(p, gt, tau=tm[g], with_shape=False)
                m.pop("f1_curve", None); out[g] = (m, h)
            pa = PRED / mdl / "authentic" / f"{k}.png"
            sa = M.image_level_scores(cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)) if pa.exists() else None
            return k, out, sa
        with ThreadPoolExecutor(threads) as ex:
            res = list(ex.map(work, list(tab.index)))
        H = {g: {} for g in GTS}; rows = []
        for k, out, sa in res:
            for g in GTS:
                m, h = out[g]; H[g][k] = h
                rows.append(dict(model=mdl, key=k, gt=g, **m, auth_top1pct=(sa or {}).get("score_top1pct", np.nan),
                                 auth_max=(sa or {}).get("score_max", np.nan)))
        df = pd.DataFrame(rows); allrows.append(df)
        subsets = {"all": tab.index, "primary": tab.index[tab.primary], "seen": tab.index[tab.doc_seen]}
        for col in ("source_dataset", "v1_tool", "area_bin"):
            for v in sorted(tab[col].dropna().unique()):
                subsets[f"{col}={v}"] = tab.index[(tab[col] == v) & tab.primary]
        subsets["primary&green"] = tab.index[tab.primary & tab.green_residue]
        subsets["primary&nogreen"] = tab.index[tab.primary & ~tab.green_residue]
        for sname, ids in subsets.items():
            ids = list(ids)
            if not ids:
                continue
            for g in GTS:
                hl = [H[g][i] for i in ids if H[g].get(i) is not None]
                d = df[(df["gt"] == g) & df.key.isin(ids) & (df.n_pos > 0)]
                pooled = M.pooled_from_hists(np.stack([h[0] for h in hl]), np.stack([h[1] for h in hl])) if hl else {}
                summ.append(dict(model=mdl, subset=sname, gt=g, n=len(ids), tau=tm[g], **pooled,
                                 **{c: float(d[c].mean()) for c in ("ap_img", "f1_tau", "iou_tau", "oracle_f1", "f1_0p5") if c in d}))
            dd = df[(df["gt"] == "bbox") & df.key.isin(ids)].dropna(subset=["auth_top1pct"])
            if len(dd) >= 10:
                pos, neg = dd.score_top1pct.values, dd.auth_top1pct.values
                thr = itau.get(mdl, {}).get("thr")
                imglev.append(dict(model=mdl, subset=sname, n_pairs=len(dd), auroc_top1pct=auroc(pos, neg),
                                   ci95=boot_auc(pos, neg), auroc_max=auroc(dd.score_max.values, dd.auth_max.values),
                                   v1_thr=thr, tpr_at_v1_thr=float((pos >= thr).mean()) if thr is not None else np.nan,
                                   fpr_at_v1_thr=float((neg >= thr).mean()) if thr is not None else np.nan))
        print(f"[eval {mdl}] " + "; ".join(f"{s['gt']} AUPRC {s.get('auprc_pooled', np.nan):.3f}" for s in summ
                                            if s["model"] == mdl and s["subset"] == "primary"), flush=True)
    tag = subset
    pd.concat(allrows).to_csv(RES / f"per_image_v2_{tag}.csv", index=False)
    pd.DataFrame(summ).to_csv(RES / f"summary_v2_{tag}.csv", index=False)
    pd.DataFrame(imglev).to_csv(RES / f"image_level_v2_{tag}.csv", index=False)
    print("yazıldı:", RES)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "infer", "eval"])
    ap.add_argument("--models", default=""); ap.add_argument("--yolo", default="")
    ap.add_argument("--subset", default="primary", choices=["primary", "all"])
    ap.add_argument("--sources", default="forged,authentic"); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    seg = [m for m in a.models.split(",") if m]; yol = [m for m in a.yolo.split(",") if m]
    if a.stage == "prep":
        prep(a.threads)
    elif a.stage == "infer":
        infer(seg, yol, a.subset, a.sources.split(","))
    else:
        evaluate(seg + yol, a.subset, a.threads)
