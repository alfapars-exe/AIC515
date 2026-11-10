"""build_manuscript.py — Makale ve raporları kaynak dosyalardan yeniden üretir (TR).
- Tablolar: phase2/experiments/results/*.csv, phase2/stats/stats_summary.csv, phase2/gt/gt_stats.json, phase2/controls/controls_summary.json,
  phase2/literature_baselines/zero_shot_results.csv, baselines/results/*.csv, dataset_audit/audit_stats.json → markdown tabloları
- Atıflar: reports/src/*.md içindeki [@bibkey] işaretleri literature/references.bib + reports/src/classical_refs.json ile numaralanır
- Çıktı: reports/Bildiri_Tam_Metin.md/.docx (İngilizce tam makale)
Kullanım: python tools/build_manuscript.py --root <Vize Proje kökü>
"""
import os, re, json, argparse, numpy as np, pandas as pd
import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(_ROOT)); a = ap.parse_args()
VP = a.root; R = os.path.join(VP,"phase2","experiments","results"); B = os.path.join(VP,"baselines","results"); SRC = os.path.join(VP,"reports","src")

def md_table(df, cols, headers=None, fmt=None):
    headers = headers or cols; fmt = fmt or {}
    out = ["| " + " | ".join(headers) + " |", "|" + "---|"*len(cols)]
    for _, r in df.iterrows():
        cells=[]
        for c in cols:
            v = r[c]
            if c in fmt: cells.append(fmt[c](v))
            elif isinstance(v,(float,np.floating)): cells.append("—" if pd.isna(v) else f"{v:.3f}")
            else: cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
def pm(m,s,nd=3): return f"{m:.{nd}f} ± {s:.{nd}f}"
lab={"bbox":"bbox","pixel_eps8_morph":"pixel","glyph_v2":"glyph"}; order={"bbox":0,"pixel":1,"glyph":2}

def build_tables():
    T = {}
    au = json.load(open(os.path.join(VP,"dataset_audit","audit_stats.json"), encoding="utf-8"))
    gs = json.load(open(os.path.join(VP,"phase2","gt","gt_stats.json"), encoding="utf-8")); v = gs["variants"]
    cs = json.load(open(os.path.join(VP,"phase2","controls","controls_summary.json"), encoding="utf-8"))
    T["dataset"] = open(os.path.join(SRC,"table_dataset.md"), encoding="utf-8").read()
    rows=[dict(variant=n, cov=vv["bbox_coverage(n_in/bbox_px)"]["median"], iou=vv["iou_bbox"]["median"], posfrac=vv["pos_frac_img"]["median"]*100 if vv["pos_frac_img"]["median"]<1 else vv["pos_frac_img"]["median"], n_out=vv["n_pairs_any_outside"], n_out1=vv["n_pairs_outside_share_gt_1pct"], ncc=vv["n_cc"]["median"]) for n,vv in v.items()]
    T["gt_variants"] = md_table(pd.DataFrame(rows), ["variant","cov","iou","posfrac","n_out","n_out1","ncc"], ["Variant","Box coverage (median)","IoU with box (median)","Positive pixels, % of image (median)","Pairs with ≥1 outside-box pixel","Pairs with >1 % outside share","Components (median)"], {"posfrac":lambda x:f"{x:.2f}","n_out":lambda x:f"{int(x)}","n_out1":lambda x:f"{int(x)}","ncc":lambda x:f"{x:.0f}"})
    cr=[]
    for k,l in [("REAL_FORGED","Real forgery (reference)"),("benign_pasteback","Benign paste-back"),("fullimage_resample","Full-image 2× resample"),("reencode_png","PNG re-encode")]:
        c=cs[k]; cr.append(dict(ctrl=l, dmax=f"{c['D_max_median']:.0f} [{c['D_max_q25']:.0f}–{c['D_max_q75']:.0f}]", n8=f"{c['n_gt8_median']:.0f} [{c['n_gt8_q25']:.0f}–{c['n_gt8_q75']:.0f}]", out=f"{c['n_gt8_out_median']:.0f}"))
    T["controls"] = md_table(pd.DataFrame(cr), ["ctrl","dmax","n8","out"], ["Image set (n = 487 each)","D_max median [IQR]","Pixels with D > 8, median [IQR]","Outside-box D > 8 pixels (median)"])
    zs = pd.read_csv(os.path.join(VP,"phase2","literature_baselines","zero_shot_results.csv")); zsel = zs[(zs["split"]=="test")&(zs["subset"].isin(["full","clean"]))]
    T["zeroshot"] = md_table(zsel, ["cond","subset","n_images","pixel_f1_05","iou_05","oracle_f1","f1_at_val_thr","pixel_auprc_pooled","pixel_auroc_pooled","auroc_img_mean","ap_img_mean"], ["Condition","Test subset","n","F1@0.5","IoU@0.5","Oracle-F1","F1@τ*(val)","Pooled AUPRC","Pooled AUROC","Per-image AUROC (mean)","Per-image AP (mean)"], {"n_images":lambda x:str(int(x))})
    sva = pd.read_csv(os.path.join(R,"summary_val_agg.csv")); sva["Train GT"]=sva["train_gt"].map(lab); sva["Eval GT"]=sva["gt"].map(lab)
    for c,(x,y) in {"AUPRC":("auprc_pooled_mean","auprc_pooled_std"),"AUROC":("auroc_pooled_mean","auroc_pooled_std"),"AP":("ap_img_mean","ap_img_std"),"F1@0.5":("f1_0p5_mean","f1_0p5_std"),"F1@τ*":("f1_tau_mean","f1_tau_std"),"IoU@τ*":("iou_tau_mean","iou_tau_std"),"Oracle-F1":("oracle_f1_mean","oracle_f1_std")}.items(): sva[c]=[pm(p,q) for p,q in zip(sva[x],sva[y])]
    sva["τ*"]=[f"{t:.2f}" for t in sva["tau_mean"]]; sva["o1"]=sva["Train GT"].map(order); sva["o2"]=sva["Eval GT"].map(order); sva=sva.sort_values(["o1","o2"])
    T["dualgt"] = md_table(sva, ["Train GT","Eval GT","AUPRC","AUROC","AP","F1@0.5","F1@τ*","IoU@τ*","Oracle-F1","τ*"])
    rect = sva[sva["Train GT"]==sva["Eval GT"]][["Train GT","rect_0p5_mean","rect_tau_mean","ncc_0p5_mean"]]
    T["rect"] = md_table(rect, ["Train GT","rect_0p5_mean","rect_tau_mean","ncc_0p5_mean"], ["Train GT","Rectangularity @0.5 (mean)","Rectangularity @τ* (mean)","Components @0.5 (mean)"])
    sta = pd.read_csv(os.path.join(R,"summary_test_agg.csv")); sta["Train GT"]=sta["train_gt"].map(lab); sta["Eval GT"]=sta["gt"].map(lab)
    for c,(x,y) in {"AUPRC":("auprc_pooled_mean","auprc_pooled_std"),"AP":("ap_img_mean","ap_img_std"),"F1@0.5":("f1_0p5_mean","f1_0p5_std"),"F1@τ*":("f1_tau_mean","f1_tau_std"),"IoU@τ*":("iou_tau_mean","iou_tau_std"),"Oracle-F1":("oracle_f1_mean","oracle_f1_std")}.items(): sta[c]=[pm(p,q) for p,q in zip(sta[x],sta[y])]
    sta["o1"]=sta["Train GT"].map(order); sta["o2"]=sta["Eval GT"].map(order); sta=sta.sort_values(["o1","subset","o2"])
    T["test"] = md_table(sta, ["Train GT","Eval GT","subset","n_images","AUPRC","AP","F1@0.5","F1@τ*","IoU@τ*","Oracle-F1"], ["Train GT","Eval GT","Subset","n","AUPRC","AP","F1@0.5","F1@τ*","IoU@τ*","Oracle-F1"], {"n_images":lambda x:str(int(x))})
    rb = pd.read_csv(os.path.join(R,"robustness.csv")); rb["train_gt"]=rb["model"].str.replace("unet_","").str.replace(r"_s\d+","",regex=True)
    rba = rb.groupby(["train_gt","gt","condition"]).agg(auprc=("auprc_pooled","mean"), dauprc=("delta_auprc_pooled","mean"), dauprc_sd=("delta_auprc_pooled","std"), ap=("ap_img","mean"), dap=("delta_ap_img","mean"), dap_sd=("delta_ap_img","std"), f1=("f1_tau","mean"), df1=("delta_f1_tau","mean")).reset_index()
    rba = rba[rba["gt"]==rba["train_gt"]].copy(); rba["Train GT"]=rba["train_gt"].map(lab); rba["ΔAUPRC"]=[pm(p,q) for p,q in zip(rba.dauprc,rba.dauprc_sd)]; rba["ΔAP"]=[pm(p,q) for p,q in zip(rba.dap,rba.dap_sd)]
    rba["o1"]=rba["Train GT"].map(order); rba=rba.sort_values(["o1","condition"])
    T["robust"] = md_table(rba, ["Train GT","condition","auprc","ΔAUPRC","ap","ΔAP","f1","df1"], ["Train GT (own GT)","Condition","AUPRC","ΔAUPRC vs clean","AP","ΔAP vs clean","F1@τ*","ΔF1@τ*"])
    pb_ = pd.read_csv(os.path.join(R,"probes_boundary.csv")); pbo=pb_[pb_["gt"]==pb_["train_gt"]]
    pba = pbo.groupby(["train_gt","condition"]).agg(rel_auprc=("rel_delta_auprc_pooled","mean"), rel_auprc_sd=("rel_delta_auprc_pooled","std"), rel_ap=("rel_delta_ap_img","mean"), rel_ap_sd=("rel_delta_ap_img","std"), rel_f1=("rel_delta_f1_tau","mean"), rel_f1_sd=("rel_delta_f1_tau","std")).reset_index()
    pba["Train GT"]=pba["train_gt"].map(lab); pba["o1"]=pba["Train GT"].map(order); pba=pba.sort_values(["o1","condition"])
    for c in ["rel_auprc","rel_ap","rel_f1"]: pba[c+"_s"]=[f"{100*p:+.1f} ± {100*q:.1f} %" for p,q in zip(pba[c],pba[c+"_sd"])]
    T["boundary"] = md_table(pba, ["Train GT","condition","rel_auprc_s","rel_ap_s","rel_f1_s"], ["Train GT (own GT)","Ablation","relΔ pooled AUPRC","relΔ per-image AP","relΔ F1@τ*"])
    pbx = pb_[(pb_["train_gt"]=="pixel_eps8_morph")&(pb_["gt"]=="bbox")&(pb_["condition"]=="inner_blur")]
    T["inner_cross"] = f"relΔ pooled AUPRC {100*pbx.rel_delta_auprc_pooled.mean():+.1f} ± {100*pbx.rel_delta_auprc_pooled.std():.1f} %, relΔ AP {100*pbx.rel_delta_ap_img.mean():+.1f} ± {100*pbx.rel_delta_ap_img.std():.1f} %"
    rf = pd.read_csv(os.path.join(R,"probes_rectfit.csv")); rfa = rf.groupby(["train_gt","gt"]).agg(f1_raw=("f1_raw","mean"), f1_rf=("f1_rectfit","mean"), d=("delta_f1","mean"), dsd=("delta_f1","std"), di=("delta_iou","mean")).reset_index()
    rfa["Train GT"]=rfa["train_gt"].map(lab); rfa["Eval GT"]=rfa["gt"].map(lab); rfa["ΔF1"]=[pm(p,q) for p,q in zip(rfa.d,rfa.dsd)]; rfa["o1"]=rfa["Train GT"].map(order); rfa["o2"]=rfa["Eval GT"].map(order); rfa=rfa.sort_values(["o1","o2"])
    T["rectfit"] = md_table(rfa, ["Train GT","Eval GT","f1_raw","f1_rf","ΔF1","di"], ["Train GT","Eval GT","F1@τ* raw","F1 rectangle-fit","ΔF1","ΔIoU"])
    pc = pd.read_csv(os.path.join(R,"probes_controls.csv")); pca = pc.groupby(["train_gt","source"]).agg(fp05=("fp_pixel_rate_0p5","mean"), fptau=("fp_pixel_rate_tau","mean"), fr05=("frac_imgs_gt0p1pct_0p5","mean"), frtau=("frac_imgs_gt0p1pct_tau","mean"), au=("auroc_score_top1pct","mean"), aug=("auroc_top1_gemini","mean"), aui=("auroc_top1_ideogram","mean")).reset_index()
    pca["Train GT"]=pca["train_gt"].map(lab); pca["o1"]=pca["Train GT"].map(order); pca=pca.sort_values(["o1","source"])
    T["controls_probe"] = md_table(pca, ["Train GT","source","fp05","fptau","fr05","frtau","au","aug","aui"], ["Train GT","Image set","FP pixel rate @0.5","FP pixel rate @τ*","Images > 0.1 % positive @0.5","@τ*","Image AUROC forged vs set (top-1 %)","Gemini","Ideogram"], {"fp05":lambda x:f"{1000*x:.2f} ‰","fptau":lambda x:f"{1000*x:.2f} ‰","fr05":lambda x:f"{100*x:.1f} %","frtau":lambda x:f"{100*x:.1f} %"})
    ho = pd.read_csv(os.path.join(R,"heldout_generator.csv")); hoa = ho.groupby(["gemini_only","train_gt","gt"]).agg(n_i=("n_ideogram","first"), iap=("ideogram_ap_img","mean"), if1=("ideogram_f1_0p5","mean"), gap_=("gemini_matched_ap_img","mean"), gf1=("gemini_matched_f1_0p5","mean"), gapd=("gap_ap_img","mean"), ns=("seed","count")).reset_index()
    hoa["Training set"]=hoa["gemini_only"].astype(str).map({"True":"Gemini only","False":"All tools","1":"Gemini only","0":"All tools"}); hoa["Train GT"]=hoa["train_gt"].map(lab); hoa["Eval GT"]=hoa["gt"].map(lab)
    hoa["o1"]=hoa["Train GT"].map(order); hoa["o2"]=hoa["Eval GT"].map(order); hoa=hoa.sort_values(["Training set","o1","o2"])
    T["heldout"] = md_table(hoa, ["Training set","Train GT","Eval GT","ns","n_i","iap","if1","gap_","gf1","gapd"], ["Training set","Train GT","Eval GT","seeds","n Ideogram","Ideogram AP","Ideogram F1@0.5","Gemini (source-matched) AP","Gemini F1@0.5","Gap AP"], {"ns":lambda x:str(int(x)),"n_i":lambda x:str(int(x))})
    h4 = pd.read_csv(os.path.join(R,"h4_results.csv")); T["h4"] = md_table(h4[h4["subset"].isin(["all","gemini","ideogram"])], ["condition","subset","n_pairs","auroc","acc_pairwise"], ["Condition","Subset","Pairs","AUROC","Pairwise accuracy"], {"n_pairs":lambda x:str(int(x))})
    st = pd.read_csv(os.path.join(VP,"phase2","stats","stats_summary.csv")); sth = st[st["p_holm"].notna()].copy(); sth["est"]=[f"{m_:.3f} [{lo:.3f}, {hi:.3f}]" for m_,lo,hi in zip(sth["mean"],sth["ci_lo"],sth["ci_hi"])]
    T["stats"] = md_table(sth, ["hypothesis","contrast","metric","est","p_boot","p_holm"], ["Hypothesis","Contrast","Metric","Estimate [95 % CI]","p (boot)","p (Holm)"], {"p_boot":lambda x:f"{x:.3f}","p_holm":lambda x:f"{x:.3f}"})
    ck = pd.read_csv(os.path.join(R,"checkpoint_summary.csv"))
    T["ckpt"] = md_table(ck, ["run","n_train","epochs","best_epoch","best_val_crop_auprc","last_val_crop_auprc","last3_slope","train_loss_last","time_min","peak_gib"], ["Run","n train","Epochs","Best epoch","Best val-crop AUPRC","Last-epoch AUPRC","Slope last 3 ep.","Train loss (last)","Time (min)","Peak GPU (GiB)"], {"n_train":lambda x:str(int(x)),"epochs":lambda x:str(int(x)),"best_epoch":lambda x:str(int(x)),"last3_slope":lambda x:f"{x:+.4f}"})
    cl = pd.read_csv(os.path.join(B,"classical_results.csv")); rn = pd.read_csv(os.path.join(B,"resnet_results.csv")); rnj = json.load(open(os.path.join(B,"resnet_results.json")))
    if "model" not in rn.columns: rn.insert(0,"model",[f"ResNet-18 {k}" for k in list(rnj.keys())][:len(rn)])
    un = pd.read_csv(os.path.join(B,"unet_results.csv")); rbp = pd.read_csv(os.path.join(B,"robustness_results.csv"))
    T["pilot_cls"] = md_table(pd.concat([cl, rn]), ["model","n","accuracy","balanced_accuracy","auroc","auprc","f1","auroc_tampered_vs_clean_paired"], ["Model","n val patches","Accuracy","Balanced acc.","AUROC","AUPRC","F1@0.5","AUROC tampered vs paired-authentic"], {"n":lambda x:str(int(x))})
    T["pilot_seg"] = md_table(un, ["variant","pixel_f1_at_0p5","iou_at_0p5","oracle_f1","pixel_auprc_pooled_subsampled","clean_crop_fp_pixel_rate_at_0p5"], ["Variant","Pixel-F1@0.5","IoU@0.5","Oracle-F1","AUPRC (subsampled)","Clean-crop FP rate"])
    T["pilot_rob"] = md_table(rbp, ["variant","condition","pixel_f1_at_0p5","iou_at_0p5","pixel_auprc_pooled_subsampled","f1_tool_gemini","f1_tool_ideogram"], ["Variant","Condition","Pixel-F1@0.5","IoU@0.5","AUPRC","F1 Gemini","F1 Ideogram"])
    gp = os.path.join(R,"green_marker_summary.csv")
    if os.path.exists(gp):
        gm = pd.read_csv(gp); gm = gm[gm.has_residue==1].copy(); gm["cond"]=pd.Categorical(gm["cond"],["forged","forged_repaired","authentic","authentic_plus_marker"]); gm["o1"]=gm["family"].map(order); gm=gm.sort_values(["o1","cond"])
        for c in ["ap_bbox","ap_pixel","bbox_cover_tau","iou_predbox_tau","pos_frac_tau","top1"]: gm[c+"_s"]=[f"{m_:.3f} [{lo:.3f}, {hi:.3f}]" for m_,lo,hi in zip(gm[c],gm[c+"_lo"],gm[c+"_hi"])]
        T["green"] = md_table(gm, ["family","cond","n","ap_bbox_s","ap_pixel_s","bbox_cover_tau_s","iou_predbox_tau_s","pos_frac_tau_s","top1_s"], ["Model family","Input","n","AP (box GT)","AP (pixel GT)","Box coverage @τ*","IoU(pred, box) @τ*","Positive fraction @τ*","Top-1 % score"], {"n":lambda x:str(int(x))})
        gc = pd.read_csv(os.path.join(R,"green_marker_contrasts.csv")); gc["est"]=[f"{m_:+.3f} [{lo:+.3f}, {hi:+.3f}]" for m_,lo,hi in zip(gc["mean"],gc["ci_lo"],gc["ci_hi"])]
        T["green_contrasts"] = md_table(gc, ["family","contrast","n","est"], ["Model family","Paired contrast (residue images)","n","Estimate [95 % CI]"], {"n":lambda x:str(int(x))})
    return T

def parse_bib(txt):
    E = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", txt, re.S):
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*\n", m.group(3)+"\n", re.S): fields[fm.group(1).lower()] = re.sub(r"\s+"," ", fm.group(2)).strip("{} ")
        E[m.group(2).strip()] = fields
    return E
def fmt_ref(e):
    au = e.get("author","").replace(" and ", ", ")
    ven = e.get("booktitle") or e.get("journal") or ("arXiv preprint" if e.get("eprint") else e.get("publisher",""))
    ident = ("doi:"+e["doi"]) if e.get("doi") else (("arXiv:"+e["eprint"]) if e.get("eprint") else e.get("url",""))
    return f'{au}. "{e.get("title","")}." {ven}, {e.get("year","")}. {ident}'.replace("..",".")

def main():
    T = build_tables(); json.dump(T, open(os.path.join(SRC,"tables_generated.json"),"w",encoding="utf-8"), indent=1)
    E = parse_bib(open(os.path.join(VP,"literature","references.bib"),encoding="utf-8").read())
    lm = pd.read_csv(os.path.join(VP,"literature","literature_matrix.csv"))
    for k,e in E.items():  # fill missing DOI from literature matrix by title prefix
        if not e.get("doi") and not e.get("eprint"):
            hit = lm[lm["paper_title"].str.lower().str[:40]==e.get("title","").lower()[:40]]
            if len(hit): e["url"] = str(hit.iloc[0]["doi_or_url"])
    E.update(json.load(open(os.path.join(SRC,"classical_refs.json"),encoding="utf-8")))
    parts = [open(os.path.join(SRC,f),encoding="utf-8").read() for f in sorted(os.listdir(SRC)) if re.match(r"ms_part\d+\.md", f)]
    text = "\n\n".join(parts)
    for k,v in T.items(): text = text.replace("{{T:"+k+"}}", v)
    for f in ["abstract.md","green_results.md"]:
        p = os.path.join(SRC,f)
        if os.path.exists(p): text = text.replace("{{"+f.split(".")[0].upper()+"}}", open(p,encoding="utf-8").read())
    for k,v in T.items(): text = text.replace("{{T:"+k+"}}", v)  # second pass: tables referenced inside inserted sections
    numbering = {}; missing = set()
    def cite(m):
        keys = [k.strip() for k in m.group(1).split(";")]; nums=[]
        for k in keys:
            k = k.lstrip("@")
            if k not in E: missing.add(k); nums.append("?"); continue
            if k not in numbering: numbering[k] = len(numbering)+1
            nums.append(str(numbering[k]))
        return "[" + ", ".join(nums) + "]"
    text = re.sub(r"\[(@[^\]]+)\]", cite, text)
    refs = "\n".join(f"[{n}] {fmt_ref(E[k])}" for k,n in sorted(numbering.items(), key=lambda kv: kv[1]))
    text = text.replace("{{REFERENCES}}", refs)
    out_md = os.path.join(VP,"reports","Bildiri_Tam_Metin.md"); open(out_md,"w",encoding="utf-8").write(text)
    print("written", out_md, "| refs:", len(numbering), "| missing keys:", sorted(missing))
    try:
        import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from md2docx import md_to_docx
        md_to_docx(out_md, out_md.replace(".md",".docx"), VP); print("docx written")
    except Exception as ex: print("docx skipped:", ex)
if __name__ == "__main__": main()
