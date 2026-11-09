"""make_package.py — Teslim paketini oluşturur (TR): proje yapısını TESLIM_PAKETI/ altına aynalar.
Dışarıda bırakılanlar (yeniden üretilebilir / lisanslı ham veri): data/ (görüntüler), phase2/*/preds/, phase2/controls/*/images|masks, baselines/cache, __pycache__.
Kullanım: python tools/make_package.py --root <Vize Proje> [--light]   (--light: ağırlıklar, klon repo'lar ve GT maskeleri hariç ~100 MB)
"""
import os, shutil, argparse, fnmatch, time
ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.environ.get("VIZE_ROOT", os.getcwd())); ap.add_argument("--out", default=None); ap.add_argument("--light", action="store_true"); a = ap.parse_args()
VP = a.root; OUT = a.out or os.path.join(VP, "TESLIM_PAKETI" + ("_LIGHT" if a.light else ""))
EXCL_DIRS = {"data", "preds", "__pycache__", "cache", "TESLIM_PAKETI", "TESLIM_PAKETI_LIGHT", "logs_master", ".git", "_INVALID_flat_id_collision"}
EXCL_CTRL = {"images", "masks"}  # under phase2/controls/<type>/
def ignore(d, names):
    rel = os.path.relpath(d, VP); ig = set()
    for n in names:
        p = os.path.join(d, n)
        if n in EXCL_DIRS and os.path.isdir(p): ig.add(n); continue
        if rel.startswith(os.path.join("phase2","controls")) and n in EXCL_CTRL and os.path.isdir(p): ig.add(n); continue
        if a.light:
            if os.path.isdir(p) and (rel == "models" or rel.startswith("models") or rel.startswith(os.path.join("phase2","literature_baselines","TruFor")) or rel.startswith(os.path.join("phase2","literature_baselines","ADCD-Net"))): ig.add(n); continue
            if rel == os.path.join("phase2","gt") and os.path.isdir(p) and n not in ("figures",): ig.add(n); continue
            if n.endswith((".pt",".pth",".pkl",".tar")) : ig.add(n); continue
            if rel.startswith(os.path.join("phase2","literature_baselines")) and os.path.isdir(p) and n in ("TruFor","ADCD-Net"): ig.add(n); continue
    return ig
t0=time.time()
if os.path.exists(OUT): shutil.rmtree(OUT)
shutil.copytree(VP, OUT, ignore=ignore, dirs_exist_ok=True)
# data pointer
os.makedirs(os.path.join(OUT,"data"), exist_ok=True)
open(os.path.join(OUT,"data","README_DATA.md"),"w",encoding="utf-8").write("# Veri seti bu pakete dahil DEĞİLDİR\n\nAIForge-Doc v1 (Hugging Face `Scam-AI/AIForge-Doc-v1`, gated, CC BY-NC-SA 4.0) lisans gereği yeniden dağıtılmaz.\nİndirme: `python tools/download_dataset.py --out data/AIForge-Doc-v1` (HF_TOKEN gerekir; ~10 GB, 12.187 dosya).\nBeklenen yapı: data/AIForge-Doc-v1/{TrainingSet,TestingSet}/{images,masks,authentic}, metadata.jsonl\n")
n=0; s=0
for r,_,fs in os.walk(OUT):
    for f in fs: n+=1; s+=os.path.getsize(os.path.join(r,f))
print(f"package: {OUT} | files {n} | {s/2**30:.2f} GB | {time.time()-t0:.0f}s")
