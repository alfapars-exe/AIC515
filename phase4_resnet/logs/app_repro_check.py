# Uygulama (kod/app) ResNet kararının Faz 4 boru hattıyla aynı olduğunu doğrular: 20 val sahte + 20 authentic.
import os, sys, json
os.environ.setdefault("USERNAME", "user")
ROOT = os.environ["VIZE_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "kod", "app")); sys.path.insert(0, os.path.join(ROOT, "kod", "phase4_resnet"))
import numpy as np
import p4common as P
import app as A
D = P.D
out = {}
for stem in ("resnet18_ft_full_s42", "resnet18_ft_full_mr_s42", "resnet18_ft_best"):
    name = ("PHASE1/" if stem.endswith("_best") else "PHASE4/") + stem
    tab = D.split_table("val").iloc[:20]
    diffs, agree, n = [], 0, 0
    for kind in ("val_forged", "val_auth"):
        z = np.load(P.PRED / stem / f"{kind}.npz"); ids = [str(i) for i in z["ids"]]; off = z["offsets"]
        ref = {i: float(z["scores"][off[k]:off[k + 1]].max()) for k, i in enumerate(ids)}
        for i, r in tab.iterrows():
            img = D.read_rgb(D.img_path(i, "train") if kind == "val_forged" else D.auth_path(r["pair_name"], "train"))
            hm, info = A.gpu_infer([], [name], [img], 112)[0][name]
            s, thr, cal = A.resnet_decision(name, info["scores"])
            assert cal, name
            diffs.append(abs(s - ref[i])); agree += int((s >= thr) == (ref[i] >= thr)); n += 1
    out[name] = dict(friendly=A.friendly(name), thr=thr, n=n, max_abs_diff=float(max(diffs)), mean_abs_diff=float(np.mean(diffs)), decisions_agree=agree)
out["_def_resnet"] = A._DEF_RESNET; out["_resnets"] = A._RESNETS
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(P.RES / "app_repro_check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
