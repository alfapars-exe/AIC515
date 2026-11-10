"""
eval_queue.py — Eğitim kuyruğunu izler; bir seed'in 3 modeli bitince run_eval'ı (tüm aşamalar) çalıştırır (TR).
Sıra: B s42 → B s43 → B s44 → C (geminionly, val+test) → H4. Tamamlanan aşamaları (results/eval_done_<tag>.json) atlar.
"""
import os, sys, time, json, subprocess
import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = str(_out_dir(__file__))
GTS = ["bbox", "pixel_eps8_morph", "glyph_v2"]
env = dict(os.environ, USERNAME="user", HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONUNBUFFERED="1")
qlog = open(os.path.join(OUT, "logs", "eval_queue.log"), "a", encoding="utf-8")


def log(msg):
    qlog.write(f"{time.strftime('%H:%M:%S')} {msg}\n"); qlog.flush()


def trained(name):
    return os.path.exists(os.path.join(OUT, "results", f"train_{name}.json"))


def run(cmd, logname):
    with open(os.path.join(OUT, "logs", logname), "a", encoding="utf-8") as lf:
        return subprocess.call([sys.executable, "-u"] + cmd, cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)


def wait_for(names):
    while not all(trained(n) for n in names):
        time.sleep(30)


jobs = []
for seed in (42, 43, 44):
    names = [f"unet_{g}_s{seed}" for g in GTS]
    # B_s42: val,test,robust,controls 15:00-15:48'de tamamlandı; ablation özetinde hata → yalnız ablation+rectfit yeniden
    stages = "ablation,rectfit" if seed == 42 else "val,test,robust,controls,ablation,rectfit"
    jobs.append((f"B_s{seed}", names, stages))
jobs.append(("C_geminionly", [f"unet_{g}_geminionly_s42" for g in GTS], "val,test"))

for tag, names, stages in jobs:
    done = os.path.join(OUT, "results", f"eval_done_{tag}.json")
    if os.path.exists(done):
        log(f"skip {tag}"); continue
    log(f"waiting for {names}"); wait_for(names)
    log(f"start eval {tag}"); t = time.time()
    rc = run(["run_eval.py", "--models", ",".join(names), "--stages", stages], f"eval_{tag}.log")
    log(f"end eval {tag} rc={rc} {time.time() - t:.0f}s")
    if rc == 0:
        json.dump(dict(tag=tag, models=names, stages=stages, seconds=time.time() - t), open(done, "w"), indent=1)

if not os.path.exists(os.path.join(OUT, "results", "h4_results.csv")):
    log("start h4"); t = time.time()
    rc = run(["h4_global_trace.py", "--epochs", "10"], "h4.log")
    log(f"end h4 rc={rc} {time.time() - t:.0f}s")
log("eval queue finished")
