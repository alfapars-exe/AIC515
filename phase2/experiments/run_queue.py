"""
run_queue.py — Sıralı eğitim kuyruğu (TR). Tamamlanmış run'ları (results/train_<run>.json var) atlar.

  python run_queue.py --stage B      # 3 GT × 3 seed
  python run_queue.py --stage C      # gemini_only × 3 GT (seed 42)
  python run_queue.py --stage BC     # ikisi sırayla
"""
import os, sys, json, time, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
GTS = ["bbox", "pixel_eps8_morph", "glyph_v2"]


def runs(stage):
    out = []
    if "B" in stage:
        for seed in (42, 43, 44):
            for gt in GTS:
                out.append((f"unet_{gt}_s{seed}", ["--gt", gt, "--seed", str(seed)]))
    if "C" in stage:
        for gt in GTS:
            out.append((f"unet_{gt}_geminionly_s42", ["--gt", gt, "--seed", "42", "--train-filter", "gemini_only"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="BC")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=4)
    args = ap.parse_args()
    env = dict(os.environ, USERNAME="user", HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONUNBUFFERED="1")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    qlog = open(os.path.join(HERE, "logs", "queue.log"), "a", encoding="utf-8")
    for name, extra in runs(args.stage):
        done = os.path.join(HERE, "results", f"train_{name}.json")
        if os.path.exists(done):
            qlog.write(f"{time.strftime('%H:%M:%S')} skip {name} (done)\n"); qlog.flush(); continue
        cmd = [sys.executable, "-u", os.path.join(HERE, "train_unet_p2.py"), "--epochs", str(args.epochs), "--patience", str(args.patience)] + extra
        qlog.write(f"{time.strftime('%H:%M:%S')} start {name}\n"); qlog.flush()
        t = time.time()
        with open(os.path.join(HERE, "logs", f"{name}.log"), "w", encoding="utf-8") as lf:
            rc = subprocess.call(cmd, cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)
        qlog.write(f"{time.strftime('%H:%M:%S')} end {name} rc={rc} {time.time() - t:.0f}s\n"); qlog.flush()
    qlog.write(f"{time.strftime('%H:%M:%S')} queue {args.stage} finished\n"); qlog.close()


if __name__ == "__main__":
    main()
