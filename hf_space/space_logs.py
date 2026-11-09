"""space_logs.py — Space çalışma/derleme loglarını yazdırır: python hf_space/space_logs.py [--build] [-n 80]"""
import sys, argparse
from huggingface_hub import HfApi
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
ap = argparse.ArgumentParser(); ap.add_argument("--space", default="infinayazilim/ai515"); ap.add_argument("--build", action="store_true"); ap.add_argument("-n", type=int, default=80)
a = ap.parse_args(); api = HfApi()
rt = api.get_space_runtime(a.space); print("stage:", rt.stage, "| hardware:", rt.hardware, "| volumes:", rt.volumes)
lines = list(api.fetch_space_logs(a.space, build=a.build))
txt = "".join(lines)
for sep in ("  File ", "Traceback", "Error", "Warning", "* Running", "====="):
    txt = txt.replace(sep, "\n" + sep)
print("\n".join(txt.splitlines()[-a.n:]))
