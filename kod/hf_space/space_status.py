import sys, json
from huggingface_hub import HfApi
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
api = HfApi(); sid = sys.argv[1] if len(sys.argv) > 1 else "infinayazilim/ai515"
info = api.space_info(sid)
rt = info.runtime
print("sdk:", info.sdk, "| private:", info.private)
print("runtime raw:", json.dumps(rt.raw if hasattr(rt, "raw") else str(rt), default=str)[:1500])
print("files:", [s.rfilename for s in info.siblings])
