import sys, time
sys.stdout.reconfigure(encoding="utf-8")
from huggingface_hub import HfApi
api = HfApi(); rid = "infinayazilim/ai515"
files = api.list_repo_files(rid, repo_type="space")
print("dosyalar:", files)
extra = [f for f in files if f in ("space_logs.py", "space_status.py", "live_test.py", "deploy_hf.py")]
if extra:
    api.delete_files(rid, delete_patterns=extra, repo_type="space", commit_message="remove helper scripts")
    print("silindi:", extra)
for i in range(60):
    rt = api.get_space_runtime(rid)
    print(i, rt.stage, rt.hardware)
    if rt.stage in ("RUNNING", "RUNTIME_ERROR", "BUILD_ERROR"): break
    time.sleep(10)
