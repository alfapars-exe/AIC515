"""download_dataset.py — AIForge-Doc v1'i Hugging Face'ten indirir (gated: HF_TOKEN gerekir).
Kullanım: python tools/download_dataset.py --out data/AIForge-Doc-v1   (HF_TOKEN ortam değişkeni ayarlı olmalı)
"""
import os, argparse
from huggingface_hub import snapshot_download
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join("data","AIForge-Doc-v1")); ap.add_argument("--revision", default=None)
a = ap.parse_args()
tok = os.environ.get("HF_TOKEN")
p = snapshot_download(repo_id="Scam-AI/AIForge-Doc-v1", repo_type="dataset", local_dir=a.out, token=tok, revision=a.revision, max_workers=8)
print("downloaded to", p)
