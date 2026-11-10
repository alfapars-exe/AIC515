r"""
deploy_hf.py — Projeyi Hugging Face'e dağıtır (TR).

  1. Özel bir Storage Bucket oluşturur (varsayılan: infinayazilim/ai515-data) ve
     veri setini (data/AIForge-Doc-v1), modelleri (models/) ve audit dosyalarını proje dizin yapısıyla senkronlar.
  2. hf_space/ klasörünü Space'e (varsayılan: infinayazilim/ai515, Gradio) yükler.
  3. Bucket'ı Space'e /data olarak bağlar, AI515_DATA_ROOT=/data değişkenini tanımlar ve Space'i yeniden başlatır.

Kullanım (proje kökünde, docforensics ortamında; önce `hf auth login` ile WRITE yetkili token):
    pip install -U "huggingface_hub>=2.4"
    python hf_space\deploy_hf.py                     # tam veri seti (≈10 GB) + modeller + Space
    python hf_space\deploy_hf.py --sample 100        # yalnız 100 örnek çift (hızlı deneme)
    python hf_space\deploy_hf.py --skip-data         # yalnız Space kodunu güncelle
    python hf_space\deploy_hf.py --dry-run           # ne yükleneceğini göster

Not: AIForge-Doc v1 CC BY-NC-SA 4.0 + kapılı (gated) bir veri setidir; bucket ÖZEL (private) tutulur, yeniden dağıtılmaz.
Bucket depolama ücretleri için https://huggingface.co/storage sayfasına bakın.
"""
from __future__ import annotations
import argparse, os, shutil, sys, time
from pathlib import Path

from huggingface_hub import HfApi, Volume

for _s in (sys.stdout, sys.stderr):  # Windows cp1254 konsolunda Unicode çıktı
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
ROOT = _ROOT  # proje kökü (Vize Proje)
SPACE_DIR = ROOT / "hf_space"  # türetilmiş Space paketi (tools/build_space.py üretir)


def log(msg):
    print(f"[deploy] {msg}", flush=True)


def make_sample(n: int, out: Path):
    """samples_forged.csv'den n adet eşleşmiş (pair_accepted) eğitim çifti seçip bucket düzeninde kopyalar."""
    import pandas as pd
    df = pd.read_csv(ROOT / "dataset_audit" / "samples_forged.csv", dtype={"new_id": str}, low_memory=False)
    df["new_id"] = df["new_id"].astype(int).map(lambda x: f"{x:09d}")
    tr = df[(df.split == "training") & (df.pair_accepted.astype(str) == "True")]
    # kaynak × generator dengeli örnekleme
    picks = []
    groups = list(tr.groupby(["source_dataset", "assigned_tool"]))
    per = max(1, n // len(groups))
    for _, g in groups:
        picks.append(g.sample(min(per, len(g)), random_state=42))
    sel = pd.concat(picks).head(n)
    src = ROOT / "data" / "AIForge-Doc-v1" / "TrainingSet"
    dst = out / "data" / "AIForge-Doc-v1" / "TrainingSet"
    for sub in ("images", "masks", "authentic"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    for r in sel.itertuples():
        shutil.copy2(src / "images" / f"{r.new_id}.png", dst / "images")
        shutil.copy2(src / "masks" / f"{r.new_id}.png", dst / "masks")
        ap = src / "authentic" / str(r.pair_name)
        if ap.exists():
            shutil.copy2(ap, dst / "authentic")
    (out / "dataset_audit").mkdir(parents=True, exist_ok=True)
    sel.to_csv(out / "dataset_audit" / "samples_forged.csv", index=False)
    shutil.copy2(ROOT / "dataset_audit" / "splits.json", out / "dataset_audit")
    log(f"örnek alt küme hazır: {len(sel)} çift → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="infinayazilim/ai515")
    ap.add_argument("--bucket", default="infinayazilim/ai515-data")
    ap.add_argument("--mount", default="/data")
    ap.add_argument("--sample", type=int, default=0, help="yalnız N örnek çift yükle (0 = tam veri seti)")
    ap.add_argument("--skip-data", action="store_true", help="bucket senkronunu atla (yalnız Space kodu)")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--no-test-set", action="store_true", help="TestingSet'i yükleme")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--message", default="AIC515 app: aşamalı ön işleme + U-Net/ResNet arayüzü", help="Space commit mesajı")
    ap.add_argument("--public-space", action="store_true", help="Space'i herkese açık yap (varsayılan: mevcut ayar)")
    ap.add_argument("--hardware", default="zero-a10g", help="Space donanımı (ZeroGPU: zero-a10g; CPU: cpu-basic)")
    args = ap.parse_args()
    sys.path.insert(0, str(_KOD / "tools")); import build_space  # hf_space/ türetilmiş pakettir
    build_space.build(check=args.dry_run, log=log)

    api = HfApi()
    me = api.whoami()
    log(f"oturum: {me['name']} (orgs: {[o['name'] for o in me.get('orgs', [])]})")

    # ---------- 1) bucket
    if not args.skip_data or not args.skip_models:
        log(f"bucket: {args.bucket} (private)")
        if not args.dry_run:
            api.create_bucket(args.bucket, private=True, exist_ok=True)
        dest = f"hf://buckets/{args.bucket}"
        if not args.skip_models:
            log("modeller senkronlanıyor (models/ → bucket/models) ...")
            api.sync_bucket(str(ROOT / "models"), f"{dest}/models", include=["*.pt"], dry_run=args.dry_run, verbose=args.dry_run)
        if not args.skip_data:
            if args.sample:
                tmp = SPACE_DIR / "_sample"
                if tmp.exists():
                    shutil.rmtree(tmp)
                make_sample(args.sample, tmp)
                api.sync_bucket(str(tmp), dest, dry_run=args.dry_run, verbose=args.dry_run)
            else:
                log("audit dosyaları senkronlanıyor ...")
                api.sync_bucket(str(ROOT / "dataset_audit"), f"{dest}/dataset_audit",
                                include=["samples_forged.csv", "splits.json"], dry_run=args.dry_run, verbose=args.dry_run)
                log("veri seti senkronlanıyor (≈10 GB; ilk yüklemede uzun sürer, kesilirse aynı komut kaldığı yerden devam eder) ...")
                exclude = [".cache/*", "*.lock"] + (["TestingSet/*"] if args.no_test_set else [])
                api.sync_bucket(str(ROOT / "data" / "AIForge-Doc-v1"), f"{dest}/data/AIForge-Doc-v1",
                                exclude=exclude, dry_run=args.dry_run, verbose=args.dry_run)
        if not args.dry_run:
            info = api.bucket_info(args.bucket)
            log(f"bucket durumu: {info.total_files} dosya, {info.size / 2**30:.2f} GiB, private={info.private}")

    # ---------- 2) Space kodu
    log(f"Space: {args.space} ← {SPACE_DIR}")
    if not args.dry_run:
        api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True,
                        private=None if not args.public_space else False)
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(SPACE_DIR), path_in_repo=".",
                          ignore_patterns=["_sample/*", "__pycache__/*", "*.pyc", "*.log", "deploy_hf.py", "space_logs.py", "space_status.py", "live_test.py"],
                          commit_message=args.message)
        # ---------- 3) volume + değişken
        api.set_space_volumes(args.space, [Volume(type="bucket", source=args.bucket, mount_path=args.mount, read_only=True)])
        api.add_space_variable(args.space, "AI515_DATA_ROOT", args.mount, description="bucket mount kökü (data/, models/, dataset_audit/)")
        try:
            api.request_space_hardware(args.space, hardware=args.hardware)
            log(f"donanım isteği: {args.hardware}")
        except Exception as e:
            log(f"donanım isteği başarısız: {e}")
        api.restart_space(args.space)
        for _ in range(60):
            rt = api.get_space_runtime(args.space)
            log(f"runtime: {rt.stage} ({rt.hardware})")
            if rt.stage in ("RUNNING", "RUNNING_APP_STARTING"):
                break
            if "ERROR" in str(rt.stage):
                log("HATA — loglar: hf spaces logs " + args.space + " --build"); break
            time.sleep(10)
    log(f"bitti → https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
