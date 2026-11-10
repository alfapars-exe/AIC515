# -*- coding: utf-8 -*-
"""TruFor variant replicating the AIForge-Doc paper protocol (§5.1): input resized to 1024x1024 with
aspect-ratio-preserving letterbox padding (zero padding, top-left anchored), prediction cropped back to the
resized content and bilinearly resized to the original resolution. Output: preds/trufor_1024pad/<split>/<id>.png
+ preds/trufor_1024pad_scores.csv (image-level score computed on the padded 1024x1024 input)."""
import os, sys, json, time, pathlib
os.environ.setdefault('USERNAME', 'user')
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn.functional as F

import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
LB = pathlib.Path(str(_ROOT)) / "phase2" / "literature_baselines"
DATA = pathlib.Path(str(_ROOT)) / "data" / "AIForge-Doc-v1"
AUD = pathlib.Path(str(_ROOT)) / "dataset_audit"
TF = _ROOT / "third_party" / "TruFor"; TF_SRC = TF / "test_docker" / "src"; S = 1024
sys.path.insert(0, str(TF_SRC)); os.chdir(str(TF_SRC))
from config import _C as config
config.defrost(); config.merge_from_file('trufor.yaml')
config.TEST.MODEL_FILE = str(TF / "test_docker" / "weights" / "trufor.pth.tar"); config.freeze()
from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
model = confcmx(cfg=config)
model.load_state_dict(torch.load(config.TEST.MODEL_FILE, map_location='cpu', weights_only=False)['state_dict']); model = model.cuda().eval()

splits = json.load(open(AUD / "splits.json"))
jobs = [(sp, 'TrainingSet' if sp == 'val' else 'TestingSet', i) for sp in ('val', 'test') for i in splits[sp]]
for sp_ in ('val', 'test'): (LB / "preds" / "trufor_1024pad" / sp_).mkdir(parents=True, exist_ok=True)
log_path = LB / "logs" / "trufor_1024pad_run.log"; rows_path = LB / "preds" / "trufor_1024pad_scores.csv"
rows = []
def log(msg):
    with open(log_path, 'a', encoding='utf-8') as f: f.write(time.strftime('%H:%M:%S ') + msg + '\n')
t_start = time.time(); peak = 0.0
with torch.no_grad():
    for k, (sp, sub, i) in enumerate(jobs):
        img = Image.open(DATA / sub / "images" / f"{i}.png").convert("RGB"); W, H = img.size
        s = S / max(W, H); nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
        small = img.resize((nw, nh), Image.BILINEAR)
        canvas = np.zeros((S, S, 3), np.uint8); canvas[:nh, :nw] = np.array(small)
        x = torch.from_numpy(canvas.transpose(2, 0, 1)).float().div(256.0).unsqueeze(0).cuda()
        t0 = time.time(); pred, conf, det, _ = model(x); torch.cuda.synchronize(); dt = time.time() - t0
        prob = F.softmax(pred, dim=1)[:, 1:2, :nh, :nw]
        prob = F.interpolate(prob, size=(H, W), mode='bilinear', align_corners=False)[0, 0]
        Image.fromarray((prob.clamp(0, 1) * 255).round().byte().cpu().numpy()).save(LB / "preds" / "trufor_1024pad" / sp / f"{i}.png")
        peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
        rows.append({'split': sp, 'id': i, 'W': W, 'H': H, 'scale': round(s, 4), 'infer_s': round(dt, 3), 'score': torch.sigmoid(det).item(),
                     'map_mean': float(prob.mean()), 'map_max': float(prob.max())})
        if (k + 1) % 50 == 0 or k == len(jobs) - 1:
            pd.DataFrame(rows).to_csv(rows_path, index=False); log(f"{k+1}/{len(jobs)} elapsed {time.time()-t_start:.0f}s peakGB {peak:.2f}")
pd.DataFrame(rows).to_csv(rows_path, index=False); log(f"DONE {len(rows)} total {time.time()-t_start:.0f}s peakGB {peak:.2f}")
print("DONE", len(rows))
