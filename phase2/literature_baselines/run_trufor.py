# -*- coding: utf-8 -*-
"""TruFor zero-shot inference on AIForge-Doc v1 (val 487 + official test 812).

Protocol (DOCFORGE-BENCH style): if H*W > 2 MP the image is downscaled (bilinear,
aspect preserved) so that H*W == 2 MP; the anomaly map is upsampled back to the
original size (bilinear) before saving as uint8 PNG under preds/trufor/<split>/<id>.png
(val and test ids share the same 9-digit space, hence the split sub-folder). Image-level integrity score
('score', sigmoid of detection head) and the confidence map are also saved.
"""
import os, sys, json, time, pathlib, argparse
os.environ.setdefault('USERNAME', 'user')
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

LB = pathlib.Path(os.environ.get("VIZE_ROOT", r"C:\Users\PARS\Desktop\PROJELER\AI 515\Vize Proje")) / "phase2" / "literature_baselines"
DATA = pathlib.Path(os.environ.get("VIZE_ROOT", r"C:\Users\PARS\Desktop\PROJELER\AI 515\Vize Proje")) / "data" / "AIForge-Doc-v1"
AUD = pathlib.Path(os.environ.get("VIZE_ROOT", r"C:\Users\PARS\Desktop\PROJELER\AI 515\Vize Proje")) / "dataset_audit"
TF_SRC = LB / "TruFor" / "test_docker" / "src"
MAX_MP = 2.0

parser = argparse.ArgumentParser()
parser.add_argument('--limit', type=int, default=0)
parser.add_argument('--splits', default='val,test')
args = parser.parse_args()

sys.path.insert(0, str(TF_SRC)); os.chdir(str(TF_SRC))
from config import _C as config
config.defrost(); config.merge_from_file('trufor.yaml')
config.TEST.MODEL_FILE = str(LB / "TruFor" / "test_docker" / "weights" / "trufor.pth.tar"); config.freeze()
from models.cmx.builder_np_conf import myEncoderDecoder as confcmx

model = confcmx(cfg=config)
ck = torch.load(config.TEST.MODEL_FILE, map_location='cpu', weights_only=False)
model.load_state_dict(ck['state_dict']); model = model.cuda().eval()

splits = json.load(open(AUD / "splits.json"))
jobs = []
for sp in args.splits.split(','):
    sub = 'TrainingSet' if sp == 'val' else 'TestingSet'
    for i in splits[sp]:
        jobs.append((sp, sub, i))
if args.limit: jobs = jobs[:args.limit]

for sp_ in ('val', 'test'):
    (LB / "preds" / "trufor" / sp_).mkdir(parents=True, exist_ok=True); (LB / "preds" / "trufor_conf" / sp_).mkdir(parents=True, exist_ok=True)
log_path = LB / "logs" / "trufor_run.log"; log_path.parent.mkdir(exist_ok=True)
rows_path = LB / "preds" / "trufor_scores.csv"
rows = []
if rows_path.exists():
    rows = pd.read_csv(rows_path, dtype={'id': str}).to_dict('records')
done = {(r['split'], r['id']) for r in rows}

def log(msg):
    with open(log_path, 'a', encoding='utf-8') as f: f.write(time.strftime('%H:%M:%S ') + msg + '\n')

log(f"start {len(jobs)} jobs, already done {len(done)}; torch {torch.__version__}")
t_start = time.time(); peak = 0.0
with torch.no_grad():
    for k, (sp, sub, i) in enumerate(jobs):
        if (sp, i) in done: continue
        p_map = LB / "preds" / "trufor" / sp / f"{i}.png"
        img = Image.open(DATA / sub / "images" / f"{i}.png").convert("RGB")
        W, H = img.size
        scale = 1.0
        if H * W > MAX_MP * 1e6:
            scale = (MAX_MP * 1e6 / (H * W)) ** 0.5
            img = img.resize((int(round(W * scale)), int(round(H * scale))), Image.BILINEAR)
        x = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float().div(256.0).unsqueeze(0).cuda()
        t0 = time.time()
        try:
            pred, conf, det, _ = model(x)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); log(f"OOM {sp} {i} at {x.shape}; retry at half size")
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False); scale *= 0.5
            pred, conf, det, _ = model(x)
        torch.cuda.synchronize(); dt = time.time() - t0
        prob = F.softmax(pred, dim=1)[:, 1:2]
        prob = F.interpolate(prob, size=(H, W), mode='bilinear', align_corners=False)[0, 0]
        cm = torch.sigmoid(conf)
        cm = F.interpolate(cm, size=(H, W), mode='bilinear', align_corners=False)[0, 0]
        score = torch.sigmoid(det).item()
        Image.fromarray((prob.clamp(0, 1) * 255).round().byte().cpu().numpy()).save(p_map)
        Image.fromarray((cm.clamp(0, 1) * 255).round().byte().cpu().numpy()).save(LB / "preds" / "trufor_conf" / sp / f"{i}.png")
        peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
        rows.append({'split': sp, 'id': i, 'W': W, 'H': H, 'scale': round(scale, 4), 'infer_s': round(dt, 3),
                     'score': score, 'map_mean': float(prob.mean()), 'map_max': float(prob.max()), 'conf_mean': float(cm.mean())})
        if (k + 1) % 25 == 0 or k == len(jobs) - 1:
            pd.DataFrame(rows).to_csv(rows_path, index=False)
            log(f"{k+1}/{len(jobs)} elapsed {time.time()-t_start:.0f}s peakGB {peak:.2f} last {sp} {i} {W}x{H} s={scale:.3f} dt={dt:.2f}")
pd.DataFrame(rows).to_csv(rows_path, index=False)
log(f"DONE {len(rows)} rows total {time.time()-t_start:.0f}s peakGB {peak:.2f}")
print("DONE", len(rows), round(time.time() - t_start), "s peakGB", round(peak, 2))
