# -*- coding: utf-8 -*-
"""ADCD-Net zero-shot inference on AIForge-Doc v1 (val 487 + official test 812), 4 conditions in one pass.

Pre-processing replicates the repo's cross-domain evaluation path (ds.GeneralValDs + pad_collate):
  cv2.imread (BGR) -> LongestMaxSize(512, bilinear) -> PIL.fromarray -> convert('L') -> JPEG(qf) -> decode -> RGB
  -> ImageNet normalisation (mean 0.485/0.455/0.406 as in repo); DCT = quantised luma coefficients of that JPEG
  (jpeglib, same as jpeg2dct normalized=False), clip(|c|,0,20); qt = the JPEG's luma quantisation table (clamp 0..63);
  pad bottom/right to a square multiple of 16 (img 0, dct 0, ocr -1).
Conditions:
  q100_ocrproxy : qf=100 (repo default for PNG cross-domain sets) + binarisation text-mask proxy for the PaddleOCR mask
  q95_ocrproxy  : qf=95 (task spec) + proxy mask
  q100_noaln    : qf=100, alignment score forced to 0 (DCT branch contribution switched off)
  q100_emptyocr : qf=100, empty OCR mask (pristine prototype from whole image)
Probability maps (softmax class-1) are cropped to the resized size, upsampled (bilinear) to the original size
and saved as uint8 PNG under preds/adcd_<condition>/<split>/<id>.png (val/test ids share one 9-digit space).
"""
import os, sys, json, time, pathlib, tempfile, argparse
os.environ.setdefault('USERNAME', 'user')
import numpy as np, pandas as pd, cv2, jpeglib
from PIL import Image
import torch, torch.nn.functional as F
import torchvision.transforms as T

import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
LB = pathlib.Path(str(_ROOT)) / "phase2" / "literature_baselines"
DATA = pathlib.Path(str(_ROOT)) / "data" / "AIForge-Doc-v1"
AUD = pathlib.Path(str(_ROOT)) / "dataset_audit"
AD = _ROOT / "third_party" / "ADCD-Net"; ADW = AD / "weights" / "ADCD-Net_exp_data"

parser = argparse.ArgumentParser()
parser.add_argument('--limit', type=int, default=0)
parser.add_argument('--splits', default='val,test')
args = parser.parse_args()

sys.path.insert(0, str(AD)); os.chdir(str(AD))
import cfg as adcfg
adcfg.docres_ckpt_path = str(ADW / "docres.pkl"); adcfg.ckpt = str(ADW / "ADCDNet.pth")
from model.model import ADCDNet
model = ADCDNet()
ck = torch.load(adcfg.ckpt, map_location='cpu', weights_only=True)
miss, unexp = model.load_state_dict({k.replace('module.', ''): v for k, v in ck['model'].items()}, strict=False)
assert not miss and not unexp, (miss, unexp)
model = model.cuda().eval()

CONDS = [("q100_ocrproxy", 100, 'proxy', None), ("q95_ocrproxy", 95, 'proxy', None),
         ("q100_noaln", 100, 'proxy', 0.0), ("q100_emptyocr", 100, 'empty', None)]
VAL_MAX = adcfg.val_max_size
img_totsr = T.Compose([T.ToTensor(), T.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])
_tmp = pathlib.Path(tempfile.mkdtemp(prefix="adcd_")) / "tmp.jpg"

def longest_max_size(arr, max_size=VAL_MAX):
    h, w = arr.shape[:2]
    if max(h, w) <= max_size: return arr
    s = max_size / max(h, w)
    return cv2.resize(arr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_LINEAR)

def jpeg_dct_pipeline(bgr_small, qf):
    Image.fromarray(bgr_small).convert('L').save(str(_tmp), 'JPEG', quality=int(qf))
    dec = Image.open(str(_tmp)); dec.load(); dec = dec.convert('RGB')
    d = jpeglib.read_dct(str(_tmp))
    bh, bw = d.Y.shape[:2]
    dct = d.Y.transpose(0, 2, 1, 3).reshape(bh * 8, bw * 8).astype(np.int32)
    return dec, dct, torch.from_numpy(d.qt[0].astype(np.int64))

def ocr_proxy_mask(bgr_small):
    g = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return (m > 0).astype(np.uint8)

def pad_square16(t, value):
    H, W = t.shape[-2:]
    S = max(H, W); S = S if S % 16 == 0 else (S // 16 + 1) * 16
    return F.pad(t, (0, S - W, 0, S - H), value=value)

def prepare(small, qf, ocr_mode):
    h, w = small.shape[:2]
    dec, dct, qt = jpeg_dct_pipeline(small, qf)
    x = img_totsr(dec)
    dct_t = torch.from_numpy(np.clip(np.abs(dct), 0, 20)).long()[:h, :w]
    ocr = ocr_proxy_mask(small) if ocr_mode == 'proxy' else np.zeros((h, w), np.uint8)
    ocr_t = torch.from_numpy(ocr).long()[None]
    return (pad_square16(x, 0.0)[None], pad_square16(dct_t, 0)[None], qt.clamp(0, 63)[None],
            pad_square16(ocr_t, -1)[None], float(ocr.mean()))

@torch.no_grad()
def forward(img, dct, qt, ocr_mask, align_override=None):
    img_size = img.size(2)
    as_feat, ms_dct_feats = model.dct_encoder(dct, qt)
    align_logits = model.dct_align_head(as_feat)
    align_score = F.softmax(align_logits, dim=1)[:, -1]
    aln_pred = align_score.clone()
    if align_override is not None:
        align_score = torch.full_like(align_score, float(align_override))
    feat, cnt_feats, frg_feats, pp_feats = model.restormer_loc(
        img=torch.cat([model.add_coord(img), torch.zeros_like(ocr_mask)], 1),
        ms_dct_feats=ms_dct_feats, dct_align_score=align_score)
    pp_maps = model.get_pp_map(pp_feats, ocr_mask, img_size)
    feat = feat * model.pp_scale_proj(pp_maps) + model.pp_bias_proj(pp_maps)
    return model.out_head(feat), aln_pred

splits = json.load(open(AUD / "splits.json"))
jobs = [(sp, 'TrainingSet' if sp == 'val' else 'TestingSet', i) for sp in args.splits.split(',') for i in splits[sp]]
if args.limit: jobs = jobs[:args.limit]
for c, *_ in CONDS:
    for sp_ in ('val', 'test'): (LB / "preds" / f"adcd_{c}" / sp_).mkdir(parents=True, exist_ok=True)
log_path = LB / "logs" / "adcd_run.log"; log_path.parent.mkdir(exist_ok=True)
rows_path = LB / "preds" / "adcd_scores.csv"
rows = pd.read_csv(rows_path, dtype={'id': str}).to_dict('records') if rows_path.exists() else []
done = {(r['split'], r['id']) for r in rows}
def log(msg):
    with open(log_path, 'a', encoding='utf-8') as f: f.write(time.strftime('%H:%M:%S ') + msg + '\n')
log(f"start {len(jobs)} jobs, done {len(done)}")
t_start = time.time(); peak = 0.0
for k, (sp, sub, i) in enumerate(jobs):
    if (sp, i) in done: continue
    bgr = cv2.imread(str(DATA / sub / "images" / f"{i}.png"))
    H, W = bgr.shape[:2]
    small = longest_max_size(bgr); h, w = small.shape[:2]
    row = {'split': sp, 'id': i, 'W': W, 'H': H, 'h512': h, 'w512': w}
    for cond, qf, ocr_mode, ovr in CONDS:
        x, d, q, o, ocr_frac = prepare(small, qf, ocr_mode)
        t0 = time.time()
        with torch.autocast('cuda', dtype=torch.float16):
            logits, aln = forward(x.cuda(), d.cuda(), q.cuda(), o.cuda(), ovr)
        prob = F.softmax(logits.float(), 1)[:, 1:2, :h, :w]
        prob = F.interpolate(prob, size=(H, W), mode='bilinear', align_corners=False)[0, 0]
        torch.cuda.synchronize(); dt = time.time() - t0
        u8 = (prob.clamp(0, 1) * 255).round().byte().cpu().numpy()
        cv2.imwrite(str(LB / "preds" / f"adcd_{cond}" / sp / f"{i}.png"), u8, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        row.update({f'{cond}_aln': float(aln.item()), f'{cond}_t': round(dt, 3), f'{cond}_mean': float(prob.mean()), f'{cond}_max': float(prob.max())})
        if cond == 'q100_ocrproxy': row['ocr_frac'] = ocr_frac
    peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
    rows.append(row)
    if (k + 1) % 25 == 0 or k == len(jobs) - 1:
        pd.DataFrame(rows).to_csv(rows_path, index=False)
        log(f"{k+1}/{len(jobs)} elapsed {time.time()-t_start:.0f}s peakGB {peak:.2f} last {sp} {i} {W}x{H}->{w}x{h}")
pd.DataFrame(rows).to_csv(rows_path, index=False)
log(f"DONE {len(rows)} rows total {time.time()-t_start:.0f}s peakGB {peak:.2f}")
print("DONE", len(rows), round(time.time() - t_start), "s")
