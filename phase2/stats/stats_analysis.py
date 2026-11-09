# -*- coding: utf-8 -*-
"""
Faz 3 — İstatistiksel analiz (AIC515 / AIForge-Doc v1 Faz 2 deneyleri)
Kaynak-belge-grubu (cluster) bootstrap güven aralıkları, eşleştirilmiş bootstrap farkları,
Holm düzeltmesi, Kendall tau sıralama kararlılığı ve H1–H4 karar tablosu.

Çalıştırma:  python stats_analysis.py  [--B 1000] [--seed 0]
Girdi:  ../experiments/results/*.csv, ../literature_baselines/zero_shot_per_image.csv
Çıktı:  stats_summary.csv, rank_kendall.csv, hypothesis_decisions.md, figures/*.png

Protokol
* Birincil görüntü-düzeyi metrikler: ap_img (per-image AP) ve f1_tau (F1@tau*, tau* val'de kalibre);
  ikincil: f1_0p5, iou_tau.
* Seed'ler sabit faktör: önce görüntü basina 3 seed ortalamasi, sonra grup-bootstrap (B=1000, seed 0).
* Grup = kaynak belge grubu (samples_forged.csv 'group'); gruplar yerine koyarak örneklenir, istatistik
  örneklenen gruplardaki tüm görüntüler üzerinden hesaplanir (cluster bootstrap, yüzdelik CI).
* Farklar: EŞLEŞTİRİLMİŞ bootstrap — ayni grup örneklemi her iki koşula uygulanir.
* p_boot = max(2*min(P(Δ<=0), P(Δ>=0)), 1/B); Holm düzeltmesi her hipotez ailesinin önceden belirlenmiş
  karar kontrastlari üzerinde (sütun holm_family).
* n_pos = 0 (boş GT) görüntüler F1/AP ortalamalarindan dişlanir ve raporlanir.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from scipy.stats import kendalltau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
BASE = os.path.abspath(os.path.join(HERE, '..'))
RES = os.path.join(BASE, 'experiments', 'results')
ZS_PATH = os.path.join(BASE, 'literature_baselines', 'zero_shot_per_image.csv')
OUT = HERE
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument('--B', type=int, default=1000)
ap.add_argument('--seed', type=int, default=0)
args, _ = ap.parse_known_args()
B, SEED = args.B, args.seed

GT_SHORT = {'bbox': 'bbox', 'pixel_eps8_morph': 'pixel', 'glyph_v2': 'glyph'}
GT_LABEL = {'bbox': 'bbox GT', 'pixel': 'pixel-change GT', 'glyph': 'glyph GT'}
MODEL_LABEL = {'bbox': 'bbox-GT-trained', 'pixel': 'pixel-GT-trained', 'glyph': 'glyph-GT-trained'}
MODEL_COLOR = {'bbox': '#d95f02', 'pixel': '#1b9e77', 'glyph': '#7570b3'}  # Dark2: orange / teal / purple (CVD-safe)
MODELS = ['bbox', 'pixel', 'glyph']
NOTES = []   # exclusion / data notes for the report

def note(s):
    print('[note]', s); NOTES.append(s)

# ----------------------------------------------------------------------------- data
def parse_model(df):
    m = df['model'].astype(str)
    df['gemonly'] = m.str.contains('geminionly')
    df['seed'] = m.str.extract(r'_s(\d+)$')[0].astype(int)
    tg = m.str.replace('unet_', '', regex=False).str.replace('_geminionly', '', regex=False).str.replace(r'_s\d+$', '', regex=True)
    df['train_gt'] = tg.map(GT_SHORT)
    if 'gt' in df.columns:
        df['eval_gt'] = df['gt'].map(GT_SHORT)
    return df

def load(name):
    return parse_model(pd.read_csv(os.path.join(RES, name)))

val = load('per_image_val.csv')
test = load('per_image_test.csv')
rob = load('per_image_robustness.csv')
bnd = load('per_image_boundary.csv')
rect = load('per_image_rectfit.csv')
ctrl = load('per_image_controls.csv')
h4 = pd.read_csv(os.path.join(RES, 'h4_per_image_val.csv'))
h4pc = pd.read_csv(os.path.join(RES, 'h4_per_image_val_poscontrol.csv'))
zs = pd.read_csv(ZS_PATH)

# exclusions: n_pos == 0 (boş GT)
for nm, df in [('val', val), ('test', test), ('robustness', rob), ('boundary', bnd)]:
    z = df[(df.n_pos == 0)]
    if len(z):
        ids = sorted(z.new_id.unique().tolist())
        note(f"{nm}: n_pos=0 (boş GT) → F1/AP ortalamalarından dışlandı: GT {sorted(z.eval_gt.unique().tolist())}, new_id {ids} ({len(ids)} görüntü; metrikler zaten NaN).")
nt = test[(test.model == 'unet_bbox_s42')]
for g in ['pixel', 'glyph']:
    d = nt[nt.eval_gt == g]
    note(f"test: {g}-GT yalnız eşleşmiş authentic'i olan 89 çiftte mevcut; n_pos>0 olan {int((d.n_pos > 0).sum())} görüntü kullanıldı, {int(d.ap_img.isna().sum())} görüntü (GT yok/boş) dışlandı.")

METRICS = ['ap_img', 'f1_tau', 'f1_0p5', 'iou_tau', 'score_top1pct', 'score_max']

def seed_avg(df, keys, metrics):
    """3 seed ortalaması (gemini-only modeller hariç). Tüm seed'ler NaN ise NaN kalır."""
    d = df[~df.gemonly]
    ms = [m for m in metrics if m in d.columns]
    out = d.groupby(keys + ['new_id'], as_index=False)[ms].mean()
    cnt = d.groupby(keys + ['new_id']).size().rename('n_seed').reset_index()
    out = out.merge(cnt, on=keys + ['new_id'])
    meta_cols = [c for c in ['group', 'tool', 'source', 'area_bin', 'clean_test', 'split'] if c in d.columns]
    meta = d.drop_duplicates(keys + ['new_id'])[keys + ['new_id'] + meta_cols]
    return out.merge(meta, on=keys + ['new_id'])

valA = seed_avg(val, ['train_gt', 'eval_gt'], METRICS)
testA = seed_avg(test, ['train_gt', 'eval_gt'], METRICS)
robA = seed_avg(rob, ['train_gt', 'eval_gt', 'condition'], METRICS)
bndA = seed_avg(bnd, ['train_gt', 'eval_gt', 'condition'], METRICS)
rectA = seed_avg(rect, ['train_gt', 'eval_gt'], ['f1_raw', 'iou_raw', 'f1_rectfit', 'iou_rectfit'])
ctrlA = seed_avg(ctrl, ['train_gt', 'condition'], ['score_top1pct', 'score_max', 'pos_frac_0p5', 'pos_frac_tau'])
assert (valA.n_seed == 3).all() and (testA.n_seed == 3).all() and (robA.n_seed == 3).all()

# ----------------------------------------------------------------------------- bootstrap engine
class GroupBoot:
    """Cluster (kaynak-belge-grubu) bootstrap. Aynı M matrisi (B×G çokluk) tüm kontrastlarda paylaşılır →
    eşleştirilmiş farklar aynı grup örneklemiyle hesaplanır."""
    def __init__(self, groups, B, seed):
        self.groups = np.asarray(sorted(set(groups)))
        self.G = len(self.groups)
        self.gid = {g: i for i, g in enumerate(self.groups)}
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self.G, size=(B, self.G))
        self.M = np.stack([np.bincount(r, minlength=self.G) for r in idx]).astype(float)  # B×G
        self.B = B

    def gidx(self, groups):
        return np.array([self.gid[g] for g in groups])

    def mean(self, vals, groups):
        """Örneklenen gruplardaki geçerli (non-NaN) görüntülerin ortalaması: B-vektör + nokta tahmini."""
        vals = np.asarray(vals, float); gi = self.gidx(groups)
        ok = ~np.isnan(vals)
        S = np.bincount(gi[ok], weights=vals[ok], minlength=self.G)
        N = np.bincount(gi[ok], minlength=self.G).astype(float)
        num = self.M @ S; den = self.M @ N
        with np.errstate(invalid='ignore', divide='ignore'):
            boot = num / den
        return float(S.sum() / N.sum()), boot, int(ok.sum()), int(len(np.unique(gi[ok])))

    def auroc(self, s_pos, g_pos, s_neg, g_neg):
        """Ağırlıklı Mann-Whitney AUROC; pozitif ve negatifler aynı grup örneklemiyle yeniden ağırlıklanır."""
        s_pos = np.asarray(s_pos, float); s_neg = np.asarray(s_neg, float)
        okp = ~np.isnan(s_pos); okn = ~np.isnan(s_neg)
        s_pos, g_pos = s_pos[okp], np.asarray(g_pos)[okp]; s_neg, g_neg = s_neg[okn], np.asarray(g_neg)[okn]
        order = np.argsort(s_neg, kind='mergesort'); sn = s_neg[order]; gn = self.gidx(g_neg[order])
        gp = self.gidx(g_pos)
        lo = np.searchsorted(sn, s_pos, 'left'); hi = np.searchsorted(sn, s_pos, 'right')
        Wn = self.M[:, gn]                                   # B×n_neg
        cum = np.concatenate([np.zeros((self.B, 1)), np.cumsum(Wn, axis=1)], axis=1)  # B×(n_neg+1)
        Wp = self.M[:, gp]                                   # B×n_pos
        less = cum[:, lo]; eq = cum[:, hi] - cum[:, lo]
        num = (Wp * (less + 0.5 * eq)).sum(1); den = Wp.sum(1) * Wn.sum(1)
        with np.errstate(invalid='ignore', divide='ignore'):
            boot = num / den
        # point estimate (unweighted)
        n_less = lo.astype(float); n_eq = (hi - lo).astype(float)
        point = float((n_less + 0.5 * n_eq).sum() / (len(s_pos) * len(s_neg)))
        return point, boot, int(len(s_pos)), int(len(np.unique(np.concatenate([gp, gn]))))

def ci(boot):
    b = boot[~np.isnan(boot)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def pval(boot, null=0.0):
    b = boot[~np.isnan(boot)]
    p = 2 * min((b <= null).mean(), (b >= null).mean())
    return float(min(1.0, max(p, 1.0 / len(b))))

ROWS = []
def add(family, hyp, contrast, point, boot, n_img, n_group, metric, condition='', null=None, holm_family='', notes=''):
    lo, hi = ci(boot)
    p = pval(boot, null) if null is not None else np.nan
    ROWS.append(dict(family=family, hypothesis=hyp, contrast=contrast, metric=metric, condition=condition,
                     n_img=n_img, n_group=n_group, mean=point, ci_lo=lo, ci_hi=hi, p_boot=p, holm_family=holm_family,
                     null=null if null is not None else np.nan, notes=notes))
    return point, lo, hi, p

def holm(df):
    df = df.copy(); df['p_holm'] = np.nan
    for fam, idx in df[df.holm_family != ''].groupby('holm_family').groups.items():
        p = df.loc[idx, 'p_boot'].values; m = len(p)
        order = np.argsort(p); adj = np.empty(m)
        running = 0.0
        for k, i in enumerate(order):
            running = max(running, (m - k) * p[i]); adj[i] = min(1.0, running)
        df.loc[idx, 'p_holm'] = adj
    return df

# populations
GB_VAL = GroupBoot(val.group.unique(), B, SEED)
GB_TEST = GroupBoot(test.group.unique(), B, SEED)
assert len(set(val.group) & set(test.group)) == 0, 'val/test grup id çakışması'
GB_ALL = GroupBoot(np.concatenate([val.group.unique(), test.group.unique()]), B, SEED)

def tab(dfA, **flt):
    d = dfA
    for k, v in flt.items():
        d = d[d[k] == v]
    return d.set_index('new_id').sort_index()

def paired(dA, dB, metric):
    """ortak (her ikisinde de geçerli) görüntüler"""
    j = dA[[metric, 'group']].join(dB[[metric]], lsuffix='_a', rsuffix='_b', how='inner').dropna()
    return j[metric + '_a'].values, j[metric + '_b'].values, j['group'].values

# ============================================================================= H1
print('== H1')
for m in MODELS:
    for g in MODELS:
        d = tab(valA, train_gt=m, eval_gt=g)
        for met in ['ap_img', 'f1_tau', 'f1_0p5', 'iou_tau']:
            pt, bt, n, ng = GB_VAL.mean(d[met].values, d.group.values)
            add('H1', 'H1', f'mean[{MODEL_LABEL[m]} | {GT_LABEL[g]}]', pt, bt, n, ng, met, 'val/clean')
for m in MODELS:
    dB = tab(valA, train_gt=m, eval_gt='bbox')
    for g in ['pixel', 'glyph']:
        dG = tab(valA, train_gt=m, eval_gt=g)
        for met in ['ap_img', 'f1_tau', 'f1_0p5', 'iou_tau']:
            a, b, grp = paired(dB, dG, met)
            _, bA, n, ng = GB_VAL.mean(a, grp); _, bB, _, _ = GB_VAL.mean(b, grp)
            add('H1', 'H1', f'Δ[{MODEL_LABEL[m]}] bbox-GT − {g}-GT', float(a.mean() - b.mean()), bA - bB, n, ng, met, 'val/clean',
                null=0.0, holm_family='H1' if met in ('ap_img', 'f1_tau') else '')

# Kendall tau: 9 model (3 train GT × 3 seed) sıralaması, per-image AP ortalaması
print('== Kendall')
v9 = val[~val.gemonly].copy()
models9 = sorted(v9.model.unique())
boots = {}; points = {}
for g in MODELS:
    for mod in models9:
        d = v9[(v9.model == mod) & (v9.eval_gt == g)].set_index('new_id').sort_index()
        pt, bt, n, ng = GB_VAL.mean(d.ap_img.values, d.group.values)
        points[(g, mod)] = pt; boots[(g, mod)] = bt
KROWS = []
def rank_desc(vals):
    return (-np.asarray(vals)).argsort().argsort() + 1
for (g1, g2) in [('bbox', 'pixel'), ('bbox', 'glyph'), ('pixel', 'glyph')]:
    x = np.array([points[(g1, m)] for m in models9]); y = np.array([points[(g2, m)] for m in models9])
    tau_pt = kendalltau(x, y).statistic
    X = np.stack([boots[(g1, m)] for m in models9], 1); Y = np.stack([boots[(g2, m)] for m in models9], 1)
    taus = np.array([kendalltau(X[b], Y[b]).statistic for b in range(B)])
    r1, r2 = rank_desc(x), rank_desc(y)
    nchg = int((r1 != r2).sum())
    lo, hi = ci(taus)
    KROWS.append(dict(level='9 models (3 train GT × 3 seed)', gt_a=g1, gt_b=g2, kendall_tau=tau_pt, ci_lo=lo, ci_hi=hi,
                      p_tau_ge_0p9=float(max(1 / B, (taus >= 0.9).mean())), n_rank_changes=nchg,
                      ranking_a=' > '.join(np.array(models9)[np.argsort(-x)]), ranking_b=' > '.join(np.array(models9)[np.argsort(-y)])))
    add('H1', 'H1', f'Kendall τ(model sıralaması {g1}-GT vs {g2}-GT; 9 model)', tau_pt, taus, 487, GB_VAL.G, 'ap_img', 'val/clean',
        notes=f'{nchg} modelin sırası değişti')
# 3 model-type level (seed avg)
pts3 = {}; bts3 = {}
for g in MODELS:
    for m in MODELS:
        d = tab(valA, train_gt=m, eval_gt=g); pt, bt, _, _ = GB_VAL.mean(d.ap_img.values, d.group.values); pts3[(g, m)] = pt; bts3[(g, m)] = bt
for (g1, g2) in [('bbox', 'pixel'), ('bbox', 'glyph'), ('pixel', 'glyph')]:
    x = np.array([pts3[(g1, m)] for m in MODELS]); y = np.array([pts3[(g2, m)] for m in MODELS])
    X = np.stack([bts3[(g1, m)] for m in MODELS], 1); Y = np.stack([bts3[(g2, m)] for m in MODELS], 1)
    taus = np.array([kendalltau(X[b], Y[b]).statistic for b in range(B)])
    lo, hi = ci(taus)
    KROWS.append(dict(level='3 model types (seed-avg)', gt_a=g1, gt_b=g2, kendall_tau=kendalltau(x, y).statistic, ci_lo=lo, ci_hi=hi,
                      p_tau_ge_0p9=float(max(1 / B, (taus >= 0.9).mean())), n_rank_changes=int((rank_desc(x) != rank_desc(y)).sum()),
                      ranking_a=' > '.join(np.array(MODELS)[np.argsort(-x)]), ranking_b=' > '.join(np.array(MODELS)[np.argsort(-y)])))
rank_df = pd.DataFrame(KROWS)
# per-model point AP table for rank file
rank_pts = pd.DataFrame([dict(model=mod, eval_gt=g, mean_ap_img=points[(g, mod)]) for g in MODELS for mod in models9]).pivot(index='model', columns='eval_gt', values='mean_ap_img')
for g in MODELS:
    rank_pts[f'rank_{g}'] = rank_desc(rank_pts[g].values)
rank_df.to_csv(os.path.join(OUT, 'rank_kendall.csv'), index=False)
rank_pts.to_csv(os.path.join(OUT, 'rank_kendall_models.csv'))

# ============================================================================= H2
print('== H2 ablation')
REL = {}
for m in MODELS:
    for g in MODELS:
        dC = tab(valA, train_gt=m, eval_gt=g)
        for cond in ['boundary_blur', 'placebo_ring', 'inner_blur']:
            dX = tab(bndA, train_gt=m, eval_gt=g, condition=cond)
            a, b, grp = paired(dX, dC, 'ap_img')   # a=ablated, b=clean
            _, bA, n, ng = GB_VAL.mean(a, grp); _, bC, _, _ = GB_VAL.mean(b, grp)
            rel = (bA - bC) / bC; pt = float((a.mean() - b.mean()) / b.mean())
            own = (g == m)
            tag = 'own-GT' if own else f'{g}-GT'
            hf = 'H2' if (own and cond in ('boundary_blur', 'placebo_ring')) else ''
            add('H2', 'H2', f'relΔAP[{MODEL_LABEL[m]} | {GT_LABEL[g]}] {cond} vs clean', pt, rel, n, ng, 'ap_img', cond, null=0.0, holm_family=hf,
                notes='own GT' if own else '')
            REL[(m, g, cond)] = (pt, rel, n, ng)
            # F1tau variant
            a2, b2, grp2 = paired(dX, dC, 'f1_tau'); _, bA2, n2, ng2 = GB_VAL.mean(a2, grp2); _, bC2, _, _ = GB_VAL.mean(b2, grp2)
            add('H2', 'H2', f'relΔF1τ[{MODEL_LABEL[m]} | {GT_LABEL[g]}] {cond} vs clean', float((a2.mean() - b2.mean()) / b2.mean()), (bA2 - bC2) / bC2, n2, ng2, 'f1_tau', cond, null=0.0)
# between-model differences in boundary relΔAP (own GT and common bbox-GT): bbox-model − other (hipotez: bbox-model daha çok düşer → fark < 0)
for other in ['pixel', 'glyph']:
    for cond in ['boundary_blur', 'inner_blur']:
        # own GT
        p1, r1, n1, _ = REL[('bbox', 'bbox', cond)]; p2, r2, n2, _ = REL[(other, other, cond)]
        add('H2', 'H2', f'relΔAP[{cond}] bbox-model(own GT) − {other}-model(own GT)', p1 - p2, r1 - r2, min(n1, n2), GB_VAL.G, 'ap_img', cond, null=0.0,
            holm_family='H2' if cond == 'boundary_blur' else '')
        # common bbox GT
        p1, r1, n1, _ = REL[('bbox', 'bbox', cond)]; p2, r2, n2, _ = REL[(other, 'bbox', cond)]
        add('H2', 'H2', f'relΔAP[{cond}] bbox-model − {other}-model (both bbox-GT)', p1 - p2, r1 - r2, min(n1, n2), GB_VAL.G, 'ap_img', cond, null=0.0,
            holm_family='H2' if cond == 'boundary_blur' else '')
        # common pixel GT
        p1, r1, n1, _ = REL[('bbox', 'pixel', cond)]; p2, r2, n2, _ = REL[(other, 'pixel', cond)]
        add('H2', 'H2', f'relΔAP[{cond}] bbox-model − {other}-model (both pixel-GT)', p1 - p2, r1 - r2, min(n1, n2), GB_VAL.G, 'ap_img', cond, null=0.0)
# boundary − placebo within model (own GT)
for m in MODELS:
    p1, r1, n1, _ = REL[(m, m, 'boundary_blur')]; p2, r2, _, _ = REL[(m, m, 'placebo_ring')]
    add('H2', 'H2', f'relΔAP[{MODEL_LABEL[m]} | own GT] boundary_blur − placebo_ring', p1 - p2, r1 - r2, n1, GB_VAL.G, 'ap_img', 'boundary−placebo', null=0.0)

print('== H2 rectfit')
for m in MODELS:
    for g in MODELS:
        d = tab(rectA, train_gt=m, eval_gt=g).dropna(subset=['f1_raw', 'f1_rectfit'])
        _, bR, n, ng = GB_VAL.mean(d.f1_rectfit.values, d.group.values); _, bW, _, _ = GB_VAL.mean(d.f1_raw.values, d.group.values)
        add('H2', 'H2', f'ΔF1τ rect-fit − raw [{MODEL_LABEL[m]} | {GT_LABEL[g]}]', float(d.f1_rectfit.mean() - d.f1_raw.mean()), bR - bW, n, ng, 'f1_tau', 'rectfit',
            null=0.0, holm_family='H2' if (m == 'bbox' and g in ('bbox', 'pixel')) else '')
# prediction rectangularity (descriptive, bbox model) — median via bootstrap of median? report mean of rect_tau with CI
for m in MODELS:
    d = tab(valA, train_gt=m, eval_gt='bbox')
    r = val[(~val.gemonly) & (val.train_gt == m) & (val.eval_gt == 'bbox')].groupby('new_id')['rect_tau'].mean()
    rr = r.reindex(d.index)
    pt, bt, n, ng = GB_VAL.mean(rr.values, d.group.values)
    add('H2', 'H2', f'mean rectangularity@τ* [{MODEL_LABEL[m]}] (pred area / its bbox area)', pt, bt, n, ng, 'rect_tau', 'val/clean',
        notes=f'median={float(np.nanmedian(rr.values)):.3f}')

print('== H2 controls')
# FP pixel rate and share of images with >0.1% positive pixels, per model × control condition (τ*_bbox threshold and 0.5)
forged = {m: tab(valA, train_gt=m, eval_gt='bbox') for m in MODELS}
CTRL = {}
for m in MODELS:
    for cond in ['authentic', 'benign_pasteback', 'fullimage_resample']:
        d = tab(ctrlA, train_gt=m, condition=cond)
        for thr in ['tau', '0p5']:
            col = f'pos_frac_{thr}'
            pt, bt, n, ng = GB_VAL.mean(d[col].values, d.group.values)
            add('H2', 'H2', f'FP pixel rate [{MODEL_LABEL[m]} | {cond}] @{thr}', pt, bt, n, ng, col, cond)
            CTRL[(m, cond, thr, 'fp')] = (pt, bt, d)
            ind = (d[col].values > 0.001).astype(float)
            pt2, bt2, _, _ = GB_VAL.mean(ind, d.group.values)
            add('H2', 'H2', f'share images >0.1% positive [{MODEL_LABEL[m]} | {cond}] @{thr}', pt2, bt2, n, ng, f'img_gt0p1pct_{thr}', cond)
            CTRL[(m, cond, thr, 'share')] = (pt2, bt2, d)
    for cond in ['benign_pasteback', 'fullimage_resample']:
        for thr in ['tau', '0p5']:
            for kind, lab in [('fp', 'FP pixel rate'), ('share', 'share images >0.1% positive')]:
                p1, b1, _ = CTRL[(m, cond, thr, kind)]; p0, b0, _ = CTRL[(m, 'authentic', thr, kind)]
                add('H2', 'H2', f'Δ{lab} [{MODEL_LABEL[m]}] {cond} − authentic @{thr}', p1 - p0, b1 - b0, 487, GB_VAL.G, f'{kind}_{thr}', cond, null=0.0,
                    holm_family='H2' if (thr == 'tau' and kind == 'fp') else '')
# image-level AUROC forged(val) vs control, score_top1pct, per seed then seed-avg
AUC = {}
for m in MODELS:
    for cond in ['authentic', 'benign_pasteback', 'fullimage_resample']:
        pts = []; bts = []
        for s in sorted(val[(~val.gemonly)].seed.unique()):
            f = val[(val.train_gt == m) & (val.seed == s) & (val.eval_gt == 'bbox') & (~val.gemonly)].set_index('new_id').sort_index()
            c = ctrl[(ctrl.train_gt == m) & (ctrl.seed == s) & (ctrl.condition == cond)].set_index('new_id').sort_index()
            pt, bt, n, ng = GB_VAL.auroc(f.score_top1pct.values, f.group.values, c.score_top1pct.values, c.group.values)
            pts.append(pt); bts.append(bt)
        pt = float(np.mean(pts)); bt = np.mean(bts, 0)
        AUC[(m, cond)] = (pt, bt)
        add('H2', 'H2', f'image-level AUROC forged vs {cond} [{MODEL_LABEL[m]}] (top-1% score)', pt, bt, 487, GB_VAL.G, 'auroc_img_top1', cond, null=0.5,
            holm_family='H2' if cond == 'authentic' else '', notes='3 seed AUROC ortalaması; null=0.5')
        # by tool
        for tool in ['gemini', 'ideogram']:
            pts_t = []; bts_t = []
            for s in sorted(val[(~val.gemonly)].seed.unique()):
                f = val[(val.train_gt == m) & (val.seed == s) & (val.eval_gt == 'bbox') & (~val.gemonly) & (val.tool == tool)].set_index('new_id').sort_index()
                c = ctrl[(ctrl.train_gt == m) & (ctrl.seed == s) & (ctrl.condition == cond) & (ctrl.tool == tool)].set_index('new_id').sort_index()
                pt_t, bt_t, n_t, _ = GB_VAL.auroc(f.score_top1pct.values, f.group.values, c.score_top1pct.values, c.group.values)
                pts_t.append(pt_t); bts_t.append(bt_t)
            add('H2', 'H2', f'image-level AUROC forged vs {cond} [{MODEL_LABEL[m]} | tool={tool}]', float(np.mean(pts_t)), np.mean(bts_t, 0), n_t, GB_VAL.G, 'auroc_img_top1', cond, null=0.5)
    for cond in ['benign_pasteback', 'fullimage_resample']:
        p1, b1 = AUC[(m, 'authentic')]; p2, b2 = AUC[(m, cond)]
        add('H2', 'H2', f'ΔAUROC [{MODEL_LABEL[m]}] authentic − {cond}', p1 - p2, b1 - b2, 487, GB_VAL.G, 'auroc_img_top1', cond, null=0.0)

# ============================================================================= H3
print('== H3 robustness')
DROP = {}
for m in MODELS:
    for g in MODELS:
        dC = tab(valA, train_gt=m, eval_gt=g)
        for cond in ['jpeg75', 'rescale075', 'blur1']:
            dX = tab(robA, train_gt=m, eval_gt=g, condition=cond)
            for met in ['ap_img', 'f1_tau']:
                a, b, grp = paired(dX, dC, met)
                _, bA, n, ng = GB_VAL.mean(a, grp); _, bC, _, _ = GB_VAL.mean(b, grp)
                own = (g == m)
                add('H3', 'H3', f'Δ{met}[{MODEL_LABEL[m]} | {GT_LABEL[g]}] {cond} − clean', float(a.mean() - b.mean()), bA - bC, n, ng, met, cond, null=0.0,
                    notes='own GT' if own else '')
                if met == 'ap_img':
                    DROP[(m, g, cond)] = (float(a.mean() - b.mean()), bA - bC, n)
for other in ['pixel', 'glyph']:
    for cond in ['jpeg75', 'rescale075', 'blur1']:
        # own GTs
        p1, b1, n1 = DROP[(other, other, cond)]; p0, b0, n0 = DROP[('bbox', 'bbox', cond)]
        add('H3', 'H3', f'ΔAP drop difference [{cond}] {other}-model(own GT) − bbox-model(own GT)', p1 - p0, b1 - b0, min(n1, n0), GB_VAL.G, 'ap_img', cond, null=0.0, holm_family='H3',
            notes='hipotez: >0 (piksel/glif-modeli daha az düşer)')
        # common pixel GT
        p1, b1, n1 = DROP[(other, 'pixel', cond)]; p0, b0, n0 = DROP[('bbox', 'pixel', cond)]
        add('H3', 'H3', f'ΔAP drop difference [{cond}] {other}-model − bbox-model (both pixel-GT)', p1 - p0, b1 - b0, min(n1, n0), GB_VAL.G, 'ap_img', cond, null=0.0, holm_family='H3',
            notes='hipotez: >0')

print('== H3 held-out generator')
allimg = pd.concat([val.assign(split='val'), test.assign(split='test')], ignore_index=True)
def heldout_sets(df, eval_gt):
    ideo = df[(df.tool == 'ideogram') & (df.eval_gt == eval_gt) & (df.n_pos > 0)]
    gem = df[(df.tool == 'gemini') & (df.eval_gt == eval_gt) & (df.n_pos > 0) & (df.source.isin(['cord', 'sroie', 'xfund']))]
    return ideo, gem
HELD = {}
for arm, sel in [('C: gemini-only (1 seed)', allimg[allimg.gemonly]), ('B: all tools (3 seeds)', allimg[~allimg.gemonly])]:
    for m in MODELS:
        for g in MODELS:
            d = sel[sel.train_gt == m]
            ideo, gem = heldout_sets(d, g)
            # seed-avg per image (1 seed for C)
            ideoA = ideo.groupby('new_id').agg(ap_img=('ap_img', 'mean'), f1_0p5=('f1_0p5', 'mean'), group=('group', 'first'), split=('split', 'first'))
            gemA = gem.groupby('new_id').agg(ap_img=('ap_img', 'mean'), f1_0p5=('f1_0p5', 'mean'), group=('group', 'first'), split=('split', 'first'))
            # NOTE: new_id collides across val/test → group by (split,new_id)
            ideoA = ideo.groupby(['split', 'new_id']).agg(ap_img=('ap_img', 'mean'), f1_0p5=('f1_0p5', 'mean'), group=('group', 'first'))
            gemA = gem.groupby(['split', 'new_id']).agg(ap_img=('ap_img', 'mean'), f1_0p5=('f1_0p5', 'mean'), group=('group', 'first'))
            pI, bI, nI, ngI = GB_ALL.mean(ideoA.ap_img.values, ideoA.group.values)
            pG, bG, nG, ngG = GB_ALL.mean(gemA.ap_img.values, gemA.group.values)
            own = (g == m); tag = f'{arm} | {MODEL_LABEL[m]} | {GT_LABEL[g]}'
            add('H3', 'H3', f'held-out Ideogram AP [{tag}]', pI, bI, nI, ngI, 'ap_img', 'ideogram', notes=('own GT; ' if own else '') + f"val {int((ideoA.index.get_level_values(0)=='val').sum())} + test {int((ideoA.index.get_level_values(0)=='test').sum())}")
            add('H3', 'H3', f'source-matched Gemini AP [{tag}]', pG, bG, nG, ngG, 'ap_img', 'gemini_matched', notes='cord/sroie/xfund, val+test')
            add('H3', 'H3', f'gap AP Gemini − Ideogram [{tag}]', pG - pI, bG - bI, nI + nG, GB_ALL.G, 'ap_img', 'gap', null=0.0, notes='own GT' if own else '')
            HELD[(arm, m, g)] = dict(ideo=ideoA, gem=gemA, pI=pI, bI=bI, pG=pG, bG=bG, nI=nI, nG=nG)
# cross-model Ideogram AP differences (paired on same Ideogram images), gemini-only models, under common pixel-GT and bbox-GT
armC = 'C: gemini-only (1 seed)'
for g in ['pixel', 'bbox', 'glyph']:
    for other in ['pixel', 'glyph']:
        A = HELD[(armC, other, g)]['ideo']; Bm = HELD[(armC, 'bbox', g)]['ideo']
        j = A[['ap_img', 'group']].join(Bm[['ap_img']], lsuffix='_a', rsuffix='_b', how='inner').dropna()
        _, bA, n, ng = GB_ALL.mean(j.ap_img_a.values, j.group.values); _, bB, _, _ = GB_ALL.mean(j.ap_img_b.values, j.group.values)
        add('H3', 'H3', f'held-out Ideogram ΔAP {other}-model − bbox-model (gemini-only; both {g}-GT)', float(j.ap_img_a.mean() - j.ap_img_b.mean()), bA - bB, n, ng, 'ap_img', 'ideogram', null=0.0,
            holm_family='H3' if g == 'pixel' else '', notes='hipotez: >0')
    # gap difference bbox-model gap − other gap (own GTs)
for other in ['pixel', 'glyph']:
    Hb = HELD[(armC, 'bbox', 'bbox')]; Ho = HELD[(armC, other, other)]
    add('H3', 'H3', f'gap difference (Gemini−Ideogram) bbox-model(own GT) − {other}-model(own GT) (gemini-only)', (Hb['pG'] - Hb['pI']) - (Ho['pG'] - Ho['pI']), (Hb['bG'] - Hb['bI']) - (Ho['bG'] - Ho['bI']),
        Hb['nI'] + Ho['nI'], GB_ALL.G, 'ap_img', 'gap', null=0.0, notes='hipotez: >0 (bbox-modelinin açığı daha büyük)')

# ============================================================================= H4
print('== H4')
def h4_rows(df, label, fam_holm):
    pt, bt, n, ng = GB_VAL.auroc(df.p_forged.values, df.group.values, df.p_authentic.values, df.group.values)
    add('H4', 'H4', f'AUROC forged vs authentic [{label} | all]', pt, bt, n, ng, 'auroc_pair', 'forged_vs_authentic', null=0.5, holm_family=fam_holm)
    bA = bt
    pt2, bt2, n2, ng2 = GB_VAL.auroc(df.p_forged_rs.values, df.group.values, df.p_resample.values, df.group.values)
    add('H4', 'H4', f'AUROC forged vs fullimage_resample [{label} | all]', pt2, bt2, n2, ng2, 'auroc_pair', 'forged_vs_resample', null=0.5, holm_family=fam_holm)
    add('H4', 'H4', f'ΔAUROC [{label}] authentic − resample', pt - pt2, bA - bt2, n, ng, 'auroc_pair', 'diff', null=0.0, holm_family=fam_holm)
    for col, vals in [('tool', ['gemini', 'ideogram']), ('source', ['cord', 'sroie', 'wildreceipt', 'xfund'])]:
        for vv in vals:
            d = df[df[col] == vv]
            if len(d) < 5: continue
            p, b, n_, ng_ = GB_VAL.auroc(d.p_forged.values, d.group.values, d.p_authentic.values, d.group.values)
            add('H4', 'H4', f'AUROC forged vs authentic [{label} | {col}={vv}]', p, b, n_, ng_, 'auroc_pair', 'forged_vs_authentic', null=0.5)
    # pairwise accuracy (p_forged > p_authentic); ties (identical crops) count as failure, then tie-excluded (exploratory)
    for colf, coln, cname in [('p_forged', 'p_authentic', 'forged_vs_authentic'), ('p_forged_rs', 'p_resample', 'forged_vs_resample')]:
        acc = (df[colf].values > df[coln].values).astype(float)
        tie = (df[colf].values == df[coln].values)
        p, b, n_, ng_ = GB_VAL.mean(acc, df.group.values)
        add('H4', 'H4', f'pairwise accuracy p(forged)>p(neg), ties=fail [{label} | all | {cname}]', p, b, n_, ng_, 'acc_pair', cname, null=0.5,
            notes=f'{int(tie.sum())}/{len(df)} çiftte skorlar özdeş (crop piksel-özdeş)')
        p, b, n_, ng_ = GB_VAL.mean(tie.astype(float), df.group.values)
        add('H4', 'H4', f'share of pairs with identical scores [{label} | all | {cname}]', p, b, n_, ng_, 'tie_share', cname)
        acc_nt = np.where(tie, np.nan, acc)
        p, b, n_, ng_ = GB_VAL.mean(acc_nt, df.group.values)
        add('H4', 'H4', f'pairwise accuracy, ties excluded [{label} | all | {cname}] (exploratory)', p, b, n_, ng_, 'acc_pair_noties', cname, null=0.5,
            notes='post-hoc; yalnız skorları farklı çiftler')
        for vv in ['cord', 'sroie', 'wildreceipt', 'xfund']:
            d = df[df.source == vv]; acc_d = np.where(d[colf].values == d[coln].values, np.nan, (d[colf].values > d[coln].values).astype(float))
            if np.isfinite(acc_d).sum() < 5: continue
            p, b, n_, ng_ = GB_VAL.mean(acc_d, d.group.values)
            add('H4', 'H4', f'pairwise accuracy, ties excluded [{label} | source={vv} | {cname}] (exploratory)', p, b, n_, ng_, 'acc_pair_noties', cname, null=0.5)
h4_rows(h4, 'outside-bbox crops (H4)', 'H4')
h4_rows(h4pc, 'positive control (bbox-centred crops)', '')

# ============================================================================= EXTRA
print('== EXTRA test leakage & zero-shot')
for m in MODELS:
    d = tab(testA, train_gt=m, eval_gt='bbox')
    for met in ['ap_img', 'f1_tau']:
        pA, bA, nA, ngA = GB_TEST.mean(d[met].values, d.group.values)
        vc = np.where(d.clean_test.values.astype(bool), d[met].values, np.nan)
        vl = np.where(~d.clean_test.values.astype(bool), d[met].values, np.nan)
        pC, bC, nC, ngC = GB_TEST.mean(vc, d.group.values)
        pL, bL, nL, ngL = GB_TEST.mean(vl, d.group.values)
        add('EXTRA', 'leakage', f'test mean [{MODEL_LABEL[m]} | bbox GT] all 812', pA, bA, nA, ngA, met, 'test')
        add('EXTRA', 'leakage', f'test mean [{MODEL_LABEL[m]} | bbox GT] clean 740', pC, bC, nC, ngC, met, 'test')
        add('EXTRA', 'leakage', f'test mean [{MODEL_LABEL[m]} | bbox GT] non-clean 72', pL, bL, nL, ngL, met, 'test')
        add('EXTRA', 'leakage', f'test Δ[{MODEL_LABEL[m]} | bbox GT] all 812 − clean 740', pA - pC, bA - bC, nA, ngA, met, 'test', null=0.0,
            holm_family='EXTRA' if (m == 'bbox') else '')
        add('EXTRA', 'leakage', f'test Δ[{MODEL_LABEL[m]} | bbox GT] non-clean 72 − clean 740', pL - pC, bL - bC, nA, ngA, met, 'test', null=0.0,
            holm_family='EXTRA' if (m == 'bbox') else '')
# zero-shot comparison (test 812, bbox GT): bbox-model (seed-avg) − zero-shot, paired by id
zt = zs[zs.split == 'test'].copy()
zt['new_id'] = zt['id'].astype(int)
d_b = tab(testA, train_gt='bbox', eval_gt='bbox')
for cond in ['trufor', 'adcd_q100_ocrproxy', 'trufor_1024pad', 'adcd_q100_emptyocr', 'adcd_q95_ocrproxy', 'adcd_q100_noaln']:
    z = zt[zt.cond == cond].set_index('new_id').sort_index()
    primary = cond in ('trufor', 'adcd_q100_ocrproxy')
    for met_m, met_z in [('ap_img', 'ap_img'), ('f1_0p5', 'f1_05')]:
        j = d_b[[met_m, 'group']].join(z[[met_z]], how='inner', rsuffix='_zs').dropna()
        zcol = met_z if met_z != met_m else met_z + '_zs'
        _, bM, n, ng = GB_TEST.mean(j[met_m].values, j.group.values); _, bZ, _, _ = GB_TEST.mean(j[zcol].values, j.group.values)
        pZ = float(j[zcol].mean())
        add('EXTRA', 'zero-shot', f'zero-shot {cond} test mean (bbox GT)', pZ, bZ, n, ng, met_z, 'test', notes='zero-shot; eşit koşul değil (eğitim yok, farklı çözünürlük protokolü)')
        add('EXTRA', 'zero-shot', f'Δ bbox-model − zero-shot {cond} (test, bbox GT)', float(j[met_m].mean() - pZ), bM - bZ, n, ng, met_m, 'test', null=0.0,
            holm_family='EXTRA' if primary else '', notes='eşit koşul değil: U-Net AIForge-Doc train ile eğitildi, zero-shot modeller görmedi')

# ============================================================================= tables
summary = holm(pd.DataFrame(ROWS))
summary = summary[['family', 'hypothesis', 'contrast', 'metric', 'condition', 'n_img', 'n_group', 'mean', 'ci_lo', 'ci_hi', 'p_boot', 'p_holm', 'holm_family', 'null', 'notes']]
summary.to_csv(os.path.join(OUT, 'stats_summary.csv'), index=False, float_format='%.6g')
print('rows:', len(summary), ' holm tests:', (summary.holm_family != '').sum())
with open(os.path.join(OUT, 'exclusions_notes.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(B=B, seed=SEED, notes=NOTES), f, ensure_ascii=False, indent=1)

# ============================================================================= figures
try:
    apply_figure_style  # figure-style skill (kernel) if present
except NameError:
    def apply_figure_style(sizes=(8, 7, 6)):
        plt.rcParams.update({'font.size': sizes[0], 'axes.titlesize': sizes[0], 'axes.labelsize': sizes[0], 'legend.fontsize': sizes[1],
                             'xtick.labelsize': sizes[2], 'ytick.labelsize': sizes[2], 'xtick.direction': 'out', 'ytick.direction': 'out',
                             'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'savefig.dpi': 300,
                             'axes.titlelocation': 'left', 'axes.titleweight': 'normal'})
apply_figure_style()
SM = summary.set_index('contrast')
def G(contrast, metric=None):
    r = SM.loc[contrast]
    if isinstance(r, pd.DataFrame):
        r = r[r.metric == metric].iloc[0] if metric else r.iloc[0]
    return float(r['mean']), float(r['ci_lo']), float(r['ci_hi'])
def errs(vals):
    m = np.array([v[0] for v in vals]); lo = np.array([v[1] for v in vals]); hi = np.array([v[2] for v in vals])
    return m, np.vstack([m - lo, hi - m])
def qa(fig, name):
    """§9.1 geometric check: text–text and text–spine overlaps (tick labels on their own spine ignored)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
    ticklabels = {ax: set(ax.get_xticklabels(which='both') + ax.get_yticklabels(which='both')) for ax in fig.axes}
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes for s in ax.spines.values() if s.get_visible()]
    ov = [(a.get_text()[:30], b.get_text()[:30]) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    ov += [(t.get_text()[:30], 'spine') for t, bt in texts for s, bs in spines if bt.overlaps(bs) and t not in ticklabels[s.axes]]
    out = [(t.get_text()[:30]) for t, bt in texts if not fig.bbox.contains(bt.x0, bt.y0) or not fig.bbox.contains(bt.x1, bt.y1)]
    print(f'[QA {name}] text overlaps: {len(ov)} {ov[:4]} | outside figure: {len(out)} {out[:3]}')

def finish(fig, name, w_pad=1.5):
    """tight_layout → panel letters above the titles (figure coords) → §9.1 QA → save (bbox tight so nothing is clipped)."""
    fig.tight_layout(w_pad=w_pad, rect=(0, 0, 1, 0.97))
    for ax, s in zip(fig.axes, 'abcdefgh'):
        pos = ax.get_position()
        fig.text(max(0.005, pos.x0 - 0.085), pos.y1 + 0.125, s, fontweight='bold', fontsize=10, va='bottom', ha='left')
    qa(fig, name)
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=300, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)

MLAB = {m: MODEL_LABEL[m].replace('-GT-trained', '-GT–trained model') for m in MODELS}
GTS = ['bbox', 'pixel', 'glyph']; GT_TICK = ['bbox GT', 'pixel-change GT', 'glyph GT']
OFF = {'bbox': -0.22, 'pixel': 0.0, 'glyph': 0.22}
EB = dict(fmt='o', ms=5, capsize=2, lw=1)

# --- Fig H1 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
for ax, met, ylab in [(axes[0], 'ap_img', 'per-image AP'), (axes[1], 'f1_tau', 'F1 at calibrated threshold τ*')]:
    for m in MODELS:
        mu, e = errs([G(f'mean[{MODEL_LABEL[m]} | {GT_LABEL[g]}]', met) for g in GTS])
        x = np.arange(3) + OFF[m]
        ax.errorbar(x, mu, yerr=e, color=MODEL_COLOR[m], label=MLAB[m], **EB)
        ax.plot(x, mu, color=MODEL_COLOR[m], lw=0.8, alpha=0.5)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(GT_TICK); ax.set_xlabel('ground truth used for scoring'); ax.set_ylabel(ylab)
    ax.set_ylim(0.15, 1.0); ax.margins(x=0.08)
axes[0].set_title('Same predictions, different GT: the bbox-GT–trained model\nfalls from AP 0.90 (bbox GT) to 0.38 (pixel-change GT)', loc='left')
axes[1].set_title('F1@τ* falls 0.84 → 0.45; the 9-model ranking reverses\n(Kendall τ bbox GT vs pixel-change GT = %.2f [%.2f, %.2f])' % tuple(
    rank_df[(rank_df.level.str.startswith('9')) & (rank_df.gt_a == 'bbox') & (rank_df.gt_b == 'pixel')].iloc[0][['kendall_tau', 'ci_lo', 'ci_hi']].astype(float)), loc='left')
axes[0].legend(loc='upper right', handletextpad=0.3)
axes[1].text(1.02, 0.5, 'higher = better', rotation=90, va='center', ha='left', fontsize=7, color='0.4', transform=axes[1].transAxes)
finish(fig, 'fig_H1_dualgt_ci')

# --- Fig H2 ablation -------------------------------------------------------
CONDS = ['boundary_blur', 'placebo_ring', 'inner_blur']; COND_TICK = ['boundary ring blur\n(5 px at bbox edge)', 'placebo ring blur\n(same ring, outside bbox)', 'inner blur\n(bbox interior)']
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
for ax, gtmode, ttl in [(axes[0], 'own', 'Own training GT: boundary blur costs the bbox-GT–trained\nmodel 1.5 % AP; pixel-trained 3.0 %, glyph-trained 3.4 %'),
                        (axes[1], 'bbox', 'All scored on bbox GT: blurring the bbox interior raises\nthe pixel-GT–trained model\'s score (+4.5 %)')]:
    for m in MODELS:
        g = m if gtmode == 'own' else 'bbox'
        mu, e = errs([G(f'relΔAP[{MODEL_LABEL[m]} | {GT_LABEL[g]}] {c} vs clean') for c in CONDS])
        ax.errorbar(np.arange(3) + OFF[m], 100 * mu, yerr=100 * e, color=MODEL_COLOR[m], label=MLAB[m], **EB)
    ax.axhline(0, color='0.5', lw=0.8, ls='--')
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(COND_TICK); ax.set_title(ttl, loc='left'); ax.margins(x=0.08)
axes[0].set_ylabel('relative change in per-image AP vs clean (%)')
axes[0].legend(loc='lower left', handletextpad=0.3)
finish(fig, 'fig_H2_ablation_ci')

# --- Fig H2 controls -------------------------------------------------------
CTRLS = ['authentic', 'benign_pasteback', 'fullimage_resample']; CTRL_TICK = ['authentic\n(untouched)', 'benign paste-back\n(no content change)', 'full-image\nresample']
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
ax = axes[0]
for m in MODELS:
    mu, e = errs([G(f'FP pixel rate [{MODEL_LABEL[m]} | {c}] @tau') for c in CTRLS])
    ax.errorbar(np.arange(3) + OFF[m], 1e3 * mu, yerr=1e3 * e, color=MODEL_COLOR[m], label=MLAB[m], **EB)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(CTRL_TICK); ax.set_ylabel('false-positive pixel rate at τ* (‰)'); ax.set_ylim(0, 7.2); ax.margins(x=0.08)
ax.set_title('Benign edits raise false positives for every model\n(paste-back +0.2–0.5 ‰; lower = better)', loc='left')
ax.legend(loc='upper left', handletextpad=0.3)
ax = axes[1]
for m in MODELS:
    mu, e = errs([G(f'image-level AUROC forged vs {c} [{MODEL_LABEL[m]}] (top-1% score)') for c in CTRLS])
    ax.errorbar(np.arange(3) + OFF[m], mu, yerr=e, color=MODEL_COLOR[m], **EB)
ax.axhline(0.5, color='0.5', lw=0.8, ls='--'); ax.text(2.42, 0.505, 'chance', fontsize=6, color='0.4', ha='right', va='bottom')
ax.set_xticks(np.arange(3)); ax.set_xticklabels(CTRL_TICK); ax.set_ylabel('image-level AUROC, forged vs control\n(top-1 % pixel score)'); ax.set_ylim(0.45, 1.0); ax.margins(x=0.08)
ax.set_title('Image-level separation drops 2–3 points against benign\ncontrols; glyph-GT–trained model is weakest (higher = better)', loc='left')
finish(fig, 'fig_H2_controls_ci')

# --- Fig H3 transfer -------------------------------------------------------
DEGS = ['jpeg75', 'rescale075', 'blur1']; DEG_TICK = ['JPEG Q75', '0.75× rescale', 'Gaussian blur σ=1']
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axes[0]
for m in MODELS:
    mu, e = errs([G(f'Δap_img[{MODEL_LABEL[m]} | {GT_LABEL[m]}] {c} − clean') for c in DEGS])
    ax.errorbar(np.arange(3) + OFF[m] - 0.05, mu, yerr=e, color=MODEL_COLOR[m], label=f'{MLAB[m]}, own GT', **EB)
    mu2, e2 = errs([G(f'Δap_img[{MODEL_LABEL[m]} | {GT_LABEL["pixel"]}] {c} − clean') for c in DEGS])
    ax.errorbar(np.arange(3) + OFF[m] + 0.05, mu2, yerr=e2, mfc='white', color=MODEL_COLOR[m], label='same, scored on pixel-change GT (open)' if m == 'bbox' else None, **EB)
ax.axhline(0, color='0.5', lw=0.8, ls='--')
ax.set_xticks(np.arange(3)); ax.set_xticklabels(DEG_TICK); ax.set_ylabel('ΔAP (degraded − clean), per-image'); ax.set_ylim(-0.27, 0.012); ax.margins(x=0.08)
ax.set_title('No GT is consistently more robust: the pixel-GT–trained model\nloses less under JPEG but more under blur (0 = no loss)', loc='left')
ax.legend(loc='lower left', handletextpad=0.3, ncol=2, columnspacing=0.8)
ax = axes[1]
for i, m in enumerate(MODELS):
    tag = f'C: gemini-only (1 seed) | {MODEL_LABEL[m]} | {GT_LABEL[m]}'
    gI = G(f'held-out Ideogram AP [{tag}]'); gG = G(f'source-matched Gemini AP [{tag}]')
    ax.errorbar(i - 0.15, gG[0], yerr=[[gG[0] - gG[1]], [gG[2] - gG[0]]], fmt='s', color=MODEL_COLOR[m], ms=5, capsize=2, lw=1)
    ax.errorbar(i + 0.15, gI[0], yerr=[[gI[0] - gI[1]], [gI[2] - gI[0]]], fmt='o', mfc='white', color=MODEL_COLOR[m], ms=5, capsize=2, lw=1)
    ax.annotate('', xy=(i + 0.15, gI[0]), xytext=(i - 0.15, gG[0]), arrowprops=dict(arrowstyle='->', color=MODEL_COLOR[m], lw=0.8, alpha=0.7))
    gap = G(f'gap AP Gemini − Ideogram [{tag}]')
    ax.text(i, min(gI[1], gG[1]) - 0.04, f'gap {gap[0]:.2f}\n[{gap[1]:.2f}, {gap[2]:.2f}]', ha='center', va='top', fontsize=6)
ax.set_xticks(np.arange(3)); ax.set_xticklabels([MODEL_LABEL[m].replace('-GT-trained', '-GT–trained\n(own GT)') for m in MODELS])
ax.set_ylabel('per-image AP'); ax.set_ylim(0.0, 1.1); ax.margins(x=0.1)
ax.errorbar([], [], fmt='s', color='0.3', label='Gemini, source-matched (seen)'); ax.errorbar([], [], fmt='o', mfc='white', color='0.3', label='Ideogram (held-out)')
ax.legend(loc='upper right', handletextpad=0.3)
ax.set_title('Gemini-only models (1 seed): every GT loses on the unseen\ngenerator; pixel-GT training does not help on Ideogram', loc='left')
finish(fig, 'fig_H3_transfer_ci')

# --- Fig H4 ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axes[0]
items = [('outside-bbox crops (H4) | all', 'forged vs authentic', 'all pairs'), ('outside-bbox crops (H4) | tool=gemini', 'forged vs authentic', 'Gemini pairs'),
         ('outside-bbox crops (H4) | tool=ideogram', 'forged vs authentic', 'Ideogram pairs'), ('outside-bbox crops (H4) | all', 'forged vs fullimage_resample', 'all pairs, symmetric\nresampled negative'),
         ('positive control (bbox-centred crops) | all', 'forged vs authentic', 'positive control:\nbbox-centred crops')]
ys = np.arange(len(items))[::-1]
for y, (lab, cond, tick) in zip(ys, items):
    v = G(f'AUROC {cond} [{lab}]')
    col = '0.25' if 'positive' in lab else ('#1b9e77' if 'resample' in cond else '#d95f02')
    ax.errorbar(v[0], y, xerr=[[v[0] - v[1]], [v[2] - v[0]]], color=col, **EB)
    ax.text(v[2] + 0.015, y, f'{v[0]:.3f}', ha='left', va='center', fontsize=6, color=col)
ax.axvline(0.5, color='0.5', lw=0.8, ls='--'); ax.set_yticks(ys); ax.set_yticklabels([t for _, _, t in items]); ax.set_xlim(0.4, 1.02); ax.set_ylim(-0.5, len(items) - 0.3)
ax.set_xlabel('AUROC, forged vs negative (ResNet-18 probe, 224 px crops)')
ax.set_title('Outside the bbox, forged vs authentic crops: AUROC 0.516\n(the same probe reaches 0.98 on bbox-centred crops)', loc='left')
ax = axes[1]
labs = ['forged vs authentic', 'forged vs resampled\n(symmetric)']
for i, (cname, col) in enumerate([('forged_vs_authentic', '#d95f02'), ('forged_vs_resample', '#1b9e77')]):
    key = f'pairwise accuracy, ties excluded [outside-bbox crops (H4) | all | {cname}] (exploratory)'
    v = G(key); n_nt = int(SM.loc[key, 'n_img'])
    ax.errorbar(i, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], color=col, **EB)
    ax.text(i, v[2] + 0.03, f'{v[0]:.2f}\n(n = {n_nt} pairs)', ha='center', va='bottom', fontsize=6)
ax.axhline(0.5, color='0.5', lw=0.8, ls='--'); ax.set_xticks([0, 1]); ax.set_xticklabels(labs); ax.set_ylim(0.3, 1.12); ax.margins(x=0.3)
ax.set_ylabel('within-pair accuracy, p(forged) > p(negative)\n(identical-crop pairs excluded)', labelpad=2)
ax.set_title('Post hoc: where outside-bbox pixels differ, forged ranks\nhigher in 92 % of pairs, but not vs a resampled negative', loc='left')
finish(fig, 'fig_H4_auroc', w_pad=3.0)
print('figures written')

# ============================================================================= decisions (markdown)
def F(contrast, metric=None, scale=1.0, nd=3, pct=False):
    m, lo, hi = G(contrast, metric)
    if pct:
        return f"{100*m:+.1f} % [{100*lo:+.1f}, {100*hi:+.1f}]"
    return f"{scale*m:.{nd}f} [{scale*lo:.{nd}f}, {scale*hi:.{nd}f}]"
def P(contrast, metric=None):
    r = SM.loc[contrast]
    if isinstance(r, pd.DataFrame): r = r[r.metric == metric].iloc[0] if metric else r.iloc[0]
    ph = r['p_holm']; pb = r['p_boot']
    s = f"p_boot={pb:.3f}" if pd.notna(pb) else ''
    if pd.notna(ph): s += f", p_Holm={ph:.3f}"
    return s
def N(contrast, metric=None):
    r = SM.loc[contrast]
    if isinstance(r, pd.DataFrame): r = r[r.metric == metric].iloc[0] if metric else r.iloc[0]
    return int(r['n_img']), int(r['n_group'])
rk9 = {(r.gt_a, r.gt_b): r for r in rank_df[rank_df.level.str.startswith('9')].itertuples()}
rk3 = {(r.gt_a, r.gt_b): r for r in rank_df[rank_df.level.str.startswith('3')].itertuples()}
def tau_s(r): return f"τ = {r.kendall_tau:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}], P(τ ≥ 0.9) = {r.p_tau_ge_0p9:.3f}, sırası değişen model: {r.n_rank_changes}"

# pooled rectangularity medians (README T6 ile karşılaştırma için; 3 seed havuzlanmış)
rect_pooled = {m: float(np.nanmedian(val[(~val.gemonly) & (val.train_gt == m) & (val.eval_gt == 'bbox')].rect_tau.values)) for m in MODELS}

BM, PM, GM = MODEL_LABEL['bbox'], MODEL_LABEL['pixel'], MODEL_LABEL['glyph']
md = []
md.append(f"""# Faz 3 — Hipotez karar tablosu (H1–H4), kaynak-belge-grubu bootstrap CI + Holm

Üretim: `stats_analysis.py` (B = {B}, seed = {SEED}); tüm sayılar `stats_summary.csv` satırlarından alınmıştır.
Veri: Faz 2 görüntü-düzeyi CSV'leri (val 487 tam görüntü, test 812, 3 GT × 9 B-modeli + 3 gemini-only C-modeli). Seed'ler sabit faktör:
önce görüntü başına 3 seed ortalaması, sonra kaynak-belge-grubu (cluster) bootstrap; farklar EŞLEŞTİRİLMİŞ bootstrap (aynı grup örneklemi iki koşula).
Grup yapısı: val 487 görüntü / {GB_VAL.G} grup (451 tekil, 18 ikili), test 812 / {GB_TEST.G} grup. CI = %95 yüzdelik. p_boot = 2·min(P(Δ≤0), P(Δ≥0)), alt sınır 1/B;
Holm düzeltmesi her hipotez ailesinin önceden belirlenmiş karar kontrastları üzerinde (`holm_family` sütunu; H1: {int((summary.holm_family=='H1').sum())}, H2: {int((summary.holm_family=='H2').sum())}, H3: {int((summary.holm_family=='H3').sum())}, H4: {int((summary.holm_family=='H4').sum())}, EXTRA: {int((summary.holm_family=='EXTRA').sum())} test).

**Dışlamalar (raporlanan):** {' '.join(NOTES)}

| Hipotez | Önceden sabitlenmiş yanlışlama ölçütü | Gözlenen (CI) | Karar |
|---|---|---|---|
| H1 metrik/sıralama | ΔF1 < 0.05 VE Kendall τ ≥ 0.9 | ΔF1@τ* (bbox-GT − piksel-GT, bbox-modeli) = {F('Δ[bbox-GT-trained] bbox-GT − pixel-GT','f1_tau')}; τ(bbox vs piksel, 9 model) = {rk9[('bbox','pixel')].kendall_tau:.2f} [{rk9[('bbox','pixel')].ci_lo:.2f}, {rk9[('bbox','pixel')].ci_hi:.2f}] | **DOĞRULANDI** |
| H2 sınır shortcut'ı | İki model benzer düşer (fark CI 0'ı kapsar) VE benign FP ≈ authentic | boundary relΔAP bbox-modeli {F('relΔAP[bbox-GT-trained | bbox GT] boundary_blur vs clean', pct=True)} vs piksel-modeli {F('relΔAP[pixel-GT-trained | pixel-change GT] boundary_blur vs clean', pct=True)}; fark (bbox − piksel, kendi GT) = {F('relΔAP[boundary_blur] bbox-model(own GT) − pixel-model(own GT)', pct=True)} (yön TERS); benign paste-back FP − authentic (bbox-modeli) = {F('ΔFP pixel rate [bbox-GT-trained] benign_pasteback − authentic @tau', scale=1e3, nd=2)} ‰ | **KISMEN** |
| H3 transfer/dayanıklılık | Fark yok / ters (CI örtüşür) | Ideogram ΔAP piksel-modeli − bbox-modeli (piksel-GT, gemini-only) = {F('held-out Ideogram ΔAP pixel-model − bbox-model (gemini-only; both pixel-GT)')}; dayanıklılık: yön koşula göre değişiyor (JPEG +, blur −) | **YANLIŞLANDI** |
| H4 global iz | AUROC ≈ 0.5 | AUROC forged-vs-authentic (bbox-dışı crop) = {F('AUROC forged vs authentic [outside-bbox crops (H4) | all]')}; simetrik resample = {F('AUROC forged vs fullimage_resample [outside-bbox crops (H4) | all]')}; pozitif kontrol = {F('AUROC forged vs authentic [positive control (bbox-centred crops) | all]')} | **YANLIŞLANDI** (pratik olarak; bkz. not) |

---

## H1 — GT türü metrikleri ve model sıralamasını değiştirir → DOĞRULANDI

**Ölçüt.** Yanlışlama: aynı tahminler için ΔF1 (bbox-GT − piksel/glif-GT) < 0.05 VE Kendall τ ≥ 0.9.

**Gözlenen (val 487; piksel/glif-GT'de 485, 2 boş-GT görüntü dışlandı):**

| Model | Metrik | bbox-GT | piksel-GT | glif-GT | Δ bbox−piksel | Δ bbox−glif |
|---|---|---|---|---|---|---|
""")
for m in MODELS:
    for met, lab in [('ap_img', 'AP'), ('f1_tau', 'F1@τ*')]:
        md.append(f"| {MODEL_LABEL[m]} | {lab} | {F(f'mean[{MODEL_LABEL[m]} | bbox GT]', met)} | {F(f'mean[{MODEL_LABEL[m]} | pixel-change GT]', met)} | {F(f'mean[{MODEL_LABEL[m]} | glyph GT]', met)} | {F(f'Δ[{MODEL_LABEL[m]}] bbox-GT − pixel-GT', met)} ({P(f'Δ[{MODEL_LABEL[m]}] bbox-GT − pixel-GT', met)}) | {F(f'Δ[{MODEL_LABEL[m]}] bbox-GT − glyph-GT', met)} ({P(f'Δ[{MODEL_LABEL[m]}] bbox-GT − glyph-GT', met)}) |")
md.append(f"""
**Sıralama kararlılığı (Kendall τ, per-image AP ortalamasına göre; `rank_kendall.csv`):**
* 9 model (3 eğitim GT × 3 seed): bbox-GT vs piksel-GT {tau_s(rk9[('bbox','pixel')])}; bbox-GT vs glif-GT {tau_s(rk9[('bbox','glyph')])}; piksel-GT vs glif-GT {tau_s(rk9[('pixel','glyph')])}.
* 3 model tipi (seed ort.): bbox vs piksel τ = {rk3[('bbox','pixel')].kendall_tau:.2f}, bbox vs glif τ = {rk3[('bbox','glyph')].kendall_tau:.2f} (sıralama tamamen tersine döner: bbox-GT altında bbox > piksel > glif; glif-GT altında glif > piksel > bbox).
* İkincil metrikler (F1@0.5, IoU@τ*) aynı yönde (stats_summary.csv, aile H1, holm_family boş).

**Karar: DOĞRULANDI.** bbox-modeli için ΔF1@τ* = {F('Δ[bbox-GT-trained] bbox-GT − pixel-GT','f1_tau')} ≫ 0.05 (CI alt sınırı 0.05'in çok üzerinde) ve τ < 0.9'un çok altında (CI üst sınırı {rk9[('bbox','pixel')].ci_hi:.2f}); 9 modelin 9'unun sırası değişir. Yanlışlama ölçütünün iki koşulu da reddedilir. Not: piksel-modelinin AP'si bbox-GT ve piksel-GT altında neredeyse aynıdır ({F('Δ[pixel-GT-trained] bbox-GT − pixel-GT','ap_img')}, {P('Δ[pixel-GT-trained] bbox-GT − pixel-GT','ap_img')}): metrik şişmesi bbox-modeline özgüdür — bbox-GT, dikdörtgenin değişmeyen pikselini "doğru" sayar.

## H2 — bbox-modeli sınır/dikdörtgen kısayolu kullanır → KISMEN

**Ölçüt.** Yanlışlama: iki model de benzer düşer (modeller arası fark CI 0'ı kapsar) VE benign kontrollerde FP ≈ authentic.

**Sınır ablasyonu, relΔAP = (ablasyon − temiz)/temiz (val 487; eşleştirilmiş bootstrap):**

| Model (değerlendirme GT) | boundary_blur | placebo_ring | inner_blur |
|---|---|---|---|
""")
for m in MODELS:
    for g in ([m] if m == 'bbox' else [m, 'bbox']):
        tag = f'{MODEL_LABEL[m]} | {GT_LABEL[g]}'
        md.append(f"| {MODEL_LABEL[m]} ({GT_LABEL[g]}{', kendi GT' if g == m else ''}) | {F(f'relΔAP[{tag}] boundary_blur vs clean', pct=True)} ({P(f'relΔAP[{tag}] boundary_blur vs clean')}) | {F(f'relΔAP[{tag}] placebo_ring vs clean', pct=True)} ({P(f'relΔAP[{tag}] placebo_ring vs clean')}) | {F(f'relΔAP[{tag}] inner_blur vs clean', pct=True)} ({P(f'relΔAP[{tag}] inner_blur vs clean')}) |")
md.append(f"""
**Modeller arası fark (bbox-modeli − diğer; hipotez: < 0, yani bbox-modeli daha çok düşer):**
* boundary_blur, kendi GT'leri: bbox − piksel = {F('relΔAP[boundary_blur] bbox-model(own GT) − pixel-model(own GT)', pct=True)} ({P('relΔAP[boundary_blur] bbox-model(own GT) − pixel-model(own GT)')}); bbox − glif = {F('relΔAP[boundary_blur] bbox-model(own GT) − glyph-model(own GT)', pct=True)} ({P('relΔAP[boundary_blur] bbox-model(own GT) − glyph-model(own GT)')}).
* boundary_blur, ortak bbox-GT: bbox − piksel = {F('relΔAP[boundary_blur] bbox-model − pixel-model (both bbox-GT)', pct=True)} ({P('relΔAP[boundary_blur] bbox-model − pixel-model (both bbox-GT)')}); bbox − glif = {F('relΔAP[boundary_blur] bbox-model − glyph-model (both bbox-GT)', pct=True)} ({P('relΔAP[boundary_blur] bbox-model − glyph-model (both bbox-GT)')}).
* boundary − placebo (bbox-modeli, kendi GT): {F('relΔAP[bbox-GT-trained | own GT] boundary_blur − placebo_ring', pct=True)} ({P('relΔAP[bbox-GT-trained | own GT] boundary_blur − placebo_ring')}).

**Rect-fit son-işleme ΔF1@τ* (rectfit − raw):** bbox-modeli bbox-GT {F('ΔF1τ rect-fit − raw [bbox-GT-trained | bbox GT]')} ({P('ΔF1τ rect-fit − raw [bbox-GT-trained | bbox GT]')}); bbox-modeli piksel-GT {F('ΔF1τ rect-fit − raw [bbox-GT-trained | pixel-change GT]')} ({P('ΔF1τ rect-fit − raw [bbox-GT-trained | pixel-change GT]')}); piksel-modeli piksel-GT {F('ΔF1τ rect-fit − raw [pixel-GT-trained | pixel-change GT]')}; glif-modeli glif-GT {F('ΔF1τ rect-fit − raw [glyph-GT-trained | glyph GT]')}.
**Tahmin dikdörtgenselliği @τ* (alan / kendi bbox alanı; seed-ortalaması, ortalama [CI]):** bbox-modeli {F('mean rectangularity@τ* [bbox-GT-trained] (pred area / its bbox area)')}, piksel-modeli {F('mean rectangularity@τ* [pixel-GT-trained] (pred area / its bbox area)')}, glif-modeli {F('mean rectangularity@τ* [glyph-GT-trained] (pred area / its bbox area)')}; 3-seed havuzlanmış medyan: bbox {rect_pooled['bbox']:.3f}, piksel {rect_pooled['pixel']:.3f}, glif {rect_pooled['glyph']:.3f} (README T6: 0.877 / 0.038 / 0.013 ile yaklaşık uyumlu).

**Benign kontroller (val 487 kaynaklı; FP = pozitif piksel oranı @τ*(bbox-GT); 3 seed ort.):**

| Model | authentic FP ‰ | benign paste-back FP ‰ | Δ paste-back − authentic ‰ | full-image resample FP ‰ | Δ resample − authentic ‰ | görüntü payı >%0.1 poz. (authentic → paste-back) |
|---|---|---|---|---|---|---|
""")
for m in MODELS:
    L = MODEL_LABEL[m]
    md.append(f"| {L} | {F(f'FP pixel rate [{L} | authentic] @tau', scale=1e3, nd=2)} | {F(f'FP pixel rate [{L} | benign_pasteback] @tau', scale=1e3, nd=2)} | {F(f'ΔFP pixel rate [{L}] benign_pasteback − authentic @tau', scale=1e3, nd=2)} ({P(f'ΔFP pixel rate [{L}] benign_pasteback − authentic @tau')}) | {F(f'FP pixel rate [{L} | fullimage_resample] @tau', scale=1e3, nd=2)} | {F(f'ΔFP pixel rate [{L}] fullimage_resample − authentic @tau', scale=1e3, nd=2)} ({P(f'ΔFP pixel rate [{L}] fullimage_resample − authentic @tau')}) | {F(f'share images >0.1% positive [{L} | authentic] @tau', nd=3)} → {F(f'share images >0.1% positive [{L} | benign_pasteback] @tau', nd=3)} (Δ {F(f'Δshare images >0.1% positive [{L}] benign_pasteback − authentic @tau', nd=3)}) |")
md.append(f"""
**Görüntü-düzeyi AUROC (top-1 % skor; forged val vs kontrol; seed başına AUROC → 3 seed ort.):**

| Model | vs authentic | vs benign paste-back | vs full-image resample | Δ(authentic − paste-back) | Δ(authentic − resample) | vs authentic, Gemini / Ideogram |
|---|---|---|---|---|---|---|
""")
for m in MODELS:
    L = MODEL_LABEL[m]
    md.append(f"| {L} | {F(f'image-level AUROC forged vs authentic [{L}] (top-1% score)')} ({P(f'image-level AUROC forged vs authentic [{L}] (top-1% score)')}) | {F(f'image-level AUROC forged vs benign_pasteback [{L}] (top-1% score)')} | {F(f'image-level AUROC forged vs fullimage_resample [{L}] (top-1% score)')} | {F(f'ΔAUROC [{L}] authentic − benign_pasteback')} | {F(f'ΔAUROC [{L}] authentic − fullimage_resample')} | {F(f'image-level AUROC forged vs authentic [{L} | tool=gemini]')} / {F(f'image-level AUROC forged vs authentic [{L} | tool=ideogram]')} |")
md.append(f"""
Not: README T7'deki AUROC_top1 değerleri (bbox 0.838 / piksel 0.896 / glif 0.814) yalnız s44 ile uyuşur; burada 3 seed ortalaması raporlanır (s42/s43/s44 bbox-modeli: 0.921 / 0.916 / 0.838).

**Karar: KISMEN.**
* Önceden sabitlenmiş *karşılaştırmalı* ölçüt başarısız: bbox-modelinin boundary_blur düşüşü piksel-/glif-modelininkinden **küçüktür** ve fark CI'ı 0'ı kapsamaz (yön ters, Holm sonrası anlamlı). "Benzer düşüş" yanlışlama koşulu da sağlanmaz (fark ≠ 0) — yani hipotezin yönsel iddiası yanlışlanmış, ama yanlışlama ölçütünün kendisi (benzer düşüş VE FP ≈ authentic) literal olarak tetiklenmemiştir.
* Shortcut lehine kanıt: bbox-modeli yalnız sınır halkasına duyarlıdır (boundary {F('relΔAP[bbox-GT-trained | bbox GT] boundary_blur vs clean', pct=True)}; plasebo ≈ 0; bbox İÇİ blur {F('relΔAP[bbox-GT-trained | bbox GT] inner_blur vs clean', pct=True)} — içerik silinince skoru değişmez), tahminleri dikdörtgenseldir (medyan {rect_pooled['bbox']:.2f} vs {rect_pooled['pixel']:.2f} / {rect_pooled['glyph']:.2f}); piksel-/glif-modelleri ise içeriğe duyarlıdır (inner_blur −7 % / −11 %). Duyarlılık *profili* (kenar-evet / içerik-hayır) shape-prior'u destekler; mutlak sınır düşüşünün küçük kalması 5 px'lik halkanın tavana yakın (AP 0.90) bir modelde sınırlı etkisiyle açıklanabilir, bu Faz 2 deneyiyle ayrıştırılamaz.
* Benign FP: paste-back ve resample her modelde FP'yi anlamlı artırır (Holm sonrası p = {SM.loc['ΔFP pixel rate [bbox-GT-trained] benign_pasteback − authentic @tau','p_holm']:.3f}), ancak artış bbox-modeline özgü değildir (piksel/glif-modellerinde de +0.2–0.5 ‰); görüntü-düzeyi AUROC 2–3 puan düşer. Yani "benign FP ≈ authentic" koşulu sağlanmaz, fakat bu bulgu H2'nin bbox-özgü iddiasını ayırt etmez.
* Metrik artefaktı (H1 ile ilişkili): piksel-modelinde bbox içini bulanıklaştırmak bbox-GT AP'sini **yükseltir** ({F('relΔAP[pixel-GT-trained | bbox GT] inner_blur vs clean', pct=True)}) ve piksel-GT AP'sini düşürür ({F('relΔAP[pixel-GT-trained | pixel-change GT] inner_blur vs clean', pct=True)}): bbox-GT, içeriği silmeyi ödüllendirir.

## H3 — piksel/glif-GT daha iyi transfer ve dayanıklılık sağlar → YANLIŞLANDI

**Ölçüt.** Yanlışlama: fark yok / ters (CI örtüşür).

**Dayanıklılık (val 487; ΔAP = bozulmuş − temiz; her model kendi GT'sinde ve ortak piksel-GT'de):**

| Model (GT) | JPEG Q75 | 0.75× rescale | blur σ=1 |
|---|---|---|---|
""")
for m in MODELS:
    for g in ([m] if m == 'pixel' else [m, 'pixel']):
        tag = f'{MODEL_LABEL[m]} | {GT_LABEL[g]}'
        md.append(f"| {MODEL_LABEL[m]} ({GT_LABEL[g]}{', kendi GT' if g == m else ''}) | {F(f'Δap_img[{tag}] jpeg75 − clean')} | {F(f'Δap_img[{tag}] rescale075 − clean')} | {F(f'Δap_img[{tag}] blur1 − clean')} |")
md.append("""
**Düşüş farkı (diğer model − bbox-modeli; hipotez > 0 = piksel/glif-modeli daha az düşer; Holm ailesi H3):**

| Kontrast | JPEG Q75 | 0.75× rescale | blur σ=1 |
|---|---|---|---|
""")
for other in ['pixel', 'glyph']:
    for mode, lab in [('own', f'{other}-model (kendi GT) − bbox-model (kendi GT)'), ('pix', f'{other}-model − bbox-model (ikisi de piksel-GT)')]:
        cells = []
        for c in DEGS:
            key = f'ΔAP drop difference [{c}] {other}-model(own GT) − bbox-model(own GT)' if mode == 'own' else f'ΔAP drop difference [{c}] {other}-model − bbox-model (both pixel-GT)'
            cells.append(f"{F(key)} ({P(key)})")
        md.append(f"| {lab} | " + ' | '.join(cells) + ' |')
md.append(f"""
**Held-out generator (gemini-only C modelleri, 1 seed; Ideogram: bbox-GT için val 41 + test 85 = 126, piksel/glif-GT için val 41; kaynak-eşlenmiş Gemini = cord/sroie/xfund, val+test):**

| Model (GT) | Ideogram AP | kaynak-eşlenmiş Gemini AP | açık (Gemini − Ideogram) |
|---|---|---|---|
""")
for m in MODELS:
    for g in ([m] if m == 'bbox' else [m, 'bbox']):
        tag = f'C: gemini-only (1 seed) | {MODEL_LABEL[m]} | {GT_LABEL[g]}'
        nI, _ = N(f'held-out Ideogram AP [{tag}]'); nG, _ = N(f'source-matched Gemini AP [{tag}]')
        md.append(f"| {MODEL_LABEL[m]} ({GT_LABEL[g]}{', kendi GT' if g == m else ''}) | {F(f'held-out Ideogram AP [{tag}]')} (n={nI}) | {F(f'source-matched Gemini AP [{tag}]')} (n={nG}) | {F(f'gap AP Gemini − Ideogram [{tag}]')} ({P(f'gap AP Gemini − Ideogram [{tag}]')}) |")
md.append(f"""
* Ideogram'da modeller arası ΔAP (aynı görüntüler, eşleştirilmiş): piksel − bbox (piksel-GT, n=41) = {F('held-out Ideogram ΔAP pixel-model − bbox-model (gemini-only; both pixel-GT)')} ({P('held-out Ideogram ΔAP pixel-model − bbox-model (gemini-only; both pixel-GT)')}); glif − bbox (piksel-GT) = {F('held-out Ideogram ΔAP glyph-model − bbox-model (gemini-only; both pixel-GT)')} ({P('held-out Ideogram ΔAP glyph-model − bbox-model (gemini-only; both pixel-GT)')}); piksel − bbox (bbox-GT, n=126) = {F('held-out Ideogram ΔAP pixel-model − bbox-model (gemini-only; both bbox-GT)')}; glif − bbox (glif-GT, n=41) = {F('held-out Ideogram ΔAP glyph-model − bbox-model (gemini-only; both glyph-GT)')}.
* Açık farkı (bbox-modeli açığı − piksel-modeli açığı, kendi GT'leri) = {F('gap difference (Gemini−Ideogram) bbox-model(own GT) − pixel-model(own GT) (gemini-only)')} ({P('gap difference (Gemini−Ideogram) bbox-model(own GT) − pixel-model(own GT) (gemini-only)')}): bbox-modelinin göreli düşüşü daha büyüktür, ama bu kendi GT'lerinde ölçülen farklı ölçeklerdeki değerlerin farkıdır; ortak GT'de Ideogram AP'si piksel-modeli lehine değildir.
* B modelleri (tüm araçlar, 3 seed) için aynı tablo stats_summary.csv'de (aile H3, 'B: all tools').

**Karar: YANLIŞLANDI.** Held-out: ortak piksel-GT altında piksel-/glif-modeli Ideogram'da bbox-modelinden yüksek AP vermez (CI'lar 0'ı kapsar, Holm p ≈ 0.86). Dayanıklılık: kendi GT'lerinde piksel-modeli JPEG'de daha az ({F('ΔAP drop difference [jpeg75] pixel-model(own GT) − bbox-model(own GT)')}), yeniden-ölçeklemede farksız, blur'da daha çok düşer; glif-modeli JPEG hariç daha çok düşer; ortak piksel-GT'de her iki model de her koşulda bbox-modelinden daha çok düşer (bbox-modelinin piksel-GT AP'si zaten düşük → taban etkisi). Tutarlı bir dayanıklılık avantajı yok → yanlışlama ölçütü ("fark yok/ters") sağlanır. Sınır: C modelleri tek seed; piksel/glif-GT Ideogram alt kümesi n = 41 (geniş CI; ~±0.07 AP'lik farklar tespit edilemez).

## H4 — bbox dışında öğrenilebilir global iz → YANLIŞLANDI (pratik olarak)

**Ölçüt.** Yanlışlama: AUROC ≈ 0.5. Hipotez: bbox-dışı crop forged-vs-authentic AUROC > 0.5 ve simetrik (resample) koşulda düşüş.

| Prob | Koşul | n çift | AUROC [CI] | p (null 0.5) |
|---|---|---|---|---|
""")
for lab in ['outside-bbox crops (H4)', 'positive control (bbox-centred crops)']:
    for cond, cl in [('forged vs authentic', 'all'), ('forged vs fullimage_resample', 'all'), ('forged vs authentic', 'tool=gemini'), ('forged vs authentic', 'tool=ideogram'),
                     ('forged vs authentic', 'source=cord'), ('forged vs authentic', 'source=sroie'), ('forged vs authentic', 'source=wildreceipt'), ('forged vs authentic', 'source=xfund')]:
        key = f'AUROC {cond} [{lab} | {cl}]'
        if key not in SM.index: continue
        n_, _ = N(key)
        md.append(f"| {lab} | {cond} | {cl}, n={n_} | {F(key)} | {P(key)} |")
md.append(f"""
* ΔAUROC(authentic − resample), bbox-dışı crop = {F('ΔAUROC [outside-bbox crops (H4)] authentic − resample')} ({P('ΔAUROC [outside-bbox crops (H4)] authentic − resample')}).
* Çift-içi doğruluk p(forged) > p(negatif), eşitlik = başarısız: vs authentic {F('pairwise accuracy p(forged)>p(neg), ties=fail [outside-bbox crops (H4) | all | forged_vs_authentic]')}; 457 çiftin {int(SM.loc['share of pairs with identical scores [outside-bbox crops (H4) | all | forged_vs_authentic]','mean']*457+0.5)}'inde skorlar özdeş (crop'lar piksel-özdeş: cord/xfund'da bbox dışı değişmiyor).
* **Post-hoc (keşifsel, önceden kayıtlı değil):** skorları farklı 197 çiftte (sroie/wildreceipt) forged, authentic'ten **{F('pairwise accuracy, ties excluded [outside-bbox crops (H4) | all | forged_vs_authentic] (exploratory)', nd=2)}** oranında yüksek skor alır; simetrik yeniden-örneklenmiş negatife karşı ise {F('pairwise accuracy, ties excluded [outside-bbox crops (H4) | all | forged_vs_resample] (exploratory)', nd=2)} (n=426). Yani bbox dışında değişen piksellerde çift-içi ayırt edilebilir bir **yeniden-kodlama/yeniden-örnekleme izi** vardır, ancak bu iz (i) sahtecilik-özgü değildir (simetrik kontrolde kaybolur) ve (ii) görüntüler arası varyansın içinde kaybolur (görüntü-düzeyi AUROC 0.516). CI'ların darlığı (±0.003) çift-düzeyi yeniden örneklemeden kaynaklanır: 260 özdeş çift her örneklemde tam 0.5 katkı verir.

**Karar: YANLIŞLANDI (pratik olarak).** Görüntü-düzeyi AUROC {F('AUROC forged vs authentic [outside-bbox crops (H4) | all]')} istatistiksel olarak 0.5'ten ayrışır (Holm p = {SM.loc['AUROC forged vs authentic [outside-bbox crops (H4) | all]','p_holm']:.3f}) ve simetrik koşulda 0.5'e iner — hipotezin iki yönsel öngörüsü de işaret olarak tutar — fakat etki büyüklüğü (+0.016) "AUROC ≈ 0.5" yanlışlama ölçütünün içindedir ve kullanılabilir bir global iz yoktur; pozitif kontrol ({F('AUROC forged vs authentic [positive control (bbox-centred crops) | all]')}) pipeline'ın öğrenebildiğini gösterir. Post-hoc çift-içi bulgu, veri kümesi inşasına ilişkin bir uyarıdır (eşleştirilmiş authentic negatiflerle eğitilen bir model yeniden-kodlama asimetrisini öğrenebilir); H4'ün "sahtecilik izi" iddiasını desteklemez.

## Ek kontrastlar

**Test sızıntısı (bbox-modeli, bbox-GT, test):** AP tam 812 = {F('test mean [bbox-GT-trained | bbox GT] all 812','ap_img')}, temiz 740 = {F('test mean [bbox-GT-trained | bbox GT] clean 740','ap_img')}, temiz-olmayan 72 = {F('test mean [bbox-GT-trained | bbox GT] non-clean 72','ap_img')}; Δ(812 − 740) = {F('test Δ[bbox-GT-trained | bbox GT] all 812 − clean 740','ap_img', nd=4)} ({P('test Δ[bbox-GT-trained | bbox GT] all 812 − clean 740','ap_img')}); Δ(72 − 740) = {F('test Δ[bbox-GT-trained | bbox GT] non-clean 72 − clean 740','ap_img')} ({P('test Δ[bbox-GT-trained | bbox GT] non-clean 72 − clean 740','ap_img')}). F1@τ*: Δ(812 − 740) = {F('test Δ[bbox-GT-trained | bbox GT] all 812 − clean 740','f1_tau', nd=4)} ({P('test Δ[bbox-GT-trained | bbox GT] all 812 − clean 740','f1_tau')}). → Sızıntı şüpheli 72 görüntünün etkisi ölçülemeyecek kadar küçük; "tam 812" ve "temiz 740" sonuçları istatistiksel olarak ayırt edilemez.

**Zero-shot literatür baseline'ları (test 812, bbox-GT; eşit koşul DEĞİL — U-Net AIForge-Doc train'de eğitildi, TruFor/ADCD-Net eğitilmedi, çözünürlük protokolleri farklı):**

| Baseline | per-image AP | F1@0.5 | Δ bbox-modeli − baseline (AP) | Δ (F1@0.5) |
|---|---|---|---|---|
""")
for cond, lab in [('trufor', 'TruFor (yerel çözünürlük)'), ('adcd_q100_ocrproxy', 'ADCD-Net (Q100, OCR-proxy)'), ('trufor_1024pad', 'TruFor (1024 pad, makale protokolü)'), ('adcd_q100_emptyocr', 'ADCD-Net (Q100, boş OCR)'), ('adcd_q95_ocrproxy', 'ADCD-Net (Q95)'), ('adcd_q100_noaln', 'ADCD-Net (Q100, hizalama kapalı)')]:
    md.append(f"| {lab} | {F(f'zero-shot {cond} test mean (bbox GT)', 'ap_img')} | {F(f'zero-shot {cond} test mean (bbox GT)', 'f1_05')} | {F(f'Δ bbox-model − zero-shot {cond} (test, bbox GT)', 'ap_img')} ({P(f'Δ bbox-model − zero-shot {cond} (test, bbox GT)', 'ap_img')}) | {F(f'Δ bbox-model − zero-shot {cond} (test, bbox GT)', 'f1_0p5')} |")
md.append(f"""
bbox-modeli test AP = {F('test mean [bbox-GT-trained | bbox GT] all 812','ap_img')}. Farklar bbox-GT ile ölçülmüştür; H1'e göre bu metrik bbox-modelini kayırır — piksel-GT ile adil karşılaştırma yalnız 88 eşleşmiş test çiftinde mümkündür (stats_summary'de yok; Faz 2 README T2'de özet).

## Hesaplanamayan / sınırlı kontrastlar
* Held-out Ideogram'da piksel/glif-GT metrikleri yalnız val'deki 41 görüntüde (test'in 89 çiftinde Ideogram yok); C modelleri tek seed → seed belirsizliği CI'lara dahil değil.
* reencode_png kontrolü authentic ile piksel-özdeş olduğundan ayrıca hesaplanmadı (dejenere).
* Test'te piksel/glif-GT yalnız 88 görüntüde; test düzeyinde H1 kontrastı tekrarlanmadı (val'deki 485 görüntü birincil).
* Bootstrap, seed'ler arası varyansı içermez (seed sabit faktör olarak ortalandı; 3 seed'lik std README tablolarında).

## Figürler
* figures/fig_H1_dualgt_ci.png — 3 model × 3 GT, per-image AP ve F1@τ* (grup-bootstrap %95 CI); Kendall τ.
* figures/fig_H2_ablation_ci.png — boundary / placebo / inner relΔAP, model başına CI (kendi GT ve ortak bbox-GT).
* figures/fig_H2_controls_ci.png — benign kontrollerde FP piksel oranı ve görüntü-düzeyi AUROC, CI.
* figures/fig_H3_transfer_ci.png — degradasyon ΔAP (kendi GT / piksel-GT) ve held-out Ideogram–Gemini açığı (gemini-only modeller).
* figures/fig_H4_auroc.png — H4 AUROC'ları (CI) ve post-hoc çift-içi doğruluk.
""")
import re as _re
_txt = _re.sub(r'\|\n\n\|', '|\n|', '\n'.join(md))
with open(os.path.join(OUT, 'hypothesis_decisions.md'), 'w', encoding='utf-8') as f:
    f.write(_txt)
print('decisions written')
