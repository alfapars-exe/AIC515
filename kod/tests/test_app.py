"""Uçtan uca fonksiyon testleri. Kullanım: python kod/tests/test_app.py  (AI515_TEST_N=35 ile hızlı koşu)"""
import os, sys, time, json, math, traceback
import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
sys.path.insert(0, os.environ.get("AI515_APP_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
OUT_DIR = str(_out_dir(__file__))  # <proje>/tests
N_MAX = int(os.environ.get("AI515_TEST_N", "0"))  # 0 = yerelde mevcut tüm örnekler (tam veri setinde saatler sürer)
import numpy as np, pandas as pd
import gradio as gr
t0 = time.time()
import app as A
print(f"import {time.time() - t0:.1f}s · modeller {len(A._REG)} · hata {A._LOAD_ERR} · cihaz {A.DEVICE} · ZeroGPU {A.ZERO_GPU}")
RES = dict(errors=[], checks=0)


def check(cond, msg):
    RES["checks"] += 1
    if not cond:
        RES["errors"].append(msg); print("  ✗", msg)


def run_sim(key, inp=A.INPUT_TYPES[0], deg=A.DEGRADATIONS[0], pipe=False, steps=None, unets=None, res=None):
    unets = A._DEF_UNETS if unets is None else unets
    res = A._DEF_RESNET if res is None else res
    last, n = None, 0
    for out in A.simulate(key, inp, deg, pipe, steps or [], unets, res, 112):
        last = out; n += 1
    return last, n


idx = A.dataset_index()
keys = [k for k in idx.index if A.sample_exists(k)]
print(f"dizin {len(idx)} · yerelde mevcut {len(keys)}")
keys_loop = keys[:N_MAX] if N_MAX else keys
check(len(idx) == 4061, "dizin 4061 satır olmalı")
check(len(keys) >= 30, "en az 30 yerel örnek")

# 1) her örnek: sahte senaryo, 3 U-Net + ResNet
repro = []
for i, k in enumerate(keys_loop, 1):
    t = time.time()
    try:
        out, n = run_sim(k)
        log, gal, df, verdict, zoom, files, state = out
        check(n >= 8, f"{k}: en az 8 ara güncelleme (adım adım) bekleniyordu, {n}")
        check(len(df) == len(A._DEF_UNETS) + len(A._DEF_RESNET), f"{k}: model satırı sayısı")
        check("7/7" in log, f"{k}: 7 adım tamamlanmalı")
        check(zoom is not None and zoom.ndim == 3, f"{k}: yakın plan görüntüsü")
        check(all(os.path.exists(f) for f in files) and len(files) == 4, f"{k}: rapor dosyaları")
        check(state and state["key"] == k, f"{k}: durum")
        check("Gerçek etiket: **SAHTE**" in verdict, f"{k}: etiket SAHTE")
        raw = pd.read_csv([f for f in files if f.endswith(".csv")][0])
        for _, r in raw.iterrows():
            if "s42" in r["Model"] and pd.notna(r.get("Ref F1@0.5 (boru hattı)")) and pd.notna(r.get("Uyg. F1@0.5")):
                repro.append(dict(key=k, model=r["Model"], ref=r["Ref F1@0.5 (boru hattı)"], app=r["Uyg. F1@0.5"]))
        has_auth = state["auth"] is not None
        print(f"[{i:02d}/{len(keys_loop)}] {k} ok {time.time() - t:.1f}s · auth={has_auth} · " +
              " | ".join(f"{r['Model'][:22]}: {r['Karar']} {r['Doğru']}" for _, r in df.iterrows()))
    except Exception as e:
        RES["errors"].append(f"{k}: {e!r}"); traceback.print_exc()

rp = pd.DataFrame(repro)
if len(rp):
    rp["abs"] = (rp["app"] - rp["ref"]).abs()
    print("\nYENİDEN ÜRETİM (F1@0.5, uygulama vs boru hattı kayıtlı):")
    print(rp.groupby("model")["abs"].agg(["count", "mean", "max"]).round(4).to_string())
    check(rp["abs"].mean() < 0.02, f"ortalama |ΔF1@0.5| < 0.02 olmalı ({rp['abs'].mean():.4f})")
    rp.to_csv(os.path.join(OUT_DIR, "repro_f1.csv"), index=False)

# 2) eşli bir örnekte tüm senaryolar + bozulmalar + pipeline
paired = [k for k in keys if idx.loc[k, "has_pair"] and idx.loc[k, "has_marker"]] or [k for k in keys if idx.loc[k, "has_pair"]]
kp = paired[0]
print(f"\nSenaryo testleri: {kp}")
scen = {}
for inp in A.INPUT_TYPES:
    out, _ = run_sim(kp, inp=inp, res=[])
    df = out[2]
    scen[inp] = df
    lab = "SAHTE" if inp in (A.INPUT_TYPES[0], A.INPUT_TYPES[2]) + tuple(A.INPUT_TYPES[5:6]) else "GERÇEK"  # Faz 3: yalnız-içerik = sahte
    check(f"Gerçek etiket: **{lab}**" in out[3], f"senaryo etiketi {inp}")
    print(" ", inp, "→", [(r["Model"][:20], r["Karar"], r["Görüntü skoru"], r["Kutu kapsama"]) for _, r in df.iterrows()])
sb = lambda inp: float(scen[inp].iloc[0]["Görüntü skoru"])
check(sb(A.INPUT_TYPES[3]) > sb(A.INPUT_TYPES[1]), "authentic+işaretçi skoru authentic'ten yüksek olmalı (kısayol)")
for deg in A.DEGRADATIONS[1:]:
    out, _ = run_sim(kp, deg=deg, unets=A._DEF_UNETS[:1], res=[])
    print(f"  bozulma {deg}: F1 bbox {out[2].iloc[0]['F1 bbox-GT']} · skor {out[2].iloc[0]['Görüntü skoru']}")
    check("Ref" not in str(out[2].iloc[0]["Ref F1@0.5 (boru hattı)"]) and out[2].iloc[0]["Ref F1@0.5 (boru hattı)"] == "—", f"bozulmada referans gösterilmemeli ({deg})")
steps = A.apply_preset(load := A.load_dataset_sample(kp)[0], "Kontrast iyileştirme (CLAHE → unsharp)", None)[2]
out, _ = run_sim(kp, pipe=True, steps=steps, unets=A._DEF_UNETS[:1], res=[])
check("pipeline" in out[0], "pipeline notu log'da")
geo = [dict(cat="9 · Geometrik", name="Yeniden boyutlandırma", info="", p=[0.5, 0, 0], mode="bilinear")]
out, _ = run_sim(kp, pipe=True, steps=geo, unets=A._DEF_UNETS[:1], res=[])
check(out[2].iloc[0]["F1 bbox-GT"] == "—", "geometrik pipeline'da GT metriği hesaplanmamalı")

# 3) hata yolları
unpaired = [k for k in keys if not idx.loc[k, "has_pair"]]
for bad, msg in ((lambda: run_sim(""), "boş anahtar"), (lambda: run_sim(unpaired[0], inp=A.INPUT_TYPES[1]), "eşsiz örnekte authentic senaryosu"),
                 (lambda: run_sim(kp, unets=[], res=[]), "model seçilmeden")):
    try:
        bad(); check(False, f"hata beklenirdi: {msg}")
    except gr.Error:
        RES["checks"] += 1
missing = next((k for k in idx.index if not A.sample_exists(k)), "testing:999999999")  # tam veri setinde eksik yoksa olmayan kimlik
try:
    run_sim(missing); check(False, "eksik dosya hatası beklenirdi")
except Exception as e:
    check("bulunamadı" in str(e) or "veri setinde yok" in str(e), f"eksik dosya mesajı: {e}")

# 4) toplu test
print("\nToplu test (dengeli, test bölümü, tüm eşli):")
last = None
for out in A.batch_test("test (testing)", "hepsi", "hepsi", "hepsi", "hepsi", 50, 1, A.BATCH_MODES[1], A.DEGRADATIONS[0], A._DEF_UNETS[:2], A._DEF_RESNET, 12):
    last = out
prog, per, agg, fig, files = last
print(prog.splitlines()[0]); print(agg.to_string())
check(fig is not None and len(files) == 4 and all(os.path.exists(f) for f in files), "toplu test çıktıları")
check((per["Girdi"] == "authentic").sum() > 0, "dengeli modda authentic satırları")
print("\nToplu test (yalnız sahte, eğitim/val, 16 örnek, JPEG Q75):")
for out in A.batch_test("eğitim (training)", "hepsi", "hepsi", "hepsi", "hepsi", 16, 3, A.BATCH_MODES[0], "JPEG Q75", A._DEF_UNETS[:1], [], 12):
    last = out
print(last[2].to_string())

# 5) diğer sekmeler
img, mask, auth, meta = A.load_dataset_sample(kp)
o = A.on_sample(kp); check(o[0] is not None and len(o) == 10, "on_sample")
r = A.run_unet(img, img, mask, auth, A._DEF_UNETS[0], "Orijinal görüntü", 0.45, False, 0, "resmî bbox maskesi"); check(len(r[3]) >= 14, "run_unet bbox")
r = A.run_unet(img, img, mask, auth, A._DEF_UNETS[1], "Orijinal görüntü", 0.45, True, 5, "piksel-değişim (authentic'ten türet, eps8_morph)"); check("piksel" in r[4], "run_unet piksel+Otsu+morf")
cur = A.apply_preset(img, "Dayanıklılık probu: JPEG Q75", auth)
c = A.compare_unet(img, cur[0], mask, auth, A._DEF_UNETS[0], 0.45, False, 0, "resmî bbox maskesi", cur[2]); check(len(c[2]) > 5, "compare_unet")
for rn in A._RESNETS:
    r = A.run_resnet(img, img, mask, rn, "Orijinal görüntü", 112); check(len(r[1]) >= 7, f"run_resnet {rn}")
r = A.run_green(img, mask, auth); check("işaretçi" in r[1].lower(), "run_green")
f = A.classical_features(img, mask); check(len(f) == 56, "56 öznitelik")
s = A.sim_random("hepsi", "hepsi", "hepsi", "hepsi", False, "hepsi", 40); check(A.sample_exists(s[0]["value"] if isinstance(s[0], dict) else s[0].value), "sim_random mevcut örnek")
pv = A.sim_preview(kp); check(pv[0] is not None, "sim_preview")
st = run_sim(kp, res=[])[0][6]; tt = A.sim_to_tabs(st); check(tt[0] is not None and len(tt) == 11, "sim_to_tabs")
fails = []
for cat, ops in A.OPS.items():
    for name, spec in ops.items():
        p = [x[5] for x in spec["params"]] + [0] * (3 - len(spec["params"]))
        for src in (img, A.to_gray(img)):
            try:
                out, _ = A.run_op(src, cat, name, *p, spec["modes"][0] if spec["modes"] else None, auth)
                assert out.dtype == np.uint8
            except Exception as e:
                fails.append((cat, name, repr(e)))
check(not fails, f"ön işleme işlemleri: {fails}")
check(len(A.results_markdown()) > 500, "sonuçlar sekmesi")

print(f"\n==== {RES['checks']} kontrol, {len(RES['errors'])} hata, {time.time() - t0:.0f}s ====")
for e in RES["errors"]:
    print(" HATA:", e)
json.dump(RES, open(os.path.join(OUT_DIR, "test_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
sys.exit(1 if RES["errors"] else 0)
