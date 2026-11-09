r"""
live_test.py — Canlı Space testi (gradio_client). Simülasyon ve toplu testi uzaktan çalıştırır, sonuçları JSON'a yazar.
Kullanım (proje kökü):  python hf_space\live_test.py [--space infinayazilim/ai515] [--n 24] [--out reports\canli_test.json]
Gerekli: pip install gradio_client ; hf auth login (Space özel).
"""
import argparse, json, sys, time, datetime
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from gradio_client import Client
from huggingface_hub import get_token

ap = argparse.ArgumentParser()
ap.add_argument("--space", default="infinayazilim/ai515")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--out", default="reports/canli_test.json")
a = ap.parse_args()

t0 = time.time()
c = Client(a.space, token=get_token(), verbose=False)
api = c.view_api(return_format="dict", print_info=False)
names = sorted(api["named_endpoints"])
print("uç noktalar:", names)
OUT = dict(space=a.space, zaman=datetime.datetime.now().isoformat(timespec="seconds"), uc_noktalar=names, simulasyon=[], toplu={})


def table(x):
    if isinstance(x, dict) and "headers" in x:
        return [dict(zip(x["headers"], row)) for row in x["data"]]
    return x


UNETS = ["PHASE2/unet_bbox_s42", "PHASE2/unet_pixel_eps8_morph_s42", "PHASE2/unet_glyph_v2_s42"]
RES = ["PHASE1/resnet18_ft_best"]
INPUTS = ["Sahte belge (forged)", "Authentic eş (negatif kontrol)", "Sahte, yeşil işaretçi onarılmış",
          "Authentic + yalnız yeşil işaretçi (kısayol testi)"]

# 1) rastgele eşli + işaretçili test örneği seç
dd, cnt = c.predict("test (testing)", "hepsi", "hepsi", "hepsi", True, "var", 12, api_name="/sim_random")
key = dd["value"] if isinstance(dd, dict) else dd
print("seçilen örnek:", key, "·", cnt)
# 2) dört senaryo
for inp in INPUTS:
    t = time.time()
    res = c.predict(key, inp, "Yok (ham PNG)", False, UNETS, RES, 112, api_name="/simulate")  # gr.State girdileri API'de yoktur
    log, gal, df, verdict, zoom, files = res[:6]
    rows = table(df)
    OUT["simulasyon"].append(dict(ornek=key, senaryo=inp, sure_s=round(time.time() - t, 1), satirlar=rows, karar=verdict.splitlines()[:8]))
    print(f"\n[{inp}] {time.time() - t:.1f}s")
    for r in rows:
        print("  ", r.get("Model"), "→", r.get("Karar"), r.get("Doğru"), "skor", r.get("Görüntü skoru"), "F1bbox", r.get("F1 bbox-GT"), "F1piks", r.get("F1 piksel-GT"), "kapsama", r.get("Kutu kapsama"))
# 3) toplu test (dengeli)
t = time.time()
prog, per, agg, fig, files = c.predict("test (testing)", "hepsi", "hepsi", "hepsi", "hepsi", a.n, 42, "Sahte + authentic eşi (dengeli tespit)",
                                      "Yok (ham PNG)", UNETS[:2], RES, 12, api_name="/batch_test")
OUT["toplu"] = dict(n=a.n, sure_s=round(time.time() - t, 1), ilerleme=prog, ozet=table(agg), ornekler=table(per))
print(f"\n[toplu test] {time.time() - t:.1f}s\n{prog}")
for r in table(agg):
    print("  ", r)
OUT["toplam_sure_s"] = round(time.time() - t0, 1)
import os
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(OUT, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("\nyazıldı:", a.out, f"({OUT['toplam_sure_s']} s)")
