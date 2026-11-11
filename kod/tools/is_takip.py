"""is_takip.py — GPU işlerinin ve testlerin durumunu tek komutla gösterir (salt okunur).

Kapsam:
  1. Yerel kuyruk (phase3/logs/queue.log): son başlatılan kuyruk, biten/hatalı adımlar, çalışan adım ve süresi,
     ortalama adım süresi; kuyruk süreci ve GPU kullanımı (nvidia-smi).
  2. Hugging Face Space (varsayılan infinayazilim/ai515): aşama ve donanım; ZeroGPU kotası (token izin verirse).
  3. Hugging Face Jobs (varsayılan infinayazilim): çalışan ve son işler (ücretli işler bu projede kapalı; liste yalnız denetim içindir).
  4. Testler: tests/run_all_summary.txt ve tests/test_result.json.

Kullanım:
  python kod/tools/is_takip.py                 # hepsi
  python kod/tools/is_takip.py --yerel         # yalnız yerel kuyruk
  python kod/tools/is_takip.py --izle 300      # 5 dakikada bir yenile (Ctrl+C ile çık)
  python kod/tools/is_takip.py --json          # makine okunur çıktı
Hiçbir işi başlatmaz, durdurmaz ya da değiştirmez.
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

sys.path.append(str(next(p for p in Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

LOG = ROOT / "phase3" / "logs"


def read_text(p: Path) -> str:
    b = p.read_bytes()
    for enc in ("utf-16", "utf-8-sig", "latin-1"):
        try:
            t = b.decode(enc)
            if enc != "utf-16" or b[:2] in (b"\xff\xfe", b"\xfe\xff"):
                return t
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "replace")


def yerel() -> dict:
    q = LOG / "queue.log"
    if not q.exists():
        return dict(hata="queue.log yok")
    lines = [l.strip() for l in read_text(q).splitlines() if l.strip()]
    starts = [i for i, l in enumerate(lines) if re.search(r"\sq\w+ start\b", l)]
    blk = lines[starts[-1]:] if starts else lines
    name = re.search(r"\s(q\w+) start", blk[0]).group(1) if starts else "?"
    pid = re.search(r"pid=(\d+)", blk[0])
    done, fail, started = [], [], {}
    for l in blk:
        m = re.match(r"(\S+) (START|DONE|FAIL) (\S+)(?: (\d+)s)?", l)
        if not m:
            continue
        ts, kind, step, sec = m.groups()
        if kind == "START":
            started[step] = ts
        elif kind == "DONE":
            done.append((step, int(sec or 0)))
        else:
            fail.append((step, l))
    ended = {s for s, _ in done} | {s for s, _ in fail}
    running = [(s, t) for s, t in started.items() if s not in ended]
    now = datetime.now()
    run = None
    if running:
        s, t = running[-1]
        el = (now - datetime.fromisoformat(t)).total_seconds()
        steplog = LOG / f"{s}.log"
        last = ""
        if steplog.exists():
            tl = [x for x in read_text(steplog).splitlines() if x.strip() and "libpng warning" not in x]
            last = tl[-1][:160] if tl else ""
        run = dict(adim=s, basladi=t, gecen_dk=round(el / 60, 1), son_satir=last)
    finished = any(re.search(r"\sq\w+ finished", l) for l in blk)
    alive = None
    if pid and sys.platform == "win32":
        r = subprocess.run(["powershell", "-NoProfile", "-Command", f"[bool](Get-Process -Id {pid.group(1)} -ErrorAction SilentlyContinue)"],
                           capture_output=True, text=True)
        alive = r.stdout.strip().lower() == "true"
    gpu = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        n, u, mu, mt, tc = [x.strip() for x in r.stdout.strip().splitlines()[0].rsplit(",", 4)]
        gpu = dict(ad=n, kullanim_yuzde=int(float(u)), bellek_mb=f"{mu}/{mt}", sicaklik_c=round(float(tc)))
    except Exception as e:
        gpu = dict(hata=f"nvidia-smi okunamadı ({type(e).__name__})")
    return dict(kuyruk=name, pid=int(pid.group(1)) if pid else None, surec_calisiyor=alive, bitti=finished,
                biten=len(done), hatali=[s for s, _ in fail], calisan=run,
                son_bitenler=[f"{s} ({sec // 60} dk)" for s, sec in done[-5:]],
                ort_adim_dk=round(sum(x for _, x in done) / len(done) / 60, 1) if done else None,
                son_bloklar=[l for l in blk if re.search(r"bitti|bloklari|blogu|finished", l)][-3:], gpu=gpu)


def space(sid: str) -> dict:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        rt = api.space_info(sid).runtime
        out = dict(space=sid, asama=getattr(rt, "stage", None), donanim=getattr(rt, "hardware", None))
        try:
            out["zerogpu_kota"] = str(api.get_zero_gpu_quota())
        except Exception as e:
            out["zerogpu_kota"] = f"okunamadı ({type(e).__name__}; tokenin bu izni yok olabilir)"
        return out
    except Exception as e:
        return dict(space=sid, hata=f"{type(e).__name__}: {str(e)[:120]}")


def jobs(ns: str) -> dict:
    try:
        from huggingface_hub import HfApi
        js = list(HfApi().list_jobs(namespace=ns))
        rows = []
        for j in js[-8:]:
            st = getattr(j, "status", None)
            rows.append(dict(id=j.id, durum=getattr(st, "stage", st), donanim=getattr(j, "flavor", None),
                             olusturuldu=str(getattr(j, "created_at", ""))[:19]))
        return dict(namespace=ns, toplam=len(js), calisan=sum(1 for r in rows if str(r["durum"]).upper() == "RUNNING"), son=rows)
    except Exception as e:
        msg = "403: bu bilgisayardaki HF tokeninin Jobs izni yok" if "403" in str(e) else f"{type(e).__name__}: {str(e).splitlines()[-1][:120]}"
        return dict(namespace=ns, hata=msg)


def testler() -> dict:
    out = {}
    t = ROOT / "tests"
    for f in ("run_all_summary.txt", "test_result.json"):
        p = t / f
        if p.exists():
            out[f] = dict(icerik=" ".join(read_text(p).split())[:300], tarih=datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes"))
    return out


def yaz(d: dict):
    y = d.get("yerel")
    if y:
        print(f"== Yerel kuyruk: {y.get('kuyruk')} (pid {y.get('pid')}, süreç {'çalışıyor' if y.get('surec_calisiyor') else 'yok' if y.get('surec_calisiyor') is False else '?'})"
              + (" — BİTTİ" if y.get("bitti") else ""))
        print(f"   biten adım: {y.get('biten')} · hatalı: {len(y.get('hatali', []))} {y.get('hatali') or ''} · ortalama adım: {y.get('ort_adim_dk')} dk")
        c = y.get("calisan")
        if c:
            print(f"   çalışan: {c['adim']} ({c['gecen_dk']} dk) · {c['son_satir']}")
        print("   son bitenler: " + ", ".join(y.get("son_bitenler", [])))
        for l in y.get("son_bloklar", []):
            print("   " + l)
        g = y.get("gpu")
        if g and "hata" in g:
            print("   GPU: " + g["hata"])
        elif g:
            print(f"   GPU: {g['ad']} · %{g['kullanim_yuzde']} · {g['bellek_mb']} MB · {g['sicaklik_c']} °C")
    s = d.get("space")
    if s:
        print(f"== Space {s.get('space')}: " + (s.get("hata") or f"{s.get('asama')} · {s.get('donanim')} · ZeroGPU kotası: {s.get('zerogpu_kota')}"))
    j = d.get("jobs")
    if j:
        print(f"== HF Jobs ({j.get('namespace')}): " + (j.get("hata") or f"{j.get('toplam')} iş, çalışan {j.get('calisan')}"))
        for r in j.get("son", []) if not j.get("hata") else []:
            print(f"   {r['id']} {r['durum']} {r['donanim']} {r['olusturuldu']}")
    for k, v in (d.get("testler") or {}).items():
        print(f"== Test {k} ({v['tarih']}): {v['icerik'][:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yerel", action="store_true")
    ap.add_argument("--space", default="infinayazilim/ai515")
    ap.add_argument("--namespace", default="infinayazilim")
    ap.add_argument("--izle", type=int, default=0, help="saniye cinsinden yenileme aralığı")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    while True:
        d = dict(zaman=datetime.now().isoformat(timespec="seconds"), yerel=yerel())
        if not a.yerel:
            d.update(space=space(a.space), jobs=jobs(a.namespace), testler=testler())
        if a.json:
            print(json.dumps(d, ensure_ascii=False, indent=1, default=str))
        else:
            print(f"--- {d['zaman']}"); yaz(d)
        if not a.izle:
            break
        time.sleep(a.izle)


if __name__ == "__main__":
    main()
