"""appcore.config — yollar (ROOT, DATA, MODELS, META ...), ZeroGPU algılama ve gpu() dekoratörü."""
from __future__ import annotations
from ._common import *  # noqa: F401,F403 — ortak içe aktarmalar

# ------------------------------------------------------------------ yollar
# Yerelde app/ proje kökünün altındadır (ROOT = üst klasör). Space'te app.py repo kökündedir; veri ve modeller
# bucket'tan AI515_DATA_ROOT altına bağlanır ve proje dizin yapısını izler (data/, models/, dataset_audit/).
APP_DIR = Path(__file__).resolve().parent.parent  # app.py'nin bulunduğu klasör (appcore/ bir alt pakettir)


IN_SPACE = bool(os.environ.get("SPACE_ID"))


# Kod çatısı: Space'te app.py repo kökündedir (baselines/ yanında); yerelde kod/app/ altındadır.
CODE_DIR = APP_DIR if (APP_DIR / "baselines").exists() else APP_DIR.parent


ROOT = Path(os.environ.get("VIZE_ROOT") or (CODE_DIR.parent if CODE_DIR.name.lower() == "kod" else CODE_DIR))


DATA_ROOT = Path(os.environ.get("AI515_DATA_ROOT") or ROOT)


DATA = DATA_ROOT / "data" / "AIForge-Doc-v1"


MODELS = DATA_ROOT / "models"


AUDIT = DATA_ROOT / "dataset_audit"


if not any(MODELS.glob("*.pt")) and any((ROOT / "models").glob("*.pt")):
    MODELS = ROOT / "models"


if not (AUDIT / "samples_forged.csv").exists() and (ROOT / "dataset_audit" / "samples_forged.csv").exists():
    AUDIT = ROOT / "dataset_audit"


META = APP_DIR / "meta"  # uygulamayla gelen küçük tablolar (eşikler, referans sonuçlar, işaretçi listesi)


RESULTS_MD = ROOT / "baselines" / "results" / "results_tables.md"


P2_SUMMARY = ROOT / "phase2" / "experiments" / "results" / "summary_test_agg.csv"


TAU_JSON = ROOT / "phase2" / "experiments" / "results" / "tau.json"


if not TAU_JSON.exists() and (META / "tau.json").exists():
    TAU_JSON = META / "tau.json"

sys.path.insert(0, str(CODE_DIR / "baselines"))


try:
    import features_classical as FC
    FC_OK = True
except Exception:
    FC_OK = False

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)


IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _zero_gpu() -> bool:
    if not (_HAS_SPACES and IN_SPACE):
        return False
    try:
        from spaces.config import Config  # type: ignore
        return bool(Config.zero_gpu)
    except Exception:
        return bool(os.environ.get("SPACES_ZERO_GPU"))


ZERO_GPU = _zero_gpu()


if TORCH_OK:
    DEVICE = torch.device("cuda" if (ZERO_GPU or torch.cuda.is_available()) else "cpu")
else:
    DEVICE = None


def gpu(duration=60):
    """@spaces.GPU sarmalayıcısı: ZeroGPU'da GPU ister, diğer ortamlarda etkisizdir. duration sayı ya da çağrılabilir."""
    def deco(fn):
        if _HAS_SPACES:
            try:
                return spaces.GPU(duration=duration)(fn)
            except Exception:
                return fn
        return fn
    return deco
