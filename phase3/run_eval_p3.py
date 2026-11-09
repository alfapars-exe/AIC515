"""run_eval_p3.py — Faz 2 run_eval.py'yi Faz 3 yamalarıyla (mimari kaydı, clahe/gray koşulları) çalıştırır (TR).
Kullanım run_eval.py ile aynı: python run_eval_p3.py --models dlv3p_bbox_s42,... --stages val,test,robust,controls
Not: 'robust' aşaması D.DEGRADATIONS'taki TÜM koşulları gezer → clahe ve gray de dahil edilir."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3ext  # noqa: F401,E402
import run_eval  # noqa: E402

if __name__ == "__main__":
    run_eval.main()
