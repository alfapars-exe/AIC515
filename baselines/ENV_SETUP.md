# Ortam Kurulumu (docforensics) — RTX 5080 / Windows

Doğrulanan yığın (19 Ekim 2026):
- Python 3.11 (conda), torch 2.11.0+cu128, torchvision 0.26.0+cu128 (CUDA 12.8; RTX 5080 = Blackwell sm_120 için cu128 ZORUNLU, PyPI'daki Windows wheel'leri CPU-only)
- timm 1.0.30, segmentation-models-pytorch 0.5.0, scikit-learn 1.9.1, opencv-python-headless 5.0.0, albumentations 2.0.8, scikit-image, imagehash, torchmetrics, huggingface_hub, datasets
- Doğrulama: torch.cuda.is_available()=True, cihaz "NVIDIA GeForce RTX 5080", capability (12,0), 15.9 GiB

Kurulum (PowerShell, kendi makinenizde):
    conda create -n docforensics python=3.11 pip -y
    conda activate docforensics
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install timm segmentation-models-pytorch scikit-learn scikit-image opencv-python-headless pillow pyarrow pandas matplotlib seaborn tqdm imagehash openpyxl python-docx huggingface_hub datasets torchmetrics albumentations

Bilinen sorun: USERNAME ortam değişkeni tanımsızsa torch import'u `ModuleNotFoundError: pwd` ile düşer (torch/_inductor cache dizini getpass.getuser() çağırır).
Çözüm: import öncesi `os.environ.setdefault("USERNAME","user")` veya site-packages/sitecustomize.py ile aynı ayar (bu ortamda uygulandı).

Tam paket listesi: requirements_freeze.txt
