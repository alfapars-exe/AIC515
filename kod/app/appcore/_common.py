"""appcore._common — ortak içe aktarmalar (numpy, OpenCV, pandas, matplotlib, gradio, torch, skimage) ve TORCH_OK bayrağı."""
from __future__ import annotations

import os, sys, io, re, json, time, math, glob, random, tempfile, datetime


from pathlib import Path

os.environ.setdefault("USERNAME", "user")  # torch inductor getpass sorunu (ENV_SETUP.md)


# ZeroGPU: `spaces` torch/CUDA'dan ÖNCE içe aktarılmalı. Paket yoksa (yerel) dekoratör etkisiz kalır.
try:
    import spaces  # type: ignore
    _HAS_SPACES = True
except Exception:
    spaces = None
    _HAS_SPACES = False

import numpy as np


import cv2


import pandas as pd


import matplotlib


matplotlib.use("Agg")


import matplotlib.pyplot as plt


import gradio as gr

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import segmentation_models_pytorch as smp
    TORCH_OK = True
    _TORCH_ERR = ""
except Exception as e:  # torch yoksa yalnız klasik işlemler çalışır
    TORCH_OK = False
    _TORCH_ERR = str(e)

from skimage.filters import threshold_sauvola


from skimage.segmentation import watershed


from skimage.feature import peak_local_max

__all__ = [n for n in dir() if not n.startswith("__")]  # alt çizgili adlar dahil (ör. _HAS_SPACES)
