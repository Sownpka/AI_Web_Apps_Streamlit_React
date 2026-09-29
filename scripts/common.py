"""Phần chung cho các script pipeline (tương ứng mục 0–1 của notebook)."""
import os
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("APP_ROOT", str(ROOT))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", line_buffering=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # chạy không cần màn hình: hình lưu vào docs/figures thay vì plt.show()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from config import ART_DIR, DATA_DIR, DEVICE  # noqa: E402

SEED = 42
FAST = DEVICE == "cpu"  # CPU: giảm epoch & dùng LLM nhỏ để pipeline vẫn chạy hết
FIG_DIR = ROOT / "docs" / "figures"
FLOWERS_DIR = DATA_DIR / "flowers" / "flower_photos"
COCO_DIR = DATA_DIR / "coco128"

for d in ["artifacts/classifier", "artifacts/detector", "artifacts/retrieval", "logs", "docs/figures"]:
    (ROOT / d).mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def download(url: str, dest: Path) -> Path:
    dest = Path(dest)
    if not dest.exists():
        print("↓", url)
        urllib.request.urlretrieve(url, dest)
    return dest


def coco_images() -> list[Path]:
    return sorted((COCO_DIR / "images" / "train2017").glob("*.jpg"))


def save_fig(name: str):
    path = FIG_DIR / name
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    print("🖼", path.relative_to(ROOT))
