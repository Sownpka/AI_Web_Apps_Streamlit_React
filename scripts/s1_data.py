"""Mục 1 — Tải dữ liệu (TF Flowers, COCO128) và khám phá dữ liệu (EDA)."""
import tarfile
import zipfile

from common import COCO_DIR, DATA_DIR, FLOWERS_DIR, coco_images, download, plt, save_fig
from PIL import Image


def main():
    # 1) TF Flowers
    if not FLOWERS_DIR.exists():
        tgz = download("https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
                       DATA_DIR / "flower_photos.tgz")
        with tarfile.open(tgz) as t:
            t.extractall(DATA_DIR / "flowers", filter="data")
        tgz.unlink()
    (FLOWERS_DIR / "LICENSE.txt").unlink(missing_ok=True)
    print("Flowers:", {d.name: len(list(d.glob("*.jpg"))) for d in sorted(FLOWERS_DIR.iterdir()) if d.is_dir()})

    # 2) COCO128 — 128 ảnh COCO kèm nhãn YOLO
    if not COCO_DIR.exists():
        z = download("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip", DATA_DIR / "coco128.zip")
        with zipfile.ZipFile(z) as f:
            f.extractall(DATA_DIR)
        z.unlink()
    images = coco_images()
    print("COCO128:", len(images), "ảnh")

    # 3) Kho tri thức ShopLite (đã tách từ notebook vào data/kb/)
    print("KB:", sorted(p.name for p in (DATA_DIR / "kb").glob("*.md")))

    # EDA: số ảnh mỗi lớp, kích thước, ảnh mẫu
    classes = sorted(d.name for d in FLOWERS_DIR.iterdir() if d.is_dir())
    counts = {c: len(list((FLOWERS_DIR / c).glob("*.jpg"))) for c in classes}
    sizes = [Image.open(p).size for p in list(FLOWERS_DIR.rglob("*.jpg"))[:300]]
    print("Số ảnh mỗi lớp:", counts, "| tổng:", sum(counts.values()))
    print("Kích thước ảnh (300 mẫu): rộng %d–%d px, cao %d–%d px" % (
        min(w for w, _ in sizes), max(w for w, _ in sizes), min(h for _, h in sizes), max(h for _, h in sizes)))

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, c in zip(axes[0], classes):
        ax.imshow(Image.open(next((FLOWERS_DIR / c).glob("*.jpg")))); ax.set_title(c); ax.axis("off")
    for ax, p in zip(axes[1], images[:5]):
        ax.imshow(Image.open(p)); ax.set_title(p.name); ax.axis("off")
    plt.suptitle("Hàng trên: TF Flowers · Hàng dưới: COCO128"); plt.tight_layout()
    save_fig("eda_samples.png")


if __name__ == "__main__":
    main()
