"""Mục 4 — Ứng dụng 3: lập chỉ mục CLIP + FAISS cho kho ảnh, chạy thử và đo Precision@k."""
import json
import random
import shutil
import time

from common import ART_DIR, DATA_DIR, FLOWERS_DIR, ROOT, SEED, coco_images, config, np, plt, save_fig
from PIL import Image
from ultralytics.utils import ASSETS

import core.detector as det
import core.retrieval as ret


def show_results(results, title, fig_name):
    fig, axes = plt.subplots(1, len(results), figsize=(3 * len(results), 3.4))
    for ax, r in zip(np.atleast_1d(axes), results):
        ax.imshow(Image.open(config.resolve_path(r["path"]))); ax.axis("off")
        ax.set_title(f"{r['label']}\n{r['score']:.3f}", fontsize=9)
    fig.suptitle(title)
    save_fig(fig_name)


def main():
    classes = sorted(d.name for d in FLOWERS_DIR.iterdir() if d.is_dir())

    # Kho ảnh riêng data/gallery/ (chép ảnh vào) → khi triển khai chỉ cần mang theo thư mục này.
    # meta.json lưu đường dẫn TƯƠNG ĐỐI so với ROOT để index chạy được trên máy khác.
    gallery = DATA_DIR / "gallery"
    gallery.mkdir(exist_ok=True)
    detector = det.ObjectDetector()
    items = []
    for p in coco_images():
        summary = detector.detect(Image.open(p), conf=0.4)[0]["summary"]  # dùng Ứng dụng 2 để gắn nhãn ảnh COCO
        label = ", ".join(sorted(summary, key=summary.get, reverse=True)[:2]) or "coco"
        dst = gallery / f"coco_{p.name}"; shutil.copy(p, dst)
        items.append({"path": dst.relative_to(ROOT).as_posix(), "label": label, "source": "coco128"})
    rng = random.Random(SEED)
    for c in classes:
        files = sorted((FLOWERS_DIR / c).glob("*.jpg"))
        for p in rng.sample(files, min(100, len(files))):
            dst = gallery / f"{c}_{p.name}"; shutil.copy(p, dst)
            items.append({"path": dst.relative_to(ROOT).as_posix(), "label": c, "source": "flowers"})
    del detector

    encoder = ret.ClipEncoder()
    t0 = time.time()
    ret.build_index(encoder, items)
    engine = ret.ImageSearch(encoder=encoder)
    print(f"Đã lập chỉ mục {engine.index.ntotal} ảnh trong {time.time() - t0:.0f}s")

    for i, q in enumerate(["yellow sunflowers in a field", "a dog", "people playing sports", "a pizza on a table"]):
        show_results(engine.search_text(q, k=5), f"Truy vấn văn bản: “{q}”", f"retrieval_text_{i}.png")
    show_results(engine.search_image(Image.open(ASSETS / "bus.jpg"), k=5), "Truy vấn bằng ảnh: bus.jpg",
                 "retrieval_image_bus.png")

    # Đánh giá định lượng trên phần ảnh hoa (có nhãn loài)
    flower_ids = [i for i, m in enumerate(engine.meta) if m["source"] == "flowers"]
    queries = random.Random(SEED).sample(flower_ids, 50)
    p_at_5 = []
    for qi in queries:  # ảnh → ảnh: bỏ chính nó, xem 5 kết quả đầu có cùng loài không
        res = [r for r in engine.search_image(Image.open(config.resolve_path(engine.meta[qi]["path"])), k=6)
               if r["id"] != qi][:5]
        p_at_5.append(np.mean([r["label"] == engine.meta[qi]["label"] for r in res]))
    text_p10 = {}
    for c in classes:  # văn bản → ảnh (zero-shot): "a photo of tulips" → 10 kết quả đầu có đúng loài không
        res = engine.search_text(f"a photo of {c}", k=10)
        text_p10[c] = float(np.mean([r["label"] == c for r in res]))

    ret_metrics = {"image_to_image_precision@5": float(np.mean(p_at_5)), "text_to_image_precision@10": text_p10,
                   "gallery_size": engine.index.ntotal, "model": config.CLIP_MODEL}
    (ART_DIR / "retrieval" / "metrics.json").write_text(json.dumps(ret_metrics, indent=2))
    print(json.dumps(ret_metrics, indent=2))


if __name__ == "__main__":
    main()
