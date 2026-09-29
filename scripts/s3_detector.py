"""Mục 3 — Ứng dụng 2: YOLO11n pretrained COCO, chạy thử và đo mAP trên COCO128."""
import json
import time

from common import ART_DIR, coco_images, plt, save_fig
from PIL import Image
from ultralytics.utils import ASSETS
from ultralytics.utils.downloads import attempt_download_asset

import core.detector as det


def main():
    attempt_download_asset(str(ART_DIR / "detector" / "yolo11n.pt"))  # tải trọng số vào artifacts/detector/
    detector = det.ObjectDetector()
    images = coco_images()

    demo_images = [ASSETS / "bus.jpg", ASSETS / "zidane.jpg", images[10], images[42]]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    outputs = []
    for ax, p in zip(axes, demo_images):
        t0 = time.perf_counter(); result, annotated = detector.detect(Image.open(p), conf=0.25)
        ms = (time.perf_counter() - t0) * 1000
        outputs.append(result)
        ax.imshow(annotated); ax.axis("off"); ax.set_title(f"{result['summary']}\n{ms:.0f} ms", fontsize=9)
    save_fig("detector_samples.png")
    print("JSON trả về cho bus.jpg (3 đối tượng đầu):")
    print(json.dumps(outputs[0]["detections"][:3], indent=2))

    # Đánh giá định lượng trên COCO128 (có nhãn) — mAP là chỉ số chuẩn của object detection
    val = detector.model.val(data="coco128.yaml", imgsz=640, batch=16, device=detector.device, plots=False,
                             verbose=False, workers=0)
    det_metrics = {"mAP50": float(val.box.map50), "mAP50_95": float(val.box.map), "dataset": "coco128", "model": "yolo11n"}
    (ART_DIR / "detector" / "metrics.json").write_text(json.dumps(det_metrics, indent=2))
    print(det_metrics)


if __name__ == "__main__":
    main()
