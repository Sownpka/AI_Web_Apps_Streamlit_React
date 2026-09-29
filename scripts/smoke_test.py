"""Mục 6 — Kiểm thử khói toàn bộ API đang chạy (mặc định http://localhost:8000). Chạy lại mỗi khi sửa code."""
import json
import os
import sys

import requests
from ultralytics.utils import ASSETS

from common import FLOWERS_DIR

API = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")


def main():
    print("health:", requests.get(f"{API}/api/health", timeout=10).json())
    flower_img = open(next((FLOWERS_DIR / "sunflowers").glob("*.jpg")), "rb").read()
    bus_img = open(ASSETS / "bus.jpg", "rb").read()

    r = requests.post(f"{API}/api/classify", files={"file": flower_img}, data={"top_k": 3}); r.raise_for_status()
    print("classify:", r.json()["predictions"][0], r.headers["X-Process-Time-ms"], "ms")

    r = requests.post(f"{API}/api/detect", files={"file": bus_img}, data={"conf": 0.3}); r.raise_for_status()
    print("detect:", r.json()["summary"])

    r = requests.post(f"{API}/api/search/text", json={"query": "a red flower", "k": 3}); r.raise_for_status()
    print("search/text:", [(x["label"], x["score"]) for x in r.json()["results"]])
    assert requests.get(API + r.json()["results"][0]["url"]).headers["content-type"].startswith("image/")

    r = requests.post(f"{API}/api/search/image", files={"file": flower_img}, data={"k": 3}); r.raise_for_status()
    print("search/image:", [(x["label"], x["score"]) for x in r.json()["results"]])

    r = requests.post(f"{API}/api/chat/sync", json={"message": "Phí giao hàng cho đơn 200.000đ là bao nhiêu?"}, timeout=120)
    r.raise_for_status()
    print("chat:", r.json()["answer"][:200])

    # Streaming: đọc các sự kiện SSE
    with requests.post(f"{API}/api/chat", json={"message": "Bảo hành đồ gia dụng bao lâu?"}, stream=True, timeout=120) as s:
        s.encoding = "utf-8"
        events = [json.loads(line[6:]) for line in s.iter_lines(decode_unicode=True) if line.startswith("data: ")]
    print("chat SSE:", [e["type"] for e in events][:5], "…", "".join(e.get("text", "") for e in events)[:120])

    # Các ca lỗi phải trả mã HTTP đúng
    assert requests.post(f"{API}/api/classify", files={"file": b"not an image"}).status_code == 400
    assert requests.post(f"{API}/api/search/text", json={"query": ""}).status_code == 422
    print("✅ Tất cả smoke test đạt")


if __name__ == "__main__":
    sys.exit(main())
