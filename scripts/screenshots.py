"""Chụp ảnh giao diện React và Streamlit cho README (cần API + Streamlit đang chạy).

python scripts/screenshots.py            # mặc định React http://localhost:8010, Streamlit http://localhost:8511
"""
import os

from playwright.sync_api import sync_playwright
from ultralytics.utils import ASSETS

from common import FLOWERS_DIR, ROOT

REACT = os.environ.get("REACT_URL", "http://localhost:8010")
STREAMLIT = os.environ.get("STREAMLIT_URL", "http://localhost:8511")
OUT = ROOT / "docs" / "screenshots"
FLOWER = str(sorted((FLOWERS_DIR / "sunflowers").glob("*.jpg"))[5])
BUS = str(ASSETS / "bus.jpg")


def shot(page, name):
    page.screenshot(path=str(OUT / name), full_page=True)
    print("📸", (OUT / name).relative_to(ROOT))


def react(page):
    page.goto(REACT)
    page.get_by_text("đang chạy").wait_for()

    page.locator("input[type=file]").set_input_files(FLOWER)
    page.locator(".bar").first.wait_for()
    shot(page, "react_classify.png")

    page.get_by_role("tab", name="Phát hiện đối tượng").click()
    page.locator("input[type=file]").set_input_files(BUS)
    page.locator("table tbody tr").first.wait_for()
    shot(page, "react_detect.png")

    page.get_by_role("tab", name="Tìm kiếm ảnh").click()
    page.get_by_role("button", name="Tìm").click()
    page.locator(".gallery img").first.wait_for()
    page.wait_for_function("[...document.querySelectorAll('.gallery img')].every(i => i.complete)")
    shot(page, "react_search.png")

    page.get_by_role("tab", name="Chatbot RAG").click()
    page.get_by_label("Câu hỏi").fill("Đổi trả trong bao lâu và cần điều kiện gì?")
    page.get_by_role("button", name="Gửi").click()
    page.get_by_role("button", name="Gửi").wait_for(timeout=120_000)  # nút "Dừng" đổi lại thành "Gửi" khi stream xong
    page.locator("details summary").first.click()
    shot(page, "react_chat.png")


def streamlit(page):
    page.goto(STREAMLIT)
    page.get_by_text("AI Web Apps").first.wait_for(timeout=60_000)
    page.locator("input[type=file]").first.set_input_files(FLOWER)
    page.get_by_text(" ms").first.wait_for(timeout=60_000)
    page.wait_for_timeout(1500)
    shot(page, "streamlit_classify.png")

    page.get_by_role("tab", name="💬 Chatbot").click()
    page.get_by_placeholder("Nhập câu hỏi…").fill("Đơn bao nhiêu tiền thì được miễn phí giao hàng?")
    page.keyboard.press("Enter")
    page.get_by_text("Nguồn đã dùng").wait_for(timeout=120_000)
    page.wait_for_timeout(1500)
    shot(page, "streamlit_chat.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 860}, device_scale_factor=1)
        react(page)
        streamlit(page)
        browser.close()


if __name__ == "__main__":
    main()
