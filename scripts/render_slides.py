"""Xuất slide (docs/slides/src/*.html) thành ảnh PNG 1920×1080 và một file PDF để nhúng vào README.

python scripts/render_slides.py
"""
import json
import re

from playwright.sync_api import sync_playwright

from common import ROOT

SRC = ROOT / "docs" / "slides" / "src"
OUT = ROOT / "docs" / "slides"

# Ảnh trong slide trỏ tới asset của bản trình chiếu online → đổi sang ảnh chụp giao diện trong dự án
BLOBS = {
    "3c7ada8d7bf150ddf35f836d09b49aa2": "react_classify.png",
    "a4d13a1e393f6c5ba03902d8e684bb60": "react_detect.png",
    "f4a567351dac08603ffd26a62bfb7d5b": "react_search.png",
    "3de32b06d167c599df49461349fbbc75": "react_chat.png",
}
ARROW_SVG = ('<svg width="72" height="36" viewBox="0 0 72 36"><polygon '
             'points="0,10 43,10 43,0 72,18 43,36 43,26 0,26" fill="{color}"/></svg>')
PAGE = """<!doctype html><html><head><meta charset="utf-8">
{links}
<style>
* {{ margin: 0; box-sizing: border-box; }}
html, body {{ width: 1920px; }}
section {{ position: relative; width: 1920px; height: 1080px; overflow: hidden; break-after: page; }}
ul {{ padding-left: 1.2em; }}
h1, h2, h3 {{ font-weight: 600; }}
@page {{ size: 1920px 1080px; margin: 0; }}
</style></head><body>{body}</body></html>"""


def to_browser_html(slide: str) -> str:
    slide = re.sub(r"<aside>.*?</aside>", "", slide, flags=re.S)  # ghi chú người nói không hiện trên ảnh
    for blob, name in BLOBS.items():
        slide = slide.replace(f"/_blob/{blob}", (ROOT / "docs" / "screenshots" / name).as_uri())

    def arrow(m):
        color = re.search(r"background:\s*(#[0-9A-Fa-f]{6})", m.group(0))
        return ARROW_SVG.format(color=color.group(1) if color else "#0B6E99")

    return re.sub(r'<x-shape kind="arrow-right"[^>]*></x-shape>', arrow, slide)


def main():
    deck = json.loads((SRC / "deck.json").read_text(encoding="utf-8"))
    links = "\n".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in deck["faces"].values() if "href" in f)
    slides = [to_browser_html((SRC / f"{sid}.html").read_text(encoding="utf-8")) for sid in deck["order"]]
    html_path = OUT / "deck_preview.html"
    html_path.write_text(PAGE.format(links=links, body="\n".join(slides)), encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        for i, sid in enumerate(deck["order"], 1):
            path = OUT / f"{i:02d}_{sid}.png"
            page.locator(f"section#{sid}").screenshot(path=str(path))
            print("🖼", path.relative_to(ROOT))
        page.pdf(path=str(OUT / "slides.pdf"), width="1920px", height="1080px", print_background=True)
        print("📄", (OUT / "slides.pdf").relative_to(ROOT))
        browser.close()
    html_path.unlink()


if __name__ == "__main__":
    main()
