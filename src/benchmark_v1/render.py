from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError:
    PlaywrightTimeoutError = RuntimeError
    sync_playwright = None


BENCHMARK_VIEWPORT_WIDTH = 1280
BENCHMARK_VIEWPORT_HEIGHT = 2400
BENCHMARK_DEVICE_SCALE_FACTOR = 1.0
BENCHMARK_USER_AGENT = (
    "PaperAlchemyBenchmarkV1/1.0 "
    "(deterministic-render; Chromium; viewport=1280x2400; dpr=1)"
)
BENCHMARK_WAIT_UNTIL = "networkidle"
BENCHMARK_NAVIGATION_TIMEOUT_MS = 45000
BENCHMARK_SETTLE_WAIT_MS = 1500


def benchmark_render_config() -> dict[str, Any]:
    return {
        "viewport": {
            "width": BENCHMARK_VIEWPORT_WIDTH,
            "height": BENCHMARK_VIEWPORT_HEIGHT,
        },
        "device_scale_factor": BENCHMARK_DEVICE_SCALE_FACTOR,
        "user_agent": BENCHMARK_USER_AGENT,
        "wait_until": BENCHMARK_WAIT_UNTIL,
        "navigation_timeout_ms": BENCHMARK_NAVIGATION_TIMEOUT_MS,
        "settle_wait_ms": BENCHMARK_SETTLE_WAIT_MS,
        "full_page": True,
    }


def take_benchmark_screenshot(html_absolute_path: str, output_image_path: str) -> str:
    html_path = Path(html_absolute_path).absolute()
    image_path = Path(output_image_path).absolute()

    if not html_path.exists():
        print(f"[BenchmarkV1] Screenshot skipped: HTML file not found at {html_path}")
        return ""
    if sync_playwright is None:
        print("[BenchmarkV1] Screenshot skipped: playwright is not installed.")
        return ""

    image_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={
                    "width": BENCHMARK_VIEWPORT_WIDTH,
                    "height": BENCHMARK_VIEWPORT_HEIGHT,
                },
                device_scale_factor=BENCHMARK_DEVICE_SCALE_FACTOR,
                user_agent=BENCHMARK_USER_AGENT,
            )
            page = context.new_page()
            try:
                page.goto(
                    html_path.as_uri(),
                    wait_until=BENCHMARK_WAIT_UNTIL,
                    timeout=BENCHMARK_NAVIGATION_TIMEOUT_MS,
                )
            except PlaywrightTimeoutError as exc:
                print(
                    "[BenchmarkV1] networkidle timeout for "
                    f"{html_path.as_uri()}: {exc}. Capturing best-effort screenshot."
                )
            _settle_page(page)
            page.screenshot(path=str(image_path), full_page=True)
            context.close()
            browser.close()
            return str(image_path)
    except Exception as exc:
        print(f"[BenchmarkV1] Screenshot failed for {html_path}: {exc}")
        return ""


def _settle_page(page: Any) -> None:
    try:
        page.evaluate(
            """
            async () => {
              if (document.fonts && document.fonts.ready) {
                try {
                  await document.fonts.ready;
                } catch (err) {
                }
              }
              if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
                try {
                  await window.MathJax.typesetPromise();
                } catch (err) {
                }
              }
            }
            """
        )
    except Exception:
        pass
    try:
        page.wait_for_timeout(BENCHMARK_SETTLE_WAIT_MS)
    except Exception:
        pass
