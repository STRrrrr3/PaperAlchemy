from pathlib import Path
import re

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError:
    PlaywrightTimeoutError = RuntimeError
    sync_playwright = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
PREVIEW_CACHE_DIR = OUTPUT_DIR / "_preview_cache"


def _sanitize_preview_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    return normalized.strip("._") or "preview"


def build_template_preview_path(candidate: dict) -> Path:
    template_id = _sanitize_preview_name(str(candidate.get("template_id") or "template"))
    entry_name = _sanitize_preview_name(Path(str(candidate.get("entry_html") or "index.html")).stem)
    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PREVIEW_CACHE_DIR / f"{template_id}_{entry_name}.png"


def build_final_preview_path(entry_html_path: Path) -> Path:
    site_dir = entry_html_path.parent
    site_dir.mkdir(parents=True, exist_ok=True)
    return site_dir / "final_render.png"


def build_visual_critic_screenshot_path(entry_html_path: Path) -> Path:
    site_dir = entry_html_path.parent
    site_dir.mkdir(parents=True, exist_ok=True)
    return site_dir / "temp_screenshot.png"


def take_local_screenshot(html_absolute_path: str, output_image_path: str) -> str:
    html_path = Path(html_absolute_path).absolute()
    image_path = Path(output_image_path).absolute()

    if not html_path.exists():
        print(f"[Preview] Screenshot skipped: HTML file not found at {html_path}")
        return ""

    if sync_playwright is None:
        print("[Preview] Screenshot skipped: playwright is not installed.")
        return ""

    image_path.parent.mkdir(parents=True, exist_ok=True)
    target_uri = html_path.as_uri()

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()

            try:
                page.goto(target_uri, wait_until="networkidle", timeout=45000)
            except PlaywrightTimeoutError as exc:
                print(f"[Preview] networkidle timeout for {target_uri}: {exc}. Capturing best-effort screenshot.")
                try:
                    page.wait_for_timeout(1500)
                except Exception:
                    pass

            page.screenshot(path=str(image_path), full_page=True)
            context.close()
            browser.close()
            return str(image_path)
    except Exception as exc:
        print(
            "[Preview] Playwright screenshot failed for "
            f"{html_path}: {exc}. Ensure `playwright` is installed and Chromium is available."
        )
        return ""
