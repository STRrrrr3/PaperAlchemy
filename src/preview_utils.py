import hashlib
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


def build_binding_template_preview_path(entry_html_path: Path, block_id: str) -> Path:
    preview_dir = PREVIEW_CACHE_DIR / "binding"
    preview_dir.mkdir(parents=True, exist_ok=True)
    entry_name = _sanitize_preview_name(entry_html_path.stem)
    block_name = _sanitize_preview_name(block_id)
    return preview_dir / f"{entry_name}_{block_name}_template.png"


def build_binding_candidate_preview_path(entry_html_path: Path, block_id: str, rank: int) -> Path:
    preview_dir = PREVIEW_CACHE_DIR / "binding"
    preview_dir.mkdir(parents=True, exist_ok=True)
    entry_name = _sanitize_preview_name(entry_html_path.stem)
    block_name = _sanitize_preview_name(block_id)
    return preview_dir / f"{entry_name}_{block_name}_candidate_{rank}.png"


def build_layout_compose_template_preview_path(entry_html_path: Path) -> Path:
    preview_dir = PREVIEW_CACHE_DIR / "layout_compose"
    preview_dir.mkdir(parents=True, exist_ok=True)
    entry_name = _sanitize_preview_name(entry_html_path.stem)
    return preview_dir / f"{entry_name}_template.png"


def build_layout_compose_section_preview_path(entry_html_path: Path, selector_hint: str) -> Path:
    preview_dir = PREVIEW_CACHE_DIR / "layout_compose"
    preview_dir.mkdir(parents=True, exist_ok=True)
    entry_name = _sanitize_preview_name(entry_html_path.stem)
    selector_hash = hashlib.sha1(str(selector_hint or "").encode("utf-8")).hexdigest()[:12]
    return preview_dir / f"{entry_name}_{selector_hash}.png"


def build_paper_figure_preview_path(paper_folder_name: str, figure_path: str) -> str:
    clean_folder = str(paper_folder_name or "").strip()
    clean_figure_path = str(figure_path or "").strip().replace("\\", "/")
    if not clean_folder or not clean_figure_path:
        return ""

    absolute_path = (OUTPUT_DIR / clean_folder / clean_figure_path).resolve()
    if absolute_path.exists() and absolute_path.is_file():
        return str(absolute_path)
    return ""


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


def take_labeled_template_screenshot(
    html_absolute_path: str,
    selector_labels: list[dict[str, str]],
    output_image_path: str,
) -> str:
    html_path = Path(html_absolute_path).absolute()
    image_path = Path(output_image_path).absolute()

    if not html_path.exists():
        print(f"[Preview] Labeled screenshot skipped: HTML file not found at {html_path}")
        return ""

    if sync_playwright is None:
        print("[Preview] Labeled screenshot skipped: playwright is not installed.")
        return ""

    labels = [
        {
            "selector": str(item.get("selector") or "").strip(),
            "label": str(item.get("label") or "").strip(),
        }
        for item in selector_labels
        if str(item.get("selector") or "").strip() and str(item.get("label") or "").strip()
    ]
    if not labels:
        return take_local_screenshot(str(html_path), str(image_path))

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
                print(f"[Preview] networkidle timeout for {target_uri}: {exc}. Capturing best-effort labeled screenshot.")
                try:
                    page.wait_for_timeout(1500)
                except Exception:
                    pass

            page.evaluate(
                """
                (items) => {
                  const overlay = document.createElement('div');
                  overlay.setAttribute('data-paperalchemy-layout-overlay', 'true');
                  overlay.style.position = 'absolute';
                  overlay.style.top = '0';
                  overlay.style.left = '0';
                  overlay.style.width = '100%';
                  overlay.style.height = '100%';
                  overlay.style.pointerEvents = 'none';
                  overlay.style.zIndex = '2147483647';

                  const body = document.body || document.documentElement;
                  if (!body) {
                    return;
                  }
                  if (getComputedStyle(body).position === 'static') {
                    body.style.position = 'relative';
                  }

                  items.forEach((item) => {
                    const element = document.querySelector(item.selector);
                    if (!element) {
                      return;
                    }
                    const rect = element.getBoundingClientRect();
                    if (!rect || rect.width <= 0 || rect.height <= 0) {
                      return;
                    }

                    const frame = document.createElement('div');
                    frame.style.position = 'absolute';
                    frame.style.left = `${window.scrollX + rect.left}px`;
                    frame.style.top = `${window.scrollY + rect.top}px`;
                    frame.style.width = `${rect.width}px`;
                    frame.style.height = `${rect.height}px`;
                    frame.style.border = '3px solid rgba(229, 72, 77, 0.92)';
                    frame.style.borderRadius = '14px';
                    frame.style.boxSizing = 'border-box';
                    frame.style.background = 'rgba(229, 72, 77, 0.04)';

                    const badge = document.createElement('div');
                    badge.textContent = item.label;
                    badge.style.position = 'absolute';
                    badge.style.left = `${window.scrollX + rect.left + 8}px`;
                    badge.style.top = `${window.scrollY + rect.top + 8}px`;
                    badge.style.padding = '4px 10px';
                    badge.style.borderRadius = '999px';
                    badge.style.background = 'rgba(15, 23, 42, 0.92)';
                    badge.style.color = '#ffffff';
                    badge.style.fontSize = '14px';
                    badge.style.fontWeight = '700';
                    badge.style.fontFamily = 'Consolas, monospace';
                    badge.style.letterSpacing = '0.02em';

                    overlay.appendChild(frame);
                    overlay.appendChild(badge);
                  });

                  body.appendChild(overlay);
                }
                """,
                labels,
            )
            page.screenshot(path=str(image_path), full_page=True)
            context.close()
            browser.close()
            return str(image_path)
    except Exception as exc:
        print(
            "[Preview] Playwright labeled screenshot failed for "
            f"{html_path}: {exc}."
        )
        fallback_path = take_local_screenshot(str(html_path), str(image_path))
        return fallback_path or ""


def take_selector_screenshot(html_absolute_path: str, selector: str, output_image_path: str) -> str:
    html_path = Path(html_absolute_path).absolute()
    image_path = Path(output_image_path).absolute()

    if not html_path.exists():
        print(f"[Preview] Selector screenshot skipped: HTML file not found at {html_path}")
        return ""

    if sync_playwright is None:
        print("[Preview] Selector screenshot skipped: playwright is not installed.")
        return ""

    clean_selector = str(selector or "").strip()
    if not clean_selector:
        print("[Preview] Selector screenshot skipped: selector is empty.")
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
                print(f"[Preview] networkidle timeout for {target_uri}: {exc}. Capturing best-effort selector screenshot.")
                try:
                    page.wait_for_timeout(1500)
                except Exception:
                    pass

            locator = page.locator(clean_selector).first
            if locator.count() == 0:
                print(f"[Preview] Selector screenshot skipped: no element matched {clean_selector}")
                context.close()
                browser.close()
                return ""
            locator.screenshot(path=str(image_path))
            context.close()
            browser.close()
            return str(image_path)
    except Exception as exc:
        print(
            "[Preview] Playwright selector screenshot failed for "
            f"{html_path} selector={clean_selector}: {exc}."
        )
        return ""
