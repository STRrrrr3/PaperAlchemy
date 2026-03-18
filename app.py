import html
import json
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote, urlparse

try:
    import gradio as gr
except ImportError as exc:
    raise RuntimeError("Gradio is required to launch the PaperAlchemy web UI.") from exc

from src.agent_coder import run_coder_agent
from src.agent_planner import run_planner_agent
from src.agent_reader import run_reader_agent
from src.deterministic_template_selector import score_and_select_templates
from src.parser import parse_pdf
from src.schemas import CoderArtifact, PagePlan, StructuredPaper
from src.template_resources import SyncedTemplateAssets, ensure_autopage_template_assets

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
TEMPLATE_TOP_K = 5

WINDOWS_ABS_PATH_PATTERN = re.compile(r"^[a-zA-Z]:[\\/]")
LOCAL_ASSET_ATTR_PATTERN = re.compile(
    r'(?P<prefix>\b(?:src|href|poster)\s*=\s*)(?P<quote>["\'])(?P<url>[^"\']+)(?P=quote)',
    flags=re.IGNORECASE,
)
INLINE_STYLE_URL_PATTERN = re.compile(
    r'url\((?P<quote>["\']?)(?P<url>[^)"\']+)(?P=quote)\)',
    flags=re.IGNORECASE,
)

APP_CSS = """
#paperalchemy-preview {
  min-height: 78vh;
  border: 1px solid #d9dde7;
  border-radius: 16px;
  overflow: auto;
  background: #ffffff;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
}

#paperalchemy-preview > div {
  min-height: 78vh;
}

#paperalchemy-logs textarea {
  font-family: Consolas, "SFMono-Regular", monospace;
}
"""


def load_cached_structured_data(path: Path) -> StructuredPaper | None:
    if not path.exists():
        return None

    print("[PaperAlchemy] Found cached structured paper, loading...")
    try:
        with open(path, "r", encoding="utf-8") as file:
            data_dict = json.load(file)
        structured_data = StructuredPaper(**data_dict)
        print(f"[PaperAlchemy] Structured paper loaded: {structured_data.paper_title}")
        return structured_data
    except Exception as exc:
        print(f"[PaperAlchemy] Structured cache is invalid, rerunning Reader: {exc}")
        return None


def save_structured_data(path: Path, structured_data: StructuredPaper) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(structured_data.model_dump(), file, indent=2, ensure_ascii=False)


def save_page_plan(path: Path, page_plan: PagePlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(page_plan.model_dump(), file, indent=2, ensure_ascii=False)


def save_coder_artifact(path: Path, artifact: CoderArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(artifact.model_dump(), file, indent=2, ensure_ascii=False)


def ensure_parsed_output(pdf_filename: str, output_dir: Path) -> bool:
    output_md_path = output_dir / "full_paper.md"
    parsed_json_path = output_dir / "parsed_data.json"
    if output_md_path.exists() and parsed_json_path.exists():
        print("[PaperAlchemy] Parsed PDF assets already exist, skipping parser.")
        return True

    print("[PaperAlchemy] Parsing PDF...")
    parse_pdf(pdf_filename)

    if output_md_path.exists() and parsed_json_path.exists():
        return True

    print("[PaperAlchemy] Parser output is incomplete: missing full_paper.md or parsed_data.json.")
    return False


def list_available_pdfs() -> list[str]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(path.name for path in INPUT_DIR.glob("*.pdf"))


def get_default_pdf() -> str:
    available_pdfs = list_available_pdfs()
    if not available_pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {INPUT_DIR}. Add a paper to data/input before starting generation."
        )
    return available_pdfs[0]


def build_user_constraints(
    background_color: str,
    density: str,
    navigation: str,
    layout: str,
) -> dict[str, str]:
    return {
        "background_color": background_color,
        "page_density": density,
        "has_navigation": navigation,
        "image_layout": layout,
    }


def build_planner_constraints(tags_json_path: Path, top_k: int = TEMPLATE_TOP_K) -> dict[str, Any]:
    return {
        "target_framework": "static-html",
        "max_templates_for_planner": 120,
        "max_entry_candidates": 3,
        "template_candidate_top_k": max(1, int(top_k)),
        "max_blocks": 10,
        "min_blocks": 6,
        "template_tags_json_path": str(tags_json_path),
    }


def append_log_lines(existing_text: str, new_lines: list[str]) -> str:
    merged = [line for line in str(existing_text or "").splitlines() if line.strip()]
    merged.extend(line for line in new_lines if str(line).strip())
    return "\n".join(merged)


def read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _is_external_url(url: str) -> bool:
    lower_url = url.lower()
    return lower_url.startswith(
        ("http://", "https://", "data:", "blob:", "mailto:", "javascript:", "#", "/file=")
    )


def _resolve_local_asset_path(url: str, base_dir: Path) -> Path | None:
    clean_url = str(url or "").strip()
    if not clean_url or _is_external_url(clean_url):
        return None

    if WINDOWS_ABS_PATH_PATTERN.match(clean_url):
        asset_path = Path(clean_url)
        return asset_path if asset_path.exists() and asset_path.is_file() else None

    parsed = urlparse(clean_url)
    if parsed.scheme or parsed.netloc:
        return None

    asset_path = (base_dir / Path(unquote(parsed.path))).resolve()
    return asset_path if asset_path.exists() and asset_path.is_file() else None


def _to_gradio_file_url(path: Path) -> str:
    normalized = path.resolve().as_posix()
    return f"/file={quote(normalized, safe='/:')}"


def rewrite_local_asset_urls_for_gradio(html_text: str, entry_html_path: Path) -> str:
    base_dir = entry_html_path.parent

    def _replace_attr(match: re.Match[str]) -> str:
        asset_path = _resolve_local_asset_path(match.group("url"), base_dir)
        if not asset_path:
            return match.group(0)
        return f"{match.group('prefix')}{match.group('quote')}{_to_gradio_file_url(asset_path)}{match.group('quote')}"

    rewritten = LOCAL_ASSET_ATTR_PATTERN.sub(_replace_attr, html_text)

    def _replace_style_url(match: re.Match[str]) -> str:
        asset_path = _resolve_local_asset_path(match.group("url"), base_dir)
        if not asset_path:
            return match.group(0)
        return f"url({match.group('quote')}{_to_gradio_file_url(asset_path)}{match.group('quote')})"

    return INLINE_STYLE_URL_PATTERN.sub(_replace_style_url, rewritten)


def load_preview_html(entry_html_path: Path) -> str:
    if not entry_html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {entry_html_path}")
    html_text = read_text_with_fallback(entry_html_path)
    return rewrite_local_asset_urls_for_gradio(html_text, entry_html_path)


def build_placeholder_preview(message: str) -> str:
    safe_message = html.escape(message.strip())
    return f"""
    <div style="padding: 32px; font-family: 'Segoe UI', sans-serif; color: #334155;">
      <h2 style="margin: 0 0 12px;">PaperAlchemy Preview</h2>
      <p style="margin: 0; line-height: 1.7;">{safe_message}</p>
    </div>
    """


def build_candidate_label(candidate: dict[str, Any]) -> str:
    template_name = str(candidate.get("display_name") or candidate.get("template_id") or "unknown-template")
    rank = int(candidate.get("rank") or 0)
    score = float(candidate.get("score") or 0.0)
    max_score = float(candidate.get("max_possible_score") or 0.0)
    return f"{rank}. {template_name} (score {score:.2f}/{max_score:.2f})"


def attach_candidate_labels(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for candidate in candidates:
        enriched = dict(candidate)
        enriched["ui_label"] = build_candidate_label(enriched)
        labeled.append(enriched)
    return labeled


def resolve_selected_candidate(
    selected_label: str | None,
    search_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not selected_label:
        return None

    for candidate in (search_state or {}).get("candidates", []):
        if str(candidate.get("ui_label") or "") == str(selected_label):
            return candidate
    return None


def ensure_template_assets() -> SyncedTemplateAssets:
    return ensure_autopage_template_assets(PROJECT_ROOT)


def run_langgraph_batch(
    pdf_filename: str,
    user_constraints: dict[str, str],
    selected_candidate: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
    log: Callable[[str], None],
) -> CoderArtifact:
    safe_pdf_filename = str(pdf_filename or "").strip()
    if not safe_pdf_filename:
        raise ValueError("A valid PDF filename is required.")

    input_path = INPUT_DIR / safe_pdf_filename
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    template_id = str(selected_candidate.get("template_id") or "").strip()
    template_path = str(selected_candidate.get("template_path") or "").strip()
    entry_html = str(selected_candidate.get("entry_html") or "").strip()
    if not (template_id and template_path and entry_html):
        raise ValueError("The selected template is missing template_id, template_path, or entry_html.")

    log(f"[UI] Using input PDF: {safe_pdf_filename}")
    log(f"[UI] Confirmed template: {template_id}")
    log(f"[UI] Template root: {template_path}")
    log(f"[UI] Entry HTML: {entry_html}")
    log(f"[UI] User constraints: {json.dumps(user_constraints, ensure_ascii=False)}")

    synced_assets = ensure_template_assets()
    log(f"[Assets] Template resources ready at {synced_assets.templates_dir}")

    paper_folder_name = Path(safe_pdf_filename).stem
    output_dir = OUTPUT_DIR / paper_folder_name
    structured_json_path = output_dir / "structured_paper.json"
    planner_json_path = output_dir / "page_plan.json"
    coder_json_path = output_dir / "coder_artifact.json"

    if not ensure_parsed_output(safe_pdf_filename, output_dir):
        raise RuntimeError("PDF parsing failed. Please inspect the parser output logs.")

    structured_data = load_cached_structured_data(structured_json_path)
    if not structured_data:
        log("[Reader] Running reader agent...")
        structured_data = run_reader_agent(paper_folder_name)
        if not structured_data:
            raise RuntimeError("Reader agent failed to produce structured paper data.")
        save_structured_data(structured_json_path, structured_data)
        log(f"[Reader] Saved structured paper to {structured_json_path}")
    else:
        log(f"[Reader] Reused cached structured paper from {structured_json_path}")

    planner_candidates = [
        {
            "template_id": item.get("template_id"),
            "template_name": item.get("template_name"),
            "template_path": item.get("template_path"),
            "entry_html": item.get("entry_html"),
            "entry_html_path": item.get("entry_html_path"),
            "score": item.get("score", 0.0),
            "max_possible_score": item.get("max_possible_score", 0.0),
            "matched_features": list(item.get("matched_features") or []),
            "mismatched_features": list(item.get("mismatched_features") or []),
            "template_tags": dict(item.get("template_tags") or {}),
            "reasons": list(item.get("reasons") or []),
        }
        for item in ranked_candidates
    ]

    generation_constraints = {
        **build_planner_constraints(
            synced_assets.tags_json_path,
            top_k=max(TEMPLATE_TOP_K, len(planner_candidates)),
        ),
        "designated_template_id": template_id,
        "designated_template_path": template_path,
        "designated_entry_html": entry_html,
        "designated_template_score": float(selected_candidate.get("score") or 0.0),
        "ui_ranked_candidates": planner_candidates,
    }

    log("[Planner] Running template-first planner graph with designated template...")
    page_plan = run_planner_agent(
        paper_folder_name=paper_folder_name,
        structured_data=structured_data,
        generation_constraints=generation_constraints,
        user_constraints=user_constraints,
        max_retry=2,
    )
    if not page_plan:
        raise RuntimeError("Planner agent failed to produce a page plan.")

    save_page_plan(planner_json_path, page_plan)
    log(f"[Planner] Saved page plan to {planner_json_path}")

    log("[Coder] Running coder agent...")
    coder_artifact = run_coder_agent(
        paper_folder_name=paper_folder_name,
        structured_data=structured_data,
        page_plan=page_plan,
        max_retry=1,
    )
    if not coder_artifact:
        raise RuntimeError("Coder agent failed to build the final webpage.")

    save_coder_artifact(coder_json_path, coder_artifact)
    log(f"[Coder] Generated entry html at {coder_artifact.entry_html}")
    log("[Done] Generation completed successfully.")
    return coder_artifact


def find_templates(
    background_color: str,
    density: str,
    navigation: str,
    layout: str,
) -> tuple[Any, dict[str, Any], Any, str, str, Any]:
    log_lines: list[str] = []
    preview_html = build_placeholder_preview(
        "Select one of the Top 5 candidates to preview it here before generation."
    )

    try:
        user_constraints = build_user_constraints(
            background_color=background_color,
            density=density,
            navigation=navigation,
            layout=layout,
        )
        synced_assets = ensure_template_assets()
        candidates = score_and_select_templates(
            user_constraints=user_constraints,
            tags_json_path=synced_assets.tags_json_path,
            top_k=TEMPLATE_TOP_K,
        )
        candidates = attach_candidate_labels(candidates)

        if not candidates:
            raise ValueError("No templates matched the provided constraints.")

        search_state = {
            "user_constraints": user_constraints,
            "tags_json_path": str(synced_assets.tags_json_path),
            "candidates": candidates,
        }

        log_lines.append(f"[Search] user_constraints={json.dumps(user_constraints, ensure_ascii=False)}")
        for candidate in candidates:
            log_lines.append(
                f"[Candidate] {candidate['ui_label']} -> {candidate['template_path']}"
            )

        radio_update = gr.update(
            choices=[candidate["ui_label"] for candidate in candidates],
            value=None,
            interactive=True,
        )
        confirm_update = gr.update(interactive=False)
        return radio_update, search_state, None, "\n".join(log_lines), preview_html, confirm_update
    except Exception as exc:
        log_lines.append(f"[Error] {exc}")
        empty_state = {"user_constraints": {}, "tags_json_path": "", "candidates": []}
        radio_update = gr.update(choices=[], value=None, interactive=False)
        confirm_update = gr.update(interactive=False)
        error_preview = build_placeholder_preview(f"Template search failed: {exc}")
        return radio_update, empty_state, None, "\n".join(log_lines), error_preview, confirm_update


def preview_selected_template(
    selected_label: str | None,
    search_state: dict[str, Any] | None,
    current_logs: str,
) -> tuple[str, dict[str, Any] | None, str, Any]:
    candidate = resolve_selected_candidate(selected_label, search_state)
    if not candidate:
        preview_html = build_placeholder_preview(
            "Select one of the Top 5 candidates to preview it here before generation."
        )
        return preview_html, None, current_logs, gr.update(interactive=False)

    try:
        entry_html_path = Path(str(candidate.get("entry_html_path") or ""))
        preview_html = load_preview_html(entry_html_path)
        updated_logs = append_log_lines(
            current_logs,
            [f"[Preview] Loaded template {candidate['template_id']} from {entry_html_path}"],
        )
        return preview_html, candidate, updated_logs, gr.update(interactive=True)
    except Exception as exc:
        updated_logs = append_log_lines(current_logs, [f"[Error] Failed to preview template: {exc}"])
        preview_html = build_placeholder_preview(f"Template preview failed: {exc}")
        return preview_html, None, updated_logs, gr.update(interactive=False)


def confirm_and_start_generation(
    pdf_filename: str | None,
    selected_label: str | None,
    search_state: dict[str, Any] | None,
    selected_candidate_state: dict[str, Any] | None,
    current_logs: str,
) -> tuple[str, str]:
    selected_candidate = selected_candidate_state or resolve_selected_candidate(selected_label, search_state)
    if not selected_candidate:
        raise gr.Error("Select and preview one of the Top 5 candidate templates before starting generation.")

    chosen_pdf = str(pdf_filename or "").strip() or get_default_pdf()
    user_constraints = dict((search_state or {}).get("user_constraints") or {})
    ranked_candidates = list((search_state or {}).get("candidates") or [])

    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    preview_html = build_placeholder_preview("Generation is running. The final webpage will appear here shortly.")

    try:
        coder_artifact = run_langgraph_batch(
            pdf_filename=chosen_pdf,
            user_constraints=user_constraints,
            selected_candidate=selected_candidate,
            ranked_candidates=ranked_candidates,
            log=log,
        )
        preview_html = load_preview_html(Path(coder_artifact.entry_html))
        log(f"[Preview] Loaded generated webpage from {coder_artifact.entry_html}")
    except Exception as exc:
        log(f"[Error] {exc}")
        preview_html = build_placeholder_preview(f"Generation failed: {exc}")

    return "\n".join(run_log_lines), preview_html


def build_app() -> gr.Blocks:
    available_pdfs = list_available_pdfs()
    default_pdf = available_pdfs[0] if available_pdfs else None

    with gr.Blocks(title="PaperAlchemy", css=APP_CSS) as demo:
        search_state = gr.State({"user_constraints": {}, "tags_json_path": "", "candidates": []})
        selected_candidate_state = gr.State(None)

        gr.Markdown("# PaperAlchemy")

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Control Panel")

                pdf_dropdown = gr.Dropdown(
                    choices=available_pdfs,
                    value=default_pdf,
                    label="Paper PDF",
                    allow_custom_value=True,
                )
                background_color = gr.Radio(
                    choices=["light", "dark"],
                    value="light",
                    label="Background Color",
                )
                density = gr.Radio(
                    choices=["spacious", "compact"],
                    value="spacious",
                    label="Page Density",
                )
                navigation = gr.Radio(
                    choices=["yes", "no"],
                    value="no",
                    label="Navigation",
                )
                layout = gr.Radio(
                    choices=["parallelism", "rotation"],
                    value="parallelism",
                    label="Layout Style",
                )

                find_templates_button = gr.Button("Find Templates")
                candidates_radio = gr.Radio(
                    choices=[],
                    label="Top 5 Candidates",
                    interactive=False,
                )
                confirm_button = gr.Button(
                    "Confirm & Start Generation",
                    variant="primary",
                    interactive=False,
                )
                system_logs = gr.Textbox(
                    label="System Logs",
                    value="Choose constraints, click Find Templates, preview a candidate, then confirm to start generation.",
                    lines=24,
                    interactive=False,
                    elem_id="paperalchemy-logs",
                )

            with gr.Column(scale=2):
                preview_html = gr.HTML(
                    value=build_placeholder_preview(
                        "Template previews and the generated webpage will appear here."
                    ),
                    elem_id="paperalchemy-preview",
                )

        find_templates_button.click(
            fn=find_templates,
            inputs=[background_color, density, navigation, layout],
            outputs=[
                candidates_radio,
                search_state,
                selected_candidate_state,
                system_logs,
                preview_html,
                confirm_button,
            ],
            api_name="find_templates",
        )

        candidates_radio.change(
            fn=preview_selected_template,
            inputs=[candidates_radio, search_state, system_logs],
            outputs=[preview_html, selected_candidate_state, system_logs, confirm_button],
            api_name="preview_template",
        )

        confirm_button.click(
            fn=confirm_and_start_generation,
            inputs=[pdf_dropdown, candidates_radio, search_state, selected_candidate_state, system_logs],
            outputs=[system_logs, preview_html],
            api_name="confirm_generation",
        )

    return demo


def main() -> None:
    allowed_paths = [str(OUTPUT_DIR.resolve())]
    try:
        synced_assets = ensure_template_assets()
        allowed_paths.append(str(synced_assets.templates_dir.resolve()))
    except Exception:
        pass

    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":
    main()
