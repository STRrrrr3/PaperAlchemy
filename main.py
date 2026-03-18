import json
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ImportError as exc:
    raise RuntimeError("Gradio is required to launch the PaperAlchemy web UI.") from exc

from src.agent_coder import run_coder_agent
from src.agent_planner import run_planner_agent
from src.agent_reader import run_reader_agent
from src.deterministic_template_selector import score_and_select_template
from src.parser import parse_pdf
from src.schemas import CoderArtifact, PagePlan, StructuredPaper
from src.template_resources import ensure_autopage_template_assets

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"


def load_cached_structured_data(path: Path) -> StructuredPaper | None:
    if not path.exists():
        return None

    print("[PaperAlchemy] Found cached structured paper, loading...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        structured_data = StructuredPaper(**data_dict)
        print(f"[PaperAlchemy] Structured paper loaded: {structured_data.paper_title}")
        return structured_data
    except Exception as exc:
        print(f"[PaperAlchemy] Structured cache is invalid, rerunning Reader: {exc}")
        return None


def save_structured_data(path: Path, structured_data: StructuredPaper) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(structured_data.model_dump(), f, indent=2, ensure_ascii=False)


def save_page_plan(path: Path, page_plan: PagePlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(page_plan.model_dump(), f, indent=2, ensure_ascii=False)


def save_coder_artifact(path: Path, artifact: CoderArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact.model_dump(), f, indent=2, ensure_ascii=False)


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


def _build_planner_constraints(tags_json_path: Path) -> dict[str, Any]:
    return {
        "target_framework": "static-html",
        "max_templates_for_planner": 120,
        "max_entry_candidates": 3,
        "template_candidate_top_k": 3,
        "max_blocks": 10,
        "min_blocks": 6,
        "template_tags_json_path": str(tags_json_path),
    }


def _format_selection_summary(
    page_plan: PagePlan,
    selection_preview: dict[str, Any],
) -> dict[str, Any]:
    selected_root_dir = page_plan.template_selection.selected_root_dir
    selected_root_path = (PROJECT_ROOT / selected_root_dir).resolve()

    return {
        "selected_template_id": page_plan.template_selection.selected_template_id,
        "selected_template_path": str(selected_root_path),
        "selected_entry_html": page_plan.template_selection.selected_entry_html,
        "selection_rationale": selection_preview.get("selection_rationale"),
        "score": selection_preview.get("score"),
        "matched_features": selection_preview.get("matched_features"),
        "mismatched_features": selection_preview.get("mismatched_features"),
        "template_tags": selection_preview.get("template_tags"),
        "ranking": list(selection_preview.get("ranking") or [])[:3],
    }


def run_generation_request(
    pdf_filename: str,
    background_color: str,
    density: str,
    navigation: str,
    layout: str,
) -> tuple[str, dict[str, Any], str, str]:
    run_log: list[str] = []

    def log(message: str) -> None:
        print(message)
        run_log.append(message)

    selection_payload: dict[str, Any] = {}
    page_plan_path_text = ""
    entry_html_path_text = ""

    try:
        safe_pdf_filename = str(pdf_filename or "").strip()
        if not safe_pdf_filename:
            raise ValueError("Please select a PDF from data/input before generating.")

        input_path = INPUT_DIR / safe_pdf_filename
        if not input_path.exists():
            raise FileNotFoundError(f"Input PDF not found: {input_path}")

        user_constraints = build_user_constraints(
            background_color=background_color,
            density=density,
            navigation=navigation,
            layout=layout,
        )
        log(f"[UI] user_constraints={json.dumps(user_constraints, ensure_ascii=False)}")

        synced_assets = ensure_autopage_template_assets(PROJECT_ROOT)
        log(f"[Assets] template resources ready at {synced_assets.templates_dir}")

        selection_preview = score_and_select_template(
            user_constraints=user_constraints,
            tags_json_path=synced_assets.tags_json_path,
        )
        log(
            "[Selector] preview selected template="
            f"{selection_preview['selected_template_id']} "
            f"(score={selection_preview['score']})"
        )

        paper_folder_name = Path(safe_pdf_filename).stem
        output_dir = OUTPUT_DIR / paper_folder_name
        structured_json_path = output_dir / "structured_paper.json"
        planner_json_path = output_dir / "page_plan.json"
        coder_json_path = output_dir / "coder_artifact.json"

        if not ensure_parsed_output(safe_pdf_filename, output_dir):
            raise RuntimeError("PDF parsing failed. Please inspect the parser output logs.")

        structured_data = load_cached_structured_data(structured_json_path)
        if not structured_data:
            log("[Reader] running reader agent...")
            structured_data = run_reader_agent(paper_folder_name)
            if not structured_data:
                raise RuntimeError("Reader agent failed to produce structured paper data.")
            save_structured_data(structured_json_path, structured_data)
            log(f"[Reader] saved structured paper to {structured_json_path}")
        else:
            log(f"[Reader] reused cached structured paper from {structured_json_path}")

        log("[Planner] running template-first planner graph...")
        page_plan = run_planner_agent(
            paper_folder_name=paper_folder_name,
            structured_data=structured_data,
            generation_constraints=_build_planner_constraints(synced_assets.tags_json_path),
            user_constraints=user_constraints,
            max_retry=2,
        )
        if not page_plan:
            raise RuntimeError("Planner agent failed to produce a page plan.")

        save_page_plan(planner_json_path, page_plan)
        page_plan_path_text = str(planner_json_path)
        log(
            "[Planner] saved page plan with template "
            f"{page_plan.template_selection.selected_template_id}"
        )

        log("[Coder] running coder agent...")
        coder_artifact = run_coder_agent(
            paper_folder_name=paper_folder_name,
            structured_data=structured_data,
            page_plan=page_plan,
            max_retry=1,
        )
        if not coder_artifact:
            raise RuntimeError("Coder agent failed to build the final webpage.")

        save_coder_artifact(coder_json_path, coder_artifact)
        entry_html_path_text = coder_artifact.entry_html
        log(f"[Coder] generated entry html at {coder_artifact.entry_html}")

        selection_payload = _format_selection_summary(page_plan, selection_preview)
        log("[Done] generation completed successfully.")
    except Exception as exc:
        log(f"[Error] {exc}")

    return "\n".join(run_log), selection_payload, page_plan_path_text, entry_html_path_text


def build_app() -> gr.Blocks:
    available_pdfs = list_available_pdfs()
    default_pdf = available_pdfs[0] if available_pdfs else None

    with gr.Blocks(title="PaperAlchemy Template-First Generator") as demo:
        gr.Markdown(
            """
            # PaperAlchemy
            Template selection now happens before LangGraph planning. Choose a local PDF and the visual constraints,
            then PaperAlchemy will deterministically select a local template and run the full pipeline.
            """
        )

        pdf_dropdown = gr.Dropdown(
            choices=available_pdfs,
            value=default_pdf,
            allow_custom_value=True,
            label="Paper PDF",
        )

        with gr.Row():
            background_color = gr.Radio(
                choices=["light", "dark"],
                value="light",
                label="Background Color",
            )
            density = gr.Radio(
                choices=["spacious", "compact"],
                value="spacious",
                label="Density",
            )

        with gr.Row():
            navigation = gr.Radio(
                choices=["yes", "no"],
                value="no",
                label="Navigation",
            )
            layout = gr.Radio(
                choices=["parallelism", "rotation"],
                value="parallelism",
                label="Layout",
            )

        generate_button = gr.Button("Generate", variant="primary")

        run_log = gr.Textbox(label="Run Log", lines=18)
        selection_json = gr.JSON(label="Selected Template")
        page_plan_path = gr.Textbox(label="Saved Page Plan Path")
        entry_html_path = gr.Textbox(label="Generated Entry HTML Path")

        generate_button.click(
            fn=run_generation_request,
            inputs=[pdf_dropdown, background_color, density, navigation, layout],
            outputs=[run_log, selection_json, page_plan_path, entry_html_path],
            api_name="generate",
        )

    return demo


def main() -> None:
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
