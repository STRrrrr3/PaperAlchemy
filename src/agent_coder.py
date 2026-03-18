import html
import re
import shutil
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agent_coder_critic import build_coder_critic_router, coder_critic_node
from src.schemas import CoderArtifact, PagePlan, StructuredPaper
from src.state import CoderState

BODY_START_MARKER = "<!-- PaperAlchemy Generated Body Start -->"
BODY_END_MARKER = "<!-- PaperAlchemy Generated Body End -->"

RUNTIME_STYLE = """
<style id="paperalchemy-runtime">
  :root {
    --pa-bg: #f7f8fa;
    --pa-card: #ffffff;
    --pa-text: #1f2937;
    --pa-muted: #6b7280;
    --pa-accent: #2563eb;
    --pa-border: #e5e7eb;
  }
  .pa-shell {
    max-width: 1080px;
    margin: 0 auto;
    padding: 28px 16px 56px;
    color: var(--pa-text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  .pa-hero {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5ff 100%);
    border: 1px solid var(--pa-border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
  }
  .pa-hero h1 {
    margin: 0 0 10px;
    font-size: 1.9rem;
    line-height: 1.25;
  }
  .pa-hero p {
    margin: 0;
    color: var(--pa-muted);
    line-height: 1.7;
  }
  .pa-nav {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 14px 0 22px;
  }
  .pa-nav a {
    text-decoration: none;
    color: var(--pa-accent);
    border: 1px solid #bfdbfe;
    background: #eff6ff;
    border-radius: 999px;
    padding: 6px 12px;
    font-size: 0.9rem;
  }
  .pa-block {
    background: var(--pa-card);
    border: 1px solid var(--pa-border);
    border-radius: 12px;
    padding: 18px;
    margin: 16px 0;
  }
  .pa-block h2 {
    margin: 0 0 10px;
    font-size: 1.28rem;
  }
  .pa-meta {
    color: var(--pa-muted);
    font-size: 0.9rem;
    margin-bottom: 10px;
  }
  .pa-block ul {
    margin: 0 0 10px 20px;
    padding: 0;
    line-height: 1.7;
  }
  .pa-media-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin-top: 10px;
  }
  .pa-media-grid figure {
    margin: 0;
    border: 1px solid var(--pa-border);
    border-radius: 8px;
    overflow: hidden;
    background: #fff;
  }
  .pa-media-grid img {
    width: 100%;
    display: block;
    object-fit: contain;
    background: #f9fafb;
  }
  .pa-media-grid figcaption {
    font-size: 0.8rem;
    color: var(--pa-muted);
    padding: 8px;
    border-top: 1px solid var(--pa-border);
  }
</style>
"""


def _normalize_page_plan(plan: Any) -> PagePlan | None:
    if isinstance(plan, PagePlan):
        return plan
    if plan is None:
        return None
    try:
        return PagePlan.model_validate(plan)
    except Exception:
        return None


def _normalize_structured_paper(paper: Any) -> StructuredPaper | None:
    if isinstance(paper, StructuredPaper):
        return paper
    if paper is None:
        return None
    try:
        return StructuredPaper.model_validate(paper)
    except Exception:
        return None


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip())
    slug = slug.strip("-").lower()
    return slug or "section"


def _extract_head_content(html_text: str) -> str:
    match = re.search(r"<head[^>]*>(.*?)</head>", html_text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else ""


def _extract_body_attrs(html_text: str) -> str:
    match = re.search(r"<body([^>]*)>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    attrs = (match.group(1) or "").strip()
    return f" {attrs}" if attrs else ""


def _sanitize_head_content(head_content: str, structured_paper: StructuredPaper) -> str:
    cleaned = head_content
    cleaned = re.sub(r"<title\b[^>]*>.*?</title>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(
        r"<meta\s+[^>]*name=[\"']description[\"'][^>]*>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<meta\s+[^>]*name=[\"']keywords[\"'][^>]*>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<meta\s+[^>]*property=[\"']og:[^\"']+[\"'][^>]*>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    title_tag = f"<title>{html.escape(structured_paper.paper_title)}</title>"
    desc = html.escape(structured_paper.overall_summary[:280])
    description_tag = f'<meta name="description" content="{desc}">'
    keywords_tag = '<meta name="keywords" content="PaperAlchemy, academic paper page">'

    return "\n".join([title_tag, description_tag, keywords_tag, cleaned.strip()])


def _collect_figure_paths(page_plan: PagePlan) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for block in page_plan.blocks:
        for path in block.asset_binding.figure_paths:
            clean = str(path).strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            ordered.append(clean)
    return ordered


def _copy_paper_assets(
    project_root: Path,
    paper_folder_name: str,
    site_dir: Path,
    figure_paths: list[str],
) -> dict[str, str]:
    copied_map: dict[str, str] = {}
    if not figure_paths:
        return copied_map

    source_root = project_root / "data" / "output" / paper_folder_name
    target_dir = site_dir / "assets" / "paper"
    target_dir.mkdir(parents=True, exist_ok=True)

    for idx, rel_path in enumerate(figure_paths, start=1):
        source_path = source_root / rel_path
        if not source_path.exists() or not source_path.is_file():
            continue

        stem = _safe_slug(source_path.stem)[:60]
        suffix = source_path.suffix or ".png"
        target_name = f"{idx:02d}-{stem}{suffix}"
        target_path = target_dir / target_name
        shutil.copy2(source_path, target_path)

        copied_rel = str(target_path.relative_to(site_dir)).replace("\\", "/")
        copied_map[rel_path] = copied_rel

    return copied_map


def _build_caption_lookup(structured_paper: StructuredPaper) -> dict[str, str]:
    captions: dict[str, str] = {}
    for section in structured_paper.sections:
        for fig in section.related_figures:
            key = str(fig.image_path or "").strip()
            if not key or key in captions:
                continue
            captions[key] = str(fig.caption or "").strip()
    return captions


def _build_generated_body(
    structured_paper: StructuredPaper,
    page_plan: PagePlan,
    copied_asset_map: dict[str, str],
) -> str:
    outline_lookup = {item.block_id: item for item in page_plan.page_outline}
    caption_lookup = _build_caption_lookup(structured_paper)

    body_parts: list[str] = [BODY_START_MARKER, RUNTIME_STYLE, '<main class="pa-shell">']
    body_parts.append('<header class="pa-hero">')
    body_parts.append(f"  <h1>{html.escape(structured_paper.paper_title)}</h1>")
    body_parts.append(f"  <p>{html.escape(structured_paper.overall_summary)}</p>")
    body_parts.append("</header>")

    body_parts.append('<nav class="pa-nav" aria-label="Section Navigation">')
    for item in sorted(page_plan.page_outline, key=lambda x: x.order):
        anchor = _safe_slug(item.block_id)
        body_parts.append(
            f'  <a href="#pa-{anchor}">{html.escape(item.title)}</a>'
        )
    body_parts.append("</nav>")

    blocks_sorted = sorted(page_plan.blocks, key=lambda b: b.responsive_rules.mobile_order)
    for block in blocks_sorted:
        anchor = _safe_slug(block.block_id)
        outline = outline_lookup.get(block.block_id)
        headline = block.content_contract.headline or (outline.title if outline else block.block_id)

        body_parts.append(f'<section class="pa-block" id="pa-{anchor}">')
        body_parts.append(f"  <h2>{html.escape(headline)}</h2>")

        if outline and outline.source_sections:
            sec_text = ", ".join(outline.source_sections)
            body_parts.append(f"  <div class=\"pa-meta\">Source sections: {html.escape(sec_text)}</div>")

        body_parts.append("  <ul>")
        for point in block.content_contract.body_points[:6]:
            body_parts.append(f"    <li>{html.escape(point)}</li>")
        body_parts.append("  </ul>")

        media_html: list[str] = []
        for src_rel in block.asset_binding.figure_paths:
            mapped_rel = copied_asset_map.get(src_rel)
            if not mapped_rel:
                continue
            caption = caption_lookup.get(src_rel) or src_rel
            media_html.append(
                "<figure>"
                f'<img src="{html.escape(mapped_rel)}" alt="{html.escape(headline)}">'
                f"<figcaption>{html.escape(caption)}</figcaption>"
                "</figure>"
            )

        if media_html:
            body_parts.append('  <div class="pa-media-grid">')
            for fragment in media_html:
                body_parts.append(f"    {fragment}")
            body_parts.append("  </div>")

        body_parts.append("</section>")

    body_parts.append("</main>")
    body_parts.append(BODY_END_MARKER)
    return "\n".join(body_parts)


def coder_node(state: CoderState) -> dict[str, Any]:
    print(
        f"[PaperAlchemy-Coder] building site "
        f"(attempt {state.get('coder_retry_count', 0) + 1})..."
    )

    page_plan = _normalize_page_plan(state.get("page_plan"))
    structured_paper = _normalize_structured_paper(state.get("structured_paper"))
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not page_plan or not structured_paper or not paper_folder_name:
        print("[PaperAlchemy-Coder] missing page_plan/structured_paper/paper_folder_name.")
        return {}

    project_root = Path(__file__).resolve().parent.parent
    template_root_rel = page_plan.template_selection.selected_root_dir
    template_root = project_root / template_root_rel
    template_entry_rel = page_plan.template_selection.selected_entry_html

    output_dir = project_root / "data" / "output" / paper_folder_name
    site_dir = output_dir / "site"

    if site_dir.exists():
        shutil.rmtree(site_dir)

    if not template_root.exists():
        print(f"[PaperAlchemy-Coder] template root not found: {template_root}")
        return {}

    shutil.copytree(template_root, site_dir)

    entry_html_path = site_dir / template_entry_rel
    if not entry_html_path.exists():
        print(f"[PaperAlchemy-Coder] template entry html not found: {entry_html_path}")
        return {}

    try:
        template_html = entry_html_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        template_html = entry_html_path.read_text(encoding="latin-1")

    head_content = _extract_head_content(template_html)
    # Avoid inheriting template-specific body class names that can leak source project identity.
    _ = _extract_body_attrs(template_html)
    body_attrs = ""
    cleaned_head = _sanitize_head_content(head_content, structured_paper)

    figure_paths = _collect_figure_paths(page_plan)
    copied_asset_map = _copy_paper_assets(
        project_root=project_root,
        paper_folder_name=paper_folder_name,
        site_dir=site_dir,
        figure_paths=figure_paths,
    )

    generated_body = _build_generated_body(
        structured_paper=structured_paper,
        page_plan=page_plan,
        copied_asset_map=copied_asset_map,
    )

    final_html = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        f"{cleaned_head}\n"
        "</head>\n"
        f"<body{body_attrs}>\n"
        f"{generated_body}\n"
        "</body>\n"
        "</html>\n"
    )
    entry_html_path.write_text(final_html, encoding="utf-8")

    artifact = CoderArtifact(
        site_dir=str(site_dir),
        entry_html=str(entry_html_path),
        selected_template_id=page_plan.template_selection.selected_template_id,
        copied_assets=list(copied_asset_map.values()),
        edited_files=[str(entry_html_path.relative_to(site_dir)).replace("\\", "/")],
        notes="v1-clean-body-rewrite: replaced template body with paper-only content.",
    )
    return {"coder_artifact": artifact}


def build_coder_graph(max_retry: int = 1):
    workflow = StateGraph(CoderState)
    workflow.add_node("coder", coder_node)
    workflow.add_node("coder_critic", coder_critic_node)

    workflow.set_entry_point("coder")
    workflow.add_edge("coder", "coder_critic")
    workflow.add_conditional_edges(
        "coder_critic",
        build_coder_critic_router(max_retry=max_retry),
        {"retry": "coder", "end": END},
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_coder_agent(
    paper_folder_name: str,
    structured_data: StructuredPaper,
    page_plan: PagePlan,
    max_retry: int = 1,
) -> CoderArtifact | None:
    app = build_coder_graph(max_retry=max_retry)
    thread = {"configurable": {"thread_id": f"coder_{paper_folder_name}"}}

    initial_state: CoderState = {
        "paper_folder_name": paper_folder_name,
        "structured_paper": structured_data,
        "page_plan": page_plan,
        "coder_feedback_history": [],
        "coder_artifact": None,
        "coder_critic_passed": False,
        "coder_retry_count": 0,
    }

    print("[PaperAlchemy-Coder] running Coder + CoderCritic graph...")
    for _ in app.stream(initial_state, thread):
        pass

    final_state = app.get_state(thread)
    artifact_result = final_state.values.get("coder_artifact")
    normalized_artifact: CoderArtifact | None = None
    if artifact_result is not None:
        try:
            normalized_artifact = CoderArtifact.model_validate(artifact_result)
        except Exception:
            normalized_artifact = None

    if not normalized_artifact or not final_state.values.get("coder_critic_passed"):
        print("[PaperAlchemy-Coder] coder completed but critic did not fully pass.")
        return None

    print(f"[PaperAlchemy-Coder] build completed: {normalized_artifact.entry_html}")
    return normalized_artifact
