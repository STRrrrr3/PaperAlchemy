import json
import re
from pathlib import Path

from src.agent_coder import run_coder_agent
from src.agent_planner import run_planner_agent
from src.agent_reader import run_reader_agent
from src.parser import parse_pdf
from src.schemas import CoderArtifact, PagePlan, StructuredPaper

PROJECT_ROOT = Path(__file__).resolve().parent


def load_cached_structured_data(path: Path) -> StructuredPaper | None:
    if not path.exists():
        return None

    print("[PaperAlchemy]发现本地已有结构化存档，正在加载...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        structured_data = StructuredPaper(**data_dict)
        print(f"[PaperAlchemy]成功加载存档: {structured_data.paper_title}")
        return structured_data
    except Exception as e:
        print(f"[PaperAlchemy]🤡存档损坏，将重新运行 Reader: {e}🤡")
        return None


def load_cached_page_plan(path: Path) -> PagePlan | None:
    if not path.exists():
        return None

    print("[PaperAlchemy]发现本地已有页面规划存档，正在加载...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            plan_dict = json.load(f)
        page_plan = PagePlan(**plan_dict)
        print(
            "[PaperAlchemy]成功加载页面规划: "
            f"{page_plan.template_selection.selected_template_id}"
        )
        return page_plan
    except Exception as e:
        print(f"[PaperAlchemy]页面规划存档损坏，将重新运行 Planner: {e}")
        return None


def load_cached_coder_artifact(path: Path) -> CoderArtifact | None:
    if not path.exists():
        return None

    print("[PaperAlchemy]发现本地已有网页构建存档，正在加载...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            artifact_dict = json.load(f)
        artifact = CoderArtifact(**artifact_dict)
        if "v1-clean-body-rewrite" not in artifact.notes:
            print("[PaperAlchemy]检测到旧版 Coder 产物，准备重新生成...")
            return None

        entry_path = Path(artifact.entry_html)
        if not entry_path.exists():
            print("[PaperAlchemy]缓存网页入口不存在，准备重新生成...")
            return None

        html_text = entry_path.read_text(encoding="utf-8", errors="ignore")
        title_count = len(re.findall(r"<title\b", html_text, flags=re.IGNORECASE))
        if title_count != 1:
            print("[PaperAlchemy]检测到缓存页面结构异常（title 标签数量不正确），准备重新生成...")
            return None

        print(f"[PaperAlchemy]成功加载网页构建存档: {artifact.entry_html}")
        return artifact
    except Exception as e:
        print(f"[PaperAlchemy]网页构建存档损坏，将重新运行 Coder: {e}")
        return None


def ensure_parsed_output(pdf_filename: str, output_md_path: Path) -> bool:
    if output_md_path.exists():
        print("[PaperAlchemy]已有解析数据，跳过。")
        return True

    print("[PaperAlchemy]解析 PDF...")
    parse_pdf(pdf_filename)

    if not output_md_path.exists():
        print(f"[PaperAlchemy]🤡解析阶段失败，未找到输出文件: {output_md_path}🤡")
        return False

    return True


def save_structured_data(path: Path, structured_data: StructuredPaper) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print("[PaperAlchemy]保存结构化数据到硬盘...")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(structured_data.model_dump(), f, indent=2, ensure_ascii=False)


def save_page_plan(path: Path, page_plan: PagePlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print("[PaperAlchemy]保存页面规划到硬盘...")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(page_plan.model_dump(), f, indent=2, ensure_ascii=False)


def save_coder_artifact(path: Path, artifact: CoderArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print("[PaperAlchemy]保存网页构建结果到硬盘...")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact.model_dump(), f, indent=2, ensure_ascii=False)


def main(pdf_filename: str) -> None:
    input_path = PROJECT_ROOT / "data" / "input" / pdf_filename
    if not input_path.exists():
        print(f"[PaperAlchemy]🤡输入文件不存在: {input_path}🤡")
        return

    paper_folder_name = Path(pdf_filename).stem
    output_dir = PROJECT_ROOT / "data" / "output" / paper_folder_name
    output_md_path = output_dir / "full_paper.md"
    structured_json_path = output_dir / "structured_paper.json"
    planner_json_path = output_dir / "page_plan.json"
    coder_json_path = output_dir / "coder_artifact.json"

    if not ensure_parsed_output(pdf_filename, output_md_path):
        return

    structured_data = load_cached_structured_data(structured_json_path)

    if not structured_data:
        print("[PaperAlchemy]启动 Reader Agent...")
        structured_data = run_reader_agent(paper_folder_name)

        if not structured_data:
            print("[PaperAlchemy]🤡Reader Agent 失败，流程终止🤡")
            return

        save_structured_data(structured_json_path, structured_data)

    print("[PaperAlchemy]Reader 阶段数据准备就绪。")

    page_plan = load_cached_page_plan(planner_json_path)
    if not page_plan:
        print("[PaperAlchemy]启动 Planner Agent...")
        planner_constraints = {
            "target_framework": "static-html",
            "max_templates_for_planner": 120,
            "max_entry_candidates": 3,
            "template_candidate_top_k": 3,
            "template_selection_mode": "human",
            "max_blocks": 10,
            "min_blocks": 6,
        }
        page_plan = run_planner_agent(
            paper_folder_name=paper_folder_name,
            structured_data=structured_data,
            generation_constraints=planner_constraints,
            max_retry=2,
        )

        if not page_plan:
            print("[PaperAlchemy]🤡Planner Agent 失败，流程终止🤡")
            return

        save_page_plan(planner_json_path, page_plan)

    print("[PaperAlchemy]Planner 阶段数据准备就绪。")

    coder_artifact = load_cached_coder_artifact(coder_json_path)
    if not coder_artifact:
        print("[PaperAlchemy]启动 Coder Agent...")
        coder_artifact = run_coder_agent(
            paper_folder_name=paper_folder_name,
            structured_data=structured_data,
            page_plan=page_plan,
            max_retry=1,
        )

        if not coder_artifact:
            print("[PaperAlchemy]🤡Coder Agent 失败，流程终止🤡")
            return

        save_coder_artifact(coder_json_path, coder_artifact)

    print("[PaperAlchemy]Coder 阶段数据准备就绪。")
    print(f"[PaperAlchemy]网页入口文件: {coder_artifact.entry_html}")


if __name__ == "__main__":
    target_pdf = "Achilles.pdf"
    main(target_pdf)
