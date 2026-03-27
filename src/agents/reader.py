import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.reader_critic import build_critic_router, critic_node
from src.services.human_feedback import extract_human_feedback_text, normalize_human_feedback
from src.services.llm import get_llm
from src.prompts import (
    READER_RETRY_FEEDBACK_APPEND_TEMPLATE,
    READER_SYSTEM_PROMPT,
    READER_USER_PROMPT_TEMPLATE,
)
from src.contracts.schemas import StructuredPaper
from src.contracts.state import ReaderState


def reader_node(state: ReaderState):
    print(f"[PaperAlchemy-Reader]Gemini 正在阅读全文提取结构 (第 {state.get('retry_count', 0)} 次尝试)...")
    llm = get_llm(temperature=0.7, use_smart_model=True)
    structured_llm = llm.with_structured_output(StructuredPaper)

    md_content = state["raw_markdown"]
    assets_context = json.dumps(state["assets_list"], indent=2, ensure_ascii=False)
    human_directives = extract_human_feedback_text(state.get("human_directives"))
    previous_structured_paper = state.get("previous_structured_paper")
    if isinstance(previous_structured_paper, StructuredPaper):
        previous_structured_paper_json = json.dumps(
            previous_structured_paper.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    elif previous_structured_paper:
        try:
            previous_structured_paper_json = json.dumps(
                StructuredPaper.model_validate(previous_structured_paper).model_dump(),
                indent=2,
                ensure_ascii=False,
            )
        except Exception:
            previous_structured_paper_json = "null"
    else:
        previous_structured_paper_json = "null"

    feedback_section = ""
    if state.get("feedback_history"):
        history_str = "\n".join([f"- {item}" for item in state["feedback_history"]])
        feedback_section = READER_RETRY_FEEDBACK_APPEND_TEMPLATE.format(
            feedback_history=history_str
        )

    system_msg = READER_SYSTEM_PROMPT + feedback_section
    user_msg = READER_USER_PROMPT_TEMPLATE.format(
        human_directives=human_directives or "(none)",
        previous_structured_paper_json=previous_structured_paper_json,
        assets_context=assets_context,
        md_content=md_content,
    )

    try:
        result = structured_llm.invoke(
            [
                SystemMessage(content=system_msg),
                HumanMessage(content=user_msg),
            ]
        )

        if not result:
            raise ValueError("Model returned empty result")

        return {"structured_paper": result}
    except Exception as e:
        print(f"[PaperAlchemy-Reader] Error: {e}")
        return {}


def build_reader_graph(max_retry: int = 3):
    workflow = StateGraph(ReaderState)
    workflow.add_node("reader", reader_node)
    workflow.add_node("critic", critic_node)

    workflow.set_entry_point("reader")
    workflow.add_edge("reader", "critic")

    workflow.add_conditional_edges(
        "critic",
        build_critic_router(max_retry=max_retry),
        {
            "retry": "reader",
            "end": END,
        },
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def _load_reader_inputs(output_dir: Path) -> tuple[str, list[dict]]:
    with open(output_dir / "full_paper.md", "r", encoding="utf-8") as f:
        raw_md = f.read()

    with open(output_dir / "parsed_data.json", "r", encoding="utf-8") as f:
        full_json = json.load(f)

    assets: list[dict] = []
    for page in full_json.get("pages", []):
        assets.extend(page.get("figures", []))
        assets.extend(page.get("tables", []))

    return raw_md, assets


def run_reader_agent(
    paper_folder_name: str,
    human_directives: str | dict = "",
    previous_structured_paper: StructuredPaper | None = None,
    max_retry: int = 3,
):
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    output_dir = project_root / "data" / "output" / paper_folder_name

    print(f"[PaperAlchemy]启动 Reader Agent，读取数据: {output_dir}")

    try:
        raw_md, assets = _load_reader_inputs(output_dir)
    except FileNotFoundError:
        print("[PaperAlchemy-Reader]🤡错误：找不到解析数据。请确保 parser.py 已运行🤡")
        return None

    app = build_reader_graph(max_retry=max_retry)
    thread = {"configurable": {"thread_id": "main_session_auto"}}

    initial_state: ReaderState = {
        "raw_markdown": raw_md,
        "assets_list": assets,
        "human_directives": normalize_human_feedback(human_directives),
        "previous_structured_paper": previous_structured_paper,
        "feedback_history": [],
        "critic_passed": False,
        "retry_count": 0,
        "structured_paper": None,
    }

    print("[PaperAlchemy-System]正在全自动执行信息提取与 Critic 自查流水线...")
    for _ in app.stream(initial_state, thread):
        pass

    final_state = app.get_state(thread)
    structured_result = final_state.values.get("structured_paper")

    if not structured_result or not final_state.values.get("critic_passed"):
        print("\n[PaperAlchemy-System]🤡提取流程最终异常或未完美通过 Critic 校验🤡")
    else:
        print("\n" + "=" * 50)
        print(f"[PaperAlchemy-System]自动化提取大成功：{structured_result.paper_title}")
        print(f"[PaperAlchemy-System]共拆解 {len(structured_result.sections)} 个结构化章节。")
        for idx, sec in enumerate(structured_result.sections, start=1):
            print(f"   {idx}. {sec.section_title[:30]}... (具有图表引用数: {len(sec.related_figures)})")
        print("=" * 50 + "\n")

    return structured_result


if __name__ == "__main__":
    run_reader_agent("All You Need is DAG")

