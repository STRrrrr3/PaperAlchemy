import os
import json
from pathlib import Path
from typing import TypedDict, List, Annotated, Optional
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage

from src.schemas import StructuredPaper
from src.llm import get_llm 

# 定义状态 (State)
class ReaderState(TypedDict):
    raw_markdown: str
    assets_list: List[dict]
    feedback_history: Annotated[List[str], operator.add] 
    structured_paper: Optional[StructuredPaper] # 允许为空
    is_approved: bool

# 定义节点 (Nodes)

def reader_node(state: ReaderState):
    print("[PaperAlchemy-Reader] Gemini 正在阅读全文并提取结构...")
    # 获取模型
    llm = get_llm(temperature=0.7, use_smart_model=True)
    
    # 绑定结构化输出
    structured_llm = llm.with_structured_output(StructuredPaper)
    
    # 准备数据上下文
    md_content = state['raw_markdown'] 
    assets_context = json.dumps(state['assets_list'], indent=2, ensure_ascii=False)
    
    # 处理历史反馈 (如果有)
    feedback_section = ""
    if state.get("feedback_history"):
        history_str = "\n".join([f"- {item}" for item in state['feedback_history']])
        feedback_section = f"""
        # !!! CRITICAL HUMAN FEEDBACK !!!
        The user has rejected your previous output. You MUST fix the following issues in this iteration:
        {history_str}
        Review your previous logic and ensure these specific points are addressed.
        """

    # 核心 System Prompt
    system_msg = f"""
    You are an expert **Academic Content Structuring Specialist**. 
    Your mission is to transform raw, unstructured Markdown text from a research paper into a clean, hierarchical, and structured format ready for web presentation.

    ###  CRITICAL RULES:
    1. **ABSTRACT HANDLING:** - Check if the paper has an explicit 'Abstract' or 'Summary' block at the beginning.
       - **IF YES:** You **MUST** extract it as the first `PaperSection`. Do not skip it even if you already wrote the `overall_summary`.
       - **IF NO:** Do NOT invent an Abstract section. Start with the first real section (e.g., Introduction).
    2. **NO MERGING:** Do NOT combine multiple sections...

    ### INPUT DATA
    1. **Raw Markdown:** The full text of the paper.
    2. **Assets List:** A JSON list of available images/tables extracted from the PDF, including their file paths.

    ### YOUR TASKS
    1. **Structure Extraction:**
       - Identify the paper's title and generate a concise overall summary.
       - Segment the paper into logical sections (e.g., Introduction, Methodology, Experiments, Conclusion).
       - For `content_summary`: Provide a high-level overview.
       - For `key_details`: Extract **comprehensive technical details**. 
         - If it's a Method section: List the specific steps of the algorithm.
         - If it's an Experiment section: Extract specific numbers. **Cite exact values from the text.
         - If it's a Theory section: Describe the core definitions or proofs in text.

    2. **Asset Mapping (CRITICAL):**
       - The raw Markdown contains text references to figures/tables (e.g., "Figure 1 shows...", "As seen in Table 2...").
       - The `Assets List` contains the actual file paths (e.g., "assets/element_5.png").
       - **Your Goal:** Intelligently map the images from the `Assets List` to the correct `PaperSection` based on context.
       - *Rule:* If a section discusses "Figure 3", look for the image in the Assets List that likely corresponds to Figure 3 (based on order or caption context) and assign it to that section.
       - *Rule:* Do NOT invent file paths. Only use paths provided in the `Assets List`.

    ### OUTPUT REQUIREMENTS
    - The output must strictly follow the `StructuredPaper` schema.
    - Summaries should be informative but easy to read for a general technical audience.
    - If a section has no relevant images, `related_figures` should be an empty list.

    {feedback_section}
    """
    
    user_msg = f"""
    ### ASSETS LIST (Use these paths):
    {assets_context}

    ### RAW MARKDOWN CONTENT:
    {md_content}
    """
    
    # 调用模型
    try:
        result = structured_llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg)
        ])
        
        if not result:
            raise ValueError("Model returned empty result")
            
        return {"structured_paper": result}
        
    except Exception as e:
        print(f"Reader Error: {e}")
        return {}

def human_review_node(state: ReaderState):
    # 断点：等待人工审核
    print("[PaperAlchemy-Reader] 流程暂停，等待用户审核结果...")
    pass

# 定义路由 (Edges)

def router_after_review(state: ReaderState):
    if state.get("is_approved"):
        print("[PaperAlchemy-Reader] 用户审核通过！Reader 任务结束。")
        return "end"
    else:
        print("[PaperAlchemy-Reader] 用户提出修改意见，Reader 返工中...")
        return "retry"

# 构建图 (Graph)

def build_reader_graph():
    workflow = StateGraph(ReaderState)
    # 添加节点
    workflow.add_node("reader", reader_node)
    workflow.add_node("human_review", human_review_node)
    # 设置流程
    workflow.set_entry_point("reader")
    workflow.add_edge("reader", "human_review")
    # 添加条件边
    workflow.add_conditional_edges(
        "human_review",
        router_after_review,
        {
            "retry": "reader",
            "end": END
        }
    )
    # 设置记忆
    memory = MemorySaver()
    # interrupt_before=["human_review"] 确保在进入审核节点前暂停
    return workflow.compile(checkpointer=memory, interrupt_before=["human_review"])

def run_reader_agent(paper_folder_name: str):
    # 动态定位路径
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent # src -> PaperAlchemy
    output_dir = project_root / "data" / "output" / paper_folder_name
    
    print(f"[PaperAlchemy] 启动 Reader Agent，读取数据: {output_dir}")

    # 2. 加载数据
    try:
        with open(output_dir / "full_paper.md", "r", encoding="utf-8") as f:
            raw_md = f.read()
        with open(output_dir / "parsed_data.json", "r", encoding="utf-8") as f:
            full_json = json.load(f)
            assets = []
            for p in full_json.get('pages', []):
                assets.extend(p.get('figures', []))
                assets.extend(p.get('tables', []))
    except FileNotFoundError:
        print(f"[PaperAlchemy-Reader] 🤡错误：找不到解析数据。请确保 parser.py 已运行🤡")
        return None

    # 启动图
    app = build_reader_graph()
    thread = {"configurable": {"thread_id": "main_session"}}
    
    initial_state = {
        "raw_markdown": raw_md,
        "assets_list": assets,
        "feedback_history": [],
        "is_approved": False,
        "structured_paper": None
    }

    # 首次运行
    print("Wait for Gemini...")
    app.invoke(initial_state, thread)

    # 进入交互循环
    while True:
        current_state = app.get_state(thread)
        structured_result = current_state.values.get("structured_paper")
        
        if not structured_result:
            print("[PaperAlchemy-Reader] 🤡警告：无结果🤡")
            break

        # 展示简报
        print("\n" + "-"*30)
        print(f"{structured_result.paper_title}")
        print(f"{len(structured_result.sections)} Sections detected.")
        for idx, sec in enumerate(structured_result.sections):
            print(f"   {idx+1}. {sec.section_title} (Img: {len(sec.related_figures)})")
        print("-" * 30 + "\n")

        # 获取输入
        user_input = input("[PaperAlchemy-Reader] 审核 (输入 'ok' 通过，或输入意见) > ").strip()
        
        if user_input.lower() in ["ok", "y", "yes"]:
            app.update_state(thread, {"is_approved": True})
            # 跑完剩余流程
            for _ in app.stream(None, thread): pass
            print("[PaperAlchemy] Reader 阶段完成。")
            return structured_result # 返回最终结果给
        else:
            app.update_state(thread, {"is_approved": False, "feedback_history": [user_input]})
            for _ in app.stream(None, thread): pass

# 直接测试
if __name__ == "__main__":
    run_reader_agent("All You Need is DAG")