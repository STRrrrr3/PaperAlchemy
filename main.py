import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))
from src.parser import parse_pdf
from src.agent_reader import run_reader_agent
from src.schemas import StructuredPaper

def main(pdf_filename):
    # 定义路径
    project_root = Path(__file__).parent
    input_path = project_root / "data" / "input" / pdf_filename
    paper_folder_name = Path(pdf_filename).stem 
    output_dir = project_root / "data" / "output" / paper_folder_name
    output_md_path = output_dir / "full_paper.md"
    structured_json_path = output_dir / "structured_paper.json"

    # 检查是否需要 Parse
    if not output_md_path.exists():
        print(f"[PaperAlchemy] 解析 PDF...")
        parse_pdf(pdf_filename)
    else:
        print(f"[PaperAlchemy] 已有解析数据，跳过。")

    # 运行 Reader Agent (Step 2)
    structured_data = None
    
    # 如果本地已经有结构化数据，直接读取，节约token
    if structured_json_path.exists():
        print(f"💾 [Cache] 发现本地已有结构化存档，正在加载...")
        try:
            with open(structured_json_path, "r", encoding="utf-8") as f:
                data_dict = json.load(f)
                # 将字典转换回 Pydantic 对象
                structured_data = StructuredPaper(**data_dict)
                print(f"[PaperAlchemy] 成功加载存档: {structured_data.paper_title}")
        except Exception as e:
            print(f"[PaperAlchemy] 🤡存档损坏，将重新运行 Reader: {e}🤡")
            structured_data = None

    # 如果没有结构化数据，进入Agent流程
    if not structured_data:
        print(f"[PaperAlchemy] 启动 Reader Agent...")
        structured_data = run_reader_agent(paper_folder_name)
        # 跑完立刻保存
        if structured_data:
            print(f"[PaperAlchemy] 保存结构化数据到硬盘...")
            with open(structured_json_path, "w", encoding="utf-8") as f:
                json.dump(structured_data.model_dump(), f, indent=2, ensure_ascii=False)
        else:
            print("[PaperAlchemy] 🤡Reader Agent 失败，流程终止🤡")
            return
            
    if structured_data:
        print("[PaperAlchemy] Reader 阶段数据准备就绪。")
        # TODO: 下一步把 structured_data 传给 Planner Agent
        # run_planner_agent(structured_data)

if __name__ == "__main__":
    target_pdf = "Achilles.pdf"
    main(target_pdf)