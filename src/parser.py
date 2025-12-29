import os
import json
import time
from pathlib import Path
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice
)
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

# 定义输入输出路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
IMAGE_SCALE = 2.0 

def get_pipeline_options():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options.lang = ["en"]
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    
    # 生成页面级截图 (用于让 Agent "看" 论文排版)
    pipeline_options.generate_page_images = True
    pipeline_options.images_scale = IMAGE_SCALE
    
    # 生成独立的图表截图 (用于提取素材)
    pipeline_options.generate_picture_images = True
    
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, 
        device=AcceleratorDevice.CUDA 
    )
    return pipeline_options

def parse_pdf(pdf_filename):
    pdf_path = os.path.join(INPUT_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        print(f"❌ 文件不存在: {pdf_path}")
        return

    print(f"⚗️ [PaperAlchemy-Docling] 深度解析: {pdf_filename}")
    start_time = time.time()

    # 转换文档
    pipeline_opts = get_pipeline_options()
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
    )
    result = converter.convert(pdf_path)
    doc = result.document

    # 准备输出目录
    paper_folder_name = Path(pdf_path).stem
    paper_output_dir = OUTPUT_DIR / paper_folder_name
    images_dir = paper_output_dir / "assets"
    os.makedirs(images_dir, exist_ok=True)

    # 构建结构化数据 (Schema for Agent)
    structured_data = {
        "metadata": {
            "filename": pdf_filename,
            "page_count": len(doc.pages),
            "parse_time": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "pages": [] # 按页面组织数据
    }

    print("[PaperAlchemy-Docling] 正在导出页面截图和提取元素...")

    # 遍历每一页
    for page_no, page in doc.pages.items():
        # 定义图片保存路径 (相对路径用于 JSON，绝对路径用于保存)
        img_filename = f"page_{page_no}.png"
        img_rel_path = f"assets/{img_filename}"
        img_abs_path = images_dir / img_filename
        
        # 保存整页截图
        page.image.pil_image.save(img_abs_path, format="PNG")
        
        structured_data["pages"].append({
            "page_number": page_no,
            "page_image": img_rel_path, # Agent 读取的路径
            "figures": [], # 稍后填充
            "tables": []   # 稍后填充
        })
   
    element_counter = 0
    for element, level in doc.iterate_items():
        if isinstance(element, (PictureItem, TableItem)):
            element_counter += 1
            # 获取图片所在页码
            # Docling 的 element 通常有 prov (provenance) 信息指向页面
            page_idx = -1
            if hasattr(element, "prov") and element.prov:
                page_idx = element.prov[0].page_no # 获取页码
            
            # 保存元素截图
            elem_filename = f"element_{element_counter}.png"
            elem_path = os.path.join(images_dir, elem_filename)
            
            # 只有当元素有图片数据时才保存
            if hasattr(element, "get_image"):
                element_img = element.get_image(doc)
                if element_img:
                    element_img.save(elem_path, "PNG")
            
            # 将元素信息添加到对应的页面数据中
            item_data = {
                "id": element_counter,
                "type": "table" if isinstance(element, TableItem) else "figure",
                "image_path": f"assets/{elem_filename}",
                "caption": "",
                "bbox": element.prov[0].bbox.as_tuple() if element.prov else None
            }
            
            # 找到对应页面的 list 并 append
            for p in structured_data["pages"]:
                if p["page_number"] == page_idx:
                    if isinstance(element, TableItem):
                        p["tables"].append(item_data)
                    else:
                        p["figures"].append(item_data)
                    break

    # 生成全文 Markdown (作为 Agent 的主要阅读材料)
    # 使用 REFERENCED 模式，这样 Markdown 里会有 ![image](assets/xxx.png)
    md_content = doc.export_to_markdown(image_mode=ImageRefMode.REFERENCED)
    
    # Docling 默认引用的图片路径可能需要调整，这里先保存标准版
    md_path = os.path.join(paper_output_dir, "full_paper.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    json_path = os.path.join(paper_output_dir, "parsed_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)

    duration = time.time() - start_time
    
    print("-" * 40)
    print(f"✅ [Success] 解析完成！耗时 {duration:.2f} 秒")
    print(f"📍 输出位置: {paper_output_dir}")
    print(f"   ├── 📄 full_paper.md (全文文本)")
    print(f"   ├── 📊 parsed_data.json (结构化数据+页面索引)")
    print(f"   └── 🖼️ assets/ (包含 {element_counter} 个图表 + {len(doc.pages)} 张整页截图)")
    print("-" * 40)

if __name__ == "__main__":

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_paper = "Achilles.pdf"
    parse_pdf(target_paper)