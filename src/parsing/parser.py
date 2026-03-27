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

# 瀹氫箟杈撳叆杈撳嚭璺緞
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
IMAGE_SCALE = 2.0 

def get_pipeline_options():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options.lang = ["en"]
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    
    # 鐢熸垚椤甸潰绾ф埅鍥?(鐢ㄤ簬璁?Agent "鐪? 璁烘枃鎺掔増)
    pipeline_options.generate_page_images = True
    pipeline_options.images_scale = IMAGE_SCALE
    
    # 鐢熸垚鐙珛鐨勫浘琛ㄦ埅鍥?(鐢ㄤ簬鎻愬彇绱犳潗)
    pipeline_options.generate_picture_images = True
    
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, 
        device=AcceleratorDevice.CUDA 
    )
    return pipeline_options

def parse_pdf(pdf_filename):
    pdf_path = os.path.join(INPUT_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        print(f"鉂?鏂囦欢涓嶅瓨鍦? {pdf_path}")
        return

    print(f"鈿楋笍 [PaperAlchemy-Docling] 娣卞害瑙ｆ瀽: {pdf_filename}")
    start_time = time.time()

    # 杞崲鏂囨。
    pipeline_opts = get_pipeline_options()
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
    )
    result = converter.convert(pdf_path)
    doc = result.document

    # 鍑嗗杈撳嚭鐩綍
    paper_folder_name = Path(pdf_path).stem
    paper_output_dir = OUTPUT_DIR / paper_folder_name
    images_dir = paper_output_dir / "assets"
    os.makedirs(images_dir, exist_ok=True)

    # 鏋勫缓缁撴瀯鍖栨暟鎹?(Schema for Agent)
    structured_data = {
        "metadata": {
            "filename": pdf_filename,
            "page_count": len(doc.pages),
            "parse_time": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "pages": [] # 鎸夐〉闈㈢粍缁囨暟鎹?
    }

    print("[PaperAlchemy-Docling] 姝ｅ湪瀵煎嚭椤甸潰鎴浘鍜屾彁鍙栧厓绱?..")

    # 閬嶅巻姣忎竴椤?
    for page_no, page in doc.pages.items():
        # 瀹氫箟鍥剧墖淇濆瓨璺緞 (鐩稿璺緞鐢ㄤ簬 JSON锛岀粷瀵硅矾寰勭敤浜庝繚瀛?
        img_filename = f"page_{page_no}.png"
        img_rel_path = f"assets/{img_filename}"
        img_abs_path = images_dir / img_filename
        
        # 淇濆瓨鏁撮〉鎴浘
        page.image.pil_image.save(img_abs_path, format="PNG")
        
        structured_data["pages"].append({
            "page_number": page_no,
            "page_image": img_rel_path, # Agent 璇诲彇鐨勮矾寰?
            "figures": [], # 绋嶅悗濉厖
            "tables": []   # 绋嶅悗濉厖
        })
   
    element_counter = 0
    for element, level in doc.iterate_items():
        if isinstance(element, (PictureItem, TableItem)):
            element_counter += 1
            # 鑾峰彇鍥剧墖鎵€鍦ㄩ〉鐮?
            # Docling 鐨?element 閫氬父鏈?prov (provenance) 淇℃伅鎸囧悜椤甸潰
            page_idx = -1
            if hasattr(element, "prov") and element.prov:
                page_idx = element.prov[0].page_no # 鑾峰彇椤电爜
            
            # 淇濆瓨鍏冪礌鎴浘
            elem_filename = f"element_{element_counter}.png"
            elem_path = os.path.join(images_dir, elem_filename)
            
            # 鍙湁褰撳厓绱犳湁鍥剧墖鏁版嵁鏃舵墠淇濆瓨
            if hasattr(element, "get_image"):
                element_img = element.get_image(doc)
                if element_img:
                    element_img.save(elem_path, "PNG")
            
            # 灏嗗厓绱犱俊鎭坊鍔犲埌瀵瑰簲鐨勯〉闈㈡暟鎹腑
            item_data = {
                "id": element_counter,
                "type": "table" if isinstance(element, TableItem) else "figure",
                "image_path": f"assets/{elem_filename}",
                "caption": "",
                "bbox": element.prov[0].bbox.as_tuple() if element.prov else None
            }
            
            # 鎵惧埌瀵瑰簲椤甸潰鐨?list 骞?append
            for p in structured_data["pages"]:
                if p["page_number"] == page_idx:
                    if isinstance(element, TableItem):
                        p["tables"].append(item_data)
                    else:
                        p["figures"].append(item_data)
                    break

    # 鐢熸垚鍏ㄦ枃 Markdown (浣滀负 Agent 鐨勪富瑕侀槄璇绘潗鏂?
    # 浣跨敤 REFERENCED 妯″紡锛岃繖鏍?Markdown 閲屼細鏈?![image](assets/xxx.png)
    md_content = doc.export_to_markdown(image_mode=ImageRefMode.REFERENCED)
    
    # Docling 榛樿寮曠敤鐨勫浘鐗囪矾寰勫彲鑳介渶瑕佽皟鏁达紝杩欓噷鍏堜繚瀛樻爣鍑嗙増
    md_path = os.path.join(paper_output_dir, "full_paper.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    json_path = os.path.join(paper_output_dir, "parsed_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)

    duration = time.time() - start_time
    
    print("-" * 40)
    print(f"[Success] Parse completed in {duration:.2f} seconds.")
    print(f"Output directory: {paper_output_dir}")
    print("  - full_paper.md")
    print("  - parsed_data.json")
    print(f"  - assets/ ({element_counter} extracted elements + {len(doc.pages)} page screenshots)")
    print("-" * 40)

if __name__ == "__main__":

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_paper = "Achilles.pdf"
    parse_pdf(target_paper)
