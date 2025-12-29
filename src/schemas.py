from typing import List, Optional
from pydantic import BaseModel, Field

# 定义最小单元（图片）
class FigureInfo(BaseModel):
    image_path: str = Field(description="Relative path of the image in the assets folder, e.g., 'assets/element_1.png'")
    caption: Optional[str] = Field(description="Caption or description of the image, e.g., 'Figure 1: System Architecture'")
    type: str = Field(description="Type, e.g., 'chart', 'table', 'photo'")

# 定义章节（把文字和图片捆绑）
class PaperSection(BaseModel):
    section_title: str = Field(description="Section title, e.g., 'Introduction' or '3. Methodology'")
    content_summary: str = Field(description="Core content summary of the section (200-300 words), for quick browsing")
    key_details: List[str] = Field(description="Detailed content points of the section. Includes steps of core mechanisms, specific experimental data comparisons, important mathematical definition descriptions, etc. Each point 50-150 words.")
    related_figures: List[FigureInfo] = Field(description="List of figures belonging to this section")

# 定义整篇论文的结构（Reader 的最终产出）
class StructuredPaper(BaseModel):
    paper_title: str = Field(description="Title of the paper")
    overall_summary: str = Field(description="Brief summary of the entire paper")
    sections: List[PaperSection] = Field(description="""
    Ordered list of extracted sections.
    CRITICAL RULES:
    1. **IF** the paper has an 'Abstract' (or 'Executive Summary'), it MUST be the first item in this list. DO NOT skip it.
    2. **IF** the paper has NO Abstract, start directly with the first actual section (e.g., '1. Introduction').
    3. DO NOT merge sections.
    """)