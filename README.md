PaperAlchemy ⚗️
================

基于多智能体协同的学术论文网页自动化构建系统  
(Automated Paper-to-Page Construction based on Multi-Agent Collaboration)

### 📖 项目简介

PaperAlchemy 旨在把静态 PDF 学术论文“点石成金”，转化为结构化、可用于网页展示的内容数据。  
目前已完成从 **PDF → Markdown + 结构化 JSON** 的自动化解析与 LLM 智能阅读阶段。

- **Parser（Docling 管线）**：从原始 PDF 中解析出全文 Markdown、页面截图以及图表切片元数据。
- **Reader Agent（基于 Gemini + LangGraph）**：阅读 Markdown 与图表元数据，生成符合 `StructuredPaper` schema 的结构化论文表示，用于后续网页/可视化渲染。
- **缓存机制**：解析结果（`parsed_data.json`、`full_paper.md`、`structured_paper.json`）落盘，避免重复消耗 token。

后续规划：在此基础上增加 **Planner Agent / Coder Agent**，自动生成前端页面代码。

### 🛠️ 技术栈

- **解析层**：Docling（PDF 多模态解析，导出 Markdown + assets）
- **Agent 编排**：LangGraph（`StateGraph` + `MemorySaver`）
- **LLM 接入**：`langchain_google_genai.ChatGoogleGenerativeAI`（Gemini 3 Pro / Flash）
- **数据建模**：Pydantic (`StructuredPaper`, `PaperSection`, `FigureInfo`)
- **运行环境**：Python 3.10+（建议）

### 📂 目录结构（核心部分）

```text
PaperAlchemy/
  ├─ main.py                # 主入口：整合 Parser + Reader Agent
  ├─ src/
  │   ├─ parser.py          # 使用 Docling 解析 PDF，导出 Markdown 和图像/表格元数据
  │   ├─ agent_reader.py    # Reader Agent：调用 Gemini 阅读并结构化论文
  │   ├─ llm.py             # LLM 封装：Gemini 模型初始化、代理设置
  │   └─ schemas.py         # Pydantic 模型：StructuredPaper / PaperSection / FigureInfo
  ├─ data/
  │   ├─ input/             # 待解析论文 PDF（本地，不会上传到 GitHub）
  │   └─ output/            # 解析与结构化结果（Markdown + JSON + 资产）
  ├─ docling/               # 与 Docling 相关的实验脚本
  ├─ requirements.txt       # Python 依赖
  └─ README.md
```

### ⚙️ 环境配置

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 在项目根目录创建 `.env` 文件，配置 Gemini 相关环境变量（示例）：

```bash
GOOGLE_API_KEY=你的_gemini_api_key

# 可选：HTTP 代理（当前代码默认尝试使用 127.0.0.1:7890）
HTTPS_PROXY=http://127.0.0.1:7890
```

> 说明：`src/llm.py` 中会自动从 `.env` 读取 `GOOGLE_API_KEY`，并设置代理与 SSL 选项。

### 🚀 运行方式

#### 1. 完整流程（推荐）

在项目根目录下运行：

```bash
python main.py
```

默认会以 `data/input/Achilles.pdf` 为示例：

- 如果尚未解析过该 PDF：
  - 调用 `parse_pdf`（`src/parser.py`），生成：
    - `data/output/Achilles/full_paper.md`
    - `data/output/Achilles/parsed_data.json`
    - `data/output/Achilles/assets/` 下的页面截图与图表切片
- 然后启动 `run_reader_agent`（`src/agent_reader.py`）：
  - 调用 Gemini 阅读 Markdown + 资产列表
  - 交互式人工审核（命令行输入 `ok` 或给出修改意见）
  - 输出结构化结果到：
    - `data/output/Achilles/structured_paper.json`

#### 2. 仅运行解析器（Parser）

```bash
python src/parser.py
```

在 `parser.py` 内修改 `target_paper` 即可切换解析目标 PDF。

#### 3. 单独测试 Reader Agent

在已有解析结果的前提下（`data/output/<PaperName>/full_paper.md` 与 `parsed_data.json` 已存在）：

```bash
python src/agent_reader.py
```

在 `agent_reader.py` 末尾的测试代码中修改 `run_reader_agent("All You Need is DAG")` 可以切换论文。

### ✅ 当前阶段成果小结

- 完成 **PDF → Markdown + 图表元数据** 的自动解析管线（Docling）。
- 完成 **Reader Agent**：基于 Gemini + LangGraph 的结构化阅读与反馈迭代机制。
- 已能产出可直接用于前端渲染的结构化 JSON（标题、章节摘要、关键要点、相关图片路径）。

### 🔭 下一步计划（TODO）

- 实现 **Planner Agent**：基于 `StructuredPaper` 规划网页信息架构与交互设计。
- 实现 **Coder Agent**：自动生成前端代码（React/Vue 组件、样式与路由）。
- 增加 Web UI，用于可视化编辑章节结构与图片映射关系。
