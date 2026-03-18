# 架构工作流分析：PaperAlchemy vs. AutoPage

本文档对两个自动化论文到网页构建代码库：**PaperAlchemy** 和 **AutoPage** 的执行管道进行了高度技术性、端到端的解构。

该分析严格遵循时间执行顺序，从五个关键维度对每一个技术细节进行了分解：核心逻辑、实现机制、工具与依赖、确切的制品输出，以及人在回路（Human-in-the-Loop）工作流。

---

## 第一部分：PaperAlchemy 工作流解构

PaperAlchemy 采用了一个高度结构化的、通过 LangGraph 编排的多智能体管道。该代码库将解析、阅读、规划和编码阶段明确分离为不同的智能体模块。

### 阶段 1：PDF 解析阶段
* **步骤定义与核心逻辑：** 位于 [src/parser.py](file:///e:/Graduation%20Design/PaperAlchemy/src/parser.py)。该模块将目标学术 PDF 转换为结构化的 Markdown，提取表格/图像，并生成全页截图，为下游智能体提供视觉上下文。
* **实现机制：** [parse_pdf](file:///e:/Graduation%20Design/PaperAlchemy/src/parser.py#41-153) 函数使用 `PdfPipelineOptions` 初始化 `DocumentConverter`。它捕获 OCR (`do_ocr=True`)、表格结构 (`do_table_structure=True`)、页面截图 (`generate_page_images=True`) 以及单个图表 (`generate_picture_images=True`)。然后它遍历文档项 (`PictureItem`, `TableItem`) 以隔离资产并建立每一页的边界框/位置映射。
* **工具、依赖与 APIs：**
  - `docling` & `docling_core` 用于多模态 PDF 解析。
  - 通过 `PIL.Image` 生成图像。
  - 如果可用，将运行 CUDA 加速 (`AcceleratorDevice.CUDA`)。
* **确切的制品输出：**
  - `data/output/<PaperName>/full_paper.md`：带有 `REFERENCED` 图像路径的全文 markdown。
  - `data/output/<PaperName>/parsed_data.json`：一个结构化的 JSON 树，详细说明了逐页的引用、边界框和资产链接。
  - `data/output/<PaperName>/assets/`：一个包含提取的图表 (`element_*.png`) 和全页截图 (`page_*.png`) 的文件夹。
* **人在回路：** 无。全自动化。

### 阶段 2：读者智能体阶段（信息提取）
* **步骤定义与核心逻辑：** 位于 [src/agent_reader.py](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader.py) 和 [src/agent_reader_critic.py](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader_critic.py)。智能体消耗原始 markdown 和资产 JSON，以提取论文的标准语义表示 (`StructuredPaper`)。它利用 actor-critic（行动者-评论者）循环来保证输出质量。
* **实现机制：**
  - 一个 `StateGraph` ([ReaderState](file:///e:/Graduation%20Design/PaperAlchemy/src/state.py#7-14)) 编排该循环。
  - **生成器节点 (Generator Node)：** 读取 `full_paper.md` 和 `parsed_data.json`，如果在重试，则将之前的评论者反馈注入到 prompt 中。使用 `with_structured_output(StructuredPaper)` 调用 LLM。
  - **评论者节点 (Critic Node)：** 对输入上下文运行确定性检查（[_run_density_checks](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader_critic.py#80-141) 用于检查章节长度、关键细节计数、资产幻觉）和语义 LLM 检查（[run_semantic_critic](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader_critic.py#155-187)）。
  - **路由器 (Router)：** 有条件地循环回生成器，直到 `critic_passed`为 True 或达到 `max_retry`（默认为 3）。
* **工具、依赖与 APIs：** LangChain (`langchain_google_genai`)、LangGraph (`StateGraph`, `MemorySaver`)、Pydantic (`StructuredPaper`)、Gemini Pro/Flash API。
* **确切的制品输出：** `data/output/<PaperName>/structured_paper.json` 包含经过 Pydantic 验证的 `StructuredPaper` schema（标题、整体摘要、包含关键细节、内容摘要和相关图表的 `PaperSection` 元素数组）。
* **人在回路：** 核心自动化节点中没有，尽管 [main.py](file:///e:/Graduation%20Design/PaperAlchemy/main.py) 的文档提到了一种交互式检查（在当前的 LangGraph 批处理运行布局中似乎被禁用了）。

### 阶段 3：规划者智能体阶段（网页布局规划）
* **步骤定义与核心逻辑：** 位于 [src/agent_planner.py](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner.py) 和 [src/agent_planner_critic.py](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner_critic.py)。负责将 `StructuredPaper` 语义映射到可用的 HTML 模板上，确定章节对应关系和块排序。
* **实现机制：**
  - 执行一个 4 节点 `StateGraph` ([PlannerState](file:///e:/Graduation%20Design/PaperAlchemy/src/state.py#16-29))。
  - **节点 1 ([semantic_planner_node](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner.py#115-157))：** 使用 Gemini 生成不受约束的 `SemanticPlan`（逻辑布局）。
  - **节点 2 ([template_selector_node](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner.py#159-209))：** 将 `SemanticPlan` 与解析出的本地模板目录进行打分对比，输出 `TemplateCandidate` 项。
  - **节点 3 ([template_binder_node](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner.py#211-281))：** 水合生成一个 `PagePlan` (schema)，将语义计划严格绑定到获胜的 `TemplateCandidate`。
  - **节点 4 ([planner_critic_node](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner_critic.py#170-211))：** 检查丢失的资产、丢失的模板文件、布局幻觉，并运行语义 LLM 评论者（[run_planner_semantic_critic](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner_critic.py#136-168)）。
* **工具、依赖与 APIs：** LangGraph、Gemini API（Pro/Flash 可变温度设置）、Pydantic (`PagePlan`, `SemanticPlan`, `TemplateCandidate`)。
* **确切的制品输出：** `data/output/<PaperName>/page_plan.json` 符合 `PagePlan` schema（包含 `template_selection`, `page_outline`, `blocks`, 和 `coder_handoff`）。
* **人在回路：** 约束条件中的 `template_selection_mode` 允许“人工(human)”模式（[_choose_template_with_human](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner.py#71-113)）。它会暂停执行，将前几个 `TemplateCandidate` 选项（带有原因和分数）打印到标准输出，并使用 [input()](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader.py#73-86) 捕获用户的整数选择或模板 ID。

### 阶段 4：编码者智能体阶段（HTML 构建）
* **步骤定义与核心逻辑：** 位于 [src/agent_coder.py](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_coder.py) 和 [src/agent_coder_critic.py](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_coder_critic.py)。直接操作文件系统布局，复制资产，并将生成的标记（markup）注入到所选模板的 HTML 文件中。
* **实现机制：**
  - 使用一个简洁的逻辑图，没有直接调用 LLM 生成用于构建字符串。
  - **节点 1 ([coder_node](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_coder.py#308-390))：** 使用 `PagePlan` 将模板静态域复制到 `data/output/<PaperName>/site/`。丢弃样板 HTML `<body>`，并生成完全自定义的、源自 `StructuredPaper` 和 `PagePlan` 的语义 `<body>`。遍历 `page_plan.blocks` 将预定义的 CSS 类附加到注入的组件上。
  - **节点 2 ([coder_critic_node](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_coder_critic.py#70-86))：** 执行确定性代码检查。确保生成的 body 标记 (`BODY_START_MARKER`) 存在，验证 `<title>` 标签的确切数量，并断言所有 `copied_assets` 都被可验证地写入磁盘并在 HTML 字符串中被引用。
* **工具、依赖与 APIs：** Python 标准库（`shutil`, [re](file:///e:/Graduation%20Design/PaperAlchemy/.gitignore), [html](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#463-489)）、LangGraph。
* **确切的制品输出：**
  - `data/output/<PaperName>/coder_artifact.json` 捕获 `CoderArtifact` schema。
  - `data/output/<PaperName>/site/` 包含完全可部署的静态站点 (HTML, CSS, Assets)。
* **人在回路：** 无。全自动化确定性渲染。

---

## 第二部分：AutoPage 工作流解构

AutoPage 作为一个迭代的程序化管道运行，主要封装在单个主线程循环中，并通过 `CAMEL` 框架智能体执行。它强调迭代的 UI 预览检查和大量的视觉/CSS 修改。

### 阶段 1：前置模板匹配与设置
* **步骤定义与核心逻辑：** 位于 [AutoPage/app.py](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/app.py) 和 [ProjectPageAgent/main_pipline.py](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/main_pipline.py)。用户约束（背景颜色、密度、导航、布局）会根据模板数据库进行打分匹配。
* **实现机制：** [matching(requirement)](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/main_pipline.py#33-55) 方法将加权分数应用于根据预定义的 [tags.json](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/tags.json) 文件解析出的用户特征。准备好本地模板目录，并且可能会启动一个本地 HTTP 预览服务器（[find_free_port](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/app.py#114-123), [CustomHTTPRequestHandler](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/app.py#107-113)）供用户选择。
* **工具、依赖与 APIs：** `gradio`（用于 UI 预览）、自定义 HTTP 服务器 (`http.server`)、[tags.json](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/tags.json)。
* **确切的制品输出：** 用于预览的活动 HTTP 服务器端口，模板本地路径。
* **人在回路：** 用户在 Gradio 中输入 GUI 参数，点击模板预览并确认执行。如果模板不明确，[main_pipline.py](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/main_pipline.py) 中的备用 CLI 会使用 [input()](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader.py#73-86)。

### 阶段 2：PDF 解析阶段
* **步骤定义与核心逻辑：** 位于 [ProjectPageAgent/parse_raw.py](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/parse_raw.py)。转换 PDF 文本，提取 Markdown，检测图像项。如果 `docling` 产生的数据过短，存在后备提取方法。通过 LLM 布局结构化判定提取。
* **实现机制：**
  - 使用 `docling` `DocumentConverter`。
  - 子例程：运行一个 `ChatAgent`（通过 `gen_page_raw_content.txt`）迫使 LLM 将原始 markdown 切割成逻辑字典“章节”。使用 `try-catch` 循环进行迭代，直到 JSON 有效并包含 "title"。
  - 子例程：如果 docling 文本少于 500 个字符，后备方法使用 `marker`（ML 模型）。
  - 子例程：使用随机切片 (`random.sample`) 将预处理的 `sections` 数组降采样到最多 9 个逻辑块。
  - 调用 [gen_image_and_table](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/parse_raw.py#139-256) 获取 PIL 裁剪图，并写入独立的 HTML/MD 制品。
* **工具、依赖与 APIs：** `docling`, `PIL`, `marker`, CAMEL (`ChatAgent`, `ModelFactory`)。
* **确切的制品输出：**
  - `project_contents/<PaperName>_raw_content.json`（包含 LLM 提取的章节字典）。
  - `generated_project_pages/images_and_tables/<PaperName>/`（包含多个 [md](file:///e:/Graduation%20Design/PaperAlchemy/README.md)、[html](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#463-489) 和 [png](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#156-167) 提取制品）。
  - 图表清单：`_images.json` 和 `_tables.json`。
* **人在回路：** 无。

### 阶段 3：内容规划者操作
* **步骤定义与核心逻辑：** 位于 [ProjectPageAgent/content_planner.py](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/content_planner.py)。在隔离的阶段（过滤 -> 章节 -> 文本内容 -> 完整内容）中迭代构建站点的文本 payload。
* **实现机制：**
  - **过滤内容：** 删除“References（参考文献）”部分。使用 CAMEL 智能体 (`filter_figures.yaml`) 删除未映射到逻辑章节的资产。
  - **章节生成：** 检测论文“领域”(`technical` vs `other`)，利用可变的 yaml 模板 (`adaptive_sections.yaml` 或 `section_generation.yaml`) 输出一个 JSON 对象，描述目标语义键（例如，`title`, `authors`, `abstract`）。
  - **完整内容生成循环：** 使用 Actor-Critic 架构。Actor ([planner_agent](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner.py#305-376)) 输出 JSON 内容 (`generated_full_content.v0.json`)。Critic ([reviewer_agent](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/content_planner.py#69-95) 配合 `full_content_review.yaml`) 对 JSON 进行批判。Actor 处理反馈 (`full_content_revise.yaml`)，并迭代 `full_content_check_times` 次。
* **工具、依赖与 APIs：** CAMEL, Jinja2 模板渲染。
* **确切的制品输出：**
  - `project_contents/<PaperName>_generated_section.json`
  - `project_contents/<PaperName>_generated_text_content.json`
  - 一系列输出：`project_contents/<PaperName>_generated_full_content.v{N}.json` 和 `review.iter{N}.json`。
* **人在回路：** 在 critic 循环完成后，如果 `args.human_input == '1'`，它会暂停字符串执行，将 `rich.Pretty` JSON 树打印到终端，并等待 [input()](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader.py#73-86)。用户可以输入英文反馈。智能体会动态迭代，直到用户输入 `yes`。

### 阶段 4：HTML 生成器和视觉反馈阶段
* **步骤定义与核心逻辑：** 位于 [ProjectPageAgent/html_generator.py](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py) 和 `css_checker.py`。将生成的内容动态映射到原始模板 HTML 中，通过纯视觉语言模型 (VLM) 操作修复 HTML 表格，并执行端到端的视觉审查。
* **实现机制：**
  - **基础 HTML ([generate_complete_html](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#372-461))：** LLM (`html_generation.yaml`) 将 `generated_content` 字典直接合并进源 HTML 模板字符串。解析并修复断开的 CSS 链接 (`check_css`)。
  - **表格修改 ([modify_html_table](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#191-329))：** VLM 解析原始表格的 `.png` 截图。一个 LLM 智能体 ([table_agent](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#124-140)) 将像素数据转换为纯 HTML 的 `<table>` 标签。然后，通过向 VLM 发送当前页面渲染的截图来绘制“颜色建议”。一个长上下文智能体将色彩调色板 + HTML 表格合并到主 HTML 中。
  - **视觉审计管道：** 通过 Playwright/Puppeteer 将 HTML 文件渲染为 `.png` (`run_sync_screenshots`)。将该 `.png` 发送回视觉智能体 ([review_agent](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#99-122)) 进行视觉检查 ([get_revision_suggestions](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#168-189))。重新生成 HTML 的循环会执行 `args.html_check_times` 次。
* **工具、依赖与 APIs：** CAMEL Vision Agents（带有长多模态参数的 `ChatAgent`），Playwright/Selenium 网页渲染 (`run_sync_screenshots`)，`PIL`，正则表达式文本处理。
* **确切的制品输出：**
  - `generated_project_pages/<PaperName>/<html_dir>/index_init.html`
  - `generated_project_pages/<PaperName>/<html_dir>/table_html/*.html`（代码段表格 blobs）。
  - 截图：`page_final_no_modify_table.png`, `page_iter{N}.png`, `page_final.png`。
  - 最终产品：`generated_project_pages/<PaperName>/<html_dir>/index.html`。
  - 元数据记录：`metadata.json`, `generation_log.json`。
* **人在回路：** 如果 `args.human_input == '1'`，在最终视觉生成后，CLI 会阻塞。指示用户在本地打开 `index.html` 和 `page_final.png`。用户在 [input()](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader.py#73-86) 中提供自由格式反馈，这将触发 `modify_html_from_human_feedback.yaml` (HTML 重新生成)，并反复执行，直到用户输入 `yes`。
