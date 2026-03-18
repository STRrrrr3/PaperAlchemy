# Architectural Workflow Analysis: PaperAlchemy vs. AutoPage

This document provides a highly technical, end-to-end deconstruction of the execution pipelines for two automated paper-to-page construction codebases: **PaperAlchemy** and **AutoPage**. 

The analysis strictly adheres to chronological execution order, breaking down every technical nuance across five critical dimensions: Core Logic, Implementation Mechanisms, Tools & Dependencies, Artifact Outputs, and Human-in-the-Loop workflows.

---

## Part 1: PaperAlchemy Workflow Deconstruction

PaperAlchemy employs a highly structured, multi-agent pipeline orchestrated via LangGraph. The codebase clearly separates parsing, reading, planning, and coding phases into distinct agent modules.

### Phase 1: PDF Parsing Phase
* **Step Definition & Core Logic:** Located in [src/parser.py](file:///e:/Graduation%20Design/PaperAlchemy/src/parser.py). The module converts target academic PDFs into structured Markdown, extracts tables/images, and generates full-page screenshots to provide visual context to downstream agents.
* **Implementation Mechanisms:** The [parse_pdf](file:///e:/Graduation%20Design/PaperAlchemy/src/parser.py#41-153) function initializes a `DocumentConverter` with `PdfPipelineOptions`. It captures OCR (`do_ocr=True`), table structure (`do_table_structure=True`), page screenshots (`generate_page_images=True`), and individual charts (`generate_picture_images=True`). It then iterates over the document's items (`PictureItem`, `TableItem`) to isolate assets and build a bounding box/location map per page.
* **Tools, Dependencies & APIs:** 
  - `docling` & `docling_core` for multi-modal PDF parsing.
  - Generates images via `PIL.Image`.
  - Runs with CUDA acceleration if available (`AcceleratorDevice.CUDA`).
* **Exact Artifact Outputs:** 
  - `data/output/<PaperName>/full_paper.md`: Full text markdown with `REFERENCED` image paths.
  - `data/output/<PaperName>/parsed_data.json`: A structured JSON tree detailing page-by-page references, bounding boxes, and asset links.
  - `data/output/<PaperName>/assets/`: A folder containing extracted figures (`element_*.png`) and full-page screenshots (`page_*.png`).
* **Human-in-the-Loop:** None. Fully automated.

### Phase 2: Reader Agent Phase (Information Extraction)
* **Step Definition & Core Logic:** Located in [src/agent_reader.py](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader.py) and [src/agent_reader_critic.py](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader_critic.py). The agent consumes the raw markdown and assets JSON to extract a canonical semantic representation (`StructuredPaper`) of the paper. It utilizes an actor-critic loop to guarantee output quality.
* **Implementation Mechanisms:** 
  - A `StateGraph` ([ReaderState](file:///e:/Graduation%20Design/PaperAlchemy/src/state.py#7-14)) orchestrates the loop. 
  - **Generator Node:** Reads `full_paper.md` and `parsed_data.json`, injecting previous critic feedback into the prompt if retrying. Calls LLM with `with_structured_output(StructuredPaper)`.
  - **Critic Node:** Runs deterministic checks ([_run_density_checks](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader_critic.py#80-141) for section length, key details count, asset hallucination) and semantic LLM checks ([run_semantic_critic](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader_critic.py#155-187)) against the input context.
  - **Router:** Conditionally loops back to Generator until `critic_passed` is True or `max_retry` (default 3) is reached.
* **Tools, Dependencies & APIs:** LangChain (`langchain_google_genai`), LangGraph (`StateGraph`, `MemorySaver`), Pydantic (`StructuredPaper`), Gemini Pro/Flash API.
* **Exact Artifact Outputs:** `data/output/<PaperName>/structured_paper.json` containing the Pydantic-validated `StructuredPaper` schema (Title, Overall Summary, Array of `PaperSection` elements with key details, content summaries, and related figures).
* **Human-in-the-Loop:** None in the core automated node, though the [main.py](file:///e:/Graduation%20Design/PaperAlchemy/main.py) documentation mentions an interactive check (which appears disabled in the current LangGraph batch run layout).

### Phase 3: Planner Agent Phase (Web Layout Planning)
* **Step Definition & Core Logic:** Located in [src/agent_planner.py](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner.py) and [src/agent_planner_critic.py](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner_critic.py). Responsible for mapping the `StructuredPaper` semantics onto an available HTML template, determining section correspondence and block ordering.
* **Implementation Mechanisms:** 
  - Executes a 4-node `StateGraph` ([PlannerState](file:///e:/Graduation%20Design/PaperAlchemy/src/state.py#16-29)).
  - **Node 1 ([semantic_planner_node](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner.py#115-157)):** Generates an unconstrained `SemanticPlan` (logic layout) using Gemini.
  - **Node 2 ([template_selector_node](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner.py#159-209)):** Scores the `SemanticPlan` against a catalog of parsed local templates to output `TemplateCandidate` items.
  - **Node 3 ([template_binder_node](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner.py#211-281)):** Hydrates a `PagePlan` (schema) that binds the semantic plan strictly to the winning `TemplateCandidate`.
  - **Node 4 ([planner_critic_node](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner_critic.py#170-211)):** Checks for missing assets, missing template files, layout hallucination, and runs a semantic LLM critic ([run_planner_semantic_critic](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner_critic.py#136-168)).
* **Tools, Dependencies & APIs:** LangGraph, Gemini API (Pro/Flash variable temp settings), Pydantic (`PagePlan`, `SemanticPlan`, `TemplateCandidate`).
* **Exact Artifact Outputs:** `data/output/<PaperName>/page_plan.json` conforming to the `PagePlan` schema (contains `template_selection`, `page_outline`, `blocks`, and `coder_handoff`).
* **Human-in-the-Loop:** `template_selection_mode` in constraints allows "human" mode ([_choose_template_with_human](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner.py#71-113)). It pauses execution, prints the top `TemplateCandidate` options (with reasons and scores) to stdout, and uses [input()](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader.py#73-86) to capture a user's integer choice or template ID.

### Phase 4: Coder Agent Phase (HTML Construction)
* **Step Definition & Core Logic:** Located in [src/agent_coder.py](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_coder.py) and [src/agent_coder_critic.py](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_coder_critic.py). Directly manipulates filesystem layouts, copies assets, and injects generated markup into the chosen template's HTML file.
* **Implementation Mechanisms:** 
  - Employs a concise logic graph without direct LLM generation for string building. 
  - **Node 1 ([coder_node](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_coder.py#308-390)):** Uses the `PagePlan` to copy template static domains into `data/output/<PaperName>/site/`. Discards boilerplate HTML `<body>` and generates a fully custom, semantic `<body>` derived directly from `StructuredPaper` and `PagePlan`. Iterates over `page_plan.blocks` appending predefined CSS classes to injected components.
  - **Node 2 ([coder_critic_node](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_coder_critic.py#70-86)):** Performs deterministic code checks. Ensures generated body markers (`BODY_START_MARKER`) exist, verifies exact count of `<title>` tags, and asserts that all `copied_assets` are verifiably written to disc and referenced in the HTML string.
* **Tools, Dependencies & APIs:** Python standard libraries (`shutil`, [re](file:///e:/Graduation%20Design/PaperAlchemy/.gitignore), [html](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#463-489)), LangGraph.
* **Exact Artifact Outputs:** 
  - `data/output/<PaperName>/coder_artifact.json` capturing the `CoderArtifact` schema.
  - `data/output/<PaperName>/site/` containing the fully deployable static site (HTML, CSS, Assets).
* **Human-in-the-Loop:** None. fully automated deterministic rendering. 

---

## Part 2: AutoPage Workflow Deconstruction

AutoPage operates as an iterative, procedural pipeline primarily encapsulated in a single main thread loop and executed via `CAMEL` framework agents. It emphasizes iterative UI preview checks and extensive visual/CSS modifications.

### Phase 1: Pre-requisite Template Matching & Setup
* **Step Definition & Core Logic:** Located in [AutoPage/app.py](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/app.py) and [ProjectPageAgent/main_pipline.py](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/main_pipline.py). User constraints (Background color, Density, Navigation, Layout) are scored against a database of templates. 
* **Implementation Mechanisms:** [matching(requirement)](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/main_pipline.py#33-55) method applies weighted scores to user features parsed against a predefined [tags.json](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/tags.json) file. The local template directory is readied, and a local HTTP preview server is potentially spawned ([find_free_port](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/app.py#114-123), [CustomHTTPRequestHandler](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/app.py#107-113)) for user selection.
* **Tools, Dependencies & APIs:** `gradio` (for UI preview), Custom HTTP servers (`http.server`), [tags.json](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/tags.json).
* **Exact Artifact Outputs:** Active HTTP server port for preview, template local path.
* **Human-in-the-Loop:** User inputs GUI parameters in Gradio, clicks a template preview, and confirms execution. Alternative CLI in [main_pipline.py](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/main_pipline.py) uses [input()](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader.py#73-86) if template is ambiguous.

### Phase 2: PDF Parsing Phase
* **Step Definition & Core Logic:** Located in [ProjectPageAgent/parse_raw.py](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/parse_raw.py). Converts PDF texts, extracts Markdown, detects image items. Fallback extraction exists if `docling` yields short data. Asserts extraction via LLM layout structuring.
* **Implementation Mechanisms:** 
  - Uses `docling` `DocumentConverter`. 
  - Sub-routine: Runs a `ChatAgent` (via `gen_page_raw_content.txt`) forcing the LLM to slice the raw markdown into logical dictionary "sections". Iterates with `try-catch` loops until JSON is valid and contains "title".
  - Sub-routine: Fallback uses `marker` (ML model) if docling text falls below 500 chars.
  - Sub-routine: Down-samples `sections` arrays to 9 logic chunks max using random slicing (`random.sample`).
  - Calls [gen_image_and_table](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/parse_raw.py#139-256) to grab PIL crops and writes independent HTML/MD artifacts.
* **Tools, Dependencies & APIs:** `docling`, `PIL`, `marker`, CAMEL (`ChatAgent`, `ModelFactory`).
* **Exact Artifact Outputs:** 
  - `project_contents/<PaperName>_raw_content.json` (Containing the LLM-extracted section dictionary).
  - `generated_project_pages/images_and_tables/<PaperName>/` (containing multiple [md](file:///e:/Graduation%20Design/PaperAlchemy/README.md), [html](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#463-489), and [png](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#156-167) extraction artifacts).
  - Figure manifests: `_images.json` and `_tables.json`.
* **Human-in-the-Loop:** None.

### Phase 3: Content Planner Operations
* **Step Definition & Core Logic:** Located in [ProjectPageAgent/content_planner.py](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/content_planner.py). Iteratively constructs the text payload of the site in isolated stages (Filter -> Section -> Text Content -> Full Content).
* **Implementation Mechanisms:** 
  - **Filter Content:** Removes "References" section. Deletes assets unmapped to logic sections using a CAMEL agent (`filter_figures.yaml`).
  - **Section Generation:** Detects paper "domain" (`technical` vs `other`), utilizes variable yaml templates (`adaptive_sections.yaml` or `section_generation.yaml`) to output a JSON object describing the target semantic keys (e.g., `title`, `authors`, `abstract`).
  - **Full Content Generation Loop:** Uses an Actor-Critic architecture. The Actor ([planner_agent](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_planner.py#305-376)) outputs JSON content (`generated_full_content.v0.json`). The Critic ([reviewer_agent](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/content_planner.py#69-95) with `full_content_review.yaml`) critiques the JSON. Actor processes the feedback (`full_content_revise.yaml`) for `full_content_check_times` iterations.
* **Tools, Dependencies & APIs:** CAMEL, Jinja2 template rendering.
* **Exact Artifact Outputs:** 
  - `project_contents/<PaperName>_generated_section.json`
  - `project_contents/<PaperName>_generated_text_content.json`
  - Series of outputs: `project_contents/<PaperName>_generated_full_content.v{N}.json` and `review.iter{N}.json`.
* **Human-in-the-Loop:** After the critic loop completes, if `args.human_input == '1'`, it pauses string execution, prints a `rich.Pretty` JSON tree to terminal, and awaits [input()](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader.py#73-86). The user can type English feedback. The agent iterates dynamically until the user types `yes`.

### Phase 4: HTML Generator & Visual Feedback Phase
* **Step Definition & Core Logic:** Located in [ProjectPageAgent/html_generator.py](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py) and `css_checker.py`. Maps generated content dynamically into original template HTML, fixes HTML tables via pure Vision-Language-Model (VLM) manipulation, and executes end-to-end visual review.
* **Implementation Mechanisms:** 
  - **Base HTML ([generate_complete_html](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#372-461)):** The LLM (`html_generation.yaml`) merges the `generated_content` dictionary directly into the source HTML template string. Parses and fixes broken CSS links (`check_css`).
  - **Table Modification ([modify_html_table](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#191-329)):** VLM parses original table `.png` screenshots. An LLM agent ([table_agent](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#124-140)) converts the pixel data to pure HTML `<table>` tags. Then, a "Color Suggestion" is drawn by sending a screenshot of the current page rendering to the VLM. A long-context agent merges the color palette + HTML tables into the master HTML.
  - **Visual Audit Pipeline:** Renders the HTML file to a `.png` via Playwright/Puppeteer (`run_sync_screenshots`). Sends the `.png` back to a Vision Agent ([review_agent](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#99-122)) for visual inspection ([get_revision_suggestions](file:///e:/Graduation%20Design/PaperAlchemy/AutoPage/ProjectPageAgent/html_generator.py#168-189)). Regenerates HTML loop for `args.html_check_times`.
* **Tools, Dependencies & APIs:** CAMEL Vision Agents (`ChatAgent` with multimodal params), Playwright/Selenium web rendering (`run_sync_screenshots`), `PIL`, regex text processing.
* **Exact Artifact Outputs:** 
  - `generated_project_pages/<PaperName>/<html_dir>/index_init.html`
  - `generated_project_pages/<PaperName>/<html_dir>/table_html/*.html` (Snippet table blobs).
  - Screenshots: `page_final_no_modify_table.png`, `page_iter{N}.png`, `page_final.png`.
  - Final product: `generated_project_pages/<PaperName>/<html_dir>/index.html`.
  - Metadata record: `metadata.json`, `generation_log.json`.
* **Human-in-the-Loop:** If `args.human_input == '1'`, upon final visual generation, the CLI blocks. The user is instructed to open `index.html` and `page_final.png` locally. The user provides free-form feedback in [input()](file:///e:/Graduation%20Design/PaperAlchemy/src/agent_reader.py#73-86), which triggers `modify_html_from_human_feedback.yaml` (HTML re-generation) repeatedly until the user types `yes`.
