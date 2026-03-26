# PaperAlchemy: Current Project Architecture Brief for ChatGPT

This document is meant to be pasted into ChatGPT before discussing the next design step.
It is based on the current repository implementation, not on an older conceptual description.

## 1. What This Project Does

PaperAlchemy is a staged multi-agent system that turns an academic PDF into a static project webpage.

At a high level, the pipeline is:

1. Parse the PDF into markdown plus extracted visual assets.
2. Use a Reader agent to convert the raw paper into a webpage-oriented semantic content pack.
3. Use a Planner agent to map that content onto a selected local HTML template.
4. Use a Coder agent to generate a full `index.html` that stays compatible with the chosen template's shell and styles.
5. Let a human review the result and request anchored revisions through Translator + Patch agents instead of regenerating the entire page each time.

The system is not a generic "ask an LLM to write a webpage from scratch" demo.
It is a controlled pipeline with typed schema boundaries, a local template inventory, critique/validation steps, and human checkpoints.

## 2. Core Product Philosophy

The current implementation follows these principles:

- Template-first generation: a human first picks a template candidate from a local catalog.
- Schema-first handoff: each stage passes typed data, mainly via Pydantic models.
- Critic-assisted generation: Reader, Planner, and Coder each have validation layers.
- Human-in-the-loop workflow: the pipeline intentionally pauses at review checkpoints.
- Revision-friendly HTML: generated pages must expose stable `data-pa-*` anchors so later edits can target precise regions safely.

This means the project is closer to a controllable webpage production pipeline than a single-shot text-to-HTML generator.

## 3. Main Entry Points

- `main.py`: launches the app.
- `app.py`: the real application entry and workflow controller. It contains:
  - the Gradio UI
  - the LangGraph HITL workflow
  - checkpoint resume logic
  - template search and preview
  - stage approval/revision handlers
- `src/`: implementation of agents, schemas, validation, template resources, and patching logic.

## 4. Current End-to-End Runtime Flow

### 4.1 Template Selection Before Generation

The UI asks the user to choose high-level style constraints:

- background color
- page density
- navigation yes/no
- layout style

These constraints are scored against local template tags through `src/deterministic_template_selector.py`.
The user is shown the top 5 ranked templates and previews one before starting generation.

Important detail: the user-selected template is treated as designated input for downstream planning. The Planner is not free to switch to another template.

### 4.2 PDF Parsing

`src/parser.py` uses Docling to parse the PDF and produce:

- `full_paper.md`
- `parsed_data.json`
- page screenshots
- extracted figure/table images under `assets/`

The parser enables:

- OCR
- table structure extraction
- page image generation
- figure/table image generation

Output location:

- `data/output/<paper_name>/full_paper.md`
- `data/output/<paper_name>/parsed_data.json`
- `data/output/<paper_name>/assets/...`

If `full_paper.md` and `parsed_data.json` already exist, the app reuses them and skips parsing.

### 4.3 Reader Phase

Files:

- `src/agent_reader.py`
- `src/agent_reader_critic.py`

Reader input:

- `full_paper.md`
- extracted assets list from `parsed_data.json`
- optional human directives from the review UI
- optional previous `StructuredPaper`

Reader output:

- `StructuredPaper`

Reader responsibilities:

- recover paper identity from noisy front matter
- preserve author and affiliation information in human-visible text
- choose only landing-page-worthy sections instead of mirroring the full paper
- produce dense `rich_web_content` for each kept section
- attach only high-value figures/tables to sections

The Reader prompt explicitly assumes downstream agents will not reread the raw markdown, so the extraction must be information-dense enough to support planning and webpage generation later.

#### Reader Critic

The Reader Critic has two layers:

- deterministic checks:
  - referenced asset paths must exist
  - section count must be reasonable
  - summaries and key sections must have enough density
  - method/evaluation sections must be technically rich
- semantic LLM critic:
  - checks whether the extraction is grounded and sufficiently useful for downstream generation

If the critic fails, the Reader loops with feedback until max retry is reached.

### 4.4 Planner Phase

Files:

- `src/agent_planner.py`
- `src/agent_planner_critic.py`

Planner input:

- `StructuredPaper`
- selected template candidate
- template catalog
- DOM outline of the selected template entry HTML
- optional previous `PagePlan`
- optional human directives

Planner output:

- `PagePlan`

Planner responsibilities:

- keep the selected template fixed
- convert paper semantics into a page blueprint
- define the page outline as stable semantic blocks
- map blocks to template regions
- define content contracts, asset bindings, interactions, and implementation handoff
- generate `dom_mapping` and `selectors_to_remove`

The Planner is "DOM-aware" but not yet doing final HTML generation. It builds the contract that the Coder must follow.

#### Planner Critic

The Planner Critic also has two layers:

- deterministic checks:
  - selected template must exist in the template catalog
  - `selected_entry_html` must be valid
  - `block_id` values must be stable snake_case
  - no positional IDs like `section_1`
  - source section references must exist
  - asset paths must exist in `StructuredPaper`
  - block IDs in `page_outline` and `blocks` must match exactly
- semantic LLM critic:
  - checks design feasibility and planning quality

If the critic fails, the Planner loops and retries.

### 4.5 Outline Review Checkpoint

After planning, the workflow pauses in the UI at the outline review stage.

The human can:

- revise the outline by giving instructions
- approve the outline
- optionally enable manual layout compose before draft generation

This stage is the main semantic review boundary before HTML is generated.

### 4.6 Optional Manual Layout Compose Stage

Files:

- `src/template_shell_resolver.py`
- related UI handlers in `app.py`

This stage is optional and only happens if the human checks `Enable Manual Layout Compose` during outline approval.

It lets the human:

- inspect candidate template sections
- choose which template section a block should bind to
- reorder blocks
- choose which extracted figures belong to each block

The state object used here is `LayoutComposeSession`.

This is important: manual layout compose is now optional.
If it is not enabled, the workflow goes from outline approval directly to Coder.

### 4.7 Coder Phase

Files:

- `src/agent_coder.py`
- `src/agent_coder_critic.py`

Coder input:

- `StructuredPaper`
- `PagePlan`
- template reference HTML
- copied paper assets
- optional human directives
- optional coder instructions
- previous generated HTML

Coder output:

- `CoderArtifact`
- optional `VisualSmokeReport`
- possibly an enriched/resolved `PagePlan`

Important correction versus older descriptions:

The current Coder is not just a deterministic BeautifulSoup injector that fills fixed selectors.
It asks the LLM to generate a complete HTML document, but under strict constraints:

- reuse the selected template's design language and structure
- preserve shell compatibility with each block's target region
- generate exactly one root element per block with `data-pa-block`
- use only the fixed slot vocabulary:
  - `title`
  - `summary`
  - `body`
  - `media`
  - `meta`
  - `actions`
- preserve required global anchors when applicable
- reference only copied paper assets
- produce a full valid HTML document with body markers

Before generation, the Coder:

- copies the entire template directory to `data/output/<paper>/site`
- copies selected paper assets into `site/assets/paper`
- enriches the `PagePlan` with shell contracts if needed

After generation, it:

- validates local image references
- extracts a `PageManifest`
- saves the HTML and manifest

Generated artifacts typically include:

- `data/output/<paper>/site/index.html`
- `data/output/<paper>/coder_artifact.json`
- `data/output/<paper>/page_manifest.json`

### 4.8 Coder Critic and Visual QA

The Coder Critic validates:

- generated site directory exists
- entry HTML exists
- exactly one `<title>` exists
- body markers exist in the correct place
- `page_manifest.json` matches current HTML
- `data-pa-block` order matches approved page outline
- copied assets actually exist and are referenced

Then the system takes a screenshot with Playwright and runs a visual QA pass.

The visual critic returns a `VisualSmokeReport` with:

- `passed`
- `issue_class`: `none`, `cosmetic`, or `structure`
- `suggested_recovery`: `accept`, `patch_or_review`, or `rerun_planner`
- issue list
- optional selectors/CSS suggestions

Important implementation detail:

The visual critic currently classifies and reports issues, but it does not run a deep automatic retry loop back into Coder.
Instead:

- if the issue is structural, the app can route back to Planner
- otherwise the workflow proceeds to human review and anchored patching

### 4.9 Webpage Review and Anchored Revision Loop

Files:

- `src/agent_translator.py`
- `src/agent_patch.py`

After first draft generation, the workflow pauses again at webpage review.

The human can:

- approve the final webpage
- request a revision with text and optional screenshots

#### Translator Agent

The Translator converts multimodal human feedback into a structured `RevisionPlan`.

It uses:

- current HTML
- current `PageManifest`
- uploaded screenshots
- human text feedback

The Translator does not directly edit HTML.
It only decides what the requested edit is and where it should apply:

- slot-level
- block-level
- global anchor-level

#### Patch Agent

The Patch Agent converts `RevisionPlan` into a `TargetedReplacementPlan` containing:

- HTML replacements
- style changes
- attribute changes
- override CSS rules
- fallback blocks

This is still schema-bound and anchor-aware.

#### Patch Executor

The Patch Executor applies changes deterministically with BeautifulSoup and guarded validation.

It can:

- replace slot inner HTML
- replace block HTML
- update attributes
- inject limited anchored CSS overrides
- fall back to block-level regeneration for hard cases

If a requested edit cannot be safely applied, it returns a safe failure instead of silently corrupting the page.

## 5. The Active HITL LangGraph Workflow

The currently wired main workflow in `app.py` is:

1. `reader`
2. `overview`
3. human checkpoint
4. `planner`
5. `outline_review`
6. human checkpoint
7. optional `layout_compose_prepare`
8. optional `layout_compose_review`
9. human checkpoint
10. `coder`
11. if structural visual issue: back to `planner`
12. otherwise `webpage_review`
13. human checkpoint
14. `translator`
15. `patch_agent`
16. `patch_executor`
17. back to `webpage_review`

Interrupt points are:

- `overview`
- `outline_review`
- `layout_compose_review`
- `webpage_review`

This means the user-facing system is designed around resumable checkpoints, not one uninterrupted batch pass.

## 6. Important Data Models

### 6.1 `StructuredPaper`

Main content pack produced by Reader.

- `paper_title`
- `overall_summary`
- `sections[]`
  - `section_title`
  - `rich_web_content`
  - `related_figures[]`

### 6.2 `PagePlan`

Planner blueprint for webpage generation.

Key parts:

- `plan_meta`
- `template_selection`
- `decision_summary`
- `adaptation_strategy`
- `global_design`
- `page_outline`
- `blocks`
- `dom_mapping`
- `selectors_to_remove`
- `coder_handoff`
- `quality_checks`
- `open_questions`

Each block is described by `BlockPlan`, which includes:

- target template region
- content contract
- asset binding
- interaction
- responsive rules
- optional shell contract

### 6.3 `CoderArtifact`

Saved output of Coder / patch flow.

- `site_dir`
- `entry_html`
- `selected_template_id`
- `copied_assets`
- `edited_files`
- `notes`

### 6.4 `PageManifest`

The manifest that enables safe anchored revision.

It records:

- all blocks and their selectors
- all slots inside each block
- all global anchors
- shell-related structural metadata

### 6.5 `RevisionPlan`

Translator output: "what the human wants changed".

### 6.6 `TargetedReplacementPlan`

Patch Agent output: "how to apply the requested change safely".

### 6.7 `VisualSmokeReport`

Visual QA classification and recovery suggestion.

### 6.8 `WorkflowState`

The top-level LangGraph state combines:

- source metadata
- generation constraints
- manual layout compose toggle
- human directives
- review flags
- structured paper
- page plan
- coder artifact
- revision and patch data
- visual smoke report

## 7. Revision Anchoring System

This is one of the most important architectural pieces.

Generated HTML must expose:

- `data-pa-block`
- `data-pa-slot`
- `data-pa-global`

This makes later changes targetable and safe.

Allowed global anchors are:

- `header_brand`
- `header_primary_action`
- `header_nav`
- `footer_meta`

Allowed slots are:

- `title`
- `summary`
- `body`
- `media`
- `meta`
- `actions`

If future design changes break these anchors, the revision system will break too.

## 8. Template System

Templates come from AutoPage-style local assets.

Relevant files/modules:

- `src/template_resources.py`
- `src/planner_template_catalog.py`
- `src/deterministic_template_selector.py`

The system syncs or reuses template assets under:

- `data/templates/autopage/`

The selector uses tagged template metadata and a small weighted scoring model over features such as:

- background color
- hero presence
- density
- image layout
- title color
- navigation

The selected template is previewed before the Reader phase starts.

## 9. LLM Provider Configuration

`src/llm.py` currently does this:

- first tries Vertex AI if it can auto-discover exactly one valid service account JSON in the project root, or if an explicit path is provided through environment variables
- otherwise falls back to `GOOGLE_API_KEY`

Current code defaults:

- smart model: `gemini-3.1-pro-preview`
- fast model: `gemini-3-flash-preview`

The project also sets proxy-related environment variables from `.env`.

## 10. What Is Actually User-Facing in the Current UI

The current Gradio UI is stage-based.

Main visible flow:

1. Choose PDF and style constraints.
2. Find and preview template candidates.
3. Run "Step 1: Extract Source Pack".
4. Review Reader output.
5. Approve to generate Planner outline.
6. Review or revise outline.
7. Optionally enable manual layout compose.
8. Generate first draft.
9. Review screenshot preview of generated webpage.
10. Request webpage revisions or approve final page.

The left control panel only shows the controls relevant to the current stage.
Layout compose has its own dedicated right-side panel.

## 11. Important Architectural Invariants

If ChatGPT is helping design the next step, it should treat these as non-negotiable unless a larger migration is explicitly intended.

### 11.1 Schema Boundaries Matter

Do not casually pass free-form strings between pipeline stages.
New cross-stage information should normally be added to `schemas.py` and `state.py`.

### 11.2 Stable Block IDs Matter

`block_id` is part of the revision contract.
It must remain stable and semantic across replans when the conceptual section still exists.

### 11.3 Anchors Matter

Do not propose changes that remove or destabilize:

- `data-pa-block`
- `data-pa-slot`
- `data-pa-global`
- `page_manifest.json`

### 11.4 Template Compatibility Matters

The system is not supposed to ignore the chosen template and invent a separate layout language.
Coder output must stay compatible with the template shell and class system.

### 11.5 Patch Scope Is Intentionally Local

The Translator/Patch system is built for targeted local corrections, not for arbitrary whole-page redesign.
Large structural changes should go back to Planner or Outline review.

### 11.6 Downstream Safety Matters

This repository is a staged pipeline.
Changing Reader output shape, Planner contracts, manifest structure, slot vocabulary, or shell assumptions can silently break later stages.

## 12. Subtle but Important Implementation Notes

### 12.1 There Are Shell Resolution Utilities Beyond the Main Flow

The repository includes:

- `shell_resolver_phase_node()`
- `binding_review_node()`

But these are not currently wired into the main HITL LangGraph path in `build_hitl_workflow()`.

In practice:

- shell-related logic is actively used by manual layout compose utilities
- batch generation helper code also invokes shell resolution
- the main interactive path relies on Planner output plus Coder-side shell enrichment and validation

So shell-aware tooling exists, but the standalone shell-binding review stage is not currently a first-class checkpoint in the active UI workflow.

### 12.2 There Are Two Runtime Styles

The repository contains:

- a checkpointed HITL workflow used by the UI
- a batch helper (`run_langgraph_batch`) for non-interactive generation

The HITL path is the primary product path.

### 12.3 Visual QA Can Send the User Back to Planning

If the visual critic reports a structural issue, the app routes back to outline/planning instead of opening the normal webpage revision loop.

This is a core distinction:

- cosmetic issue -> patch/review loop
- structural issue -> planner rerun

## 13. Known Strengths

- Strong staged decomposition
- Good human review checkpoints
- Anchored revision system is much safer than raw HTML regeneration
- Template-first workflow gives controllability
- Rich technical extraction aims to preserve enough detail for serious academic pages
- Optional manual layout compose gives a controlled bridge between automation and explicit layout editing

## 14. Known Limitations / Design Tensions

These are not necessarily bugs, but they are real design constraints worth knowing before proposing the next plan.

- The Coder is still LLM-generated full HTML, so quality depends heavily on prompt discipline and validation.
- Visual QA currently classifies issues more than it automatically repairs them.
- Shell resolution exists in code, but its dedicated review node is not fully integrated into the main interactive graph.
- The patch system is intentionally conservative and best for local revisions, not major layout overhauls.
- Author and affiliation metadata are still embedded inside summary/section text rather than stored in dedicated schema fields.
- The project contains both interactive and batch logic, which may drift if future features are added to only one path.

## 15. Best Summary for ChatGPT

If you need one compact mental model, use this:

PaperAlchemy is a template-first, schema-driven, multi-agent academic webpage generation system with four major production layers:

1. Reader extracts a landing-page-oriented semantic paper pack.
2. Planner turns that pack into a template-bound page blueprint.
3. Coder generates a full anchored HTML page that stays compatible with the chosen template shell.
4. Translator + Patch convert human feedback into safe targeted revisions over stable HTML anchors.

The most important architectural contracts are:

- `StructuredPaper`
- `PagePlan`
- `PageManifest`
- `CoderArtifact`
- `RevisionPlan`
- `TargetedReplacementPlan`
- `data-pa-block / slot / global`

If you propose next-step designs, optimize for:

- preserving staged contracts
- keeping template compatibility
- preserving revision anchors
- making human checkpoints stronger instead of bypassing them
- pushing large structural changes upstream to Planner instead of overloading Patch
