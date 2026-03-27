# PaperAlchemy: Current Architecture Context Pack for ChatGPT

This document is intended to be pasted into ChatGPT before discussing the next design step.
It is based on the repository state inspected on March 27, 2026.
It should be treated as a code-derived architecture brief, not a marketing summary.

## 1. Project Purpose

PaperAlchemy is a multi-stage system that transforms an academic PDF into a static project webpage.

The product goal is not "let an LLM freestyle an academic site from scratch."
The goal is controlled page generation with:

- a fixed local template inventory
- staged semantic extraction
- explicit planning contracts
- human review checkpoints
- revision-safe HTML anchors
- deterministic patch execution for local edits

In practical terms, the system takes a PDF, extracts a webpage-oriented semantic representation, maps that representation onto a chosen template, generates a webpage draft, and then supports human-guided iterative revision.

## 2. The Most Important Current Architecture Fact

The project has evolved beyond the older "template-first planner plus fullpage coder" description.

The current architecture now has two rendering strategies:

1. `compiled_block_assembly`
2. `legacy_fullpage`

`compiled_block_assembly` is the preferred path.
In that mode, the system:

- compiles the selected template into a `TemplateProfile`
- plans against compiled shell candidates instead of raw DOM outline text
- renders each page block separately as a constrained HTML fragment
- assembles the final page programmatically from block fragments

`legacy_fullpage` still exists as a fallback when the template is too risky or compilation confidence is too low.

This is the single most important thing ChatGPT should know before proposing future changes.

## 3. High-Level Runtime Pipeline

The active user-facing flow is:

1. Select a paper PDF and style constraints.
2. Rank and preview template candidates.
3. Parse the PDF into markdown plus assets.
4. Run Reader to produce `StructuredPaper`.
5. Pause for human review of extracted content.
6. Run `template_compile` to produce `TemplateProfile`.
7. Run Planner to produce `PagePlan`.
8. Pause for human review of the outline.
9. Optionally enter manual `layout_compose`.
10. Run Coder to generate the first draft.
11. Run deterministic critic plus visual QA.
12. Pause for webpage review.
13. On revision requests, run Translator.
14. Route feedback through `edit_intent_router`.
15. If patchable, run Patch Agent and Patch Executor.
16. If not patchable, return a safe non-patch message to the webpage review stage.

## 4. Main Runtime Entry Points

- `main.py`
  - thin entry point that launches the app
- `app.py`
  - the real orchestration layer
  - Gradio UI
  - LangGraph HITL workflow
  - stage transitions
  - review-state management
  - persistence helpers
- `src/`
  - all agents, schemas, template compilation, validation, page manifest logic, and patch execution

## 5. Current HITL LangGraph Workflow

The current active graph in `app.py` is:

1. `reader`
2. `overview`
3. human checkpoint
4. `template_compile`
5. `planner`
6. `outline_review`
7. human checkpoint
8. optional `layout_compose_prepare`
9. optional `layout_compose_review`
10. human checkpoint
11. `coder`
12. `webpage_review`
13. human checkpoint
14. `translator`
15. `edit_intent_router`
16. either:
   - `patch_agent -> patch_executor -> webpage_review`
   - or `non_patch_feedback -> webpage_review`

Routing details:

- after `overview`
  - approve -> `template_compile`
  - revise -> back to `reader`
- after `outline_review`
  - revise -> back to `planner`
  - approve + manual compose enabled -> `layout_compose_prepare`
  - approve + manual compose disabled -> `coder`
- after `coder`
  - if visual smoke suggests structural recovery -> back to `planner`
  - otherwise -> `webpage_review`
- after `webpage_review`
  - approve -> end
  - revise -> `translator`

Interrupt points are:

- `overview`
- `outline_review`
- `layout_compose_review`
- `webpage_review`

This means the product is explicitly checkpointed and resumable.

## 6. Core Agent and Node Architecture

### 6.1 Parser Layer

Primary file:

- `src/parser.py`

Responsibilities:

- parse PDF with Docling
- export `full_paper.md`
- export `parsed_data.json`
- export page screenshots
- export cropped figure/table images

Artifacts:

- `data/output/<paper>/full_paper.md`
- `data/output/<paper>/parsed_data.json`
- `data/output/<paper>/assets/...`

The parser is a preprocessing stage, not an LLM agent, but the rest of the pipeline depends on its output shape.

### 6.2 Reader Agent

Primary files:

- `src/agent_reader.py`
- `src/agent_reader_critic.py`

Input:

- parser markdown
- extracted asset list
- optional human directives
- optional previous `StructuredPaper`

Output:

- `StructuredPaper`

Reader responsibilities:

- convert raw paper text into a landing-page-oriented semantic content pack
- recover paper identity from front matter
- preserve title, authors, and affiliations inside human-visible text
- select only webpage-worthy sections
- produce dense `rich_web_content` instead of thin summaries
- attach only high-value visual assets

Reader Critic responsibilities:

- validate asset grounding
- validate section density and section coverage
- validate method/evaluation richness
- run semantic LLM review before downstream planning

The Reader still follows an actor-critic pattern with retry support.

### 6.3 Template Compile Stage

Primary file:

- `src/template_compile.py`

This is now a major architectural layer, not a helper.

Input:

- designated template choice
- generation constraints
- user style constraints
- local AutoPage template resources

Output:

- `TemplateCandidate[]`
- selected `TemplateCandidate`
- `TemplateProfile`
- compile cache path
- cache-hit flag

`TemplateProfile` is a compiled structural summary of the selected template.
It contains:

- `archetype`
- `global_preserve_selectors`
- `shell_candidates`
- `optional_widgets`
- `removable_demo_selectors`
- `unsafe_selectors`
- `compile_confidence`
- `risk_flags`
- `notes`
- `source_fingerprint`

What template compilation actually does:

- chooses the designated template or a deterministic fallback
- inspects the entry HTML
- discovers stable shell candidates for major sections
- detects global preserve anchors like header/nav/footer
- identifies demo content that can be removed later
- detects risky widgets and runtime dependencies
- infers an archetype such as `hero_bulma`, `bootstrap_navbar`, `single_column_article`, or `chart_fetch_dashboard`
- caches the compiled result under the template root

The compile cache lives under the template directory:

- `.paperalchemy/template_compile_cache/`

This stage is important because the Planner no longer works primarily from an ad hoc DOM outline.
It now plans against a compiled template understanding.

### 6.4 Planner Agent

Primary files:

- `src/agent_planner.py`
- `src/agent_planner_critic.py`

Input:

- `StructuredPaper`
- selected `TemplateCandidate`
- `TemplateProfile`
- template catalog
- optional previous `PagePlan`
- human directives

Output:

- `PagePlan`

The Planner now uses `TemplateProfile` directly.

Current Planner responsibilities:

- keep the selected template fixed
- plan stable semantic `block_id`s
- bind each block to a compiled shell candidate
- preserve global template anchors through compatibility `dom_mapping`
- decide whether the Coder should use:
  - `compiled_block_assembly`
  - or `legacy_fullpage`
- surface known template/render risks into `coder_handoff.known_risks`

Important current behavior:

- `dom_mapping` is no longer the main rendering interface
- it is now mainly a compatibility field for preserved global selectors
- block binding comes from `TemplateProfile.shell_candidates`

#### Planner Critic

The Planner Critic checks:

- stable semantic block IDs
- source section validity
- asset-path grounding
- selected template consistency
- selector hints must exist in `TemplateProfile.shell_candidates`
- `dom_mapping` must only reference template-profile global preserve selectors
- `plan_meta.render_strategy` must match template compile risk

This means planning is now coupled to template compilation quality, not just semantic structure.

### 6.5 Optional Manual Layout Compose

Primary files:

- `src/template_shell_resolver.py`
- UI handlers in `app.py`

This stage is still optional and only appears if the human enables it during outline approval.

What the user can do here:

- inspect available section-shell candidates
- reassign a block to a different shell candidate
- reorder blocks
- choose which extracted figures belong to a block

This stage now works on top of `TemplateProfile`-derived shell candidates.

So manual layout compose is no longer just "pick selectors from raw HTML."
It is a human override layer over compiled template structure.

### 6.6 Coder Agent

Primary file:

- `src/agent_coder.py`

Current Coder architecture has two modes.

#### Preferred mode: `compiled_block_assembly`

This is the new main path.

Pipeline inside this mode:

1. Read the selected template and copy it into the output site directory.
2. Copy selected paper assets into `site/assets/paper`.
3. Build a `BlockRenderSpec` for each planned block.
4. Call the Block Renderer prompt once per block.
5. Validate each rendered block fragment.
6. Persist block specs and block artifacts to disk.
7. Assemble the page programmatically by inserting block fragments into template shells.
8. Preserve global anchors and remove known demo/template garbage.
9. Extract and save `PageManifest`.
10. Save `CoderArtifact`.

This is much more structured than the old full-page coder path.

#### Fallback mode: `legacy_fullpage`

This is still present and is used when:

- template compile confidence is too low
- template widgets are risky
- or compiled block assembly fails and fallback is triggered

In this mode, the LLM generates the whole page HTML subject to shell contracts and anchor rules.

#### Key Coder outputs

- `CoderArtifact`
- possibly updated `PagePlan`
- `BlockRenderSpec[]`
- `BlockRenderArtifact[]`

#### New compiled-path data objects

`BlockRenderSpec` includes:

- block identity
- order
- source sections
- resolved block binding
- content contract
- asset binding
- interaction
- responsive rules
- shell HTML
- allowed slots

`BlockRenderArtifact` includes:

- rendered fragment HTML
- output paths
- validation errors
- notes

### 6.7 Coder Critic and Visual QA

Primary file:

- `src/agent_coder_critic.py`

Responsibilities:

- validate generated files
- validate anchors
- validate manifest synchronization
- validate copied asset references
- validate page title/body markers
- run Playwright screenshot capture
- run visual smoke QA

Visual QA returns `VisualSmokeReport`:

- `passed`
- `issue_class`
- `suggested_recovery`
- `issues`
- `selectors_to_remove`
- `css_rules_to_inject`

Current recovery policy:

- structural visual issue -> route back to Planner
- cosmetic/local issue -> keep webpage review open for patch loop

### 6.8 Translator Agent

Primary file:

- `src/agent_translator.py`

Responsibilities:

- read human text plus optional screenshots
- read current HTML and `PageManifest`
- translate human feedback into structured `RevisionPlan`

This remains the "intent extraction" stage for revisions.

### 6.9 Edit Intent Router

Primary file:

- `src/agent_translator.py`
  - `edit_intent_router_node`

This is a newer and important node.

Its purpose is to decide whether the revision request is patchable or not.

Current routing classes:

- `patch`
- `non_patch`

Examples of requests classified as `non_patch`:

- whole-page redesign
- global theme/style retuning
- template replacement
- replan/rebind requests
- page-wide density/rhythm/style changes

If the request is `non_patch`, the system does not attempt DOM patching.
Instead it returns a safe message and loops back to webpage review.

This is important because it prevents the patch path from being overloaded with requests that should really go back upstream to planning or template selection.

### 6.10 Patch Agent and Patch Executor

Primary file:

- `src/agent_patch.py`

This part of the architecture is still anchored and deterministic.

The Patch Agent converts `RevisionPlan` into `TargetedReplacementPlan`, containing:

- replacements
- style changes
- attribute changes
- override CSS rules
- fallback blocks

The Patch Executor then applies those edits with BeautifulSoup while enforcing:

- anchor safety
- slot/block/global scope correctness
- asset safety
- manifest validity
- shell-contract compatibility

For hard cases, it can regenerate a single block instead of rewriting the whole page.

## 7. Current Core Data Contracts

### 7.1 `StructuredPaper`

Reader output.

Key fields:

- `paper_title`
- `overall_summary`
- `sections[]`

Each section includes:

- `section_title`
- `rich_web_content`
- `related_figures[]`

### 7.2 `TemplateCandidate`

Represents a user-selected or deterministically selected template candidate.

Key fields:

- `template_id`
- `root_dir`
- `chosen_entry_html`
- `score`
- `reasons`

### 7.3 `TemplateProfile`

Compiled template understanding.

This is now one of the most important contracts in the system.

Key fields:

- `template_id`
- `template_root_dir`
- `entry_html`
- `archetype`
- `global_preserve_selectors`
- `shell_candidates`
- `optional_widgets`
- `removable_demo_selectors`
- `unsafe_selectors`
- `compile_confidence`
- `risk_flags`
- `notes`
- `source_fingerprint`

### 7.4 `PagePlan`

Planner output and Coder blueprint.

Important fields:

- `plan_meta`
  - now includes `render_strategy`
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

### 7.5 `BlockRenderSpec`

Compiled-block Coder input for each block fragment.

### 7.6 `BlockRenderArtifact`

Compiled-block Coder output for each block fragment.

### 7.7 `CoderArtifact`

Final build artifact.

Important fields now include:

- `render_mode`
- `template_profile_path`
- `page_manifest_path`
- `block_artifact_dir`

### 7.8 `PageManifest`

Stable revision manifest extracted from final HTML.

This is central to patch safety.

### 7.9 `RevisionPlan`

Translator output.

### 7.10 `TargetedReplacementPlan`

Patch Agent output.

### 7.11 `VisualSmokeReport`

Visual QA output.

## 8. Output Artifacts and Persistence Layout

For each paper, the main output directory is:

- `data/output/<paper_name>/`

Typical files now include:

- `full_paper.md`
- `parsed_data.json`
- `structured_paper.json`
- `template_profile.json`
- `page_plan.json`
- `coder_artifact.json`
- `page_manifest.json`
- `site/index.html`
- `site/assets/paper/...`
- `block_specs/<block_id>.json`
- `block_renders/<block_id>.html`
- `block_renders/<block_id>.json`

This is important for future tooling because the system now persists both plan-level and block-level generation artifacts.

## 9. Template Compilation Details That Matter for Future Design

The compile layer is deliberately doing more than selector discovery.

It also tries to answer:

- Which major shells are safe to reuse?
- Which parts of the template must be preserved globally?
- Which parts are just demo content and should be removed?
- Which widgets introduce runtime risk?
- Is the template safe enough for block assembly, or should the system fall back to fullpage rendering?

Current risk signals include things like:

- `fetch_runtime_dependency`
- `chart_runtime_dependency`
- `math_runtime_dependency`

Current practical rule:

- if `compile_confidence < 0.70` or risky widget/runtime flags are present, Planner tends to choose `legacy_fullpage`
- otherwise Planner prefers `compiled_block_assembly`

## 10. Current Prompt Strategy

The system now has three different LLM generation styles in the rendering layer:

1. full semantic extraction
   - Reader
2. structured planning
   - Planner
3. constrained rendering
   - Block Renderer or legacy fullpage Coder

Most important rendering prompt distinction:

- `BLOCK_RENDER_SYSTEM_PROMPT`
  - render exactly one block fragment
  - preserve shell contract
  - preserve exact `data-pa-block`
  - use only allowed slots
- `CODER_SYSTEM_PROMPT`
  - whole-page fallback path
  - still shell-constrained and anchor-constrained

So the system now prefers fragment-level rendering over page-level rendering whenever the template is safe enough.

## 11. Current Revision Philosophy

Revision is intentionally conservative.

The patch path is for:

- local content fixes
- local layout fixes
- slot-level or block-level changes
- global-anchor updates for header/nav/footer/button regions

The patch path is not for:

- replacing the template
- redesigning the whole page
- global theme overhaul
- major replanning
- shell rebinding as a casual feedback request

That is why `edit_intent_router` exists.

## 12. Important Architectural Invariants

If ChatGPT is helping design the next step, it should assume these are important unless the goal is an explicit large migration.

### 12.1 Staged Contracts Matter

Do not bypass `StructuredPaper -> TemplateProfile -> PagePlan -> CoderArtifact -> PageManifest`.

### 12.2 `TemplateProfile` Is Now a First-Class Contract

Planning and rendering both depend on it.
Future changes should not treat template understanding as an informal side-input anymore.

### 12.3 `PagePlan.blocks[*].target_template_region.selector_hint` Must Stay Grounded

In the current design, block selectors should come from compiled shell candidates, not invented selectors.

### 12.4 `plan_meta.render_strategy` Is Not Cosmetic

It controls which render path the Coder uses.
Any future design proposal must respect that strategy boundary.

### 12.5 Anchors Must Survive

Do not break:

- `data-pa-block`
- `data-pa-slot`
- `data-pa-global`
- `page_manifest.json`

If those break, the revision path breaks.

### 12.6 Stable Semantic `block_id` Values Matter

They are revision targets.
They must not drift into positional or template-coupled names.

### 12.7 Patch Scope Must Stay Local

If a change is page-wide or architectural, it should be routed upstream instead of stuffed into patch execution.

## 13. Current Auxiliary or Partially Integrated Parts

There are still shell-resolution utilities in the codebase:

- `shell_resolver_phase_node()`
- `binding_review_node()`

But they are not part of the active main HITL workflow.

Current reality:

- shell logic is still used by manual layout compose helpers
- `TemplateProfile` has become the main source of shell understanding
- the standalone shell-binding review stage is not currently a top-level checkpoint in the user-facing graph

This is important if future work wants to formalize shell review as a first-class stage again.

## 14. Existing Regression Coverage

Two important test areas exist right now:

- `tests/test_patch_mode.py`
- `tests/test_template_compile_refactor.py`

The template compile refactor tests currently cover:

- template profile cache hit/invalidation
- archetype detection
- selector uniqueness and preservation logic
- planner render-strategy downgrade behavior
- compiled page assembly preserving globals and outline order

This suggests the "template compile + block assembly" refactor is not just conceptual; it has dedicated regression coverage.

## 15. Best Compact Mental Model for ChatGPT

Use this if you need to reason quickly:

PaperAlchemy is now a template-compiled, schema-driven, human-in-the-loop academic webpage generation system.

The four most important production layers are:

1. Reader
   - turns PDF text into a webpage-oriented semantic paper pack
2. TemplateCompile + Planner
   - compiles the chosen template into reusable shell facts and plans blocks against those facts
3. Coder
   - preferably renders one block at a time and assembles the page, with fullpage fallback when the template is risky
4. Translator + EditIntentRouter + Patch
   - turns human feedback into either safe local DOM patches or an explicit non-patch response

The most important current contracts are:

- `StructuredPaper`
- `TemplateProfile`
- `PagePlan`
- `BlockRenderSpec`
- `BlockRenderArtifact`
- `CoderArtifact`
- `PageManifest`
- `RevisionPlan`
- `TargetedReplacementPlan`

The most important current invariants are:

- compiled template understanding is first-class
- render strategy must be explicit
- block IDs must stay stable
- anchors must survive
- patch scope must remain local
- large structural changes should go upstream to planning instead of overloading the patch path

## 16. What ChatGPT Should Optimize For in the Next Design Step

If proposing the next architecture step, optimize for:

- preserving `TemplateProfile` as a stable planning/rendering interface
- strengthening the compiled block assembly path rather than removing it
- improving render-strategy selection and fallback clarity
- making unsupported feedback types route more explicitly upstream
- preserving anchored revision compatibility
- keeping block-level artifacts useful for debugging and evaluation
- avoiding any change that makes future revisions depend on fragile free-form HTML matching
