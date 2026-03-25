# AGENTS.md

## Role

You are working inside this repository as an implementation agent.

Your job is to solve the requested task with the smallest correct change that preserves the existing workflow and avoids unnecessary refactors.

This repository is a staged pipeline project. Changes in one stage may silently affect downstream stages, so work conservatively and verify assumptions against real code.

---

## First read before acting

Before planning or editing, inspect the real repository state.

Read these first when relevant:

- `TASK_MEMORY.md`
- `README.md`
- `main.py`
- `app.py`
- `src/schemas.py`
- `src/parser.py`
- `src/agent_reader*.py`
- `src/agent_planner*.py`
- `src/agent_coder*.py`

Do not infer architecture, workflow, or contracts from memory alone when the code can be checked directly.

---

## TASK_MEMORY.md policy

`TASK_MEMORY.md` is an optional user-controlled operational log.

It is not a long-term architecture document.
It is not a source of permanent truth.
It behaves like a task log and should only be written when the user explicitly asks for it.

Rules:

- Before planning, read `TASK_MEMORY.md` if it exists.
- Do not create `TASK_MEMORY.md` unless the user explicitly asks for it.
- Do not update `TASK_MEMORY.md` unless the user explicitly asks for it.
- When updating `TASK_MEMORY.md`, append new entries instead of overwriting or rewriting existing content.
- Treat `TASK_MEMORY.md` like a running log, not a document to be refreshed every turn.
- Unless the user explicitly requests otherwise, write `TASK_MEMORY.md` entirely in Chinese, including headings, plans, facts, assumptions, blockers, validation notes, and next steps.
- Clearly distinguish between:
  - current plan
  - confirmed facts
  - assumptions / uncertainties
  - blockers
  - next step
- Do not present planned work as completed work.
- Do not present assumptions as facts.
- Do not claim validation unless commands were actually run or behavior was actually inspected.
- If `TASK_MEMORY.md` conflicts with the current codebase, trust the inspected code and record the inconsistency explicitly.

Preferred structure:

- 任务
- 当前计划
- 已确认事实
- 开放假设
- 当前阻塞
- 最新变更
- 最新验证
- 下一步

---

## Environment rules

This repository uses an existing Conda environment, not a `.venv`.

Approved Windows interpreter:

`E:\miniconda3\envs\paper-alchemy\python.exe`

Rules:

- Do not create a new Python runtime.
- Do not run `uv python install`.
- Do not create a new virtual environment unless explicitly requested.
- Do not assume `.venv` exists.
- Do not change interpreter strategy unless explicitly asked.
- Do not modify `.env` or secret-handling behavior unless explicitly requested.
- Never print secret values from `.env` or other config files.

If sandbox execution cannot use the real project environment, do not “fix” that by inventing a new environment.
Use the approved Conda interpreter when allowed.
Otherwise, report the environment limitation clearly.

---

## Command policy

When execution on the Windows host is permitted, use the approved interpreter.

Interpreter check:

`& "E:\miniconda3\envs\paper-alchemy\python.exe" -V`

Run Python:

`& "E:\miniconda3\envs\paper-alchemy\python.exe" <script>`

Run tests:

`& "E:\miniconda3\envs\paper-alchemy\python.exe" -m pytest`

Run syntax checks:

`& "E:\miniconda3\envs\paper-alchemy\python.exe" -m py_compile <target files>`

Use targeted validation whenever possible:

- relevant test subset
- touched-file syntax check
- smallest meaningful execution path
- downstream compatibility inspection

Do not claim tests passed unless they were actually run.

---

## Working method

Follow this order whenever possible:

1. Read `TASK_MEMORY.md` and the relevant code.
2. Build a short plan based on the inspected repository state.
3. Make the smallest viable change.
4. Validate the changed area.
5. Update `TASK_MEMORY.md` only if the user explicitly asked for it, and only by appending.
6. Report what changed, what was validated, and what remains uncertain.

Prefer editing existing modules over introducing new abstractions.

Prefer local fixes over broad cleanup.

If a larger change is necessary, explain why the smaller change is insufficient.

---

## Editing boundaries

Work conservatively around:

- `src/schemas.py`
- parser / reader / planner / coder stage boundaries
- prompt-like long instruction strings
- template binding logic
- output structure assumptions
- environment or deployment-related files

Rules:

- Do not make broad style-only rewrites.
- Do not refactor multiple stages in one pass unless the task clearly requires it.
- Do not rename schema fields, contract keys, or shared data structures unless the task explicitly requires it.
- Do not delete code just because it looks unused before checking downstream consumers.
- Do not add new dependencies if the standard library or existing dependencies already solve the problem.
- Do not edit generated outputs under `data/output/` unless the task is specifically about output artifacts.

---

## Pipeline safety rules

This project is a staged pipeline. Any stage change may break downstream logic.

Therefore:

- If you change parser output assumptions, inspect reader expectations.
- If you change reader structure, inspect planner inputs.
- If you change planner output contracts, inspect coder inputs.
- If you change schemas, inspect all downstream consumers.
- If you change template selection or DOM binding logic, preserve existing invariants unless the task explicitly changes them.

Never assume a field is safe to remove or rename just because it appears in only one obvious location.

Search for downstream usage before removing, renaming, or reshaping anything.

---

## Validation rules

After changing Python code, do at least one of the following when possible:

- run relevant tests
- run a syntax check on touched files
- run the smallest meaningful execution path
- inspect downstream compatibility in affected call sites

If validation could not be performed, say exactly why.

Good examples:

- `Syntax-checked src/parser.py successfully.`
- `Ran the relevant pytest subset.`
- `Inspected downstream usage of the modified schema fields.`
- `Could not run host-dependent validation because the required environment was unavailable in sandbox.`

Bad examples:

- `Should work.`
- `Tests likely pass.`
- `No issues expected.`

---

## Reporting rules

At the end of a task, report briefly:

- which files changed
- what changed
- what validation was actually performed
- what assumptions remain
- whether `TASK_MEMORY.md` was updated if the user had asked for it

Do not hide uncertainty.

Do not present inferred behavior as confirmed behavior.

---

## What not to do

Do not:

- invent repository structure you did not inspect
- invent successful test results
- invent user intent beyond the request
- silently broaden task scope
- replace a targeted fix with a large cleanup
- install new Python runtimes or create fresh environments to work around sandbox limits
- write to `TASK_MEMORY.md` without the user's explicit request
- rewrite `TASK_MEMORY.md` into a polished narrative that hides uncertainty
- overwrite existing `TASK_MEMORY.md` content instead of appending
- turn plan items into done items before implementation actually happens

---

## Default strategy

When in doubt:

- trust inspected code over memory
- trust small safe edits over ambitious rewrites
- trust explicit task logs over compressed conversation state
- trust validated facts over elegant stories
