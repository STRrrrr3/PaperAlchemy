# TASK_MEMORY.md

## 任务
将仓库当前的 Gemini 接入从仅支持 `GOOGLE_API_KEY` 扩展为支持 Vertex AI service account JSON，并验证用户更新后的本地 JSON 是否已经可以真实调用 Vertex。

## 当前计划
- 步骤 1：检查当前 LLM 接入方式、依赖和本地 JSON 凭据的必要字段。
- 步骤 2：修改 `src/llm.py`，优先自动发现并加载 Vertex service account JSON，保留 `GOOGLE_API_KEY` 作为回退路径。
- 步骤 3：做本地语法检查与实例化检查，确认 Vertex 代码路径可正常初始化。
- 步骤 4：用真实网络请求验证 Vertex 最小调用，并在用户替换新凭据后再次复测。

## 已确认事实
仅记录实际检查、修改或执行过的内容。

### 已检查文件
- `TASK_MEMORY.md`
- `src/llm.py`
- `requirements.txt`
- `E:\Graduation Design\PaperAlchemy\project-5f3fe41d-a593-460f-928-36f5f2a9d496.json`
- `E:\Graduation Design\PaperAlchemy\project-5f3fe41d-a593-460f-928-11b2a4272480.json`

### 已修改文件
- `src/llm.py`
- `TASK_MEMORY.md`

### 实际执行过的命令
- `Get-Content src/llm.py`
- `Get-Content requirements.txt`
- `Get-Content TASK_MEMORY.md`
- `Get-ChildItem *.json | Select-Object Name, LastWriteTime, Length`
- `@' ... '@ | & "E:\miniconda3\envs\paper-alchemy\python.exe" -`
  说明：读取本地 JSON 的非敏感字段，仅确认 `type/project_id/client_email/private_key` 是否存在。
- `& "E:\miniconda3\envs\paper-alchemy\python.exe" -m py_compile src/llm.py`
- `@' ... '@ | & "E:\miniconda3\envs\paper-alchemy\python.exe" -`
  说明：本地导入 `src.llm`，检查是否能自动发现 Vertex JSON、加载运行时配置并实例化 `ChatGoogleGenerativeAI`。
- `@' ... '@ | & "E:\miniconda3\envs\paper-alchemy\python.exe" -`
  说明：在提权联网后，分别对 `global` 和 `us-central1` 做最小 Vertex 调用测试。
- `@' ... '@ | & "E:\miniconda3\envs\paper-alchemy\python.exe" -`
  说明：用户替换新 JSON 后，再次检查自动发现路径、运行时配置与真实最小调用结果。

### 观察结果
- 当前环境安装了 `langchain_google_genai`，但没有安装 `langchain_google_vertexai` 或 `vertexai`。
- 现有 `langchain_google_genai.ChatGoogleGenerativeAI` 支持 `credentials`、`vertexai`、`project`、`location` 参数，因此不需要额外引入新的 SDK 就能接 Vertex。
- `src/llm.py` 已改为：
  - 优先自动发现根目录中的 service account JSON，走 Vertex AI；
  - 如果没有可用的 Vertex 凭据，再回退到原有 `GOOGLE_API_KEY` 路径；
  - 支持通过环境变量覆盖 `VERTEX_SERVICE_ACCOUNT_JSON`、`VERTEX_PROJECT`、`VERTEX_LOCATION`、模型名。
- Vertex 默认模型已设置为 `gemini-2.5-pro` / `gemini-2.5-flash`，API Key 路径仍保留原来的默认模型名。
- `src/llm.py` 语法检查通过。
- 本地实例化检查通过，`get_llm(use_smart_model=False)` 会正确初始化为 `vertexai=True`，并自动识别项目根目录中的 service account JSON。
- 第一份 JSON 曾在真实联网调用中返回 `403 PERMISSION_DENIED`，阻塞点是缺少 `aiplatform.endpoints.predict` 权限。
- 用户替换新 JSON 后，项目根目录中当前只检测到一份新的 service account JSON：`project-5f3fe41d-a593-460f-928-11b2a4272480.json`。
- 用新 JSON 重新做真实 Vertex 调用后，`global` 区域已经成功返回 `OK`，说明当前 Vertex 凭据和代码路径都可用。

## 开放假设
记录仍待确认的不确定项或假设。

- 既然 `global` 已经可用，后续大概率不需要再改接入层；只有当某些特定模型或区域出现限制时，才需要继续微调默认配置。
- 如果以后要把这套配置写入 `.env` 或做成显式 provider 开关，可以再做一轮小整理，但当前不是功能阻塞项。

## 当前阻塞
记录当前真正阻塞推进的事项。

- 当前无硬阻塞，Vertex 最小调用已经通过。

## 最新变更
概述最近一次有意义的实现变更。

- 重写了 `src/llm.py` 的凭据选择逻辑，使其支持 Vertex service account JSON 自动发现与加载。
- 保留了原有 `GOOGLE_API_KEY` 回退路径，避免影响没有 Vertex 凭据时的旧用法。
- 用户更换新 JSON 后，已确认当前自动发现到的新凭据可以真实调用 Vertex。

## 最新验证
仅记录实际执行过的验证。

- 用 `py_compile` 验证了 `src/llm.py` 语法正确。
- 在本地验证了 `get_llm()` 会正确进入 Vertex 路径，并能实例化 `ChatGoogleGenerativeAI`。
- 第一轮真实 Vertex 最小调用在 `global` 和 `us-central1` 都返回了 `403 PERMISSION_DENIED`。
- 用户替换新 JSON 后，重新验证了自动发现路径与实例化结果。
- 用新 JSON 发起真实 Vertex 最小调用，`global` 区域成功返回 `OK`。

## 下一步
说明下一个具体动作。

- 若继续推进，可考虑把当前 Vertex 凭据路径和默认区域整理成更明确的项目配置；就可用性而言，当前已经可以直接使用。

---

## 任务
- 将 Coder 的人工样式选择保留为可选项：仅在 Outline 阶段勾选时进入手工 `layout_compose`，未勾选时直接生成网页初版。
- 调整前端控制区：每个阶段结束后，隐藏该阶段的操作控件；截图上传仅在网页修订阶段显示。

## 当前计划
- 本轮代码实现已完成。
- 若继续推进，下一步优先做一次真实 Gradio 页面人工回归，分别验证“勾选手工 compose”和“不勾选直接出初版”两条路径。

## 已确认事实
- 本轮实际修改文件为：`app.py`、`src/state.py`、`tests/test_patch_mode.py`。
- `src/state.py` 已新增 `manual_layout_compose_enabled: bool`。
- `app.py` 中 `outline_review_router()` 已支持三路分支：
  - 未批准大纲时回到 `planner`
  - 已批准且勾选手工 compose 时进入 `layout_compose_prepare`
  - 已批准且未勾选时直接进入 `coder`
- `approve_outline_and_generate_draft()` 已接收 Outline 阶段的复选框值，并将该值写入工作流状态。
- 左侧控制区已改为单个“当前阶段操作区”：
  - Reader 阶段显示文本反馈与 Reader 按钮
  - Outline 阶段显示文本反馈、手工 compose 复选框与 Outline 按钮
  - Webpage 阶段显示文本反馈、截图上传与网页修订按钮
  - Layout Compose 阶段隐藏该操作区，仅保留右侧 compose 面板
- `TASK_MEMORY.md` 在当前 shell 输出中显示为乱码；这说明当前终端读取时存在编码显示问题。本次追加仅依据当前代码与实际命令结果记录。
- 当前 `git status --short` 还显示除本轮修改外，工作区内还有其他未提交变更；这些文件不在本轮实现范围内。

## 开放假设
- 尚未在真实浏览器中完整人工点击验证 Gradio 的控件显隐，只完成了代码检查与自动化测试。
- 当前 Conda 环境缺少 `pytest`；后续若要严格按 `pytest` 命令复验，需要先补齐测试依赖或切换到已有带 `pytest` 的环境。

## 当前阻塞
- 当前无功能实现阻塞。
- 自动化验证存在环境限制：`pytest` 不可用，但已用 `unittest` 覆盖目标测试类。

## 最新变更
- 新增工作流布尔状态 `manual_layout_compose_enabled`，将 Coder 前的人工 `layout_compose` 改为可选。
- 重构 `app.py` 中阶段控件的显隐逻辑，新增统一 helper，根据当前阶段返回一组 `gr.update(...)`。
- 在 Outline 阶段加入 `Enable Manual Layout Compose` 复选框，默认不勾选。
- 更新测试：
  - 增加阶段控件显隐 helper 的测试
  - 增加“不勾选手工 compose 时直接到 webpage review”的工作流测试
  - 调整原有 compose 路径测试，使其显式走勾选分支

## 最新验证
- 实际执行：`& "E:\miniconda3\envs\paper-alchemy\python.exe" -m py_compile app.py src\state.py tests\test_patch_mode.py`
- 实际执行：`& "E:\miniconda3\envs\paper-alchemy\python.exe" -m pytest tests\test_patch_mode.py -k "ReviewFormatterTests or WorkflowPatchRoutingTests"`
  - 结果：失败，原因是当前环境缺少 `pytest` 模块。
- 实际执行：`& "E:\miniconda3\envs\paper-alchemy\python.exe" -m unittest tests.test_patch_mode.ReviewFormatterTests tests.test_patch_mode.WorkflowPatchRoutingTests`
  - 结果：通过，运行 10 个测试，全部成功。

## 下一步
- 若需要继续验证，可在本机启动 Gradio，手动走一次以下流程：
  - Reader -> Outline -> 不勾选 compose -> 直接进入网页初版
  - Reader -> Outline -> 勾选 compose -> 进入手工 compose -> 再生成网页初版
- 若你要，我也可以继续帮你逐个查看当前工作区里那些“非本轮修改”的未提交文件分别改了什么。
---

## 任务
- 补记“PaperAlchemy 模板编译 + Block 渲染重构”这一轮架构改造，便于后续维护 Git 版本描述与阶段事实追踪。

## 当前计划
- 本条仅追加历史记录，不新增实现，不改动当前架构代码。
- 依据当前仓库代码现状与此前已经实际执行过的验证结果，补记这轮架构改造的关键事实。

## 已确认事实
- 当前主流程已经接入 `template_compile` 节点，工作流顺序中存在 `overview -> template_compile -> planner -> outline_review -> [layout_compose] -> coder -> webpage_review -> translator -> patch_agent -> patch_executor`，对应 `app.py`。
- 当前公共类型中已经存在 `TemplateProfile`、`TemplateShellCandidate`、`TemplateWidget`、`ResolvedBlockBinding`、`BlockRenderSpec`、`BlockRenderArtifact`，对应 `src/schemas.py`。
- 当前 `WorkflowState`、`PlannerState`、`CoderState` 已扩展 `template_candidates`、`selected_template`、`template_profile`、`template_profile_path`、`template_compile_cache_hit`、`block_render_artifacts` 等字段，对应 `src/state.py`。
- 当前仓库中存在独立模板编译模块 `src/template_compile.py`，负责模板候选选择、模板 Profile 编译、缓存命中与复用准备。
- 当前 Planner 已从主要依赖原始模板 DOM outline 切换为消费 `TemplateProfile`；`plan_meta.render_strategy` 会按模板编译置信度与风险在 `compiled_block_assembly` 和 `legacy_fullpage` 之间切换，对应 `src/agent_planner.py`。
- 当前 `dom_mapping` 仍然保留，但只作为兼容字段存在，不再是新架构的页面主体生成接口。
- 当前 Coder 已拆分为块级渲染与页面组装主路径，并保留 legacy 整页生成兜底；代码中存在 `_run_compiled_block_assembly()` 与 `_run_legacy_fullpage_render()`，对应 `src/agent_coder.py`。
- 当前 `layout_compose` 与 shell resolver 已能以上游 `TemplateProfile.shell_candidates` 作为候选事实源，对应 `src/template_shell_resolver.py`。
- 当前架构仍保留 `PageManifest`、`data-pa-block`、`data-pa-slot`、`data-pa-global`、translator / patch / patch_executor 这条修订链路兼容性。
- 当前测试中已包含与这轮架构改造对应的覆盖，包含 `tests/test_template_compile_refactor.py`，且 `tests/test_patch_mode.py` 已适配新增 workflow 节点。

## 开放假设
- 尚未在当前记录中补写一条真实论文样本从 Reader 到最终网页的在线端到端回归结论，因此块级渲染主路径的生产观感仍主要依赖结构测试与单元测试结果。
- `TASK_MEMORY.md` 在当前 shell 输出里仍显示乱码；这更像终端编码显示问题，不直接等同于文件内容损坏，后续若继续维护此文件，仍需留意编码一致性。

## 当前阻塞
- 当前无新增实现阻塞。
- 如需进一步确认这轮架构改造的实际产出质量，阻塞点主要是是否安排一次真实样本端到端回归，而不是当前代码缺失。

## 最新变更
- 新增模板编译层，将模板理解从 Planner / Coder 中抽离为可缓存的 `TemplateProfile`。
- 重构 workflow 接线，把模板选择与模板编译提升为主图中的独立阶段。
- 重构 Planner，使 block 绑定以上游编译得到的 shell candidates 为准，并按模板风险决定渲染策略。
- 重构 Coder，使其拆分为 render spec 准备、block 渲染、page 组装、manifest 抽取校验与 critic，旧整页 LLM coder 下沉为 `legacy_fullpage` fallback。
- 打通 `layout_compose` 与新模板编译层的衔接，使人工 compose 继续可选，但不再是默认主路径。

## 最新验证
- 之前已实际执行语法检查：
  - `& "E:\miniconda3\envs\paper-alchemy\python.exe" -m py_compile app.py src\schemas.py src\state.py src\template_compile.py src\agent_planner.py src\agent_planner_critic.py src\agent_coder.py src\template_shell_resolver.py src\prompts.py tests\test_patch_mode.py tests\test_template_compile_refactor.py`
- 之前已实际执行单元测试：
  - `& "E:\miniconda3\envs\paper-alchemy\python.exe" -m unittest tests.test_patch_mode tests.test_template_compile_refactor`
  - 结果：共 36 个测试全部通过。
- 本次补记前已再次实际检查当前代码入口与关键符号，确认上述架构改造仍真实存在于 `app.py`、`src/schemas.py`、`src/state.py`、`src/template_compile.py`、`src/agent_planner.py`、`src/agent_coder.py`、`src/template_shell_resolver.py`。

## 下一步
- 如果后续要继续维护这轮架构的稳定性，优先补一条真实论文样本的端到端回归记录，覆盖 `template_compile -> planner -> coder -> patch` 全链路。
- 如果只是维护 Git 版本描述或阶段里程碑，当前这条补记已经足以作为“模板编译上移 + Block 装配主路径落地”的事实依据。

---

## 任务
- 在“零功能变更、零流程语义漂移、零输出契约漂移”的前提下，对当前仓库做结构重组：
  - 给 `src/` 做分层
  - 缩小 `app.py`
  - 保留旧导入路径与旧测试兼容

## 当前计划
- 本轮结构重组实现已完成。
- 若继续推进，下一步优先做一次宿主机上的真实 Gradio 启动回归，以及一次真实样本的人工端到端检查；这属于后续验证，不属于本轮已完成范围。

## 已确认事实
- 本轮已新增分层目录：`src/agents/`、`src/contracts/`、`src/parsing/`、`src/patching/`、`src/services/`、`src/template/`、`src/ui/`、`src/utils/`、`src/validators/`、`src/workflows/`。
- 原先平铺在 `src/` 根目录下的核心实现，已迁移到新目录中的对应模块；旧路径文件仍保留，但已改为兼容 wrapper。
- 兼容 wrapper 不是简单复制导出，而是桥接到新实现模块；这样旧 import 与测试中的 `patch("src.xxx...")` 仍可命中新实现。
- `app.py` 已从“大而全”实现文件重构为兼容门面：
  - 继续保留 `build_hitl_workflow()`、`HITL_WORKFLOW`、`build_app()`、`main()`
  - 继续保留测试直接引用或 monkeypatch 的关键符号
  - 实际实现已迁入 `src/workflows/` 与 `src/ui/`
- 本轮新增的主要实现模块包括：
  - `src/services/artifact_store.py`
  - `src/ui/formatters.py`
  - `src/ui/updates.py`
  - `src/ui/constraints.py`
  - `src/workflows/hitl_nodes.py`
  - `src/workflows/hitl_routes.py`
  - `src/workflows/hitl_graph.py`
  - `src/workflows/batch_runtime.py`
  - `src/ui/review_handlers.py`
  - `src/ui/layout_compose_handlers.py`
  - `src/ui/app_builder.py`
- `app.build_hitl_workflow()` 仍是显式包装层，并从 `app` 命名空间传入节点与路由；这保证了现有测试对 `app.reader_phase_node`、`app.coder_phase_node` 等符号的 monkeypatch 仍然有效。
- `app.coder_phase_node` 已保留显式兼容包装逻辑，以兼容测试对 `app.run_coder_agent_with_diagnostics`、`app.save_page_plan`、`app.save_coder_artifact` 的 patch 行为。
- 本轮未改动以下行为层约束：
  - LangGraph 主节点顺序
  - 条件路由逻辑
  - `interrupt_after`
  - schema 字段名
  - JSON / artifact 落盘路径约定
  - prompt 语义
  - 默认参数与 fallback 策略
- 本轮还新增了结构重组专用回归测试：`tests/test_refactor_compat.py`。

## 开放假设
- 本轮已验证 `import app; app.build_app()` 与 `import app; app.build_hitl_workflow()` 能成功装配，但没有实际执行 `app.launch()`，因此未把“真实 UI 服务已成功启动”记录为已确认事实。
- 本轮没有做真实论文样本的人工端到端点击回归，因此“结构更清晰”已经确认，但“宿主机 UI 交互观感完全无漂移”仍有待人工确认。
- `TASK_MEMORY.md` 在 shell 中可能仍受终端编码显示影响；本次追加内容依据实际文件追加，不依据 shell 乱码外观判断文件损坏。

## 当前阻塞
- 当前无新的实现阻塞。
- 若继续做更高层验证，主要阻塞不是代码缺失，而是是否安排宿主机上的真实 UI / 样本回归时间。

## 最新变更
- 迁移并分层了以下实现职责：
  - contracts：`schemas`、`state`
  - services：`llm`、`human_feedback`、`preview`、artifact 持久化
  - template：`compile`、selector、catalog、resource、shell resolver
  - validators：`page_manifest`、`page_validation`
  - agents：reader / planner / coder / translator
  - patching：patch pipeline
  - workflows：节点、路由、图构建、batch runtime
  - ui：updates、formatters、constraints、review handlers、layout compose handlers、app builder
- 保留了旧路径兼容文件，例如：
  - `src/agent_coder.py`
  - `src/agent_patch.py`
  - `src/template_compile.py`
  - `src/schemas.py`
  这些文件现在作为兼容桥接层存在，不再承载主实现。
- 重写了 `app.py`，使其只保留兼容门面、工作流装配入口和启动逻辑。
- 新增 `tests/test_refactor_compat.py`，覆盖：
  - 旧导入兼容
  - `app.build_app()` 最小装配
  - `app.build_hitl_workflow()` 最小装配
  - `src/**/*.py` 导入图无循环依赖
- 在迁移过程中修复了两处纯机械性语法问题：
  - `src/services/llm.py` 中复制后受旧编码文本影响的异常字符串
  - `src/parsing/parser.py` 中复制后受旧编码文本影响的日志字符串
  这些修复只涉及日志/报错文本可解析性，不涉及业务逻辑调整。

## 最新验证
- 实际执行了大范围语法检查，覆盖：
  - `app.py`
  - `main.py`
  - 旧 `src/` 兼容 wrapper
  - 新分层模块
  - `tests/test_patch_mode.py`
  - `tests/test_template_compile_refactor.py`
  - `tests/test_refactor_compat.py`
- 实际执行了单元测试：
  - `& "E:\miniconda3\envs\paper-alchemy\python.exe" -m unittest tests.test_patch_mode tests.test_template_compile_refactor tests.test_refactor_compat`
  - 结果：`Ran 41 tests ... OK`
- 实际新增并通过了以下验证目标：
  - 旧导入兼容 smoke
  - `import app; app.build_app()`
  - `import app; app.build_hitl_workflow()`
  - `src/**/*.py` 导入图无循环依赖

## 下一步
- 若继续推进，建议优先补一条真实宿主机回归记录：
  - 启动 Gradio
  - 手动走一次 Reader -> Outline -> Draft -> Revision 的最小流程
  - 确认 UI 装配与事件绑定在真实环境中无行为漂移
- 若继续维护代码结构，可再考虑第二轮小范围整理，但本轮不建议继续拆 `prompts.py`、`patch_pipeline.py`、`agents/coder.py`、`template/shell_resolver.py` 的内部大函数，以避免超出“零行为变更”范围。
