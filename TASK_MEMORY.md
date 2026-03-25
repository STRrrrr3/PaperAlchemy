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
