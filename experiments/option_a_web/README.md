# Option A 网页测试接入层

这层只用于后续 Codex／人工在真实浏览器中执行冻结的 Option A 六个脚本。
它不是产品功能，不修改生产文件，也不修改 `experiments/option_a/` 冻结包。

本目录的隔离测试只使用 fake client、临时 SQLite 和进程内 ASGI，不启动真实监听，
不调用真实 API／local 模型，也不做浏览器验收。

## 冻结实验兼容契约（2026-11-05）

启动器在验证独立测试数据库和冻结 manifest 后，为本实例显式注入
`FrozenReceiveTeachingPolicy`。请求使用原始冻结 system 和版本，recorder 的
逐字/哈希校验与快照校验继续执行。语言遵循按原实验口径观察和评分，本实验不采用
正式 Receive 的 `l1` 语言硬约束，不追加语言指令或 `+l1` 版本。

新运行的 `run_meta.json` 包含 `receive_policy.kind=frozen-experiment-v1`、
`reply_language=frozen-prompt-only` 和 `history_tools=false`。该配置仅通过构造实验
服务实例传入，不由网页请求或环境变量启用，不能与历史检索工具同时配置。
正式服务默认不传此配置，继续执行语言解析和写库前校验。

旧 Prompt、manifest 与真实 pilot 证据保持不变；新运行的兼容实现和源码版本应另行记录。

## 1. 结构

| 文件 | 作用 |
| --- | --- |
| `launcher.py` | 独立入口：冻结校验、环境隔离、Prompt 覆盖、测试 app 组装、PID 停止 |
| `asgi.py` | 测试 ASGI wrapper：A 横幅、arm 闸门、消息请求关联、未计划入口阻断 |
| `gate.py` | 整批有序、单次使用的 arm/consume/complete 闸门 |
| `recording.py` | 真实 LLM client 的记录包装器；参数、模型、响应、诊断脱敏记录 |
| `tests/test_web_layer.py` | 隔离验证 |
| `README.md` | 本文 |

## 2. 架构

```text
浏览器
  -> TestApp（本目录，127.0.0.1:测试端口）
     GET /   -> 真实 frontend/index.html 的测试副本 + A 横幅
     /static -> 真实 FastAPI app 的静态资源
     /api    -> 真实 FastAPI app -> ChatService -> RecordingLLMClient -> 真实 API
     /__option_a__/arm, /__option_a__/snapshot -> 测试控制
```

隔离边界：

- 独立进程、独立本地端口；单 worker，`reload=False`。
- 运行目录、SQLite DB、证据目录都在仓库外。
- `DATABASE_URL` 在导入任何 `backend` 模块前设置；启动器在导入前拒绝已加载的
  `backend`／`backend.*`。
- 只替换测试进程内 `backend.chat_service` 的 Receive 提示词和版本 getter。
- 会话仍使用真实 `receive_teaching` 模式；不新增第三模式。
- 消息快照的 `prompt_version_snapshot` 记录 A 版本；
  `interaction_mode_snapshot` 仍是 `receive_teaching`。
- 真实 `frontend/app.js`、真实 `ChatService`、真实历史组装、真实 API client 都保留。
- 测试 HTML 只在内存中由当前 `frontend/index.html` 注入横幅；不改生产文件。
- 未计划入口（`/api/chat`、History Review、切换 interaction mode／LLM profile）
  在 TestApp 层被阻断；普通 `/api/sessions/{id}/messages` 必须先通过 arm 闸门。

## 3. 启动前检查

启动器会：

1. 在任何 backend 导入前调用
   `experiments.option_a.freeze_manifest.load_and_verify_manifest()`；
2. 核对 condition=A、asset_version、六个脚本的 2/2/2/2/4/4 和 16 总数；
3. 读取 `experiments/option_a/prompts/system.txt`，核对字节数与 SHA-256；
4. 独占创建仓库外运行目录；
5. 设置合成测试环境（`LOCAL_LLM_ENABLED=false`，真实 API 才使用显式传入的
   key/base/model）；
6. 导入 backend 后核对 engine 绝对路径指向本次 `test.sqlite3`，并核对
   `SessionLocal` 绑定同一个 engine；
7. 只在测试进程覆盖 Receive 的提示词和版本解析。

当前冻结 A system：

- 字节数：13571
- SHA-256：`16236223af1b3f813e5e27aa881e27d4b45039b58db6be2f44d4a4f9137f26a4`
- 测试进程的 A 版本字符串：
  `option-a-isolated-v0.1+sys-16236223af1b`

## 4. 启动（后续 Codex／人工执行）

本仓库本轮没有执行下面命令；不要在本轮启动监听服务。

```bash
cd /Users/luo/Documents/teachable_agent

LLM_API_KEY=... \
LLM_API_BASE_URL=... \
LLM_MODEL=... \
.venv/bin/python -m experiments.option_a_web.launcher \
  --real-api \
  --run-dir /Users/luo/Documents/teachable_agent_runs/option-a-web/run-001 \
  --host 127.0.0.1 \
  --port 8765
```

要求：

- `--run-dir` 必须不存在且位于仓库外。
- API key 只从环境读取；不要写进命令行历史、README、证据或 Git。
- 端口必须空闲；启动器不会替你杀未知进程。
- 不要使用 reload；启动器固定 `workers=1`，`reload=False`。
- 启动后先看页面顶部 A 横幅，再开始创建会话。
- 页面标题和横幅会明确显示 `A Prompt 测试变体`、条件 A、A 版本、system
  hash 和请求模型；横幅不隐藏、替换或包裹助手回复。

## 5. 浏览器执行流程

六个脚本仍然是六个独立会话，每个 turn 一次发送，总计 16 次：

| 脚本 | 会话 | turn 数 |
| --- | --- | --- |
| S1 | 独立 | 2 |
| S2 | 独立 | 2 |
| S3 | 独立 | 2 |
| S4 | 独立 | 2 |
| S5 | 独立 | 4（T1/Q1/T2/Q2） |
| S6 | 独立 | 4（T1/Q1/T2/Q2） |

对每个脚本：

1. 在真实页面新建一个会话（`POST /api/sessions`），记下响应中的 `id`。
2. 对脚本中每个 turn，先从
   `experiments/option_a/scripts/scripts.json` 读取**原样**的
   `content`；不得改写、截断或把 A Prompt 拼进 user 消息。
3. 调用一次 arm：

   ```js
   const banner = document.querySelector('#option-a-test-banner');
   const instanceId = banner.dataset.instanceId;
   const sessionId = /* 上一步创建的会话 id */;
   const scriptId = 'S1';
   const turnId = 'T1';

   await fetch('/__option_a__/arm', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({ instance_id: instanceId, session_id: sessionId,
                            script_id: scriptId, turn_id: turnId })
   });
   ```

4. 在聊天输入框填入该 turn 的 `content`，点击发送**一次**。
5. 等待真实 `POST /api/sessions/{id}/messages` 返回 200；不要刷新、不要重发。
6. 检查返回的 `user_message.content` 与脚本逐字一致、assistant 回复存在。
   服务端只有在 HTTP 状态、调用次数、user/assistant message id、消息快照和证据
   写入全部通过后，才调用 `gate.complete()` 推进到下一 turn。上一步未完成前
   arm 下一 turn 会被拒绝。
7. 建议调用一次快照检查（非生成，不产生模型调用）：

   ```js
   await fetch('/__option_a__/snapshot', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({ instance_id: instanceId, session_id: sessionId,
                            script_id: scriptId, turn_id: turnId })
   });
   ```

8. 保存页面截图到 `evidence/screenshots/`；至少每个脚本结束时一张。

不要用浏览器重试按钮、页面刷新或重新点击来处理失败。任何请求失败、空回复、
`finish_reason` 不是 `stop`（包括 `length`、`content_filter`、`tool_calls`、
缺失或未知值）、快照不匹配、重复请求或 arm 拒绝，都应停止整批并保留现场。

## 6. A Prompt／版本／快照核对

- 启动时：冻结 manifest、system bytes/hash、脚本总数全部通过才创建 server。
- 每次模型调用前：`RecordingLLMClient` 检查 messages 第 0 条 system
  与冻结 `system.txt` 完全一致；不一致则不调用 inner。
- 每次调用记录 `request.system_sha256`，必须等于冻结 hash。
- 消息落库后，`prompt_version_snapshot` 应为
  `option-a-isolated-v0.1+sys-16236223af1b`，不是 `receive-teaching-v7`。
- `interaction_mode_snapshot` 应为 `receive_teaching`；这是真实模式，
  A 是测试 Prompt 变体，不伪造第三模式。
- `llm_profile_id_snapshot=default`，`llm_profile_kind_snapshot=api`，
  `llm_model_snapshot` 是请求模型名，不是上游返回模型。

## 7. 生成参数与模型记录

测试包装器把原参数与实际发送参数分开记录：

- 原参数（`parameters.original_from_chat_service`）：
  `temperature=None`、`max_tokens=None`、`timeout_seconds=None`、
  `reasoning_effort=None`；这是真实 `ChatService` 未显式传入的值。
- 包装器实际发送（`parameters.effective_sent_by_wrapper`）：
  `temperature=0.0`、`max_tokens=800`、`timeout_seconds=30`、
  `reasoning_effort=None`、`reasoning_effort_sent=false`。
- 请求模型：`model.request_model`。
- 上游返回模型：`model.upstream_response_model`。当前生产
  `OpenAICompatibleLLMClient.generate()` 只返回 content，不暴露响应 model，
  因此真实 API 路径该字段为 `null`，并带
  `upstream_response_model_reason`。不要用返回模型覆盖数据库请求模型快照。
- `diagnostics` 只记录可获得的诊断字段；不记录原始响应体、reasoning_content、
  `str(exc)`、上游 `error.message` 或 Authorization。
- 参数“已发送”只表示包装器传给了 client，不声称提供方一定执行了该参数。

## 8. 证据

运行目录示例：

```text
/Users/luo/Documents/teachable_agent_runs/option-a-web/run-001/
  test.sqlite3
  server.pid
  evidence/
    run_meta.json
    calls.jsonl
    http.jsonl
    snapshots.jsonl
    screenshots/
```

- `run_meta.json`：实例 ID、冻结版本/hash、模型、参数、DB 绝对路径、测试 HTML 源
  hash、host/port、启动时间。
- `calls.jsonl`：每次实际 client 调用一行；request messages、system hash、
  原/有效参数、请求模型、上游模型（或 null 原因）、回复、response hash、
  脱敏 diagnostics、call_id/request_id/session/script/turn。
- `http.jsonl`：每个消息 HTTP 请求一行；arm 关联、状态码、response body hash、
  call ids、user/assistant message id、snapshot 结果、失败原因。
- `snapshots.jsonl`：显式快照检查结果。
- `test.sqlite3`：测试数据库，永久保留以核对消息和快照。
- `screenshots/`：人工保存的页面证据。

重点核对：

1. `http.jsonl` 恰好 16 行、`calls.jsonl` 恰好 16 行；
2. 六个 session，每个 session 的 turn 数正确；
3. 所有 call 的 `request.system_sha256` 等于冻结 hash；
4. 所有消息的 `prompt_version_snapshot` 是 A 版本；
5. `model.upstream_response_model` 为 null 时有原因，不伪造；
6. `failure_reason` 全为 null，`gate.run_state.failed` 在正常跑完后为 false；
7. 没有 raw error message、密钥或 Authorization。

## 9. 停止

前台运行时优先 `Ctrl-C`，让 uvicorn 正常退出；`prepare_instance.close()` 会恢复
Prompt getter、关闭证据文件并恢复环境。

如果以后由 Codex 在后台启动，只可用本次 run 的 PID 安全停止：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from experiments.option_a_web.launcher import stop_instance
stop_instance(Path('/Users/luo/Documents/teachable_agent_runs/option-a-web/run-001'))
PY
```

停止器会核对：

- `server.pid` 的 PID、`pid_start`、instance_id、host、port；
- `run_meta.json` 的 instance_id、run 目录、host/port 和 SHA-256；
- 规范化后的 `launch_argv`；
- 进程命令包含 `experiments.option_a_web.launcher` 和本次 run 目录；
- 命令包含本次测试端口；
- `lsof` 确认该 PID 确实拥有测试监听端口。

任一不符就拒绝发送信号。停止使用 `SIGTERM`，等待最多 5 秒；不使用 `pkill`，
不按端口杀未知进程。停止后保留整个运行目录和 `test.sqlite3`；不要删除证据。

## 10. 验证命令

```bash
cd /Users/luo/Documents/teachable_agent
.venv/bin/python -m unittest discover -s experiments/option_a_web/tests -v
```

当前隔离测试：34/34 通过。测试只使用 fake client／临时 SQLite／进程内 ASGI；
没有启动监听服务、没有真实 API、没有浏览器，也没有 Git 写操作。
隔离集成测试数据库实际包含 6 个脚本会话和 1 个“未计划入口”测试会话（共 7 个），
报告和断言不会把 fixture 中的 7 个会话误报成 6 个。
`StarletteDeprecationWarning` 来自当前 TestClient 与 httpx 的组合，不是失败。

## 11. 限制与最小生产改动评估

- 当前生产代码没有公开的 Prompt 注入点；本接入层在测试进程内 patch
  `backend.chat_service` 的两个 getter。这能保留真实 ChatService，但依赖内部
  名称，升级后端时需重新核对。
- 不新增第三 interaction mode：`ChatSession`/`Message` 的 DB CHECK、Pydantic
  Literal 和前端都只认两个模式。要显示真正的 mode=A，需要 DB 迁移、schema、
  API 和前端改动，不属于最小接入。
- 生产 client 不暴露上游响应 model 或原始 HTTP 响应；所以这些字段为 null，
  不声称能证明提供方实际执行了参数。
- 测试 HTML 是当前 `frontend/index.html` 的内存副本加横幅；静态资源和聊天
  代码仍来自真实 app。它不修改、隐藏或替换助手回复。
- fake 隔离测试通过不等于真实 A 行为通过；真实浏览器、真实 API 和证据核对
  仍由 Codex／人工在明确授权后执行。
- 如果后续不接受运行时 patch，最小生产改动是为 `ChatService` 增加可注入的
  Prompt/version resolver（默认行为不变），再从独立启动器传入；仍不需要 DB
  迁移或第三模式。本轮没有做这个改动。
