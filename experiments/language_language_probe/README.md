# Receive teaching 语言偏差离线实验

该目录位于 Codex 项目工作区，独立于正式 Teachable Agent 仓库。现有网页验收证据作为只读输入。默认演示通过 MockTransport 生成合成材料；获批的真实回放使用独立入口，并在输出中记录数据来源。

## 冻结口径

- **49 条主样本**，会话 56 有 39 条，会话 57 有 10 条。排除两个无历史块的首轮和三个其他路由的问答轮。已知 11 条中文回复的对应关系、来源文件哈希和排除理由列在 [冻结清单](plan.json)。
- **7 个合成控制目标**分别覆盖中英文显式要求、持续偏好、单轮覆盖、恢复偏好、保留原文引用和默认跟随英文。见 [控制输入和评分标准](controls.json)。
- 每轮最多取 20 条原始存储消息，再过滤 assistant。回放使用内存 SQLite 重建主样本边界，当前用户原文最后发送。每个控制项也包含非空教师历史。
- A 使用冻结的生产 Receive teaching 提示和普通历史块。B 替换冻结示例区中的中文对话，其余系统文本和原英文示例按原字节保留。C 使用冻结的英文 `note`，历史 JSON 中其他字段按原字节保留。
- 主实验 147 项，控制组 21 项。主实验 49 个原始轮次按 ID 排序，每轮的三项按固定排列调用。六种排列各使用 8 次，第 49 轮额外使用 ABC。七个控制项独立轮换六种排列，最后一个也使用 ABC。
- 冻结清单的消息哈希基于 UTF-8、排序对象键、紧凑 JSON。消息数组的顺序与字符串内容保留。运行时从 MockTransport 收到的实际请求体重算哈希，并记录请求体原始字节的哈希。

## 执行方式

从仓库根目录先执行 `cd experiments/language_language_probe`，再使用仓库 Python 环境。

```bash
PYTHON="$(cd ../.. && pwd)/.venv/bin/python"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -p no:cacheprovider tests -q
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" runner.py demo /path/to/new-demo-dir
```

`demo` 先与冻结清单比对，再通过 MockTransport 执行 168 项。每项只尝试一次，独立记录成功和失败。请求体只含 `model` 和 `messages`，省略 `temperature`、`max_tokens` 及工具字段。HTTP 客户端各阶段超时均配置为 30 秒，记录在请求体之外。实验将截断、空正文、无效响应、超时、非 2xx 和请求契约失败分别记账。

输出目录的 `blind.json` 用于评分。它只含随机编号、当前输入、按 `S1`、`S2` 编号的教师证据、回复和评分标准。`internal/` 保存条件映射、原始来源 ID、顺序、历史说明和请求证据，应与评分人分开保管。[盲评样例](synthetic_sample/blind.json)是一次完整合成运行导出的五条材料。[合成运行汇总](synthetic_sample/run-summary.json)和[测试输出](test-results.txt)仅证明离线链路通过。

### 评分汇总

完成盲评后，把评分写成 JSON 数组。每项以材料的 `random_id` 关联。

```json
[
  {
    "random_id": "替换为盲评材料编号",
    "language": "English",
    "main_clause_language": "English",
    "contains_chinese_prose": false,
    "quoted_original_preserved": null,
    "fidelity_score": 0,
    "notes": "此回复仅是合成占位文案。"
  }
]
```

`language` 和 `main_clause_language` 取 `English`、`Chinese` 或 `Mixed`。原文引用中的字符与正文语言分开评分。涉及指定原文时，`quoted_original_preserved` 记为 `true` 或 `false`。

```bash
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" runner.py score /path/to/demo-dir /path/to/annotations.json /path/to/new-summary.json
```

汇总的 `main.pairs.AB.n` 和 `main.pairs.AC.n` 分别是有效配对分母。双方均有成功回复和语言标注才进入配对。按条件分别显示计划、成功、失败、未尝试和已评分数，并按会话分组。七个控制项目单独报告。A 组没有足够评分时记为 `insufficient_scores`。

### 范围

默认输入为本目录 `fixtures/` 内三个逐字归档的合成教学验收文件，哈希与原冻结清单一致。迁移该工具时，可通过 `runner.py --source /path/to/evidence demo ...` 指定同一批证据的副本，文件哈希必须匹配。生产对照测试只调用生产序列化函数，用隔离环境禁用 dotenv，并设置内存 SQLite URL。真实 API 回放每批都需明确审批。独立入口 `real_replay.py` 使用同一冻结清单，要求显式启用、配置中的模型与基线一致，以及尚不存在的证据目录。它仅在进程内加载现有凭据，逐项在发送前记账，不重试，不跟随重定向。授权或额度等整批阻断错误会停止后续项。评分命令同时支持真实与合成盲评材料。

## 2026-11-06 仓库归档

本工具最初位于 Codex 项目目录。本次归档仅调整夹具、仓库及配置的定位，冻结提示词、计划、教师输入与哈希不变。已有真实结果摘要见 [本周更新](../../docs/updates/2026-11-06.md)。归档和离线测试不执行真实回放；历史运行额度不可复用。
