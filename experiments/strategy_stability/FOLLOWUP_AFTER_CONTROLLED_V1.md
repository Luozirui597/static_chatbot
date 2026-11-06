# 真实预实验后续事项

2026-11-06：controlled_v1 已冻结并完成 32 会话 / 44 次真实生成。

报告：
[本周更新与已知问题](../../docs/updates/2026-11-06.md#6-独立实验冻结资产与评分)；[正文复核结果](../../docs/updates/evidence/controlled-prepilot-reviewed-outcomes.json)。完整原始调用归档仍保存在本机项目的 reviews/strategy-stability-controlled-real-pilot-20261106/。

1. **先修缺口状态刷新**：补教成功后，GoalContext 的 missing_relations/missing_rule 未更新，B 的 I2/I3 FINAL 被渲染为“仍缺已教内容”。原始 B 候选是合法 insufficient；错误由服务器上下文和渲染器产生。A+B 的同类旧上下文暂未出现在 infer 正文中。补充 B 问中间事实/规则、补齐后合法停止的两条回归；当前 fake-seek 的 B 走 direct_target，未覆盖此分支。
2. **复核显式策略身份**：当前共同候选合同引用策略名，Standard 原正文没有显式同名头、公开数据也没有组名。应在新版本 system 中统一给出策略身份和有效权限，任务 user 输入保持四组相同；缺口答案/gold 仍不外发。
3. **保留冻结 v1**：不得修改 controlled_v1 的冻结数据后覆写原结果或补跑。真实证据目录 frozen-sources 包含本批完整源码快照。修复需要新版本/哈希，44 次额度已用完，不自动增加调用。

本次程序关系初筛 39/44 通过，但正文复核发现 B 两条错误后为 37/44。不要把 summary.json 的 appropriate_turns 当成最终全文正确率；以 reviewed-outcomes.json 和报告为准。

A+B 的两次真实缺口→补教→推导均成功；这是受控已知任务各一次的观察，不能声称稳定性、普遍优势或协同效应。正式链路和业务数据库未接入。
