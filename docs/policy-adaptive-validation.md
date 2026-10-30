# 实施与验证记录

日期：2026-10-28。运行环境：Windows、Python 3.14.6，本地 `.venv`。
项目目录没有 Git 元数据；本次直接修改工作区文件，没有创建提交或 PR。

## 实际执行结果

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| 相关单元、Agent 集成与回归 | **94 passed** | test_policy_followup / test_policy / test_agent / test_intent / test_tools |
| 全量 pytest | **183 passed，6 failed** | 189 项，13 条第三方弃用警告 |
| 关闭补检的失败对照 | **相同 6 项失败** | 与政策流程无关的商品搜索/日志用例 |
| Ruff check / format | **通过** | 两个新增生产模块、评测脚本和新测试文件 |
| Mypy | **3 source files 通过** | 两个新增生产模块、评测脚本；follow-imports=silent、ignore-missing-imports |
| compileall | **通过** | RetailPilot、tests、新评测脚本 |
| pip check | **通过** | 项目基础依赖，不含可选向量模型栈 |
| 修改的既有模块 Ruff 扩展检查 | **43 项问题** | 主要为原有未使用导入、异常边界、拼接和旧代码风格；未做无关清理 |
| 离线 Baseline/Enhanced 示例 | **执行成功** | mock_control_flow_only，独立语义发布门槛保持未通过 |

未删除、跳过或放宽原有断言。工具 Mock 增加 deadline 参数；旧 Prompt 断言改为检查
缓存消息中的实际 system 文本，不再对消息对象列表做字符串成员判断。

全量测试未通过的用例：

- `test_api_smoke`
- `test_low_candidate_policy_relaxes_before_zero_results`
- `test_search_logging_persists_routes_impressions_and_sample`
- `test_hybrid_search_respects_budget_and_negative_brand`
- `test_search_uses_controlled_attribute_relaxation_when_strict_pool_is_empty`
- `test_approximate_budget_keeps_nearby_exact_catalog_match`

当前环境原本没有数据库，使用仓库已有 `scripts/bootstrap_ci.py` 生成 4,203 条合成商品，
没有引入真实商品数据。该合成库中文“耳机”标题行数为 0，无法满足相关商品检索测试的中文
命中要求；日志测试只产生 1 条展示，断言需要 5 条。禁用 POLICY_FOLLOWUP_ENABLED 后，
这六项失败仍复现。本次没有修改商品检索、Query 解析或这些用例的业务断言。
因此不能宣称全仓回归或生产发布验收已经全部通过。

## 自动化证明的行为

新测试覆盖：首轮直接回答；补检一轮和两轮；预算耗尽；业务信息澄清；冲突；
重复查询；稳定 ID 去重；新 ID 但相同信息；语义无进展和振荡；非法 JSON、字段、状态和虚构 ID；
Grader 不可用；工具错误与超时；跨主题允许/拒绝；不可自由引入查询事实；
真实 SQL/FTS 对日期、版本状态、平台、类目、受众和陈旧 Dense 候选的过滤；
多轮证据进入同一个生成集合；三条 Citation 与 Claim；精确来源校验；
非政策路径；公开 API 脱敏；Telemetry 的调用数和实际 Token 记录；网络前 Token 预算拒绝；
ToolRegistry 共享截止时间；无独立指标时质量门槛不能通过。

## 可复现命令与产物

```powershell
New-Item -ItemType Directory -Force .test-tmp | Out-Null
.venv/Scripts/python.exe -m pytest tests/test_policy_followup.py tests/test_policy.py tests/test_agent.py tests/test_intent.py tests/test_tools.py --basetemp=.test-tmp/reproduce-policy -o addopts='' -q
.venv/Scripts/python.exe -m pytest --basetemp=.test-tmp/reproduce-all -o addopts='' -q
.venv/Scripts/python.exe -m ruff check RetailPilot/policy_evidence.py RetailPilot/policy_followup.py tests/test_policy_followup.py scripts/evaluate_policy_followup.py
.venv/Scripts/python.exe -m ruff format --check RetailPilot/policy_evidence.py RetailPilot/policy_followup.py tests/test_policy_followup.py scripts/evaluate_policy_followup.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --ignore-missing-imports RetailPilot/policy_evidence.py RetailPilot/policy_followup.py scripts/evaluate_policy_followup.py --cache-dir=.test-tmp/mypy-cache
.venv/Scripts/python.exe scripts/evaluate_policy_followup.py --mock
```

真实结果保存在 `artifacts/policy-adaptive/`：

- `targeted-tests.xml`：相关测试结果。
- `regression-tests.xml`、`regression-output.txt`：完整回归结果及失败原因。
- `disabled-regression-control.txt`：关闭补检后相同失败的对照结果。
- `modified-modules-ruff.txt`：对既有模块扩大检查的结果。
- `report.json`：单轮/自适应离线对照指标；缺失语义、价格、过滤审计数据为 null。
- `example-trace.json`：完整、可复现的多条件补检 Trace 与最终原文回复。

## 示例 Trace 摘要（固定夹具）

```json
{
  "question": "定制商品超过七天出现质量问题还能退货吗？",
  "initial": {"topic": "退换货", "chunk_ids": ["general"]},
  "assessment_0": {
    "next_action": "retrieve_more",
    "covered_aspects": ["定制限制"],
    "missing_aspects": ["质量例外与期限"]
  },
  "followup_1": {
    "topic": "保修",
    "query": "定制商品超过七天出现质量问题还能退货吗？；补充核对：质量例外与期限",
    "new_information_count": 1
  },
  "assessment_1": {
    "next_action": "answer",
    "aspect_evidence_map": {
      "定制限制": ["general"],
      "质量例外与期限": ["exception"]
    }
  },
  "stop_reason": "answer",
  "extra_retrieval_calls": 1,
  "assessment_calls": 2
}
```

固定夹具最终材料：

> 定制商品不适用七天无理由退货。
>
> 超过七天出现质量问题可申请核验，是否退货须按检测结果审核。

Agent 集成测试另行验证上述两条原文与三条多条件原文实际进入 compose_reply、Citation 和 Claim，
不只是留在 Trace 中。

## 尚未验证的真实效果

环境没有 Agent 或独立 Judge 凭据，没有安装 Qdrant/sentence-transformers/torch 可选栈，
因此未运行真实模型语义效果、真实 Dense/Rerank 权重、真实 Token 费用或线上 P50/P95 验收。
离线示例使用固定语义标签，两个模式使用同一 Query、证据和评估逻辑；其中耗时只有本机 Mock
控制流程意义。没有将其解释为真实效果提升，也没有编造改善百分比。

发布前应按技术文档用同一政策快照与独立 Judge 执行真实 Replay 消融，补齐语义与过滤审计数据，
解决上述商品回归数据依赖。新增 policy_required 门槛会在指标缺失时保持不通过。
