# Evidence-Driven Adaptive Policy RAG

实现源代码在 `RetailPilot/`，统一使用 `RetailPilot` 导入命名空间。

## 执行链

`plan_after_sales` → 原有订单依赖解析 → `registry.execute(policy_retrieve)` →
`run_policy_followup` → Evidence Selector → `assess_policy_evidence` →
按需补检、重新评估 → `compose_reply` → Claim/Citation 校验 → 最终合成 → `verify_grounding` →
[`verify_answer_faithfulness`](answer-faithfulness.md) → 持久化及返回。

仅 `route=policy_question` 启用。Planner 不预测检索次数；首轮充分即回答。
订单查询、用户隔离、SKU 权威类目解析、其它意图和 LangGraph 主结构保持原样。

`PolicyEvidenceAssessment` 保留 decision/sufficient/conflict/reason/clarification_question；
增加 next_action、missing_aspects、covered_aspects、required_user_fields、supporting_chunk_ids、
aspect_evidence_map 和 suggested_queries。每个已覆盖条件必须映射到送审快照中的真实 Chunk。
程序校验结构、一致性和 ID，语义充分性仍由 LLM 判断，检索分数不构成覆盖证明。
旧版 clarify/refuse 可以兼容；旧版 answer 缺少条件映射时关闭回答通道。

四类动作：answer 为完整覆盖；retrieve_more 为缺失具体政策条件；ask_user 为缺失业务事实；
refuse 为冲突、无法可靠判断或预算耗尽。空检索结果也交给 Grader，不能据此猜测需要订单号。
评估采用严格 JSON 和 Pydantic 校验，非法输出/超时/不可用都不能放行答案。

## 查询与权限

每轮最多执行一次只读 policy_retrieve。查询为原独立问题加 `；补充核对：缺失条件`，
或仅缺失条件，不执行任意自由改写的查询。不另开 Query Rewrite LLM。
Grader 只能从 missing_aspects 提议查询和规范主题；不能传平台、日期、类目、权限或工具名。
跨主题相对于 Planner 原主题验证，禁止逐轮传递扩张：

| 原主题 | 允许补检主题 |
| --- | --- |
| 退换货 | 退换货、保修、退款 |
| 保修 | 保修、退换货 |
| 退款 | 退款、退换货 |
| 配送 | 配送 |
| 发票 | 发票 |

每次仍调用原 BM25/Dense/RRF/Rerank 检索和 SQL 权威过滤；过期、错误平台/类目、
内部受众、失效文档及陈旧向量候选不能绕过 SQL。首轮返回的权威商品上下文冻结，
执行器固定平台和当天 as_of，后续复用同一天，避免跨午夜改变有效政策集合。

## 证据集合与生成

池中按稳定 chunk_id 去重并记录 first_seen_hop，原始元数据与 Parent Context 保留。
相同适用范围/版本下，忽略空白后完全相同的正文不计为新增信息；不同范围/版本保留。
不使用近似相似度删除政策文本，避免吞掉仅一字不同的否定、限制或冲突。
选择器先安排条件覆盖，再安排新一轮与不同章节，最后参考 Rerank/RRF、priority。
未选入的独立证据可能包含例外或冲突，所以存在预算截断时不能宣布充分，安全拒答。

送审和生成均使用完整叶子正文；Parent Context 保存在池中但不作为叶子引用之外的隐藏事实。
正文不截断。上下文预算按 JSON UTF-8 字节加余量保守计算，不假装是供应商精确 Token 数。
若支持集合小于送审集合，最多增加一次对最终集合的确认评估，要求覆盖和支持关系稳定。
最终生成与这次评估使用同一快照，Trace 记录实际选择的 ID。

删除了生成阶段固定取前两条的逻辑。compose_reply 政策路径返回结构化逐字引用，
程序验证每条 chunk_id 和完整原文，所有最终材料均须引用；失败回退为这些原文与来源。
这是一项有意的保守取舍：政策答案以可核验原文为主，不允许自由改写限制与承诺。
Citation、Claim 继续通过 attach_citation_ids、validate_claim_citations 和原有图的关闭回答机制。
完整送审快照确认并不证明 LLM 的语义判断正确；须用独立 Judge 和人工标签验证。

## 预算和终止

| 配置 | 默认值 | 范围 |
| --- | --- | --- |
| POLICY_FOLLOWUP_ENABLED | true | false 用作单轮消融 |
| POLICY_MAX_EXTRA_HOPS | 2 | 0–4，超过配置值强制截断 |
| POLICY_FOLLOWUP_TIMEOUT_SECONDS | 45 | 首次政策检索起，涵盖检索和评估 |
| POLICY_MAX_EVIDENCE_CHUNKS | 12 | 1–24 |
| POLICY_EVIDENCE_TOKEN_BUDGET | 10000 | 证据上下文的保守字节上界 |
| POLICY_FOLLOWUP_LLM_TOKEN_BUDGET | 80000 | 全部证据评估调用的输入+输出预留上界 |
| LLM_MAX_TOKENS_POLICY_EVIDENCE | 1600 | 单次评估输出上限 |

最多 H+1 次逐轮评估，另有最多一次最终集合确认；最多 H 次额外检索。
独立的 Planner 和回答生成继续使用原有输出上限；上述评估预算不计入它们。
每次评估在网络调用前，按实际 Prompt UTF-8 字节、消息余量和输出上限预留预算；
失败仍保留预留。评估关闭 SDK 重试以及普通模式二次调用，避免超时后的隐藏成本。
实际 Token/缓存/费用仍由 invoke_with_telemetry 持久化；无价格配置时费用是 null。

停止原因含 answer、ask_user、policy_conflict、max_extra_hops、followup_disabled、
assessment_timeout/error、retrieval_timeout/error、timeout_budget、repeated_query、
no_new_chunks、no_new_information、no_semantic_progress、assessment_oscillation、
cross_topic_denied、invalid_followup_query、evidence_context_budget 和最终集合确认失败。
Grader 的预算拒绝体现在 assessment_error 中，预留/调用记录用于区分。
ToolRegistry 的补检重试也受同一截止时间限制。Python 无法强制取消已经发送的只读外部调用；
截止时间后不再接受其结果、不再启动下一轮；在途请求可能稍后完成并记录 Telemetry。

内部 Trace 记录首轮数量、每轮动作/原因/缺失条件、Query/主题、工具与检索 Trace、
新增 Chunk 和有效信息数、累计数、最终 ID、停止原因、评估次数、实际/预留 Token、费用与耗时。
公共 API 递归删除完整检索 Trace 与评估内容，仅保留安全摘要，避免暴露内部提示及业务上下文。

## 验证与消融复现

普通 pytest 使用 Fake LLM、固定政策、Mock 工具和隔离 SQLite，不调用真实模型。
首次运行空仓库环境可用现有脚本生成合成 CI 数据，勿对已有业务数据库运行初始化脚本：

```powershell
.venv/Scripts/python.exe scripts/bootstrap_ci.py
New-Item -ItemType Directory -Force .test-tmp | Out-Null
.venv/Scripts/python.exe -m pytest --basetemp=.test-tmp/policy-run
.venv/Scripts/python.exe -m ruff check RetailPilot/policy_evidence.py RetailPilot/policy_followup.py tests/test_policy_followup.py scripts/evaluate_policy_followup.py
.venv/Scripts/python.exe scripts/evaluate_policy_followup.py --mock
```

离线报告：`artifacts/policy-adaptive/report.json`；示例 Trace：`example-trace.json`。
mock_control_flow_only 表示只验证流程，不代表真实模型效果改善；不计算虚构提升百分比。

真实消融使用相同输入、数据库快照、模型和独立会话分别运行已有 Replay：

```powershell
$env:POLICY_FOLLOWUP_ENABLED='false'
.venv/Scripts/python.exe scripts/run_multiturn_replay.py --input config/conversation-replay-example.jsonl --format jsonl --output-dir artifacts/policy-baseline
$env:POLICY_FOLLOWUP_ENABLED='true'
.venv/Scripts/python.exe scripts/run_multiturn_replay.py --input config/conversation-replay-example.jsonl --format jsonl --output-dir artifacts/policy-enhanced
.venv/Scripts/python.exe scripts/evaluate_policy_followup.py --baseline artifacts/policy-baseline/results.jsonl --enhanced artifacts/policy-enhanced/results.jsonl
```

请选用政策问题数据集，并配置独立 JUDGE_*，沿用 Replay 的 Judge、BadCase 和质量门槛，
不得将 Grader 自评当作答案正确性。baseline 为关闭补检的单轮消融，使用相同强化后的安全检查，
不是对旧版本二进制的历史性能重测。
逐轮可附 expected_policy：required_chunk_ids、expected_action、经审计的 hard_filter_violations；
没有标签的 Coverage/正确率/过滤违规率返回 null，不能用于通过质量门槛。独立 Judge 通过率门槛沿用现有 cluster_pass_rate 的 0.95，不能仅因 Judge 被调用过就通过。
报告复用 evaluation.py，包含覆盖率、Judge 正确性/忠实性、结构 Grounding、应答率、
拒答精确率、多条件成功率、首轮充分率、补检触发/成功率、重复次数、平均调用/Token/费用和 P50/P95。
延迟是政策检索到评估结束的耗时；完整 Agent 延迟继续查看已有 Replay 的统计。

## 文件职责

- `RetailPilot/policy_evidence.py`：严格决策 Schema、兼容及证据 ID 校验。
- `RetailPilot/policy_followup.py`：只读控制器、预算、跨主题边界、去重及证据选择。
- `RetailPilot/llm.py`：评估 Prompt/严格 JSON，政策生成材料校验。
- `RetailPilot/agent.py`：接入控制器、冻结上下文、更新 Citation/Claim/Trace。
- `RetailPilot/tools.py`：工具可选截止时间与政策 as_of 参数。
- `RetailPilot/config.py`、`.env.example`：运行参数。
- `RetailPilot/llm_config.py`：每请求预算与原有 Telemetry 衔接。
- `RetailPilot/api.py`：递归去除内部检索/评估详情。
- `RetailPilot/grounding.py`：显式 Chunk 引用必须同时匹配真实 Chunk 和文档，禁止错误 ID 回退到文档级验证。
- `RetailPilot/evaluation.py`、`scripts/evaluate_policy_followup.py`：Replay 侧车统计和离线示例。
- `tests/test_policy_followup.py`、`tests/test_policy.py`：流程、Agent、生成与真实过滤回归。
- `tests/test_agent.py`：工具 Mock 支持可选 deadline；`tests/test_intent.py`：检查缓存消息内的实际 Prompt 文本。
- `config/quality-gates.json`：在原有门槛之外增加政策安全与独立语义评估门槛。
- `README.md`、本技术文档与 `docs/policy-adaptive-validation.md`：入口、设计与真实执行记录。

当前环境的测试结果、未通过项和模型验证限制见 [验证记录](policy-adaptive-validation.md)。
