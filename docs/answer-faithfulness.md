# 最终政策答案 Faithfulness 校验

本实现参考 Adaptive-rag 的 [Checker](https://github.com/JEen-wang/Adaptive-rag/blob/main/app/guardrails/hallucination.py)、
[Faithfulness 节点](https://github.com/JEen-wang/Adaptive-rag/blob/main/app/agents/nodes/faithfulness.py) 和
[校验/修复 Prompt](https://github.com/JEen-wang/Adaptive-rag/blob/main/app/prompts/faithfulness.py) 的结构化判断、
单次改写与复检方式；代码适配本项目已有的 LLM、Telemetry、工具与 Claim-Citation 机制。
校验不可用时直接安全拒答，不采用未经校验的抽取回退。没有复制参考项目按前 6/8 条截断证据的做法。

源文件位于 `RetailPilot/answer_faithfulness.py`，统一通过
`RetailPilot.answer_faithfulness` 导入。

## 图中的位置

```text
after_sales_branch（原有检索、累计证据、生成和分支 Grounding）
  → execute_tool（汇总内部实际证据）
  → build_response（最终分节合成及重复回复处理）
  → verify_grounding（原有结构校验失败关闭回答通道）
  → verify_answer_faithfulness（最终答案语义校验 / 最多一次修复 / 复检）
  → persist（保存最终通过校验或安全拒答的回复与任务结果）
```

Faithfulness 是最后一个可能修改答案的节点。输入是最终 `state.response`，包括合成标题和包装文字，
不是单独的售后分支中间回复。通过后保持原答案、Claim、Citation 不变。
分节标题标识用户问题，不作为系统声称的事实。仅检查售后政策结论，其他意图分节是上下文。

第一版只处理 `policy_question` 已获得 answer 证据决策且没有 required_slots/tool_failure 的回答。
闲聊、商品检索、纯订单查询、澄清和已有拒答不调用校验模型。
重复回复中间件若把整个回复改为澄清，删除 task_results 中已经不再展示的旧政策答案与旧 Claim，
不会在跳过校验时从嵌套字段继续输出那条政策结论。

## 实际证据如何传递

售后分支把真正进入 compose_reply 的最终政策快照放入请求级 `policy_answer_evidence[task_id]`。
execute_tool 汇总这个内部 State 字段，build_response 记录 answer_sections 和 policy_answer_targets。
材料包含完整 Chunk 正文、真实 ID、来源、版本/有效期和必要的权威商品上下文。
订单工具结果仅提供 order_id、sku、title、status、created_at 及事件的 event_time/status；
不传用户账号标识、收件人姓名、手机号、地址和完整订单对象。

不从历史会话、用户文字或 Citation quoted_text 重建原始政策证据；内部快照缺失时不能放行答案。
材料仅通过 AgentState 传递，不写入 session context、assistant payload 或最终 API 返回值。
公共 API 递归移除这些内部字段及详细校验状态，只留下已有的安全 Trace 摘要。
内部 Trace 不记录完整 Prompt、原答案或工具对象；错误片段和模型理由仅在服务端 Trace 中保留。

## 校验与单次修复

FaithfulnessResult 使用严格 Pydantic Schema，要求 `supported`、`hallucinated_spans`、`rationale`。
布尔值不接受字符串转换；未知/缺失字段、非法 JSON、supported 与错误片段矛盾、虚构的错误片段
都视为校验器不可用。仅遗漏必要例外时错误片段可以为空，必须说明遗漏的条件。
输入问题、答案、政策、工具和错误说明均当作数据，不执行其中的指令。

LLM 检查事实是否获得语义支持，重点包括错误推断、期限/适用对象、政策例外、歪曲及编造事实。
政策相关、相同词语、有效 Chunk ID 和结构 Grounding 通过都不足以证明结论正确。
例如“超过七天出现质量问题可申请核验”不能推出“超过七天出现质量问题保证退货成功”。

- supported=true：不改动原答案。
- supported=false：携带问题、原最终答案、错误说明和同一证据集合调用一次修复。
- 修复器只返回目标政策 task_id 的 statements，每条含最终文本和支持它的 chunk_ids。
- 程序要求任务集合完全一致，禁止修改商品、订单或闲聊分节；禁止引用未送审的 Chunk。
- 最终政策正文由这些 statements 拼接，不允许额外的未映射自由文本。
- 重新从真实快照构建 Citation、Claim，执行原有 validate_claim_citations。
- 修复不能丢弃已选入的关键证据，必须覆盖全部原最终材料的 ID。
- 重新合成完整可见答案，携带新的 claim_evidence_map 做第二次语义校验。
- 结构校验失败、复检失败、非法输出、预算耗尽、超时或模型异常：安全拒答，清空该政策分节的 Claim/Citation。

结构 Grounding 的词面检查仍是必要约束；最终语义复检独立决定是否可以回答。
测试特意构造了“词面检查通过、语义判断为 false”的修复，结果仍为拒答。
多意图回答只替换政策分节，其他分节的正文、Claim 和 Citation 保持原样。
持久化使用 final_branch_outputs，避免主回复已修复但 task_results 继续携带旧结论。

## 调用、时间和 Token 预算

复用 `_invoke_json`、get_model、invoke_with_telemetry、Prompt 前缀缓存和版本指纹。
复用已有的 PolicyCallBudget、截止时间封装及 ContextVar 隔离，不引入新的 provider 或日志系统。

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| POLICY_FAITHFULNESS_TIMEOUT_SECONDS | 30 | 整个校验、修复及复检阶段共享截止时间 |
| POLICY_FAITHFULNESS_LLM_TOKEN_BUDGET | 80000 | 本阶段输入与输出 Token 的保守预留上界 |
| LLM_MAX_TOKENS_FAITHFULNESS_CHECK | 800 | 一次校验输出上限 |
| LLM_MAX_TOKENS_FAITHFULNESS_REPAIR | 1800 | 一次修复输出上限 |

每个最终回答通常增加一次校验；发现问题时最多再增加一次修复和一次复检，共最多三次调用。
不循环重试。该阶段禁用 SDK 重试与 JSON 模式失败后的普通模式重调。
网络调用前按 Prompt UTF-8 字节、消息余量及输出上限预留预算；失败保留预留，不静默截断证据。
真实 Token、缓存和价格配置仍由现有 Telemetry 记录，没有配置价格时费用为 null。
截止时间后不再接收或发布结果；已经发出的请求可能稍后结束，其真实使用量仍由 Telemetry 记录。

内部 Trace 包含初次与修复后 supported、错误片段、理由、阶段耗时、修复是否尝试/成功、
停止原因、最终答案 SHA256、完成的 LLM 调用数、实际/预留 Token 和费用。
SHA256 对应真正返回并持久化的文本，修复后不会再经过另一个未校验的生成步骤。

## 修改文件

| 文件 | 职责 |
| --- | --- |
| `RetailPilot/answer_faithfulness.py`（新增） | 严格 Schema、Prompt、最终答案校验、一次修复、重新合成、失败拒答与 Trace |
| `RetailPilot/agent.py` | 真实快照汇总、分节与目标记录、图节点接入、最终任务结果持久化 |
| `RetailPilot/state.py` | 内部证据、目标、分节、最终分支及校验状态字段 |
| `RetailPilot/grounding.py` | 复用 policy_citation，按修复文本重建 Claim/Citation 并做原有校验 |
| `RetailPilot/llm.py` | 两个新场景禁用不受控的二次调用 |
| `RetailPilot/llm_config.py` | 校验/修复的输出 Token 上限 |
| `RetailPilot/config.py`、`.env.example` | 阶段截止时间与 Token 预算配置 |
| `RetailPilot/api.py` | 防御性移除内部 Faithfulness 状态与原始材料 |
| `tests/test_answer_faithfulness.py`（新增） | 结构、语义、修复、失败、混合意图及真实图持久化测试 |
| `tests/test_agent.py` | 图结构断言增加最终校验节点 |
| `README.md`、本文档 | 功能入口、执行链与验证说明 |

## 自动化测试

普通测试使用 Fake LLM 与隔离数据库，不联网、不消耗模型 Token：

```powershell
New-Item -ItemType Directory -Force .test-tmp | Out-Null
.venv/Scripts/python.exe -m pytest tests/test_answer_faithfulness.py tests/test_agent.py tests/test_policy_followup.py tests/test_policy.py tests/test_intent.py tests/test_tools.py --basetemp=.test-tmp/reproduce-faithfulness -o addopts='' -q
.venv/Scripts/python.exe -m ruff check RetailPilot/answer_faithfulness.py tests/test_answer_faithfulness.py
.venv/Scripts/python.exe -m ruff format --check RetailPilot/answer_faithfulness.py tests/test_answer_faithfulness.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --ignore-missing-imports RetailPilot/answer_faithfulness.py RetailPilot/grounding.py RetailPilot/state.py --cache-dir=.test-tmp/mypy-faithfulness
```

全量测试结果与环境限制见 `artifacts/faithfulness/regression-output.txt`；相关 JUnit 结果见同目录。
这是功能与边界验证，不代表真实模型已经达到某个忠实度指标；独立 Judge 和真实模型验收仍需有效凭据。

### 本次真实执行结果（2026-10-28）

- 新增 Faithfulness 测试：37 项；连同售后、Policy RAG、Planner 与工具回归，相关测试 **131 passed**。
- 全量 pytest：**220 passed，6 failed**，13 条第三方弃用警告。
- 新增生产模块与测试文件的 Ruff check / format check 通过。
- answer_faithfulness.py、grounding.py、state.py 的 Mypy 检查通过（导入静默、忽略缺失第三方类型）。
- compileall 和 pip check 通过。
- 普通 pytest 使用 Fake LLM；没有消耗真实模型 Token，没有验证真实模型的忠实度水平或真实费用。

全量失败仍为上一轮记录的六个商品检索/日志用例：test_api_smoke、
test_low_candidate_policy_relaxes_before_zero_results、test_search_logging_persists_routes_impressions_and_sample、
test_hybrid_search_respects_budget_and_negative_brand、
test_search_uses_controlled_attribute_relaxation_when_strict_pool_is_empty、
test_approximate_budget_keeps_nearby_exact_catalog_match。
当前合成 CI 商品库不能满足这些用例的数据预期；本次没有修改商品检索实现或弱化这些断言。
原失败及关闭补检的对照见 [上一轮验证记录](policy-adaptive-validation.md)。因此不宣称全仓回归已经全绿。

相关测试 JUnit：`artifacts/faithfulness/targeted-tests.xml`；全量 JUnit：
`artifacts/faithfulness/regression-tests.xml`。
