# RetailPilot 快速开始

要求：Python 3.11+、Git。基础模式不需要 GPU，也不需要 LLM Key。

下载或克隆你维护的 RetailPilot 仓库后，在包含 `main.py` 的项目根目录运行以下命令。

## 1. 最快离线启动

Windows PowerShell：

```powershell
python scripts/quickstart.py
$env:RETAILPILOT_ALLOW_OPEN_API='true'
.\.venv\Scripts\python.exe main.py serve
```

macOS / Linux：

```bash
python3 scripts/quickstart.py
RETAILPILOT_ALLOW_OPEN_API=true .venv/bin/python main.py serve
```

打开 `http://127.0.0.1:8000`。快速模式构建确定性 CI 商品库，适合运行、测试和
Harness 回归；不会下载第三方原始数据。

## 2. 真实商品和真实多轮客服 Query

先阅读上游仓库的数据条款。当前集成时两个上游仓库均未暴露清晰 LICENSE 文件，
因此脚本要求显式确认，并且只下载到 `.gitignore` 覆盖的 `data/external/`：

```powershell
python scripts/quickstart.py --data public --with-real-dialogues --accept-upstream-terms
```

该命令会：

1. 构建本地 SQLite/FTS/Hash 向量基础库；
2. 下载并导入 ShopSimulator 的公开中文商品和购物任务；
3. 下载 USS 发布的 JDDC 子集：3,300 段真实中文电商客服对话；本地文件包含
   54,517 个用户 utterance、106,508 个用户/客服 utterance；
4. 保存来源 URL、SHA256 和本地 manifest，不提交原始数据。

也可以分开执行：

```powershell
.\.venv\Scripts\python.exe scripts\download_public_data.py `
  --dataset shopsimulator --dataset jddc-uss --accept-upstream-terms
.\.venv\Scripts\python.exe scripts\import_shopsimulator.py `
  data\external\shopsimulator\fine_items_eval_train_all.json.gz --replace
```

上游来源：

- [ShopSimulator](https://github.com/ShopAgent-Team/ShopSimulator)
- [USS / JDDC 子集](https://github.com/sunnweiwei/user-satisfaction-simulation)
- [JDDC 论文](https://aclanthology.org/2020.lrec-1.58/)

## 3. 批量多轮 Query、Judge 和 Badcase

### 先用批量 LLM 筛选项目范围

对于 JDDC 这类包含大量非目标业务的真实客服语料，先运行独立 Scope Classifier，
再让 Agent 和 Judge 只消费筛选后的主线 Conversation：

```powershell
.\.venv\Scripts\python.exe scripts\classify_project_scope.py `
  --input data\external\jddc-uss\JDDC.txt --format jddc-uss `
  --limit 3300 --max-user-turns 0 --batch-size 80 --min-batch-size 5 `
  --workers 4 --output-dir artifacts\conversation_replay_runs\jddc-full-llm-scope-v1 `
  --require-complete
```

Scope Classifier 一次批量判断多个 Query，不逐条请求；默认关闭 DeepSeek Thinking，且只返回
`case_id + label + confidence`，不生成理由。每批必须完整返回全部 `case_id` 才算成功；失败批次
自动拆成更小的 40/20/10/5 条批次。成功结果实时写入 `scope_batch_results.jsonl`，中断后加
`--resume` 即可续跑。

分类标签为 `pre_sales / after_sales_rag / order_read / out_of_scope /
insufficient_context`。前三类进入 `eligible_conversations.jsonl`；范围外和上下文不足分别进入
`scope_filtered_turns.jsonl` 与 `scope_uncertain_turns.jsonl`。Scope Classifier 只看原始 Query
及其先前对话，不看 Agent Intent、回复、工具或 Badcase；它与回答质量 LLM Judge 是两个隔离阶段。

筛选结束后，将 `eligible_conversations.jsonl` 作为下一节 Replay 的 JSONL 输入。

### 固定工程回归

```powershell
.\.venv\Scripts\python.exe scripts\run_customer_service_eval.py
```

### 真实客服 Query shadow replay

先用 100 段检查配置：

```powershell
.\.venv\Scripts\python.exe scripts\run_multiturn_replay.py `
  --input data\external\jddc-uss\JDDC.txt `
  --format jddc-uss --limit 100 --max-user-turns 12 --workers 4
```

跑 3,300 段、对全部范围内 Turn 使用 LLM Judge（默认全量）：

```powershell
.\.venv\Scripts\python.exe scripts\run_multiturn_replay.py `
  --input data\external\jddc-uss\JDDC.txt `
  --format jddc-uss --limit 3300 --max-user-turns 20 --workers 4 `
  --llm-judge --require-llm-judge
```

`--workers` 需要根据 SQLite 和模型 API 的限流设置；发生中断时使用固定
`--output-dir` 并增加 `--resume`。每完成一个 Conversation 就追加 checkpoint。

输出目录为 `artifacts/conversation_replay_runs/<run-id>/`：

- `manifest.json`：版本、数据源、运行参数和 Judge 请求状态；
- `results.jsonl`：每个 Conversation 的 Agent 回复、Intent、Tool、Grounding 和 Trace；
- `summary.json` / `summary.md`：通过率、根因、Intent/Tool 分布和 P50/P95；
- `badcases.jsonl`：硬检查或语义 Judge 失败的完整多轮轨迹；
- `badcase_turns.jsonl`：只保留失败 Turn 的扁平记录，包含 Query、回复、工具、
  硬检查证据和真实 LLM Judge 证据，便于脚本处理；
- `badcase_clusters.json`：按 `hard_check / llm_judge + failure_type` 聚类的机器可读结果；
- `badcase_analysis.md`：按 failure type 聚类的可读代表样本及建议修复层。
- `scope_filtered_turns.jsonl`：不属于项目主线的 Turn 及过滤原因；这些 Turn 不送
  LLM Judge，不算通过或失败，也不会进入 Badcase 修复队列。

回放在 Judge 之前执行独立的 `project-scope-gate-v1`。范围门禁不读取 Agent 自己预测的
Intent 或工具，否则一次错误路由会同时污染“是否该评”的判断。当前只评估：

- 商品搜索/推荐、详情和对比；
- 退换货、退款、配送/运费、保修、发票和质量问题等售后知识 RAG；
- 只读订单与物流查询；
- 上述业务 Frame 内有明确承接关系的多轮省略句。

纯闲聊/模型身份、提示词注入、火车票等第三方服务、额度/支付、价保、购物车页面故障、
自主下单付款等范围外 Query 会标记为 `scope_filtered`。过滤是“本项目不处理”，不是 Agent
答对，更不能用来提高通过率。`summary.json` 分开记录 `in_scope_turns`、
`scope_filtered_turns` 和 `scope_filter_counts`。

这属于 shadow replay：用户 Query 来自真实人工客服会话，但原人工客服回复不会注入
Agent，只在产物中作为弱参考上下文。它用于发现 Context、Intent、Tool、RAG 和回复质量
问题，不代表线上准确率，也不能把原回复当作本项目商品、订单或政策的事实 Gold。

### 任意自有 JSONL

格式是一行一段 Conversation：

```json
{"conversation_id":"c1","turns":[{"role":"user","content":"查订单 JD20260801"},{"role":"assistant","content":"参考回复"},{"role":"user","content":"买的什么"}]}
```

```powershell
.\.venv\Scripts\python.exe scripts\run_multiturn_replay.py `
  --input your-dialogues.jsonl --format jsonl --limit 1000 --workers 4
```

仓库包含两个无隐私示例：`config/conversation-replay-example.jsonl`。

## 4. 配置真正的 LLM Judge

复制 `.env.example` 为 `.env`，填写你自己的 OpenAI 兼容端点。Judge 可以用
独立的 `JUDGE_*` 配置；**不回退到 `LLM_*`**，没有 `JUDGE_API_KEY` 时评测直接失败：

```dotenv
# Fill this only in your local .env; never commit the value.
LLM_API_KEY=
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_TIMEOUT_SECONDS=20
LLM_MAX_RETRIES=1
INTENT_MODE=llm
LLM_MODE=fallback

# 推荐：让被测 Agent 与评审模型配置相互独立
JUDGE_API_KEY=
JUDGE_BASE_URL=https://api.openai.com/v1
JUDGE_MODEL=gpt-4o-mini
JUDGE_TIMEOUT_SECONDS=20
JUDGE_MAX_RETRIES=1
# 推理模型的隐藏 reasoning token 也会计入该预算
JUDGE_MAX_TOKENS=3000
JUDGE_JSON_MODE=true
```

真实 Key 只能放 `.env` 或环境变量，`.env` 已被 Git 忽略。报告中：

批量执行前先做一次真实连通性和 JSON 契约检查；输出只包含模型、Endpoint Host、
耗时等脱敏信息，不会打印 Key：

```powershell
.\.venv\Scripts\python.exe scripts\check_llm_judge.py
```

- `judge_mode=llm`：模型调用和结构化解析真实成功；
- `judge_mode=deterministic_fallback`：请求了 LLM，但超时、无 Key或解析失败；
- `judge_mode=deterministic`：没有请求 LLM；
- `judge_mode=not_selected`：受 `--judge-limit` 控制，本 Turn 未抽样。
- `judge_mode=scope_filtered`：不属于当前项目范围，未调用 Judge，且不进入
  通过率或 Badcase。

正式结果建议总是加 `--require-llm-judge`；如果任何选中 Turn 降级或未真正调用模型，
命令会返回非零状态，CI/脚本不会把降级结果误报为 LLM Judge 成功。

如果 Agent 回放已经完成、只有部分 Judge 因接口波动或结构化输出失败而 fallback，
不需要重新执行 Agent。对同一运行目录只重评失败 Turn，并原子刷新全部报告：

```powershell
.\.venv\Scripts\python.exe scripts\retry_replay_judge.py `
  --run-dir artifacts\conversation_replay_runs\<run-id> `
  --max-passes 3 --require-complete
```

该命令复用已经固化的 Query、Agent 回复、Intent 和历史，只替换
`deterministic_fallback` verdict；`manifest.json` 会记录尝试数、恢复数、未恢复数以及
`agent_rerun=false`。仍有 fallback 时 `--require-complete` 返回非零。

`summary.json` 中 `llm_judged_turns`、`llm_turns_passed` 和 `llm_pass_rate` **只统计
`judge_mode=llm`**。无 Key、超时或解析失败产生的 `deterministic_fallback` 单独计入
`fallback_turns`，不进入 LLM 通过率，也不会仅凭 fallback 判定一个对话为语义 Badcase。
每条 `results.jsonl` 的 `evaluation_evidence` 会同时保留：

- `hard_check`：程序断言、失败类型及可复现证据；
- `llm_judge`：是否实际执行模型、结构化分项分数、失败类型与一句话依据；
- fallback 时只保留 `fallback_diagnostic`，`counted_as_llm_judge=false`。

Judge 负责需求覆盖、上下文使用、相关性和机械重复等开放语义；工具选择、重复任务、
Claim-Citation、订单和政策事实仍由程序硬检查，不让 LLM 推翻确定性结论。

## 5. 检索任务和完整验证

```powershell
# ShopSimulator 200 条固定检索 Query
.\.venv\Scripts\python.exe scripts\run_real_query_batch.py --split final --limit 200

# 对检索 Badcase 跑 LLMJudge
.\.venv\Scripts\python.exe scripts\judge_real_query_run.py `
  artifacts\real_query_runs\<run-id> --limit 100 --workers 4 --require-complete

# 全量测试
.\.venv\Scripts\python.exe -m pytest
```

神经检索是可选能力，见主 README；基础 Harness 只依赖轻量 `requirements.txt`。

## 6. 可选启用 GitHub Actions

仓库内保留 `config/github-workflows/*.yml.example` 模板。当前推送凭据缺少 GitHub
`workflow` scope，因此模板未放入会被立即激活的 `.github/workflows/`。使用有相应权限的
凭据时，可将模板复制并去掉 `.example` 后缀：

```powershell
New-Item -ItemType Directory -Force .github\workflows
Copy-Item config\github-workflows\quality-gate.yml.example .github\workflows\quality-gate.yml
Copy-Item config\github-workflows\low-risk-automerge.yml.example .github\workflows\low-risk-automerge.yml
```

然后提交推送，并把 `quality-gate / test`、`quality-gate / policy` 配为 Required Checks。


## 受保护网页与多用户部署

本地 Demo 的 `RETAILPILOT_ALLOW_OPEN_API=true` 仅适用于单用户本机体验。公网部署保持 false，
运行 `python scripts/create_web_user.py --user alice` 生成服务端密码哈希 JSON，配置
`RETAILPILOT_AUTH_USERS_JSON`，并启用 HTTPS 与 Secure Cookie。浏览器登录后使用 HttpOnly
会话 Cookie 和 CSRF Token，无需把长期 API Key 写入前端。完整说明见
[契约与会话升级](docs/contracts-and-sessions.md)。
