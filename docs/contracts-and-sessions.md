# 前后端契约与会话可靠性升级

## 本次范围

四项 P0 均已处理：受保护网页登录、搜索参数覆盖、公开 Trace 对齐、会话归属。
同时完成图模块等价迁移、事务及版本控制、压缩后按消息 ID 加载、固定容量执行池、公开响应模型、
前端请求取消和防重复提交、服务端历史恢复、流式节点进度及客户端模块化。
保留原生 HTML/CSS/JS、原图顺序、Policy RAG、Faithfulness 与 `chat()` 调用接口。

## Demo 与受保护部署

Demo 必须明确设置 `RETAILPILOT_ALLOW_OPEN_API=true`，只供本机单用户体验。
该模式不提供多用户身份认证，不应发布为公网多用户服务。默认配置仍为关闭。

受保护模式保持 `RETAILPILOT_ALLOW_OPEN_API=false`：

1. 执行 `python scripts/create_web_user.py --user alice`，通过隐藏输入生成密码哈希。
2. 将输出的 JSON 配置为服务端环境变量 `RETAILPILOT_AUTH_USERS_JSON`。不要提交实际配置。
3. 使用 HTTPS 反向代理，保持 `RETAILPILOT_AUTH_COOKIE_SECURE=true`，正确配置受信代理和转发协议。
4. 网页输入账号密码，通过 `/api/auth/login` 换取随机会话 Cookie；密码和长期 API Key 不存入源码、localStorage 或 Cookie。

密码使用 PBKDF2-SHA256（随机盐、310000 轮），服务端仅存密码哈希。Cookie 为随机短期凭据，
HttpOnly、SameSite=Strict、Secure，数据库只保存凭据 SHA256；默认一小时到期，退出即失效。
写请求还校验同源 Origin（存在时）及内存中的 CSRF Token。登录按来源地址进行有界进程内限速；
多实例生产部署还应在反向代理配置统一限速。API 响应禁止缓存。
本地 HTTP 验证受保护模式时才临时将 Secure 设为 false，不能沿用到公网。

保留服务端调用的 Bearer / X-API-Key 方式，但该 Key 固定绑定 `RETAILPILOT_API_USER_ID`。
请求体省略 user_id 时注入认证身份，显式伪造其他 user_id、访问他人画像或会话均返回 403。
浏览器只使用 same-origin Cookie 和内存 CSRF Token，不读取或保存长期 API Key。
这是一套自托管账号配置入口，不包含注册、找回密码或第三方 OAuth；可在边界上接入现有身份系统。

## 搜索与公开 Schema

`POST /api/search` 增加 `query_mode=auto|manual`，与 `mode=hybrid|fts` 独立：

- auto 为默认值，保留 parse_query / maybe_refine_query 的结果，不再用缺省空列表清空检索词。
- manual 不调用 Query Refine LLM，使用显式 target_terms / concepts，显式空列表代表清空。
- 兼容旧客户端：未传 query_mode 但显式给出非 null 的检索词字段时按 manual 处理。
- 显式 auto 时不采用调用者的手动检索词。类目、价格和品牌筛选仍按已有语义生效。
- 自动解析仍依赖现有解析器及模型能力；不会为了增加命中率关闭类目、权限或事实校验。

FastAPI 用 ChatResponse、SearchResponse、ProductResponse、CompareResponse 和 TraceSummary 做
公开响应校验与字段投影，OpenAPI 是契约入口。内部 State 的未知字段不会透传到顶层响应。
公开 Trace 固定为 trace_summary，含 node/tool/status/latency_ms/hop/next_action 及安全数字统计；
查询正文、模型理由、Prompt、原始材料、完整内部 Trace 不对浏览器返回。没有新增公开调试后门。
前端轨迹面板和 SSE 统一使用这些公开字段，显示节点、状态、耗时与统计。

## 会话、原子写入与并发

迁移在现有 sessions 表增加 version，保留 updated_at。已有会话只能由原 user_id 读取和更新，
save_session 不再更改所有者。拒绝归属错误的请求不读取其他用户消息。

`persist_turn` 在一个 `BEGIN IMMEDIATE` 事务中完成：校验 owner 与已加载 version、
上下文/待澄清状态更新、version + 1、用户消息、助手消息。任何一步失败全部回滚。
同一版本的并发请求只有一个可以提交，另一个返回 409（流接口以 error 事件表达）；
不会把旧上下文覆盖到新版本。前端阻止同一聊天的重复提交。
使用 CAS 而不是在整个 LLM 调用期间持有数据库锁；冲突仍可能消耗已经执行的模型调用。

80% Context Compaction 规则保持不变。运行时压缩结果随最终轮次一起提交；
get_session 直接按 history_summary_through_id 查询未压缩消息，避免每轮加载全部原始记录。
人工直接调用的旧维护适配器仍可单独保存压缩结果。恢复界面使用独立的 owner 校验历史接口，
每页最多 100 条，用 before_id 继续加载；原始记录不会因为模型压缩被删除。

## 执行池与真实截止时间

删除 Policy RAG 和工具层每次调用启动一个线程的实现，使用 execution.py 的固定 daemon worker：
policy 4 worker / 12 个在途或排队槽，tool 8 / 24，SSE chat 4 / 8。分池避免嵌套等待死锁。
满载立即拒绝，不扩张线程或队列。排队任务可取消；运行任务超时后继续占槽直至真正退出。
不能声称 Python 线程已经被强制取消，CPU 模型推理仍可能在截止后结束。

ContextVar 传播最早截止时间；LLM SDK timeout 按剩余预算缩短并关闭隐藏重试，
SQLite 用 busy_timeout 和 progress_handler 中断超时查询，Qdrant query_points 使用剩余秒级超时。
这些限制与原有 Policy hop、Token 预算和安全拒答并行生效，不修改检索过滤或充分性判断。

## 图与前端模块

`agent.py` 缩为兼容入口及共享依赖，节点实现迁移到 graphs/routing.py、pre_sales.py、
after_sales.py、response.py、persistence.py；builder.py 只负责原有拓扑。
节点通过 runtime 兼容入口取依赖，保留原有 monkeypatch/依赖注入方式，不在模块间复制业务实现。
后续可在这些依赖边界上进一步引入显式服务容器，本次不改变既有节点语义。

前端 client.js 封装 Cookie/CSRF、错误、SSE、LatestRequest；session.js 管理按认证用户分区的 Session ID；
app.js 负责现有 DOM 渲染与界面绑定。无第三方打包或运行时前端依赖。
新搜索会 abort 旧请求并通过序号拒绝旧结果；聊天防重复提交，支持取消等待和切换新会话。
收藏、购物车信号、比较和历史操作统一捕获异常并恢复按钮状态。

`POST /api/chat/stream` 发出真实图节点完成进度；只在 persist 之后发出经过 Grounding、Faithfulness
与响应模型校验的 final 事件。不会把未经审核的模型 Token 先流给用户。
断开连接后取消排队任务，并在节点边界停止继续执行；正在运行的节点可能完成，
尤其提交已进入事务时，客户端取消并不等于回滚。界面明确提示可通过“恢复会话”确认结果。
该接口是安全的进度流，不是未验证文本的 Token 流。生产代理需关闭 SSE 缓冲。

## 验证命令与边界

```powershell
.venv/Scripts/python.exe -m pytest tests/test_contracts.py --basetemp=.test-tmp/contracts-reproduce -o addopts='' -q
npm test
npm run check
.venv/Scripts/python.exe -m pytest --basetemp=.test-tmp/full-reproduce -o addopts='' -q
.venv/Scripts/python.exe -m mypy --follow-imports=silent --ignore-missing-imports RetailPilot/auth.py RetailPilot/api_models.py RetailPilot/execution.py --cache-dir=.test-tmp/mypy-contracts
```

自动化覆盖 Cookie、CSRF、身份伪造、归属、事务回滚、并发 CAS、压缩加载、自动/手动参数、
公开 Trace、响应模型、SSE 隐私/错误和有界执行；前端测试覆盖 CSRF、取消、旧响应屏蔽、
分段 UTF-8 SSE 解码、错误恢复和按用户隔离 Session ID。
本地浏览器已验证模块加载、Demo 聊天进度、公开轨迹、刷新后消息恢复及手动 FTS 查询参数。
未配置真实模型凭据，浏览器聊天为离线路径安全回复；合成商品库仍存在之前记录的中文命中缺失。
没有把测试或空召回解释为真实模型、真实商品数据的效果验收。

### 2026-10-30 本地验收记录

| 检查 | 结果 |
| --- | --- |
| 新增后端契约、身份、事务、并发、SSE 和执行池测试 | 16 passed |
| 前端 Node 测试 | 6 passed |
| 当前代码全量 Pytest | 243 passed、6 failed |
| 原始提交 `57c232a` 全量 Pytest（同一商品数据库、同一虚拟环境） | 227 passed、6 failed |
| JS 三个模块语法检查 | 通过 |
| `auth.py`、`api_models.py`、`execution.py` 的 Mypy | 通过 |
| 上述三个新模块、新后端测试和账号脚本的 Ruff | 通过 |
| Python compileall、pip check、git diff --check | 通过 |

两次全量测试失败项完全相同，未删除、跳过或放宽原有断言：

- `tests/test_api.py::test_api_smoke`
- `tests/test_evaluation_pipeline.py::test_low_candidate_policy_relaxes_before_zero_results`
- `tests/test_evaluation_pipeline.py::test_search_logging_persists_routes_impressions_and_sample`
- `tests/test_retrieval.py::test_hybrid_search_respects_budget_and_negative_brand`
- `tests/test_retrieval.py::test_search_uses_controlled_attribute_relaxation_when_strict_pool_is_empty`
- `tests/test_retrieval.py::test_approximate_budget_keeps_nearby_exact_catalog_match`

这些失败涉及现有商品数据的检索命中、候选扩展和曝光数量；本次改动没有将其修复，
也不宣称全量质量门槛已经通过。完整仓库的旧代码风格告警不在本次定向 Ruff 通过范围内。
本地 JUnit 记录保存在忽略提交的 `artifacts/contracts/publish-final.xml` 与
`artifacts/contracts/publish-baseline.xml`，浏览器截图同样保存在本地 artifacts 目录。

### 文件职责

| 文件 | 职责 |
| --- | --- |
| `RetailPilot/auth.py`、`api_models.py`、`execution.py` | 身份/Cookie/CSRF、公开响应 Schema、有界执行与截止时间 |
| `RetailPilot/agent.py`、`graphs/__init__.py`、`graphs/builder.py` | 保持原有入口与拓扑的兼容层、延迟导出和组图 |
| `RetailPilot/graphs/routing.py`、`pre_sales.py`、`after_sales.py`、`response.py`、`persistence.py` | 路由与压缩、售前、售后、回复/Grounding、原子持久化节点 |
| `RetailPilot/api.py`、`schemas.py` | 认证身份绑定、响应投影、历史/SSE 接口、自动与手动搜索参数 |
| `RetailPilot/catalog.py`、`database.py` | 归属校验、版本 CAS、事务、分页、数据库迁移与查询截止时间 |
| `RetailPilot/config.py`、`llm_config.py`、`policy_followup.py`、`policy_index.py`、`tools.py` | 认证配置及已有 LLM/政策/工具链的有界执行集成 |
| `web/assets/app.js`、`client.js`、`session.js` | 界面控制、请求/进度/错误、用户分区 Session ID |
| `web/index.html`、`web/assets/styles.css` | 登录、取消、历史、检索模式及进度控件 |
| `tests/test_contracts.py`、`tests/frontend/client.test.js`、`package.json` | 可复现的后端/前端测试与前端检查命令 |
| `scripts/create_web_user.py` | 交互生成服务端密码哈希配置 |
| `.env.example`、`QUICKSTART.md`、本文档 | 配置说明、启动入口、契约与验收记录 |
