
# 源码阅读与技术面试：统一多模态证据驱动 RAG

完整实现与配置见 [升级报告](unified-evidence.md)。下述链路对应当前代码，真实模型语义效果尚未在本机验证。

## 1. 从三种对象开始

KnowledgeUnit 是持久化检索单元，EvidencePackage 是一次来源加载结果，EvidenceBundle 是当前任务要判断的全部证据和可信约束。EvidenceAssessment 是模型或确定性规则提出的决策，程序验证后才可执行。

目录字段、知识单元、授权订单是三种原始事实来源。VisualObservation 是针对本轮原图的观察，保留真实 Unit/Asset/版本/快照关系；它不会变成商品数据库或政策原文。

## 2. 售前搜索完整路径

1. api.search() 或 agent.chat() 的意图层生成 ParsedQuery。
2. pre_sales_branch() 的 catalog_search 工具调用 SearchEngine.search()。
3. 原 SQL 价格、品牌、类目、库存和排除项先限定范围；知识层继续 BM25/Text Dense/Visual、Unit RRF、SKU 聚合。
4. Query Guard 只做候选相关性准入。统一模式的视觉近邻标记为待核验候选，暂不单独调用 VLM。
5. evaluate_products() 为受控 SKU 建立 product_bundle()，包括当前 Catalog、已检索 Unit、必要条件、硬约束和允许补检的同 SKU 范围。
6. run_evidence_workflow(ProductEvidencePolicy) 校验来源。明确的目录事实走 Fast Path；复杂功能或视觉证据走 Gateway。
7. 缺少方面时，模型只能给 FollowupQuery。product_retrieve() 用原条件和明确 SKU 构造 RetrievalFilters，回到同一个统一知识索引。
8. 新证据合并后重新评估，重复/无进展/预算耗尽安全停止。只有 answer 的商品进入原评分、Candidate Policy 和多样性处理。
9. 审核后的 Unit 与视觉观察进入商品结果；_add_asset_answer() 复用它们，摘录真实原文或已验证观察，不重跑第二套资产搜索或视觉模型。
10. _ground_output() 和外层 verify_grounding 校验字段与 Claim-Citation，之后沿原拓扑持久化和返回。

详情和比较也在确定 SKU 后、生成前调用 evaluate_products()。比较的事实按 SKU 分开验证，最终价格比较等仍由可信业务代码计算。

证据阶段零模型调用不等于所有聊天请求零调用：启用 LLM 的意图识别、查询理解仍然有原来的成本。

## 3. 售后完整路径

1. after_sales_branch() 调用已有 plan_after_sales()，执行 policy_retrieve / order_query 只读计划。
2. 第一次政策工具返回真实 Metadata-filtered Unit，商品/订单上下文由可信代码冻结。
3. run_policy_followup() 作为兼容入口调用 policy_workflow()。统一模式在此构建 Policy Bundle。
4. PolicyEvidencePolicy 校验活动文档、版本、平台、日期、受众、类目、Authority 和真实来源。授权订单从 SQL 再确认 user_id。
5. 同一个 run_evidence_workflow() 执行选择、评估、补检、重评和停止。
6. Policy FollowupQuery 不能更改平台、日期、版本或商品范围；相关 Topic 仅来自最初 Topic 的白名单，不可传递扩权。
7. 问题涉及维修/安装且已解析真实 SKU 时，可以授权 product 技术方面，并通过同一个 Product Knowledge 检索说明书和图示。
8. 政策权益引用政策原文/结构化条款；技术步骤单独引用商品 Unit/Asset；订单事实不伪装成知识单元。
9. 生成后执行既有 Claim-Citation Grounding。technical_evidence 也进入真实工具上下文。
10. 外层 Answer Faithfulness 审查完整可见回答；修复使用结构化 Statement，修复后重新 Grounding，失败安全拒绝。

没有在 execute_tool 之后才补一个评估节点；评估发生在业务分支生成之前，外层 LangGraph 顺序保持。

## 4. 图片实际如何到模型

knowledge_ref(unit_id) 调用原 hydrate()，验证 Asset/Element 与版本、文件存在性、SKU/doc_id 归属。Controller 只加载预算内的原图/区域图，不把整个候选池逐张送入模型。

Model ImageInput 的 evidence 类型要求真实身份。Gateway 校验 bytes 并构造：

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "{unit_id, asset_id, version}"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]
}
```

正文、表格单元格、问题和硬约束作为数据输入；系统指令明确不执行图片/OCR/正文中的命令。模型输出必须将观察绑定到实际输入的图片、目标和快照。程序不能从 Caption 自动认证图像结论。

入库描述在 Unit 存在之前，使用 asset_bytes 类型；探针使用 probe 类型。二者不是可被答案引用的虚假 Unit。

## 5. 为什么统一售前和售后的证据决策

两者都需要回答相同的控制问题：证据是否真实、必要条件是否覆盖、是否缺用户信息、是否补检、何时停止。分别维护循环会让预算、去重、错误退出和最终快照规则逐渐不一致。

统一的是流程和协议。商品仍由 Product 策略处理，政策仍受适用性与 Authority 约束，订单仍受权限约束。共用执行器不意味着证据可以跨领域证明任何结论。

## 6. 为什么仍保留 Query Guard

Query Guard 解决“这个商品是否值得进入候选”。Evidence Assessment 解决“这个商品的具体功能是否已经有足够证据”。

例如一张绿色背包图可以满足视觉相关性，却不能单凭视觉近邻证明具备防水等级。硬预算和库存更不是模型的语义判断。保留分层能避免不必要模型调用，也避免模型推翻业务约束。

## 7. 如何避免无限补检

代码同时限制跳数、Deadline、预留 Token、图片、上下文和证据数量。还记录已查询文本、已见 ID、独立信息和覆盖签名。

重复查询、同文换 ID、没有新信息、覆盖停滞或振荡都停止。模型只能提议缺失方面和受控查询，真正的过滤由初始范围重建。上下文集合缩小后需要确认最终精确快照，不能借已丢弃的证据回答。

## 8. 一个 VLM 为什么能处理纯文字

支持图像输入的生成式服务通常也接受纯文本消息。Gateway 的文字请求完全不加载图片；只有需要视觉时才追加真实图片块。

统一的是生成式服务和预算/Telemetry，不是把 Embedding、SigLIP2、Reranker 或独立 Judge 合并。任务模型覆盖是显式配置，默认统一部署模型。

Provider 的能力必须实际验证；配置声称支持视觉不等于模型理解能力达标。探针只检查基础输入与输出契约，业务语义仍需评估。

## 9. 为什么模型统一后仍需 Grounding

模型可能引用不存在的 ID、跨 SKU 借证据、混淆政策版本、把广告当权益，或把“可申请审核”扩大成“保证赔付”。

严格 Schema 只能约束输出形状。validate_assessment() 约束决策能否执行；Claim-Citation Grounding 核对具体事实与来源；Policy Faithfulness 检查最终语言是否超出证据。一次综合 VLM 不能替代这些独立职责。

对于复杂原图，可以额外开启独立提示的视觉复核。它有单独的预留图片槽位，并按本请求快照缓存；默认同模型复核并不代表错误完全独立。

## 10. 如何权衡质量、延迟和成本

先用可信结构化数据解决确定性问题。对复杂候选只评估前几项和必要原图。图片 OCR/Caption 留在离线处理；在线综合评估尽量复用观察。

补检可能提高必要证据覆盖，也会增加调用和延迟。Shadow 计算两份决策更贵，因此后台限并发且不控制旧回答。合成对照中平均调用数没有下降，不应把“统一模型接口”解释为必然省钱。

真实取舍需要：固定模型版本、人工标注的充分/不充分例子、误拒答/误放行成本、Token/Image Usage，以及多次延迟测量。当前脚本化结果只证明工程控制链可执行。

## 11. 推荐源码阅读顺序

1. evidence_schema.py：Bundle、Aspect、观察和独立验证。
2. evidence_policies.py：来源真实性、跨领域约束、快路径和策略。
3. evidence_workflow.py：唯一的评估/补检/停止循环。
4. evidence_runtime.py：售前、政策、订单和跨域的真实适配器。
5. evidence_shadow.py：后台审计、ContextVar 模式隔离和限流。
6. multimodal_model.py：能力配置、字节输入、严格输出、单次 HTTP 传输。
7. llm_config.py / asset_parsing.py：旧入口如何进入 Gateway。
8. graphs/pre_sales.py / after_sales.py：生成之前如何接入证据工作流。
9. grounding.py / answer_faithfulness.py：原有事实校验和修复闭环。
10. tests/test_evidence_engine.py / scripts/evidence_bench.py：可运行验证与测量边界。
