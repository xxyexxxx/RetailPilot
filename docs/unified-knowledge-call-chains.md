# 统一知识检索：端到端代码调用链与面试说明

建议结合 [实施报告](unified-knowledge.md) 和源码阅读。这里描述当前真实调用，不把待验证的真实模型能力当作完成结果。

## 1. 先分清四种身份

假设 SKU 为 PACK-01，宣传图中写“500mL 水杯侧袋”。

- products 的 PACK-01 保存当前价格、库存、品牌等商品业务事实。
- assets 的稳定 asset_id 标识这张原始宣传图，图片文件在内容寻址 Store。
- asset_elements 的 element_id 标识原图或长图区域，并保留 OCR、模型观察、bbox 等。
- knowledge_units 的 unit_id 标识可召回证据，search_text 用于检索，evidence_text 用于回答引用。

一个 Asset 可以贡献多个 Unit；一个 Unit 可以引用多个 Element。来源通过 knowledge_sources 关联 SKU 或政策 doc_id。商品 Profile 也是 Unit，它引用目录字段，并不需要图片 Asset。

数据流可以概括为：

```text
Source -> Parser / Source Adapter -> Element / Profile
       -> KnowledgeUnit -> FTS / Text Vector / optional Visual Vector
       -> RetrievalHit(unit_id) -> Unit RRF -> Domain Business Logic
       -> hydrate(unit_id) -> EvidencePackage -> Answer / Grounding
```

## 2. 离线：商品目录如何进入索引

1. 目录导入从 catalog_adapters.upsert_product() 进入。数据库使用 UPSERT 保留商品身份及关联资产。
2. products 的 INSERT/文本 UPDATE 触发 knowledge_dirty。
3. knowledge.sync_products() 读取权威商品字段，将标题、品牌、类目、SKU、描述、Features、Attributes、标签转成稳定文本窗口。
4. 每个窗口调用 put_unit()，得到 profile:SKU:ordinal，source_id 为 catalog:SKU；保存字段路径引用。
5. put_unit() 原子写 knowledge_units、元素关系及 knowledge_fts。FTS 把 SKU、标题、品牌、类目放入不同列，以保留权重。
6. 若明确配置 Hash，离线向量可以同步生成。神经模型通过 knowledge_index.build(domain='product') 批量构建。
7. 向量和 Unit 共用 unit_id，记录模型名/revision 与内容 Hash。价格或库存查询回查当前数据库，无需复制到 Caption 或重新做 Embedding。

## 3. 离线：图片、PDF 和表格如何进入索引

`scripts/assets.py import` 与受保护的上传 API 都调用 stage_asset()，后续由处理任务调用 process_asset()。

stage_asset() 验证所有者、类型、体积和内容；原始 bytes 写 Asset Store。source_key 决定稳定资产身份，SHA-256 决定二进制去重。只在显式提供素材时导入远程图片，远程入口保留 HTTPS 白名单、私网地址和重定向检查。

process_asset() 调用 asset_parsing.parse_asset()。独立图片读取原图，长图切区域；PDF 由有时间限制的子进程使用 Docling，轻量 pypdf 仅提供文本降级。Docling 原生结构保留为 DocumentElement，而非只导出 Markdown。

图片元素的 OCR 原文、模型可见观察、宣传内容、未知信息分开保存。enrich_context() 在同章节和有限 Token Window 内加入附近正文，并记录上下文 Element ID。它不覆盖原文。

公共 build_chunks() 的输出：

| 模态 | 检索内容 | 回答证据 | 原始回溯 |
|---|---|---|---|
| 正文 | 有界文本窗口 | 原始正文 | Element、章节、页码/bbox |
| 图片 | OCR + 描述 + 附近上下文 | OCR 原文；按需原图观察 | 原图/区域预览及 Element |
| 表格 | 带列名的行文本 + 附近上下文 | 真实结构化单元格 | 表格截图、页码/bbox |

表格每 8 行一段，保留表头、单元格行列和跨度。structured_data 存在 Unit 中，原始完整表结构留在 Element；图片 bytes 在 Store。

发布只写一套 knowledge_units。商品素材的 source.domain 为 product；政策素材为 policy。Text Dense 对同域所有文字使用同一模型。Visual Dense 只为有真实图片的 Unit 建立。

## 4. 售前：用户查询的完整执行过程

### 入口和 Query Understanding

直接搜索由 api.search() -> search_engine.search()；聊天由 api.chat_endpoint()/chat_stream() -> agent.chat() -> LangGraph -> graphs/pre_sales.pre_sales_branch()。

聊天的意图分析先提供 intent、category、target_terms 和约束，售前节点构建 ParsedQuery。商品搜索调用 tools.registry.execute('catalog_search') -> tools.catalog_search() -> SearchEngine.search()。直接搜索还可使用原 parse_query()。

SearchEngine 保留错字纠正、意图/目标词、结构化约束和历史上下文，不重新设计 LangGraph。

### 一次领域语料的三种方法召回

SearchEngine._sql_filters() 形成参数化价格、品牌、类目、排除条件；追加库存条件。query.raw_query 与已确认目标词组成检索文本。

调用：

```python
retrieve(
    RetrievalQuery(text, "product"),
    RetrievalFilters(product_where=where, product_params=params),
)
```

内部先同步 dirty 商品 Profile，之后：

- BM25Retriever：一次查询 knowledge_fts，搜索 Profile、图片文字、说明书和参数表的所有有效 Unit。
- TextDenseRetriever：一次对应领域的 Query Encoding，在同一文本空间寻找 Unit。
- VisualRetriever：使用对应视觉模型的 Text Encoder，在图像向量空间找 Unit；查询图片则使用 Image Encoder。

每条路线返回 RetrievalHit(unit_id, route, score, rank, metadata)。不存在 SKU 与 Asset ID 在 RRF 内混合的情况。某个可选模型或远端失败只影响对应路线，错误进入 Trace。

### Unit Fusion 和证据相关性

fuse() 按 unit_id 计算 RRF：sum(1/(60+rank))。票来自 BM25、Text Dense、Visual Dense，而不是“目录来源、图片来源”。同一方法重复返回同一 Unit 不重复加票。

三种方法命中同一 Unit 时合为一条证据。可选 CrossEncoder 对 Unit 的 query/search_text 相关性重排；其原始分数不会成为商品业务推荐分。

### SKU Aggregation 和 Query Guard

aggregate_skus() 根据来源 SKU 分组；每个商品保留 Top-N 去重证据。最佳证据分 + 有上限的独立证据奖励，额外奖励最多最佳证据的 10%。图片数量本身不能无限加分。

查询 Guard 处理三类 Anchor：

- structured：已验证 SKU、品牌、类目等业务锚点。
- text：检索证据与查询存在有效文字支持。
- visual：视觉相似度通过门槛后，必要原图核验产生与本次查询有关的观察。

对于纯视觉召回，缺少 Caption/OCR 不会因为没有词面重叠直接丢弃；必须进行原图验证。VLM 无法完成核验时，该纯视觉候选不能借一个普通类目绕过 Guard。所有 Anchor 都无法覆盖价格、库存、排除条件等硬约束。

### 商品业务排序与生成

聚合后的商品进入已有 _score_candidates()、_apply_candidate_policy()、_diversify()。流行度、个人偏好、当前价格和属性匹配在这里参与排序。明确浏览推荐有独立业务候选生成，避免空语义目标强行走文字检索。

每件结果保留 knowledge_evidence；旧 asset_evidence 为其中带 Asset 的兼容视图，不是第二份索引。普通推荐理由仍基于原字段级 Citation 校验。

聊天的 pre_sales_branch() 调用 product_asset_answer()。搜索已经携带 Unit 证据时直接复用，不再次运行另一套资产搜索。详情/比较由真实 SKU 读取商品，再在相同统一语料中按这些 SKU 检索证据。

product_asset_answer() 引用 OCR 原句或表格单元格；关系/流程问题按需 verify_image()。结果标明商家宣传文字、原文或图像观察。与权威商品参数冲突时提示冲突。

最后 _ground_output() / validate_claim_citations() 校验事实，graphs/response.build_response() 合并分支，verify_grounding() 做最终验证，API 返回商品、引用和 Agent Trace。

## 5. 售后：从 Markdown/PDF 到答案

### 统一 Ingestion

Markdown 通过 policy_ingestion.ingest_policy_documents() 保存身份和适用性，再 write_markdown() -> ParsedDocument/DocumentElement -> 公共 Builder -> KnowledgeUnit。

PDF 注册同样的政策身份，再经 process_asset() -> DocumentElement -> KnowledgeUnit。PDF 不再复制成一份 policy_chunks。Markdown/PDF 差异只体现在来源、元素和位置关系。

### 查询和适用性

聊天路由到 graphs/after_sales.after_sales_branch()，plan_after_sales() 生成只读工具计划。节点执行 policy_retrieve 工具 -> policy.retrieve_policies()。

retrieve_policies() 确认 SKU/类目上下文，生成 RetrievalFilters：Topic、平台、司法辖区、用户受众、类目、生效日期、版本、状态和 Authority。

调用同一个 retrieve(domain='policy', methods=BM25/TextDense)。过滤在 SQL 召回阶段执行，融合前再次拦截过期或不适用候选。source.version 与 policy_documents.version 不一致的记录不进入结果；政策 PDF 的 asset.source_version 也必须一致。

### Rerank、Hydration 与 Assessment

公共 RRF 得到 Unit，再经现有 PolicyCrossEncoder 排序。hydrate(unit_id) 获取原始正文、结构化单元格或图片位置。输出 documents 的 chunk_id 只是 unit_id 的旧契约别名。

after_sales 节点调用原 assess_policy_evidence() 和 validate_assessment()：

- supporting_chunk_ids 必须属于实际召回集合。
- 需要图片的证据必须真实加载核验。
- 图片 Caption 不能单独支持政策权益、期限和金额。
- 冲突、失效、缺失信息不能被高检索分数覆盖。

### Adaptive Retrieval 与最终 Grounding

证据不足且可补检时，policy_followup.run_policy_followup() 在既有 hop、时间、证据和 Token 预算内补充查询；调用的仍是同一个 retrieve_policies() 和规范 Unit 索引。

需要用户订单、商品或适用条件时，保留 clarify 语义；知识库证据不足且无法补齐时拒绝无根据结论。不会把视觉 API 超时标为 sufficient。

答案先由真实 evidence snapshot 构建 Claim/Citation，再经过字段/版本/位置/原文验证。原 Answer Faithfulness 继续检查答案是否超出证据，失败时按现有修复或降级流程处理。

## 6. hydrate(unit_id) 怎么定位原图

以一个 PDF 表格 Unit 为例：

```text
unit_id
  -> knowledge_units
  -> knowledge_sources: doc_id / asset_id / source_version
  -> knowledge_unit_elements: primary + context elements
  -> asset_elements: structured table / page_no / bbox / preview_key
  -> Asset Store: original PDF / table crop / page image
```

Hydration 首先验证政策状态和版本，再验证 Asset 状态、文件存在性、Element 归属。返回 EvidencePackage，不把本地路径暴露给客户端。受保护的 /api/assets/... 接口负责实际预览与权限。

表格完整时使用结构化单元格。图像空间关系不在正文中时，verify_image() 根据 preview_key 读取原图 bytes，通过 VisionProvider.invoke() 构造真正的视觉消息。

视觉请求把知识数据标为不可信内容，独立系统指令要求忽略图片/OCR 中的指令。Provider 返回值仍需经过本次检索集合、来源版本和 Claim 校验。图片“存在”与“支持结论”是两个独立判断。

## 7. 更新与删除如何防止旧证据污染

商品描述更新 -> dirty SKU -> 重建变化 Profile -> 清理受影响本地索引。图片文件更新 -> 稳定 Asset ID + 新 revision -> 旧 Unit/Element 撤销 -> 解析发布新 Unit -> 按模型版本构建向量。政策版本更新后，旧 Source 和旧 PDF 在 SQL 过滤阶段失效，直到新内容导入。

Qdrant 是派生镜像。Unit/向量变更增加 Generation，旧镜像立即不可查询。sync() 持有租约，上传当前数据并删除无效 Point，最后 CAS 确认 Generation 未变化；失败可以重跑。同步完成前使用本地当前版本向量。

旧 Citation 的 Chunk ID 在迁移时保留。真正内容更新后，旧版本 Citation 应被拒绝，而不是为了兼容继续允许引用旧原文。

## 8. 面试中怎样解释“统一后为什么少了重复计算”

可以这样回答：

> 我们统一的是检索单元和派生索引的生命周期。不同格式在离线解析时输出同一 KnowledgeUnit，同业务域的商品字段和图片文本共用一次 BM25 和一个文本向量空间。线上 RRF 按方法融合 Unit，随后售前聚合 SKU、售后执行适用性和证据评估。原图、表格和文档位置由统一 Unit 来源关系加载。

减少的工作包括：

- 售前目录 BM25 与资产 BM25 合为一次同域查询。
- 目录 Dense 与资产 Text Dense 合为同域一次 Query Encoding。
- PDF 不再复制一套 Policy Chunk。
- 变更只维护 Unit 的 FTS/向量，不再同步多套功能重叠 Chunk。
- 相同检索结果复用证据，生成时不重复跑资产检索。
- 来源关系共用，不再要求调用方判断旧 Chunk 表。

需要同时说明代价：单位数、过滤和证据 Hydration 都有成本；本次单机测量的新路径延迟高于旧路径，不能据架构简化就声称更快。真实语义质量和视觉阈值还需要预训练模型与标注集验证。合成测试证明的是链路和边界行为。

## 9. 源码阅读顺序

1. knowledge_schema.py -> knowledge.py：约束、Unit、来源、迁移。
2. catalog_adapters.py / policy_ingestion.py / asset_ingestion.py：真实写入入口。
3. asset_parsing.py / asset_store.py：解析、原始证据、安全与文件保存。
4. knowledge_index.py / knowledge_qdrant.py：统一向量生命周期。
5. knowledge_retrieval.py：统一 Hit、方法召回、RRF、SKU 聚合。
6. retrieval.py：售前业务过滤、Guard、排序和行为日志。
7. policy.py -> policy_evidence.py -> policy_followup.py：适用性、评估和补检。
8. knowledge_evidence.py -> asset_evidence.py -> grounding.py：加载、核验和引用。
9. graphs/pre_sales.py / graphs/after_sales.py / graphs/response.py：Agent 如何消费检索。
10. tests/test_knowledge.py / test_knowledge_fusion.py / test_multimodal.py：可运行边界证据。
