# RetailPilot 多模态 RAG 运行手册

当前证据决策与生成式模型基础设施已进一步升级，见[统一证据报告](unified-evidence.md)和[源码调用链](unified-evidence-call-chains.md)。以下知识层说明及历史测量保留其阶段语境。

当前实现采用统一 KnowledgeUnit。商品字段、图片 OCR/描述和说明书共用商品文本索引；Markdown 和政策 PDF 共用政策文本索引。视觉向量保持独立空间。完整架构、迁移、测量和已知边界见 [统一知识检索报告](unified-knowledge.md)，代码执行过程见 [调用链](unified-knowledge-call-chains.md)。

## 启用

安装基础依赖与轻量资产依赖：

```text
python -m pip install -r requirements.txt -r requirements-multimodal.txt
python scripts/knowledge.py migrate --backup data/before-unified.db
```

已有库先备份迁移，再启动新版。空库可运行 scripts/bootstrap_ci.py 生成合成目录。

在 .env 设置 RETAILPILOT_MM_ENABLED=1。复杂 PDF 使用 Docling，安装 requirements-multimodal-models.txt，并按需设置 RETAILPILOT_MM_DOCLING_ARTIFACTS。轻量文本降级使用 RETAILPILOT_MM_PARSER=pypdf；该路径不能代替布局和扫描件理解。

独立图片 OCR 可设 RETAILPILOT_MM_OCR=tesseract，并安装 Tesseract 与 chi_sim+eng 语言数据。视觉 API 使用 RETAILPILOT_MM_VLM_URL、KEY、MODEL，需模型真正接受图片输入。视觉召回使用 RETAILPILOT_MM_VISUAL_ENABLED=1 以及配置的 SigLIP2 模型/revision。

文本索引配置为 RETAILPILOT_KNOWLEDGE_PRODUCT_MODEL 与 RETAILPILOT_KNOWLEDGE_POLICY_MODEL。Hash 仅用于显式离线模式，disabled 禁用 Dense。

## 导入与索引

尖括号参数替换成现有文件或数据库身份：

```text
python scripts/assets.py import --sku <SKU> --type product_main_image --path <hero.png>
python scripts/assets.py import --sku <SKU> --type product_poster --path <poster.png>
python scripts/assets.py import --sku <SKU> --type product_document --path <manual.pdf>
python scripts/assets.py register-policy <metadata.json> --markdown <policy.md>
python scripts/assets.py import --doc-id <DOC> --type document_page --path <install.pdf> --source-version <VERSION>
python scripts/assets.py batch <manifest.json>
python scripts/assets.py build
python scripts/knowledge.py build --domain all --space text
python scripts/knowledge.py build --domain product --space visual
python scripts/knowledge.py check
python scripts/knowledge.py sync
python scripts/assets.py search "绿色背包 水杯侧袋" --sku <SKU>
python scripts/assets.py search "" --image <query.png>
```

register-policy 接收 PolicyDocumentInput JSON。独立 PDF 可省略 --markdown，先登记元数据：

```json
{
  "doc_id": "manual-001",
  "policy_id": "manual",
  "topic": "保修",
  "title": "设备安装与故障排查",
  "source_name": "厂商说明书",
  "platform": "demo-mall",
  "category_scope": "*",
  "effective_from": "2026-01-01",
  "version": "1.2",
  "authority": "manufacturer"
}
```

批量清单为 JSON 数组，使用 owner_type、owner_id、kind、path 或 url，以及可选 source_key / source_version。同一归属、类型和 source_key 的更新沿用 Asset ID，原文件按内容 Hash 去重。

远程导入只处理显式 --url 或 --catalog-main 指定的素材，HTTPS 主机必须在 RETAILPILOT_MM_REMOTE_HOSTS 白名单内。不会主动爬取商品网站。

assets.py retry ASSET_ID 恢复失败任务；build --all 用于解析配置变更后的重处理；delete ASSET_ID 级联清理来源、元素、Unit 与派生本地索引。内容寻址文件可能被多个资产/缓存引用，记录删除后保留对象文件。

可选 Qdrant 使用 RETAILPILOT_KNOWLEDGE_QDRANT_URL/KEY。knowledge.py sync 发布所有当前领域/模型空间，assets.py sync-visual 是同一个统一同步操作的旧 CLI 别名。旧 POLICY_QDRANT_* / MM_QDRANT_* 不再用于查询。

## 数据与证据

assets / asset_elements 保存原始资源、元素及位置。knowledge_sources / knowledge_units / knowledge_unit_elements 保存统一来源与检索单元。knowledge_fts 和 knowledge_vectors 为派生索引；PDF 没有另一份 policy_chunks 投影。

图片 OCR 原文、模型观察、商家宣传、权威目录字段保持来源区别。表格保留单元格、表头和跨度，每 8 行形成一个带表头 Unit；截图另存 Asset Store。

检索输出 unit_id，BM25/Text Dense/Visual 在 Unit 层融合，再按业务域做 SKU 聚合或政策适用性判断。hydrate(unit_id) 定位原文、表格、页图或图片。视觉核验发送真实 PNG bytes，数据库与向量 Payload 不存 Base64。

证据不足、模型超时、文件缺失和参数冲突会降级或拒绝结论。图片描述和检索相似度不能代替真实政策条款。

## API 与权限

POST /api/assets 接收原始 PDF、PNG、JPEG、WebP 请求体和归属/类型参数，返回 202 与状态地址。GET /api/assets?sku=... 或 ?doc_id=... 查询已发布素材。预览、内容、状态和单元证据保留原 API；chunk_id 参数当前就是 Unit ID。

上传、重试和删除要求策展权限；RETAILPILOT_MM_CURATORS 配置允许账号。浏览器 Cookie 使用既有 CSRF，服务 API Key 绑定实际身份。文件和预览要求授权，不暴露服务端存储路径。原始附件、OCR 和 Caption 作为不可信知识数据处理。

远程导入拒绝重定向和私网 IP，连接固定到已校验 DNS 地址，限制 MIME、大小与时间。PDF 子进程限制处理时间、页数和页面尺寸。上传界面在策展账号登录后显示。

任务队列仍在现有 API 进程中，受限并发；重启后可通过 build/retry 恢复，无额外消息队列。

## 验证

```text
python -m pytest tests/test_knowledge.py tests/test_knowledge_fusion.py tests/test_multimodal.py
npm test
python scripts/multimodal_bench.py --output benchmark.json
```

当前实测与历史失败见 [验证记录](unified-knowledge-validation.json)。[旧多模态 Benchmark](multimodal-benchmark.json) 属于重构前阶段；当前对比与消融在 [统一 Benchmark](unified-knowledge-benchmark.json)。

合成测试使用可运行的 OCR/VLM/视觉替身。真实 Docling 结构适配、pypdf 子进程、Pillow、FTS、Qdrant 内存实例和引用校验已有覆盖；本机缺少完整预训练权重和真实视觉 Endpoint 验证，因此没有宣称真实模型召回质量或扫描 PDF 精度。
