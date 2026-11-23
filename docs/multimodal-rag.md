# RetailPilot 多模态 RAG

多模态能力默认关闭。开启后，商品索引和现有售后政策索引分别检索，在引用环节使用统一的资产与元素来源。上传接口面向策展账号，入库完成的素材发布到共享知识库。

## 启用和导入

安装基础依赖与轻量资产依赖。Windows PowerShell 和 Unix shell 都可在项目根目录运行以下 Python 命令。

```text
python -m pip install -r requirements.txt -r requirements-multimodal.txt
python scripts/assets.py init
```

已有数据库可以直接初始化。空库可运行 `python scripts/bootstrap_ci.py` 生成合成目录，或使用仓库现有目录导入流程。

在 `.env` 中设置 `RETAILPILOT_MM_ENABLED=1`。轻量 PDF 文本解析可使用 `RETAILPILOT_MM_PARSER=pypdf`。生产复杂 PDF 或视觉向量需要安装 `requirements-multimodal-models.txt`。独立图片 OCR 可设置 `RETAILPILOT_MM_OCR=tesseract`，并安装 Tesseract 和 `chi_sim+eng` 语言数据。若使用视觉模型，配置 `RETAILPILOT_MM_VLM_URL` 为兼容 Chat Completions 的基址（通常包含 `/v1`），指定图片输入模型及所需密钥。

```text
python scripts/assets.py import --sku SKU123 --type product_main_image --path samples/hero.png
python scripts/assets.py import --sku SKU123 --type product_poster --path samples/poster.png
python scripts/assets.py import --sku SKU123 --type product_document --path samples/manual.pdf
python scripts/assets.py register-policy ./policy-metadata.json --markdown ./policy-original.md
python scripts/assets.py import --doc-id manual-001 --type document_page --path samples/install.pdf --source-version 1.2
python scripts/assets.py batch ./asset-manifest.json
python scripts/assets.py build
python scripts/assets.py check
python scripts/assets.py search "绿色背包 水杯侧袋" --sku SKU123
python scripts/assets.py search "" --image ./samples/hero.png
python scripts/multimodal_bench.py --output ./benchmark.json
python -m uvicorn RetailPilot.api:app --host 127.0.0.1 --port 8000
```

`register-policy` 接收 `PolicyDocumentInput` JSON。对于独立 PDF，可省略 `--markdown`，它会先登记元数据，然后从 PDF 元素生成政策块。已有 Markdown 继续使用原导入流程。示例元数据如下。

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

批量清单是 JSON 数组，每项使用 `owner_type`、`owner_id`、`kind`、`path` 或 `url`，以及可选的 `source_key` 和 `source_version`。同一归属、类型及 `source_key` 的内容更新沿用 Asset ID。远程导入需要显式指定 `--url` 或 `--catalog-main SKU123`，并且 URL 的 HTTPS 主机在 `RETAILPILOT_MM_REMOTE_HOSTS` 白名单内。

`assets.py retry ASSET_ID` 重跑单项。默认 `build` 处理待处理、失败、部分成功和超时中断记录。模型或解析配置切换后运行 `build --all`。删除命令是 `assets.py delete ASSET_ID`，它级联删除元素、映射、FTS 和本地向量。内容寻址对象可以被多个资产或解析缓存引用，删除记录后文件由存储保留。

配置独立视觉 Qdrant 后运行 `assets.py sync-visual`，也可添加 `build --sync-visual`。迁移视觉模型到不同向量维度时使用新集合。政策 PDF 先进入现有 Policy FTS5，构建其语义索引继续运行 `python scripts/build_policy_index.py --prune`。此命令在成功写入后清理旧的 RetailPilot 政策点。预下载的 Docling 权重可通过 `RETAILPILOT_MM_DOCLING_ARTIFACTS` 指定。

## 数据与调用链

| 模块 | 职责 |
|---|---|
| `asset_schema.py`、`asset_store.py` | 数据库增量建表，外键关联 SKU 或政策文档，SHA-256 对象去重和安全远程导入 |
| `asset_ingestion.py`、`asset_parsing.py` | 原始对象、页图和裁剪图，解析缓存，章节内上下文，文档元素与稀疏索引原子发布 |
| `asset_embeddings.py`、`asset_retrieval.py`、`asset_qdrant.py` | 三路商品召回，分模型版本向量，SKU 去重，独立视觉镜像及 SQLite 降级 |
| `asset_evidence.py`、`grounding.py` | 回查来源、提取原句或单元格、视觉核验、约束冲突与引用校验 |
| `asset_api.py`、`web/assets/assets.js` | 授权上传、索引状态、资源预览和可展开引用 |

`assets` 的 SKU 与文档外键互斥。 `asset_elements` 保存原始内容、页码、坐标系与坐标、章节、阅读顺序、原来源引用以及解析版本。 `asset_chunks` 保存可检索文本和结构化表格分段。 `asset_chunk_elements` 保存主元素与进入上下文窗口的邻接元素。文本、图片和政策的现有数据保持在各自的表与集合。SQLite 写入在一个发布事务中替换对应元素、FTS 和向量。

1. **售前。** `SearchEngine.search` 在原有查询锚点检查前运行商品资产 BM25、文本 Dense、可选 SigLIP2。每路每个 SKU 最多贡献一个 RRF 名次，同内容资产去重。业务硬过滤、个性化和目录排序继续执行。售前 LangGraph 分支在搜索、详情和比较结果内显式加载相关资料。
2. **售后。** PDF 元素投影为现有 `policy_chunks` 的 leaf 块。 `retrieve_policies` 按平台、主题、有效日期、类目、受众及资产绑定版本筛选，再进入 Policy FTS5、Qdrant、RRF 和 reranker。 `run_policy_followup` 取得原始文本、表格或经核验的图示观察，之后沿用证据评估、补检、原有答案生成与 Faithfulness 检查。
3. **核验。** 结构化表格保留行列、表头与单元格跨度，每段带表头。目录参数和图片内容发生可检测的数值冲突时提供冲突提示。OCR 是提取到的图片文字，商家宣传图有独立标签。VLM 用于为检索生成描述以及对少量必要图像复核。它收到经过尺寸限制的 PNG 字节，通过 Chat Completions 多模态消息的 `image_url` 内联发送。数据库及 Qdrant 保留引用与向量。核验超时或失败时生成原文引用或明确证据不足。
4. **引用。** 引用携带真实 SKU 或 Document ID、Asset/Chunk/Element ID、页码、BBox、源版本和授权访问地址。Grounding 校验活动记录、归属、实际证据快照和摘录。售后 `supporting_asset_ids` 从通过评估的 Chunk 推导。图示观察可支持故障流程与安装图关系，涉及费用、期限和资格的条款采用政策原文或条款表格。

Docling 提供结构化 PDF 元素、阅读顺序、表格及页图。轻量 `pypdf` 路径读取 PDF 文本层并报告 `partial`。可配置 SigLIP2 以同一训练空间的图像和文本塔处理跨模态召回，特征做 L2 归一化后比较余弦分数。早期开发可用资产文本 Hash 向量保持可重复性，实库可设为 `BAAI/bge-small-zh-v1.5` 等 Sentence Transformers 模型。［[Docling 图表导出](https://docling-project.github.io/docling/_generated/examples/export_figures/)］［[SigLIP2 模型接口](https://huggingface.co/docs/transformers/model_doc/siglip2)］

## API 与权限

`POST /api/assets` 接收原始 PDF、PNG、JPEG 或 WebP 请求体和包含归属、类型、来源标识的查询参数，返回 202 和状态地址。 `GET /api/assets?sku=...` 或 `?doc_id=...` 列出已发布资产。状态、原始文件、元素预览及单块证据均从 `/api/assets/{asset_id}` 下访问。删除和重试要求策展权限。

`RETAILPILOT_MM_CURATORS` 接收逗号分隔的账号。用 `python scripts/create_web_user.py --user curator` 生成 `RETAILPILOT_AUTH_USERS_JSON` 的值，并将 `curator` 加入策展名单。本地 HTTP 网页登录使用 `RETAILPILOT_AUTH_COOKIE_SECURE=false`，HTTPS 部署使用默认值。Cookie 登录请求沿用现有 CSRF 机制，服务调用沿用绑定身份的 API Key。资产原件和预览要求真实身份，错误响应和 API 证据使用逻辑地址。远程导入拒绝重定向和非公共 IP，连接固定到已校验的 DNS 地址，并限制 MIME、大小和时间。PDF 解析在有页数、页尺寸和执行时间上限的子进程中运行。上传界面在策展账号登录后可见。

完整配置见 `.env.example` 的 `RETAILPILOT_MM_` 段。

## 验证

```text
python -m pytest tests/test_multimodal.py --basetemp=.test-tmp/mm-feature
npm test
python scripts/multimodal_bench.py --output ./benchmark.json
```

[基准 JSON](multimodal-benchmark.json) 记录 11 条合成场景、三种配置、逐例结果及测得的耗时。SKU 和 Asset 的 Recall、MRR、nDCG 各自按有标签的查询计算。答案基于预期原文及 Grounding 判定。索引体积记录 SQLite 文件和索引载荷字节，支持 dbstat 的 Python/SQLite 还会记录独立索引页大小。成本数据对应这个离线配置中的真实外部调用数。OCR、Caption、视觉编码、图示核验及复杂布局输出使用确定性替身。PDF 文本子进程、Pillow、FTS5、向量索引、SKU RRF 和实际引用逻辑都由运行代码执行。

在同一份 4,203 SKU 的 CI 合成目录上，`origin/main` 为 243 通过、6 失败，本次为 253 通过、6 失败。两次失败的是同一组既有目录检索断言。前端为 8 通过。端到端测试使用真实 FastAPI、SQLite、Pillow、pypdf 及 Qdrant 内存实现，并用确定性替身隔离 OCR、VLM 和共享视觉编码模型。

本次安装并探测了 Docling 2.151.1、Torch 2.15.0 和 Transformers 5.24.0。真实 Docling 数据结构适配通过了元素、章节及边界框检查。Hugging Face 网络探测超时且本地缺少所需权重，未完成 Docling 全模型转换和预训练 SigLIP2 中文检索验证。真实 Tesseract 和远程 VLM 验证也未执行。

## 运行边界

- 当前任务队列在 API 进程内，受限并发下后台处理。重启遗留任务可通过 `build` 或 `retry` 恢复。对象存储目录应仅由服务账号和运维人员管理。
- 视觉查询没有足够的目录锚点时仍可能触发现有查询保护。查询时视觉核验使用预算内的原图或区域图。表格抽取的精度依赖源文档与解析器，参数冲突诊断目前覆盖已命名的常见单位字段。
- SQLite 本地向量检索在过滤后的向量集合精确扫描。Qdrant 模式建立在已同步的独立集合及 SQLite 过滤之上。配置变更或镜像不可用时，SQLite 可以继续召回。
- VLM 凭证和部署数据由运行方配置。包含跨页复杂表格、中文扫描件及大量商品图片的真实质量与成本应使用业务素材另行评估。
