import { api, withButton } from "./client.js";

const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (key) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[key]);
const localAsset = (value) => /^\/api\/assets\/[a-f0-9]{32}\/(?:content(?:\?revision=[a-f0-9]{32})?|elements\/[a-f0-9]{32}\/preview)$/.test(value || "");
const labels = {
  image: "图片召回",
  visual_observation: "核验图示",
  ocr_observed: "OCR 文字",
  structured_table: "结构化表格",
  document_text: "文档原文",
  seller_advertised: "商家宣传",
  product_main_image: "商品主图",
  product_detail_image: "商品详情图",
  product_poster: "商品宣传图",
  product_comparison_image: "商品对比图",
  product_document: "商品说明书",
};

function preview(value, label) {
  return localAsset(value) ? `<img class="asset-preview" loading="lazy" src="${escape(value)}" alt="${escape(label)}">` : "";
}

export function assetEvidenceCard(item) {
  if (!item) return "";
  const label = labels[item.evidence_type] || "资产证据";
  const location = item.source_location || {};
  const image = preview(location.preview_url || location.page_preview_url, label);
  const description = item.matched_description || "";
  const routes = Object.keys(item.routes || {}).map((route) => ({
    knowledge_bm25: "关键词", knowledge_dense: "文本向量", knowledge_visual: "视觉向量",
  })[route] || "").filter(Boolean);
  return `<details class="asset-evidence"><summary>${escape(label)}${routes.length ? " · " + escape(routes.join("、")) : ""}</summary>
    ${image}${description ? `<p>${escape(description)}</p>` : ""}
    <small>检索命中 · ${item.verification_status === "source_text" ? "源文摘录" : "未核验"}</small></details>`;
}

export function assetCitations(items = []) {
  return items.filter((item) => item.asset_id).map((item) => {
    const location = [item.sku || item.doc_id, item.page_no && `第 ${item.page_no} 页`,
      item.section_path, item.source_version && `版本 ${item.source_version}`].filter(Boolean).join(" · ");
    const link = localAsset(item.source_uri)
      ? `<a href="${escape(item.source_uri)}" target="_blank" rel="noopener noreferrer">打开原始资产</a>` : "";
    const page = localAsset(item.page_preview_url)
      ? `<a href="${escape(item.page_preview_url)}" target="_blank" rel="noopener noreferrer">查看原页</a>` : "";
    return `<details class="asset-evidence"><summary>${escape(labels[item.evidence_type] || "资产引用")} · ${escape(location)}</summary>
      ${preview(item.preview_url || item.page_preview_url, location)}
      <blockquote>${escape(item.quoted_text)}</blockquote>
      <small>${escape(item.asset_id)} / ${escape(item.element_id)}</small>
      <div class="asset-links">${link}${page}</div></details>`;
  }).join("");
}

export function assetGallery(items = []) {
  if (!items?.length) return "";
  return `<section class="asset-gallery" aria-label="商品资料">${items.map((asset) => {
    const label = labels[asset.asset_type] || "商品资料";
    const cover = asset.elements?.find((el) => el.element_type === "page" && el.preview_url)
      || asset.elements?.find((el) => el.preview_url);
    const link = localAsset(asset.source_uri) ? `<a href="${escape(asset.source_uri)}"
      target="_blank" rel="noopener noreferrer">${escape(label)}</a>` : escape(label);
    return `<figure>${preview(cover?.preview_url, label)}<figcaption>${link}</figcaption></figure>`;
  }).join("")}</section>`;
}

export async function configureAssetsPanel(userId, isCurrent) {
  const button = document.querySelector("#asset-open");
  button.hidden = true;
  if (!userId) return;
  const config = await api("/api/assets/capabilities").catch(() => null);
  if (config && isCurrent()) button.hidden = !config.can_publish;
}

const jobStates = {
  queued: "等待处理", processing: "正在解析和索引", ready: "已发布",
  partial: "已发布，可用能力受限", failed: "解析或索引失败",
};

export function bindAssetsPanel() {
  const dialog = document.querySelector("#asset-dialog");
  const form = document.querySelector("#asset-upload");
  const status = document.querySelector("#asset-status");
  const type = form.elements.namedItem("owner_type");
  const kind = form.elements.namedItem("asset_type");
  type.addEventListener("change", () => {
    for (const option of kind.options) option.hidden = option.value.startsWith('document_') !== (type.value === 'policy');
    kind.selectedIndex = [...kind.options].findIndex((option) => !option.hidden);
  });
  document.querySelector("#asset-open").addEventListener("click", () => dialog.showModal());
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.submitter;
    await withButton(button, async () => {
      const file = form.elements.namedItem("file").files[0];
      const mime = file.type || ({
        pdf: "application/pdf", png: "image/png", webp: "image/webp", jpeg: "image/jpeg", jpg: "image/jpeg",
      })[file.name.split(".").pop().toLowerCase()];
      if (!mime) throw new Error("请选择 PDF、PNG、JPEG 或 WebP 文件。");
      const params = new URLSearchParams({
        owner_type: type.value, asset_type: kind.value,
        owner_id: form.elements.namedItem("owner_id").value.trim(),
        source_key: form.elements.namedItem("source_key").value.trim() || file.name.replace(/[^\p{L}\p{N}_.-]/gu, "_"),
      });
      status.textContent = "正在上传";
      const job = await api(`/api/assets?${params}`, {
        method: "POST", body: file, headers: { "Content-Type": mime },
      });
      for (let attempt = 0; attempt < 90; attempt++) {
        const info = await api(job.status_url);
        status.textContent = [jobStates[info.status], info.error_code, job.asset_id].filter(Boolean).join(" · ");
        if (["ready", "partial", "failed"].includes(info.status) || !dialog.open) return;
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
      status.textContent = `任务仍在处理中 · ${job.asset_id}`;
    }, (error) => { status.textContent = error.message; });
  });
}
