const SESSION_STORAGE_KEY = "retailpilot-session";

const state = {
  userId: "demo-user",
  sessionId: localStorage.getItem(SESSION_STORAGE_KEY)
    || localStorage.getItem("commerce-session")
    || `web-${crypto.randomUUID().slice(0, 8)}`,
  mode: "hybrid",
  catalog: null,
  profile: null,
  selectedCompare: new Map(),
  lastQuery: "",
  products: new Map(),
};

localStorage.setItem(SESSION_STORAGE_KEY, state.sessionId);
localStorage.removeItem("commerce-session");

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function renderEmptyChatInspector() {
  $("#chat-inspector").innerHTML = `
    <div class="inspector-header">
      <div><p class="eyebrow">Agent trace</p><h3>决策轨迹</h3></div>
      <span class="mode-tag">Hybrid</span>
    </div>
    <div class="empty-state compact">
      <i data-lucide="route"></i>
      <p>发送消息后显示意图、计划和节点轨迹。</p>
    </div>
  `;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value) {
  return new Intl.NumberFormat("zh-CN").format(value ?? 0);
}

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons();
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `请求失败: ${response.status}`);
  }
  return response.json();
}

function toast(message) {
  const item = document.createElement("div");
  item.className = "toast";
  item.textContent = message;
  $("#toast-region").append(item);
  setTimeout(() => item.remove(), 2600);
}

function setLoading(container, active) {
  let line = $(".loading-line", container);
  if (active && !line) {
    line = document.createElement("div");
    line.className = "loading-line";
    container.prepend(line);
  } else if (!active && line) {
    line.remove();
  }
}

function switchView(name) {
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  refreshIcons();
}

function updateCatalog(catalog) {
  state.catalog = catalog;
  const stats = catalog.stats;
  $("#stat-products").textContent = formatNumber(stats.products);
  $("#stat-categories").textContent = formatNumber(stats.categories);
  $("#stat-brands").textContent = formatNumber(stats.brands);
  $("#stat-searches").textContent = formatNumber(stats.searches);
  $("#db-count").textContent = `${formatNumber(stats.products)} products`;
  $("#catalog-status").textContent = `${stats.categories} 个类目 / SQLite FTS5`;
  $("#dataset-time").textContent = catalog.dataset?.imported_at?.slice(0, 10) || "-";
  const select = $("#search-category");
  select.innerHTML = '<option value="">全部类目</option>' + catalog.categories
    .map((item) => `<option value="${escapeHtml(item.category)}">${escapeHtml(item.category_label)} (${item.count})</option>`)
    .join("");
}

function updateProfile(profile) {
  state.profile = profile;
  const preferences = profile.preferences || [];
  const memories = profile.memories || [];
  $("#profile-content").innerHTML = `
    <section class="profile-group">
      <h3>${escapeHtml(profile.display_name || profile.user_id)}</h3>
      <div class="preference-item"><span>偏好条目</span><span>${preferences.length}</span></div>
      <div class="preference-item"><span>长期记忆</span><span>${memories.length}</span></div>
    </section>
    <section class="profile-group">
      <h3>偏好权重</h3>
      ${preferences.length ? preferences.map((item) => `
        <div class="preference-item">
          <span>${escapeHtml(item.preference_type)} / ${escapeHtml(item.preference_key)}</span>
          <span>${Number(item.weight).toFixed(2)}</span>
        </div>
      `).join("") : '<p class="detail-description">暂无偏好。</p>'}
    </section>
    <section class="profile-group">
      <h3>记忆</h3>
      ${memories.length ? memories.map((item) => `
        <div class="preference-item">
          <span>${escapeHtml(item.memory_key)}: ${escapeHtml(item.memory_value)}</span>
          <span>${Number(item.confidence).toFixed(2)}</span>
        </div>
      `).join("") : '<p class="detail-description">暂无长期记忆。</p>'}
    </section>
  `;
}

function rememberProducts(products) {
  (products || []).forEach((product) => state.products.set(product.sku, product));
}

function scoreRow(label, value) {
  const percent = Math.max(0, Math.min(100, Number(value || 0) * 100));
  return `
    <div class="score-row">
      <span>${escapeHtml(label)}</span>
      <span class="score-track"><i style="width:${percent}%"></i></span>
      <span>${percent.toFixed(0)}</span>
    </div>
  `;
}

function productCard(product, compact = false) {
  const score = product.score_breakdown || {};
  const selected = state.selectedCompare.has(product.sku);
  return `
    <article class="product-card" data-sku="${escapeHtml(product.sku)}">
      <div class="product-image">
        <img src="${escapeHtml(product.image_url)}" alt="${escapeHtml(product.title)}" loading="lazy">
        <span class="product-category">${escapeHtml(product.category_label)}</span>
      </div>
      <div class="product-content">
        <h3 class="product-title">${escapeHtml(product.title)}</h3>
        <div class="product-meta">
          <span class="product-price">$${Number(product.price).toFixed(2)}</span>
          <span class="product-rating">★ ${Number(product.rating).toFixed(1)} · ${formatNumber(product.rating_count)}</span>
        </div>
        <p class="product-reason">${escapeHtml(product.reason || product.features?.[0] || "")}</p>
        ${compact ? "" : scoreRow("相关性", score.term_match ?? product.score)}
        ${compact ? "" : scoreRow("个性化", (Number(score.personalization || 0) + 1) / 2)}
        <div class="product-actions">
          <button data-action="detail" title="查看详情" aria-label="查看详情"><i data-lucide="panel-top-open"></i></button>
          <button data-action="favorite" title="收藏" aria-label="收藏"><i data-lucide="heart"></i></button>
          <button data-action="cart" title="加入购物车" aria-label="加入购物车"><i data-lucide="shopping-cart"></i></button>
          <button data-action="compare" class="${selected ? "selected" : ""}" title="加入对比" aria-label="加入对比"><i data-lucide="columns-3"></i></button>
        </div>
      </div>
    </article>
  `;
}

function renderProductGrid(container, products, compact = false) {
  rememberProducts(products);
  if (!products?.length) {
    container.innerHTML = '<div class="empty-state"><i data-lucide="package-search"></i><p>没有商品结果。</p></div>';
  } else {
    container.classList.toggle("compact-grid", compact);
    container.innerHTML = products.map((product) => productCard(product, compact)).join("");
  }
  refreshIcons();
}

// 商品卡片混排（业界做法）：LLM 在推荐每个商品后输出 `[[sku:xxx]]` 标记，
// 前端解析标记，用 sku 精确索引 products 里的商品并替换成卡片。
// 卡片自然跟随其引用位置，不依赖模糊匹配，也不依赖数组顺序。
function interleaveProducts(text, products) {
  if (!products.length) return `<p>${escapeHtml(text)}</p>`;
  const bySku = new Map(products.map((p) => [p.sku, p]));
  // 把文本按 `[[sku:xxx]]` 切分成 文字段 / sku 标记 交替的序列
  const parts = text.split(/\[\[sku:([^\]]+)\]\]/);
  // parts 长度 2n+1：偶数下标是文字，奇数下标是 sku
  const out = [];
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 0) {
      const chunk = parts[i];
      if (!chunk) continue;
      // 把文字按空行/编号拆成段落，保留换行
      out.push(`<p>${escapeHtml(chunk).replace(/\n+/g, "<br>")}</p>`);
    } else {
      const sku = parts[i].trim();
      const item = bySku.get(sku);
      if (item) {
        out.push(`<div class="message-products inline-products"><div class="product-grid compact-grid">${productCard(item, true)}</div></div>`);
      }
      // sku 无匹配时静默丢弃标记，避免展示 [[sku:...]] 原文
    }
  }
  return out.join("");
}

function addChatMessage(role, text, products = [], interleave = false) {
  const stream = $("#message-stream");
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const icon = role === "user" ? "user-round" : "sparkles";
  let bodyHtml;
  if (role === "assistant" && interleave && products.length) {
    bodyHtml = interleaveProducts(text, products);
  } else {
    bodyHtml = `<p>${escapeHtml(text)}</p>${
      products.length
        ? `<div class="message-products"><div class="product-grid compact-grid">${products.slice(0, 3).map((item) => productCard(item, true)).join("")}</div></div>`
        : ""
    }`;
  }
  article.innerHTML = `
    <div class="message-avatar"><i data-lucide="${icon}"></i></div>
    <div class="message-body">${bodyHtml}</div>
  `;
  stream.append(article);
  stream.scrollTop = stream.scrollHeight;
  refreshIcons();
}

function renderChatInspector(payload) {
  const parsed = payload.parsed_queries?.[0] || {};
  const decisions = payload.decisions || [];
  const trace = payload.trace || [];
  const hardAttributes = (parsed.must_have_groups || [])
    .map((group) => group.slice(0, 2).join(" / "))
    .join(", ");
  $("#chat-inspector").innerHTML = `
    <div class="inspector-header">
      <div><p class="eyebrow">Agent trace</p><h3>决策轨迹</h3></div>
      <span class="mode-tag">${escapeHtml((payload.intents || []).join(" + "))}</span>
    </div>
    <div class="query-block">
      <strong>${escapeHtml(parsed.rewritten_query || payload.reply.slice(0, 80))}</strong>
      <div class="query-grid">
        <div><b>Category</b><span>${escapeHtml(parsed.category || "-")}</span></div>
        <div><b>Budget</b><span>${parsed.price_max ? `$${parsed.price_max}` : "-"}</span></div>
        <div><b>Concepts</b><span>${escapeHtml((parsed.concepts || []).slice(0, 4).join(", ") || "-")}</span></div>
        <div><b>Citations</b><span>${payload.citations?.length || 0}</span></div>
        <div><b>Retrieval terms</b><span>${escapeHtml((parsed.grounded_target_terms || []).slice(0, 3).join(", ") || "-")}</span></div>
        <div><b>Brand pref</b><span>${escapeHtml((parsed.preferred_brands || []).join(", ") || "-")}</span></div>
        <div><b>Hard attrs</b><span>${escapeHtml(hardAttributes || "-")}</span></div>
        <div><b>Excluded</b><span>${escapeHtml((parsed.negative_terms || []).slice(0, 4).join(", ") || "-")}</span></div>
      </div>
    </div>
    <div class="trace-list">
      ${decisions.map((item, index) => `
        <div class="trace-item">
          <span class="trace-index">${index + 1}</span>
          <div>
            <strong>${escapeHtml(item.domain)} / ${escapeHtml(item.action)}</strong>
            <span>${escapeHtml(item.tool || "none")} · route only</span>
          </div>
        </div>
      `).join("")}
      ${trace.slice(-8).map((item, index) => `
        <div class="trace-item">
          <span class="trace-index">${decisions.length + index + 1}</span>
          <div><strong>${escapeHtml(item.node)}</strong><span>${escapeHtml(JSON.stringify(item).slice(0, 150))}</span></div>
        </div>
      `).join("")}
    </div>
  `;
}

async function sendChat(message) {
  addChatMessage("user", message);
  const form = $("#chat-form");
  setLoading(form, true);
  try {
    const payload = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        session_id: state.sessionId,
        user_id: state.userId,
        message,
      }),
    });
    rememberProducts(payload.products);
    addChatMessage("assistant", payload.reply, payload.products || [], true);
    renderChatInspector(payload);
    const row = $("#chat-suggestions");
    row.innerHTML = (payload.suggestions || []).map((item) => `<button data-prompt="${escapeHtml(item)}">${escapeHtml(item)}</button>`).join("");
    if (payload.needs_clarification) toast("会话进入澄清状态");
    updateProfile(await api(`/api/profile/${state.userId}`));
  } catch (error) {
    addChatMessage("assistant", `请求失败：${error.message}`);
  } finally {
    setLoading(form, false);
  }
}

function renderSearchInspector(payload) {
  const parsed = payload.parsed_query;
  const top = payload.products?.[0];
  const factClass = payload.judge?.pass ? "fact-pass" : "fact-fail";
  const hardAttributes = (parsed.must_have_groups || [])
    .map((group) => group.slice(0, 2).join(" / "))
    .join(", ");
  $("#search-inspector").innerHTML = `
    <div class="inspector-header">
      <div><p class="eyebrow">Request ${escapeHtml(payload.request_id)}</p><h3>检索诊断</h3></div>
      <span class="${factClass}">${payload.judge?.pass ? "Facts pass" : "Facts fail"}</span>
    </div>
    <div class="query-block">
      <strong>${escapeHtml(parsed.rewritten_query)}</strong>
      <div class="query-grid">
        <div><b>Intent</b><span>${escapeHtml(parsed.intent)}</span></div>
        <div><b>Category</b><span>${escapeHtml(parsed.category || "-")}</span></div>
        <div><b>Price max</b><span>${parsed.price_max ? `$${parsed.price_max}` : "-"}</span></div>
        <div><b>Latency</b><span>${payload.latency_ms} ms</span></div>
        <div><b>Retrieval terms</b><span>${escapeHtml((parsed.grounded_target_terms || []).slice(0, 3).join(", ") || "-")}</span></div>
        <div><b>Brand pref</b><span>${escapeHtml((parsed.preferred_brands || []).join(", ") || "-")}</span></div>
        <div><b>Hard attrs</b><span>${escapeHtml(hardAttributes || "-")}</span></div>
        <div><b>Excluded</b><span>${escapeHtml((parsed.negative_terms || []).slice(0, 4).join(", ") || "-")}</span></div>
      </div>
    </div>
    ${(payload.relaxations || []).length ? `
      <div class="query-block constraint-note">
        <strong>Controlled relaxation</strong>
        <span>${escapeHtml(payload.relaxations[0].message)}</span>
      </div>
    ` : ""}
    ${top ? `
      <div class="query-block">
        <strong>Top 1 score</strong>
        ${Object.entries(top.score_breakdown || {}).map(([key, value]) => scoreRow(key, key === "personalization" ? (Number(value) + 1) / 2 : value)).join("")}
      </div>
    ` : ""}
    <div class="trace-list">
      ${(payload.trace || []).map((item, index) => `
        <div class="trace-item">
          <span class="trace-index">${index + 1}</span>
          <div><strong>${escapeHtml(item.node)}</strong><span>${escapeHtml(JSON.stringify(item).slice(0, 180))}</span></div>
        </div>
      `).join("")}
    </div>
  `;
}

async function runSearch() {
  const query = $("#search-query").value.trim();
  if (!query) return;
  state.lastQuery = query;
  const results = $("#search-results");
  setLoading(results, true);
  try {
    const payload = await api("/api/search", {
      method: "POST",
      body: JSON.stringify({
        query,
        user_id: state.userId,
        limit: 12,
        mode: state.mode,
        category: $("#search-category").value || null,
        price_max: $("#search-price").value ? Number($("#search-price").value) : null,
      }),
    });
    renderProductGrid(results, payload.products);
    const relaxation = payload.relaxations?.length ? " · 最小属性放宽" : "";
    $("#search-summary").textContent = `${payload.total_candidates} 个候选 · 返回 ${payload.products.length} 款 · ${payload.latency_ms} ms${relaxation}`;
    renderSearchInspector(payload);
    updateCatalog(await api("/api/catalog"));
  } catch (error) {
    results.innerHTML = `<div class="empty-state"><p>${escapeHtml(error.message)}</p></div>`;
  } finally {
    setLoading(results, false);
  }
}

function updateCompareBar() {
  const items = [...state.selectedCompare.values()];
  const bar = $("#compare-bar");
  bar.classList.toggle("open", items.length > 0);
  $("#compare-count").textContent = `${items.length} / 4`;
  $("#compare-items").innerHTML = items.map((item) => `<span class="compare-chip">${escapeHtml(item.title)}</span>`).join("");
  $("#compare-button").disabled = items.length < 2;
  $$('[data-action="compare"]').forEach((button) => {
    const card = button.closest("[data-sku]");
    button.classList.toggle("selected", state.selectedCompare.has(card?.dataset.sku));
  });
}

function toggleCompare(product) {
  if (state.selectedCompare.has(product.sku)) {
    state.selectedCompare.delete(product.sku);
  } else if (state.selectedCompare.size >= 4) {
    toast("最多比较 4 款商品");
    return;
  } else {
    state.selectedCompare.set(product.sku, product);
  }
  updateCompareBar();
}

function showProduct(product) {
  const attributes = Object.entries(product.attributes || {}).slice(0, 18);
  $("#product-dialog-content").innerHTML = `
    <div class="detail-layout">
      <div class="detail-image"><img src="${escapeHtml(product.image_url)}" alt="${escapeHtml(product.title)}"></div>
      <div class="detail-content">
        <p class="eyebrow">${escapeHtml(product.category_label)} / ${escapeHtml(product.sku)}</p>
        <h2>${escapeHtml(product.title)}</h2>
        <div class="product-meta">
          <span class="product-price">$${Number(product.price).toFixed(2)}</span>
          <span class="product-rating">★ ${Number(product.rating).toFixed(1)} · ${formatNumber(product.rating_count)}</span>
        </div>
        <p class="detail-description">${escapeHtml(product.description || product.features?.join("；") || "")}</p>
        <dl class="attribute-grid">
          ${attributes.map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}
        </dl>
      </div>
    </div>
  `;
  $("#product-dialog").showModal();
  refreshIcons();
}

async function showComparison() {
  const skus = [...state.selectedCompare.keys()];
  const payload = await api("/api/products/compare", {
    method: "POST",
    body: JSON.stringify({ skus }),
  });
  const products = payload.products;
  const basicRows = [
    { field: "价格", values: Object.fromEntries(products.map((item) => [item.sku, `$${Number(item.price).toFixed(2)}`])) },
    { field: "评分", values: Object.fromEntries(products.map((item) => [item.sku, `${Number(item.rating).toFixed(1)} / ${formatNumber(item.rating_count)}`])) },
    { field: "品牌", values: Object.fromEntries(products.map((item) => [item.sku, item.brand || "-"])) },
  ];
  const rows = [...basicRows, ...(payload.fields || [])];
  $("#compare-dialog-content").innerHTML = `
    <div class="compare-content">
      <p class="eyebrow">Product compare</p>
      <h2>商品对比</h2>
      <div class="table-wrap">
        <table class="compare-table">
          <thead>
            <tr>
              <th>维度</th>
              ${products.map((item) => `<th class="compare-product-head"><img src="${escapeHtml(item.image_url)}" alt=""><span>${escapeHtml(item.title)}</span></th>`).join("")}
            </tr>
          </thead>
          <tbody>
            ${rows.map((row) => `<tr><th>${escapeHtml(row.field)}</th>${products.map((item) => `<td>${escapeHtml(row.values[item.sku] ?? "-")}</td>`).join("")}</tr>`).join("")}
          </tbody>
        </table>
      </div>
    </div>
  `;
  $("#compare-dialog").showModal();
  refreshIcons();
}

async function recordInteraction(product, eventType) {
  await api("/api/interactions", {
    method: "POST",
    body: JSON.stringify({
      user_id: state.userId,
      sku: product.sku,
      event_type: eventType,
      query: state.lastQuery,
    }),
  });
  toast(eventType === "favorite" ? "已记录收藏偏好" : "已加入购物车信号");
  updateProfile(await api(`/api/profile/${state.userId}`));
}

async function handleProductAction(button) {
  const card = button.closest("[data-sku]");
  const product = state.products.get(card?.dataset.sku);
  if (!product) return;
  const action = button.dataset.action;
  if (action === "detail") showProduct(product);
  if (action === "favorite") await recordInteraction(product, "favorite");
  if (action === "cart") await recordInteraction(product, "cart");
  if (action === "compare") toggleCompare(product);
}

async function init() {
  $("#chat-session-label").textContent = `Session / ${state.sessionId}`;
  try {
    const [catalog, profile] = await Promise.all([
      api("/api/catalog"),
      api(`/api/profile/${state.userId}`),
    ]);
    updateCatalog(catalog);
    updateProfile(profile);
  } catch (error) {
    $("#catalog-status").textContent = error.message;
    toast(error.message);
  }
  refreshIcons();
}

document.addEventListener("click", async (event) => {
  const nav = event.target.closest(".nav-item");
  if (nav) switchView(nav.dataset.view);

  const prompt = event.target.closest("[data-prompt]");
  if (prompt) {
    $("#chat-input").value = prompt.dataset.prompt;
    await sendChat(prompt.dataset.prompt);
    $("#chat-input").value = "";
  }

  const action = event.target.closest("[data-action]");
  if (action) await handleProductAction(action);

  if (event.target.closest("[data-close-dialog]")) {
    event.target.closest("dialog").close();
  }
});

$("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("#chat-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  await sendChat(message);
});

$("#chat-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#chat-form").requestSubmit();
  }
});

$("#search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await runSearch();
});

$$(".segmented button").forEach((button) => {
  button.addEventListener("click", () => {
    state.mode = button.dataset.mode;
    $$(".segmented button").forEach((item) => item.classList.toggle("active", item === button));
  });
});

$("#profile-button").addEventListener("click", () => {
  $("#profile-drawer").classList.add("open");
  $("#drawer-backdrop").classList.add("open");
  $("#profile-drawer").setAttribute("aria-hidden", "false");
});

function closeProfile() {
  $("#profile-drawer").classList.remove("open");
  $("#drawer-backdrop").classList.remove("open");
  $("#profile-drawer").setAttribute("aria-hidden", "true");
}

$("#close-profile").addEventListener("click", closeProfile);
$("#drawer-backdrop").addEventListener("click", closeProfile);

$("#clear-chat").addEventListener("click", () => {
  state.sessionId = `web-${crypto.randomUUID().slice(0, 8)}`;
  localStorage.setItem(SESSION_STORAGE_KEY, state.sessionId);
  $("#chat-session-label").textContent = `Session / ${state.sessionId}`;
  $("#message-stream").innerHTML = `
    <article class="message assistant">
      <div class="message-avatar"><i data-lucide="sparkles"></i></div>
      <div class="message-body"><p>新会话已开始。</p></div>
    </article>
  `;
  renderEmptyChatInspector();
  refreshIcons();
});

$("#compare-button").addEventListener("click", async () => {
  try {
    await showComparison();
  } catch (error) {
    toast(error.message);
  }
});

$("#clear-compare").addEventListener("click", () => {
  state.selectedCompare.clear();
  updateCompareBar();
  toast("已清空对比商品");
});

init();
