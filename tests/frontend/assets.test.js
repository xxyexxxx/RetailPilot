import test from "node:test";
import assert from "node:assert/strict";
import { assetCitations, assetEvidenceCard, assetGallery } from "../../web/assets/assets.js";
import { api, setCsrfToken } from "../../web/assets/client.js";

test("asset evidence escapes quotes and links previews to the same origin", () => {
  const asset = "a".repeat(32);
  const element = "b".repeat(32);
  const url = `/api/assets/${asset}/elements/${element}/preview`;
  const rendered = assetCitations([{
    asset_id: asset, element_id: element, doc_id: "Document", page_no: 3,
    source_uri: "javascript:alert(1)", quoted_text: "<img onerror=alert(1)>",
    preview_url: url, evidence_type: "visual_observation",
  }]);
  assert.match(rendered, /第 3 页/);
  assert.match(rendered, /&lt;img onerror=alert\(1\)&gt;/);
  assert.ok(rendered.includes(url));
  assert.ok(!rendered.includes("javascript:"));
  assert.match(assetEvidenceCard({routes: {knowledge_visual: {}}, matched_description: "<script>"}), /视觉向量/);
  assert.ok(assetGallery([{asset_type: "product_document", elements: [{preview_url: url}]}]).includes(url));
});

test("raw upload preserves content type and CSRF", async () => {
  setCsrfToken("fixture");
  globalThis.fetch = async (_, options) => {
    assert.equal(options.headers.get("Content-Type"), "image/png");
    assert.equal(options.headers.get("X-CSRF-Token"), "fixture");
    assert.ok(options.body instanceof Blob);
    return Response.json({status: "queued"}, {status: 202});
  };
  const payload = await api("/api/assets", {method: "POST", body: new Blob(["png"]),
    headers: {"Content-Type": "image/png"}});
  assert.equal(payload.status, "queued");
});
