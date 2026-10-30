import test from "node:test";
import assert from "node:assert/strict";
import { api, setCsrfToken, LatestRequest, streamChat, withButton } from "../../web/assets/client.js";
import { loadSession, newSession } from "../../web/assets/session.js";

test("requests use cookies and in-memory CSRF, never a source API key", async () => {
  setCsrfToken("csrf-fixture");
  globalThis.fetch = async (path, options) => {
    assert.equal(options.credentials, "same-origin");
    assert.equal(options.headers.get("X-CSRF-Token"), "csrf-fixture");
    assert.equal(options.headers.get("Authorization"), null);
    return Response.json({ ok: true });
  };
  assert.deepEqual(await api("/api/chat", { method: "POST", body: "{}" }), { ok: true });
});

test("new requests abort old ones and reject late results", () => {
  const requests = new LatestRequest();
  const first = requests.start();
  const second = requests.start();
  assert.equal(first.signal.aborted, true);
  assert.equal(first.current(), false);
  assert.equal(second.current(), true);
  requests.cancel();
  assert.equal(second.signal.aborted, true);
});

test("SSE is decoded across byte boundaries without publishing partial answers", async () => {
  const wire = new TextEncoder().encode('event: progress\ndata: {"node":"verify_answer_faithfulness"}\n\nevent: final\ndata: {"reply":"通过核验"}\n\n');
  globalThis.fetch = async () => new Response(new ReadableStream({ start(controller) {
    for (let i = 0; i < wire.length; i += 3) controller.enqueue(wire.slice(i, i + 3));
    controller.close();
  } }));
  const progress = [];
  const final = await streamChat({}, { onProgress: (item) => progress.push(item) });
  assert.deepEqual(final, { reply: "通过核验" });
  assert.equal(progress[0].reply, undefined);
});

test("SSE errors and incomplete connections never become answers", async () => {
  globalThis.fetch = async () => new Response('event: error\ndata: {"detail":"conflict","status":409}\n\n');
  await assert.rejects(() => streamChat({}), /conflict/);
  globalThis.fetch = async () => new Response('event: progress\ndata: {"node":"execute_tool"}\n\n');
  await assert.rejects(() => streamChat({}), /没有通过校验/);
});

test("action errors restore button state", async () => {
  const button = { disabled: false };
  const errors = [];
  await withButton(button, async () => { throw new Error("failure"); }, (error) => errors.push(error.message));
  assert.equal(button.disabled, false);
  assert.deepEqual(errors, ["failure"]);
});

test("session IDs are isolated by authenticated user", () => {
  const data = new Map();
  const storage = { getItem: (key) => data.get(key), setItem: (key, value) => data.set(key, value) };
  const first = loadSession("alice", storage);
  assert.equal(loadSession("alice", storage), first);
  assert.notEqual(loadSession("bob", storage), first);
  assert.notEqual(newSession("alice", storage), first);
});
