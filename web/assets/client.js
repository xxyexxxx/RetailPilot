let csrfToken = "";

export function setCsrfToken(value) { csrfToken = value || ""; }

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

export async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body) headers.set("Content-Type", "application/json");
  if (csrfToken) headers.set("X-CSRF-Token", csrfToken);
  const response = await fetch(path, { ...options, credentials: "same-origin", headers });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : `请求失败: ${response.status}`;
    throw new ApiError(detail, response.status);
  }
  return response;
}

export async function api(path, options) { return (await request(path, options)).json(); }

export class LatestRequest {
  constructor() { this.sequence = 0; this.controller = null; }
  start() {
    this.cancel();
    const sequence = this.sequence;
    this.controller = new AbortController();
    return { signal: this.controller.signal, current: () => this.sequence === sequence };
  }
  cancel() { this.sequence += 1; this.controller?.abort(); }
}

export async function streamChat(body, { signal, onProgress } = {}) {
  const response = await request("/api/chat/stream", { method: "POST", body: JSON.stringify(body), signal });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let final = null;
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      buffer = buffer.replaceAll("\r\n", "\n");
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const lines = frame.split("\n");
        const event = lines.find((line) => line.startsWith("event:"))?.slice(6).trim();
        const data = lines.filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
        if (!data) continue;
        const payload = JSON.parse(data);
        if (event === "progress") onProgress?.(payload);
        if (event === "error") throw new ApiError(payload.detail, payload.status);
        if (event === "final") final = payload;
      }
      if (done) break;
    }
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
  if (!final) throw new ApiError("连接结束但没有通过校验的最终回答，请恢复会话确认结果。", 502);
  return final;
}

export async function withButton(button, action, onError) {
  if (button?.disabled) return;
  if (button) button.disabled = true;
  try { return await action(); }
  catch (error) { if (error.name !== "AbortError") onError(error); }
  finally { if (button) button.disabled = false; }
}
