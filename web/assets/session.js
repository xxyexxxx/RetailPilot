export function sessionKey(userId) { return `retailpilot-session:${userId}`; }
export function loadSession(userId, storage = localStorage) {
  const key = sessionKey(userId);
  const id = storage.getItem(key) || `web-${crypto.randomUUID()}`;
  storage.setItem(key, id);
  return id;
}
export function newSession(userId, storage = localStorage) {
  const id = `web-${crypto.randomUUID()}`;
  storage.setItem(sessionKey(userId), id);
  return id;
}
