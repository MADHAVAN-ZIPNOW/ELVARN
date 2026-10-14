// Talking to the ELVARN server.
const params = new URLSearchParams(location.search);
/* Served by the ELVARN server itself (local port 8000, or Render) -> same origin.
   Only when opened from a separate dev server (Vite on 5173, Live Server, file://) talk to the local backend. */
const DEV_STATIC = location.protocol === "file:" || ["5500", "5501", "5502", "3000", "5173", "8080"].includes(location.port);
export const API_BASE = params.get("api") || (DEV_STATIC ? "http://127.0.0.1:8000" : "");

function errorFrom(data, status) {
  const d = data && data.detail;
  return new Error(typeof d === "string" ? d : d ? JSON.stringify(d) : "Request failed (" + status + ")");
}

export async function api(path, opts) {
  const r = await fetch(API_BASE + path, opts);
  let data = null;
  try { data = await r.json(); } catch (e) { /* no body */ }
  if (!r.ok) throw errorFrom(data, r.status);
  return data;
}

/** POST /api/chat/stream: the server sends one JSON event per line; onEvent is called for each. */
export async function streamChat(body, onEvent, signal) {
  const r = await fetch(API_BASE + "/api/chat/stream", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
  });
  if (!r.ok) {
    let d = null;
    try { d = (await r.json()).detail; } catch (e) { /* no body */ }
    throw new Error(typeof d === "string" ? d : d ? JSON.stringify(d) : "Request failed (" + r.status + ")");
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
}
