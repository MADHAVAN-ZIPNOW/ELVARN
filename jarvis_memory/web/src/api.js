// Talking to the Jarvis Memory API. The owner key lives only in this browser's localStorage and is sent as a header.
const KEY = "jm.ownerKey";
export const BASE = "/api/jarvis/memory";

export class AuthError extends Error {}

/** Open the page once with ?key=<your key> and this browser remembers it (the address bar is cleaned). */
export function captureKey() {
  try {
    const url = new URL(location.href);
    const k = url.searchParams.get("key");
    if (k) {
      localStorage.setItem(KEY, k);
      url.searchParams.delete("key");
      history.replaceState(null, "", url.pathname + (url.search || "") + url.hash);
    }
  } catch (e) { /* private mode: the user types the key instead */ }
}
export const getKey = () => { try { return localStorage.getItem(KEY) || ""; } catch (e) { return ""; } };
export const setKey = (k) => { try { localStorage.setItem(KEY, k.trim()); } catch (e) { /* ignore */ } };
export const forgetKey = () => { try { localStorage.removeItem(KEY); } catch (e) { /* ignore */ } };

export async function api(path, { method = "GET", body } = {}) {
  let res;
  try {
    res = await fetch(BASE + path, {
      method,
      headers: { "X-Owner-Key": getKey(), ...(body !== undefined ? { "Content-Type": "application/json" } : {}) },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    throw new Error("Can't reach Jarvis Memory. Check your connection (a sleeping free server can take ~30 s to wake).");
  }
  if (res.status === 401) throw new AuthError("That owner key was not accepted.");
  if (res.status === 204) return null;
  let data = null;
  try { data = await res.json(); } catch (e) { /* no body */ }
  if (!res.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === "string" ? d : d ? "Please check the highlighted fields." : "Request failed (" + res.status + ").");
  }
  return data;
}

export const listMemories = (source) => api("?limit=30" + (source ? "&source=" + source : ""));
export const createMemory = (body) => api("", { method: "POST", body });
export const getMemory = (id) => api("/" + id);
export const updateMemory = (id, body) => api("/" + id, { method: "PATCH", body });
export const deleteMemory = (id) => api("/" + id, { method: "DELETE" });
export const reExtract = (id) => api("/" + id + "/extract", { method: "POST" });
export const search = (body) => api("/search", { method: "POST", body });
export const stats = () => api("/stats");
export const exportAll = () => api("/export", { method: "POST" });
export const deleteAll = () => api("/all?confirm=true", { method: "DELETE" });
