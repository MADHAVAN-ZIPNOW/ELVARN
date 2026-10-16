// Calls to /api/collect/*. Every request carries the owner key; the server answers 403 without it.
import { API_BASE } from "../api";

export async function collectCall(ownerKey, path, opts) {
  const o = opts || {};
  const headers = { "X-Owner-Key": ownerKey, ...(o.body ? { "Content-Type": "application/json" } : {}) };
  const r = await fetch(API_BASE + "/api/collect" + path, { method: o.method || "GET", headers, body: o.body, signal: o.signal });
  let data = null;
  try { data = await r.json(); } catch (e) { /* no body */ }
  if (!r.ok) {
    const d = data && data.detail;
    const err = new Error(r.status === 403 ? "The owner key was not accepted." : typeof d === "string" ? d : "Request failed (" + r.status + ")");
    err.status = r.status;
    throw err;
  }
  return data;
}
