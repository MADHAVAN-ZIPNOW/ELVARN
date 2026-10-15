// Owner mode: open the site once with ?owner=<your key> and this browser remembers it (the address bar is cleaned).
// With the key the owner can get a colourful greeting email for every new chat. Nobody else can: the server checks the key.
import { API_BASE } from "./api";

const KEY = "elvarn.ownerKey";

export function captureOwnerKey() {
  try {
    const u = new URL(location.href);
    const k = u.searchParams.get("owner");
    if (k) {
      localStorage.setItem(KEY, k);
      u.searchParams.delete("owner");
      history.replaceState(null, "", u.pathname + (u.search || "") + u.hash);
    }
  } catch (e) { /* private mode: no owner mode */ }
}

export function getOwnerKey() {
  try { return localStorage.getItem(KEY) || ""; } catch (e) { return ""; }
}

/** Ask the server to email the owner a greeting for this new chat (fire and forget). */
export function notifyNewChat() {
  const key = getOwnerKey();
  if (!key) return;
  fetch(API_BASE + "/api/notify/newchat?hour=" + new Date().getHours(), { method: "POST", headers: { "X-Owner-Key": key } }).catch(() => {});
}
