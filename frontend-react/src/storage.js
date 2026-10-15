// Saved chats and settings live in the browser (same key and shape as the earlier version, so existing sessions keep working).
import { DEFAULT_SETTINGS } from "./constants";

const KEY = "elvarn.v2";
const SESSION_KEY = "elvarn.session";

/** A random id for this browser, sent with each question so the server can remember follow-ups. */
export function sessionId() {
  try {
    let v = localStorage.getItem(SESSION_KEY);
    if (!v) {
      v = crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2);
      localStorage.setItem(SESSION_KEY, v);
    }
    return v;
  } catch (e) {
    return null;
  }
}

export function saveState({ settings, mode, currentId, convs }) {
  try {
    localStorage.setItem(KEY, JSON.stringify({ s: settings, mode, currentId, convs: convs.slice(0, 60) }));
  } catch (e) { /* storage full or blocked: the chat still works */ }
}

export function loadState() {
  const out = { settings: { ...DEFAULT_SETTINGS }, mode: "chat", currentId: null, convs: [] };
  try {
    const d = JSON.parse(localStorage.getItem(KEY) || "null");
    if (d) {
      Object.assign(out.settings, d.s || {});
      if (out.settings.settingsVersion !== DEFAULT_SETTINGS.settingsVersion) {        // older saved settings: take the steadier sampling once
        out.settings.temp = DEFAULT_SETTINGS.temp;
        out.settings.topk = DEFAULT_SETTINGS.topk;
        out.settings.settingsVersion = DEFAULT_SETTINGS.settingsVersion;
      }
      out.mode = d.mode || "chat";
      out.convs = Array.isArray(d.convs) ? d.convs.filter((c) => c && Array.isArray(c.messages)) : [];
      out.currentId = d.currentId;
    }
  } catch (e) { /* corrupt data: start fresh */ }
  return out;
}
