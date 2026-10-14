// Small pure helpers.
export const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
export const pad2 = (n) => String(n).padStart(2, "0");
export const pct = (x) => (x == null ? "—" : Math.round(x * 100) + "%");

export function ago(ts) {
  const m = Math.round((Date.now() - ts) / 60000);
  if (m < 1) return "just now";
  if (m < 60) return m + "m ago";
  const h = Math.round(m / 60);
  if (h < 24) return h + "h ago";
  return Math.round(h / 24) + "d ago";
}

export function fmtSize(n) {
  return n < 1024 ? n + " B" : n < 1048576 ? (n / 1024).toFixed(n < 10240 ? 1 : 0) + " KB" : (n / 1048576).toFixed(1) + " MB";
}

/** Only http and https links are ever made clickable. */
export function safeHttpUrl(u) {
  try {
    const x = new URL(String(u));
    return x.protocol === "https:" || x.protocol === "http:" ? x.href : null;
  } catch (e) {
    return null;
  }
}

/** Copy text; works over plain http too (falls back to a hidden textarea). */
export async function copyPlain(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (e) { /* fall through */ }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.setAttribute("aria-hidden", "true");
  ta.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0";
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch (e) { /* ignore */ }
  ta.remove();
  return ok;
}

/** The sources of an answer that can be shown as cards (a search ran and the url is a safe http/https link). */
export function sourceList(info) {
  return ((info && info.searched && info.sources) || [])
    .map((s) => ({ s, href: safeHttpUrl(s.url) }))
    .filter((x) => x.href);
}

export function searchInfo(e) {
  return e && e.search_status
    ? { status: e.search_status, searched: !!e.searched, message: e.search_message || "", sources: e.sources || [],
        kb: e.kb_facts !== undefined, facts: e.kb_facts || [], entities: e.kb_entities || [] }
    : null;
}

export const compact = (e) => ({
  generated_tokens: e.generated_tokens, prompt_tokens: e.prompt_tokens, seconds: e.seconds, tokens_per_second: e.tokens_per_second,
  mean_token_probability: e.mean_token_probability, stop_reason: e.stop_reason, checkpoint: e.checkpoint,
  weights_from_step: e.weights_from_step, raw_text: e.raw_text,
});

/** Where did this answer come from? (text only, shown under the reply) */
export function originInfo(meta) {
  if (!meta || meta.file_action) return null;
  const s = meta.search;
  if (meta.stop_reason === "knowledge_base")
    return { cls: "refused", label: "Answered from: nothing.", text: " The stored records have no match for this (or the knowledge base is unavailable), so no answer was written." };
  if (s && s.kb && s.searched)
    return { cls: "kb", label: "Answered from: stored records.", text: " Knowledge-graph facts and documents are cited below; the small model wrote the wording, which can still be wrong." };
  if (s && s.searched)
    return { cls: "web", label: "Answered from: live web search.", text: " The small model wrote the text from the results below; check them." };
  return { cls: "model", label: "Answered by the model alone.", text: " No stored records and no web search were used, so nothing here has been checked." };
}

/** Session preview text: the last answer without markdown marks. */
export function plainSnippet(text) {
  return String(text || "")
    .replace(/[*`#]+/g, "")
    .replace(/(^|\n)\s*([-•]|\d{1,2}[.)])\s+/g, "$1")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 140);
}
