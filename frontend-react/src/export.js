// Download a chat (or one answer) as PDF / Word / text. The server builds the file (POST /api/export).
import { API_BASE } from "./api";
import { sourceList } from "./utils";

function exportPayload(msgs) {
  return msgs
    .filter((m) => m.text)
    .map((m) => ({
      role: m.role === "user" ? "user" : "bot",
      text: String(m.text).slice(0, 20000),
      sources: sourceList(m.meta && m.meta.search).slice(0, 12).map((x) => ({
        title: String(x.s.title || x.s.domain || "").slice(0, 120), url: x.href, domain: x.s.domain || "",
      })),
    }));
}

/** Returns an error message, or "" when the file was downloaded. */
export async function downloadAs(title, msgs, fmt) {
  try {
    const r = await fetch(API_BASE + "/api/export", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: String(title || "ELVARN chat").slice(0, 120), format: fmt, messages: exportPayload(msgs).slice(-200) }),
    });
    if (!r.ok) {
      let why = "";
      try { why = (await r.json()).detail || ""; } catch (e) { /* no body */ }
      return "Could not make the file. " + (typeof why === "string" ? why : "");
    }
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    const cd = r.headers.get("content-disposition") || "";
    const m = cd.match(/filename\*=UTF-8''([^;]+)/);
    a.download = m ? decodeURIComponent(m[1]) : "ELVARN." + fmt;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    return "";
  } catch (e) {
    return "Could not make the file. Check your connection and try again.";
  }
}
