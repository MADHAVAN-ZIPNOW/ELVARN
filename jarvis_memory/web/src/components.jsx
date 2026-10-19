import { useState } from "react";

export const SOURCE_ICON = { manual: "✍️", voice: "🎙️", email: "✉️", daily_review: "📅" };
export const SOURCE_LABEL = { manual: "Typed", voice: "Voice", email: "Email", daily_review: "Daily review" };

const fmt = new Intl.DateTimeFormat("en-IN", { dateStyle: "medium", timeStyle: "short", timeZone: "Asia/Kolkata" });
export const formatWhen = (iso) => { try { return fmt.format(new Date(iso)); } catch (e) { return iso; } };

/** Renders search highlights from the server's safe markers (\u0002 ... \u0003). Never uses innerHTML. */
export function Highlight({ text }) {
  const parts = String(text || "").split(/(\u0002[^\u0003]*\u0003)/);
  return parts.map((p, i) => (p.startsWith("\u0002")
    ? <mark key={i}>{p.slice(1).replace(/\u0003$/, "")}</mark>
    : <span key={i}>{p}</span>));
}

export function Chips({ items, kind }) {
  if (!items || !items.length) return null;
  return <>{items.map((x) => <span key={kind + x} className={"chip chip-" + kind}>{x}</span>)}</>;
}

export function MemoryCard({ memory, snippet, onOpen, fresh }) {
  const m = memory;
  return (
    <article className={"card src-" + m.source + (fresh ? " fresh" : "")} onClick={() => onOpen(m.id)} tabIndex={0}
      onKeyDown={(e) => { if (e.key === "Enter") onOpen(m.id); }} role="button" aria-label={"Open memory: " + m.title}>
      <header>
        <span className="src" title={SOURCE_LABEL[m.source]} aria-hidden>{SOURCE_ICON[m.source]}</span>
        <h3>{m.title || "(untitled)"}</h3>
        {m.attachments && m.attachments.length > 0 && <span title="Has attachments" aria-label="Has attachments">📎</span>}
      </header>
      <time dateTime={m.created_at}>{formatWhen(m.created_at)}</time>
      <p className="snippet">{snippet !== undefined ? <Highlight text={snippet} /> : (m.summary || m.content).slice(0, 220)}</p>
      <div className="chips"><Chips items={m.projects} kind="project" /><Chips items={m.people} kind="person" /><Chips items={m.tags} kind="tag" /></div>
      {m.metadata_json && m.metadata_json.extraction && m.metadata_json.extraction.status === "failed" &&
        <small className="warn">AI summary unavailable — saved as written. Open to retry.</small>}
    </article>
  );
}

export function ConfirmButton({ label, confirmLabel = "Tap again to confirm", onConfirm, className = "danger" }) {
  const [armed, setArmed] = useState(false);
  return (
    <button className={"btn " + className} onClick={() => { if (armed) { setArmed(false); onConfirm(); } else { setArmed(true); setTimeout(() => setArmed(false), 4000); } }}>
      {armed ? confirmLabel : label}
    </button>
  );
}

export function Banner({ kind = "error", children, onClose }) {
  if (!children) return null;
  return <div className={"banner " + kind} role={kind === "error" ? "alert" : "status"}>{children}{onClose && <button className="link" onClick={onClose}>Dismiss</button>}</div>;
}
