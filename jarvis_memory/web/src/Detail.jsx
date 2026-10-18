import { useEffect, useState } from "react";
import { deleteMemory, getMemory, reExtract, updateMemory } from "./api";
import { Banner, ConfirmButton, SOURCE_ICON, SOURCE_LABEL, formatWhen } from "./components";

const list = (v) => v.split(",").map((s) => s.trim()).filter(Boolean);

export default function Detail({ id, onBack }) {
  const [m, setM] = useState(null);
  const [f, setF] = useState(null);
  const [msg, setMsg] = useState({ kind: "", text: "" });
  const [busy, setBusy] = useState(false);

  const adopt = (memory) => {
    setM(memory);
    setF({ title: memory.title, summary: memory.summary, tags: memory.tags.join(", "), projects: memory.projects.join(", "),
      people: memory.people.join(", "), next_steps: memory.next_steps.join("\n"), tasks: memory.tasks.map((t) => ({ ...t })),
      sentiment: memory.sentiment || "" });
  };
  useEffect(() => { getMemory(id).then(adopt).catch((e) => setMsg({ kind: "error", text: e.message })); }, [id]);

  if (!m || !f) return <section><button className="link" onClick={onBack}>← Back</button><Banner>{msg.kind === "error" ? msg.text : ""}</Banner><p className="muted">Loading…</p></section>;
  const set = (k, v) => setF((s) => ({ ...s, [k]: v }));
  const ex = (m.metadata_json && m.metadata_json.extraction) || {};

  const save = async () => {
    setBusy(true); setMsg({ kind: "", text: "" });
    try {
      const patch = { title: f.title, summary: f.summary, tags: list(f.tags), projects: list(f.projects), people: list(f.people),
        next_steps: f.next_steps.split("\n").map((s) => s.trim()).filter(Boolean),
        tasks: f.tasks.filter((t) => t.task.trim()).map((t) => ({ task: t.task, status: t.status })) };
      if (f.sentiment) patch.sentiment = f.sentiment; else patch.clear_sentiment = true;
      adopt(await updateMemory(id, patch));
      setMsg({ kind: "ok", text: "Saved ✓" });
    } catch (e) { setMsg({ kind: "error", text: e.message }); } finally { setBusy(false); }
  };
  const retry = async () => {
    setBusy(true); setMsg({ kind: "", text: "" });
    try { adopt(await reExtract(id)); setMsg({ kind: "ok", text: "AI extraction re-run (fields you edited were kept)." }); }
    catch (e) { setMsg({ kind: "error", text: e.message }); } finally { setBusy(false); }
  };
  const remove = async () => { try { await deleteMemory(id); onBack(true); } catch (e) { setMsg({ kind: "error", text: e.message }); } };

  return (
    <section className="detail">
      <button className="link" onClick={() => onBack()}>← Back</button>
      <h2>{SOURCE_ICON[m.source]} {m.title || "(untitled)"}</h2>
      <p className="muted">{SOURCE_LABEL[m.source]} · {formatWhen(m.created_at)}{m.updated_at !== m.created_at && ` · edited ${formatWhen(m.updated_at)}`}</p>
      <Banner kind={msg.kind || "error"}>{msg.text}</Banner>

      <h3 className="section">Original {m.source === "voice" ? "(your edited transcript)" : ""}</h3>
      {/* memory text is always rendered as text, never as HTML */}
      <pre className="original">{m.content}</pre>
      {m.raw_transcript && <details><summary>What speech recognition heard (unedited)</summary><pre className="original">{m.raw_transcript}</pre></details>}

      <h3 className="section">Summary &amp; details <small className="muted">(AI-generated, yours to edit)</small></h3>
      {ex.status === "failed" && <Banner kind="warn">AI extraction failed ({ex.error || "unknown error"}). Your text is safe. <button className="link" onClick={retry}>Retry now</button></Banner>}
      {ex.status === "skipped" && <p className="muted">Saved without AI.</p>}
      <label>Title<input value={f.title} onChange={(e) => set("title", e.target.value)} maxLength={200} /></label>
      <label>Summary<textarea rows={4} value={f.summary} onChange={(e) => set("summary", e.target.value)} maxLength={2000} /></label>
      <div className="two">
        <label>Projects <small>(comma separated)</small><input value={f.projects} onChange={(e) => set("projects", e.target.value)} /></label>
        <label>People<input value={f.people} onChange={(e) => set("people", e.target.value)} /></label>
      </div>
      <label>Tags<input value={f.tags} onChange={(e) => set("tags", e.target.value)} /></label>
      <label>Next steps <small>(one per line)</small><textarea rows={3} value={f.next_steps} onChange={(e) => set("next_steps", e.target.value)} /></label>
      <label>Sentiment
        <select value={f.sentiment} onChange={(e) => set("sentiment", e.target.value)}>
          <option value="">Not set</option><option>positive</option><option>neutral</option><option>negative</option><option>mixed</option>
        </select></label>

      <h3 className="section">Tasks</h3>
      {f.tasks.map((t, i) => (
        <div className="task-row" key={i}>
          <input value={t.task} onChange={(e) => set("tasks", f.tasks.map((x, j) => (j === i ? { ...x, task: e.target.value } : x)))} aria-label={"Task " + (i + 1)} maxLength={300} />
          <select value={t.status} onChange={(e) => set("tasks", f.tasks.map((x, j) => (j === i ? { ...x, status: e.target.value } : x)))} aria-label="Status">
            <option value="pending">Pending</option><option value="done">Done</option><option value="blocked">Blocked</option>
          </select>
          <button className="btn small" onClick={() => set("tasks", f.tasks.filter((_, j) => j !== i))} aria-label={"Remove task " + (i + 1)}>✕</button>
        </div>
      ))}
      <button className="btn small" onClick={() => set("tasks", [...f.tasks, { task: "", status: "pending" }])}>+ Add task</button>

      <div className="actions">
        <button className="btn primary" onClick={save} disabled={busy}>{busy ? "Working…" : "Save changes"}</button>
        <button className="btn" onClick={retry} disabled={busy}>Re-run AI</button>
        <span className="grow" />
        <ConfirmButton label="Delete" onConfirm={remove} />
      </div>
      {m.thread_id && <p className="muted">Linked to update request <code>{m.thread_id}</code></p>}
    </section>
  );
}
