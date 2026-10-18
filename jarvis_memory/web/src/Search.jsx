import { useState } from "react";
import { search } from "./api";
import { Banner, MemoryCard, SOURCE_ICON, SOURCE_LABEL } from "./components";

const EMPTY = { q: "", date_from: "", date_to: "", sources: [], project: "", person: "", tag: "", task_status: "", sentiment: "",
  has_transcript: false, has_attachment: false, sort: "newest" };

// <input type="datetime-local"> is local time with no zone: interpret it as India time (UTC+05:30) like the rest of the app.
const toIso = (v) => (v ? new Date(v + ":00+05:30").toISOString() : null);

function body(f, page) {
  const b = { q: f.q, sort: f.sort, page, page_size: 20 };
  if (f.date_from) b.date_from = toIso(f.date_from);
  if (f.date_to) b.date_to = toIso(f.date_to);
  if (f.sources.length) b.sources = f.sources;
  for (const k of ["project", "person", "tag", "task_status", "sentiment"]) if (f[k]) b[k] = f[k];
  if (f.has_transcript) b.has_transcript = true;
  if (f.has_attachment) b.has_attachment = true;
  return b;
}

export default function Search({ onOpen }) {
  const [f, setF] = useState(EMPTY);
  const [open, setOpen] = useState(false);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const set = (k, v) => setF((s) => ({ ...s, [k]: v }));

  const run = async (page = 1) => {
    setBusy(true); setError("");
    try {
      const r = await search(body(f, page));
      setRes(page === 1 ? r : { ...r, items: [...res.items, ...r.items] });
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  };
  const toggleSource = (s) => set("sources", f.sources.includes(s) ? f.sources.filter((x) => x !== s) : [...f.sources, s]);
  const active = Object.keys(EMPTY).filter((k) => JSON.stringify(f[k]) !== JSON.stringify(EMPTY[k]) && k !== "q" && k !== "sort").length;

  return (
    <section>
      <h2>Search memories</h2>
      <form className="searchbar" onSubmit={(e) => { e.preventDefault(); run(1); }} role="search">
        <input value={f.q} onChange={(e) => set("q", e.target.value)} placeholder='Search words, "exact phrase", -exclude, or'
          aria-label="Search" maxLength={300} />
        <button className="btn primary" disabled={busy}>{busy && !res ? "Searching…" : "Search"}</button>
      </form>
      <button className="link" onClick={() => setOpen(!open)} aria-expanded={open}>{open ? "Hide filters" : "Filters"}{active ? ` (${active})` : ""}</button>
      {open && (
        <div className="filter-grid">
          <label>From<input type="datetime-local" value={f.date_from} onChange={(e) => set("date_from", e.target.value)} /></label>
          <label>To<input type="datetime-local" value={f.date_to} onChange={(e) => set("date_to", e.target.value)} /></label>
          <fieldset><legend>Source</legend>
            {Object.keys(SOURCE_LABEL).map((s) => (
              <label key={s} className="check"><input type="checkbox" checked={f.sources.includes(s)} onChange={() => toggleSource(s)} />{SOURCE_ICON[s]} {SOURCE_LABEL[s]}</label>
            ))}
          </fieldset>
          <label>Project<input value={f.project} onChange={(e) => set("project", e.target.value)} placeholder="e.g. ELVARN" /></label>
          <label>Person<input value={f.person} onChange={(e) => set("person", e.target.value)} placeholder="e.g. Ravi" /></label>
          <label>Tag<input value={f.tag} onChange={(e) => set("tag", e.target.value)} placeholder="e.g. research" /></label>
          <label>Task status
            <select value={f.task_status} onChange={(e) => set("task_status", e.target.value)}>
              <option value="">Any</option><option value="pending">Pending</option><option value="done">Done</option><option value="blocked">Blocked</option>
            </select></label>
          <label>Sentiment
            <select value={f.sentiment} onChange={(e) => set("sentiment", e.target.value)}>
              <option value="">Any</option><option>positive</option><option>neutral</option><option>negative</option><option>mixed</option>
            </select></label>
          <label className="check"><input type="checkbox" checked={f.has_transcript} onChange={(e) => set("has_transcript", e.target.checked)} />Has a transcript</label>
          <label className="check"><input type="checkbox" checked={f.has_attachment} onChange={(e) => set("has_attachment", e.target.checked)} />Has an attachment</label>
          <label>Sort
            <select value={f.sort} onChange={(e) => set("sort", e.target.value)}>
              <option value="newest">Newest first</option><option value="oldest">Oldest first</option><option value="relevance">Best match</option>
            </select></label>
          <button className="btn" onClick={() => { setF(EMPTY); setRes(null); }}>Clear all</button>
        </div>
      )}
      <Banner onClose={() => setError("")}>{error}</Banner>
      {res && <p className="muted" role="status">{res.total.toLocaleString()} {res.total === 1 ? "memory" : "memories"} found</p>}
      {res && res.items.length === 0 && <p className="empty">No memories match. Try fewer filters or different words.</p>}
      <div className="list">{(res ? res.items : []).map((h) => <MemoryCard key={h.memory.id} memory={h.memory} snippet={h.snippet} onOpen={onOpen} />)}</div>
      {res && res.has_more && <button className="btn" onClick={() => run(res.page + 1)} disabled={busy}>{busy ? "Loading…" : "Load more"}</button>}
    </section>
  );
}
