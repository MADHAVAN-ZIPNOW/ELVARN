// The Sessions page: search, sort, date filters, and one row per chat with pin / rename / download / delete.
import { useEffect, useMemo, useRef, useState } from "react";
import { Icon, SpanIcon } from "../icons";
import { MODES, MONTHS, MONTH_FULL } from "../constants";
import { ago, pad2, plainSnippet } from "../utils";
import { useTilt } from "../hooks/useTilt";
import { Logo } from "./Logo";

const when = (c) => c.updated || c.created;

function RenameInput({ value, onDone }) {
  const ref = useRef(null);
  const [v, setV] = useState(value);
  const done = useRef(false);
  useEffect(() => { ref.current.focus(); ref.current.select(); }, []);
  const finish = (save) => {
    if (done.current) return;
    done.current = true;
    onDone(save ? v.replace(/\s+/g, " ").trim() : null);
  };
  return (
    <input ref={ref} className="stitle-edit" type="text" value={v} maxLength={80} aria-label="Session name"
      onChange={(e) => setV(e.target.value)}
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === "Enter") { e.preventDefault(); finish(true); } else if (e.key === "Escape") { e.preventDefault(); finish(false); }
      }}
      onBlur={() => finish(true)} />
  );
}

function SessionRow({ conv: c, editing, onOpen, onPin, onStartRename, onRename, onDownload, onDelete }) {
  const mode = MODES.find((m) => m.id === c.mode) || MODES[0];
  const last = [...c.messages].reverse().find((m) => m.role === "bot" && !m.error);
  const d = new Date(when(c));
  const openRow = () => onOpen(c.id);
  return (
    <div className={"srow" + (c.pinned ? " pinnedrow" : "")} tabIndex={0} role="button" onClick={openRow}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openRow(); } }}>
      <span className={"stag k-" + mode.col}>{mode.label}</span>
      <div className="smid">
        {editing ? <RenameInput value={c.title} onDone={(v) => onRename(c.id, v)} /> : <div className="stitle">{c.title}</div>}
        {last && <div className="ssnip">{plainSnippet(last.text)}</div>}
      </div>
      <span className="stime" title={d.toLocaleString()}>
        <b>{`${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}`}</b>
        {ago(when(c))}
      </span>
      <span className="sacts">
        <button className={"sdel" + (c.pinned ? " pinned" : "")} type="button" title={c.pinned ? "Unpin session" : "Pin session"}
          aria-label={c.pinned ? "Unpin session" : "Pin session"} aria-pressed={!!c.pinned}
          onClick={(e) => { e.stopPropagation(); onPin(c.id); }}>
          <Icon name="pin" />
        </button>
        <button className="sdel" type="button" title="Rename session" aria-label="Rename session"
          onClick={(e) => { e.stopPropagation(); onStartRename(c.id); }}>
          <Icon name="pen" />
        </button>
        <button className="sdel" type="button" data-dl="1" title="Download this session as PDF, Word or text" aria-haspopup="menu" aria-label="Download"
          onClick={(e) => { e.stopPropagation(); onDownload(e.currentTarget, c); }}>
          <Icon name="download" />
        </button>
        <button className="sdel" type="button" title="Delete session" aria-label="Delete session"
          onClick={(e) => { e.stopPropagation(); onDelete(c.id); }}>
          <Icon name="trash" />
        </button>
      </span>
    </div>
  );
}

export function SessionsView({ hidden, convs, onOpen, onNew, onPin, onRename, onDownload, onDelete }) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("new");
  const [filters, setFilters] = useState({ year: "", month: "", date: "" });
  const [editingId, setEditingId] = useState(null);
  const tiltNew = useTilt(8, 12);

  const all = useMemo(() => convs.filter((c) => c.messages.length), [convs]);
  const years = useMemo(() => [...new Set(all.map((c) => new Date(when(c)).getFullYear()))].sort((a, b) => b - a), [all]);
  const q = query.trim().toLowerCase();
  const list = useMemo(() => all.filter((c) => {
    const d = new Date(when(c));
    if (filters.year && d.getFullYear() !== +filters.year) return false;
    if (filters.month && d.getMonth() + 1 !== +filters.month) return false;
    if (filters.date && `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}` !== filters.date) return false;
    return !q || c.title.toLowerCase().includes(q) || c.messages.some((m) => m.text.toLowerCase().includes(q));
  }).sort((a, b) => (!!b.pinned - !!a.pinned) || (sort === "old" ? when(a) - when(b) : when(b) - when(a))), [all, filters, q, sort]);

  const bits = [];
  if (filters.year) bits.push(filters.year);
  if (filters.month) bits.push(MONTH_FULL[+filters.month - 1]);
  if (filters.date) bits.push(filters.date);
  if (q) bits.push(`"${query.trim()}"`);
  const filtered = !!(q || filters.year || filters.month || filters.date);

  return (
    <section className="sessions" id="sessionsView" hidden={hidden} aria-label="All sessions">
      <div className="sess-logo"><Logo /></div>
      <div className="sess-top">
        <div>
          <h2>Sessions</h2>
          <div className="sess-sub">{`${list.length} session${list.length === 1 ? "" : "s"}` + (bits.length ? ` · ${bits.join(" · ")}` : "")}</div>
        </div>
        <div className="sess-tools">
          <div className="search sessearch">
            <span style={{ display: "inline-grid" }}><Icon name="search" /></span>
            <input id="search" placeholder="Search sessions" aria-label="Search sessions" value={query} onChange={(e) => setQuery(e.target.value)} />
          </div>
          <div className="seg" role="group" aria-label="Sort sessions">
            <button className={"segbtn" + (sort === "new" ? " active" : "")} type="button" onClick={() => setSort("new")}>New</button>
            <button className={"segbtn" + (sort === "old" ? " active" : "")} type="button" onClick={() => setSort("old")}>Old</button>
          </div>
          <button className="key solid k-blue small" id="sessNew" type="button" onClick={onNew} {...tiltNew}>
            <span className="kicon"><SpanIcon name="plus" /></span>
            <span className="klabel">New chat</span>
          </button>
        </div>
      </div>
      <div className="sess-filters" aria-label="Filter sessions by date">
        <label className="fld">Year
          <select value={filters.year} onChange={(e) => setFilters({ ...filters, year: e.target.value })}>
            <option value="">All</option>
            {years.map((y) => <option key={y} value={String(y)}>{y}</option>)}
          </select>
        </label>
        <label className="fld">Month
          <select value={filters.month} onChange={(e) => setFilters({ ...filters, month: e.target.value })}>
            <option value="">All</option>
            {MONTH_FULL.map((m, i) => <option key={m} value={String(i + 1)}>{m}</option>)}
          </select>
        </label>
        <label className="fld">Date
          <input type="date" value={filters.date} onChange={(e) => setFilters({ ...filters, date: e.target.value })} />
        </label>
        <button className="fclear" type="button" hidden={!(filters.year || filters.month || filters.date)}
          onClick={() => setFilters({ year: "", month: "", date: "" })}>
          Clear filters
        </button>
      </div>
      <div className="slist">
        {list.length === 0 ? (
          <div className="noconv">{filtered ? "No sessions match these filters." : "No sessions yet. Start a new chat and it will appear here."}</div>
        ) : (
          list.map((c) => (
            <SessionRow key={c.id} conv={c} editing={editingId === c.id} onOpen={onOpen} onPin={onPin}
              onStartRename={setEditingId}
              onRename={(id, title) => { setEditingId(null); if (title) onRename(id, title); }}
              onDownload={onDownload} onDelete={onDelete} />
          ))
        )}
      </div>
    </section>
  );
}
