// "Politician by politician": overall progress, the person in progress, recent and next people, and the searchable roster.
import { useEffect, useRef, useState } from "react";
import { collectCall } from "../pipeline/collectApi";
import { NA, STAGE_LABEL, fmtAgo, fmtNum, isNum, pct } from "../pipeline/format";

const STEPS = ["wikipedia_en", "wikipedia_ta", "wikidata", "corruption_articles", "finishing"];
const STATUSES = ["", "pending", "in_progress", "done", "failed"];
const PAGE = 25;

function Steps({ stage }) {
  const at = STEPS.indexOf(stage);
  return (
    <ol className="pl-steps" aria-label="Stages for this person">
      {STEPS.map((s, i) => (
        <li key={s} className={i < at ? "done" : i === at ? "now" : ""} aria-current={i === at ? "step" : undefined}>{STAGE_LABEL[s]}</li>
      ))}
    </ol>
  );
}

export function Corrupt({ n }) {
  if (!isNum(n)) return <span className="pl-badge off" title={NA}>-</span>;
  return <span className={"pl-badge " + (n > 0 ? "hot" : "off")} title={n + " corruption-related documents"}>{n}</span>;
}

function Person({ p, onOpen }) {
  return (
    <button type="button" className="pl-person" onClick={() => onOpen(p.key)}>
      <span className="pl-pname">{p.name || p.key}</span>
      {p.status && <span className={"pl-pill s-" + p.status}>{p.status}</span>}
    </button>
  );
}

export function Overview({ people, onOpen }) {
  if (!people) return <p className="pl-na">Roster numbers {NA}</p>;
  const { total, done, failed, in_progress: inProg, current: cur, recent, queue } = people;
  const p = pct(done, total);
  return (
    <div className="pl-overview">
      <div className="pl-prog">
        <div className="pl-barhead"><span>People finished</span><b>{isNum(done) && isNum(total) ? fmtNum(done) + " of " + fmtNum(total) : NA}</b></div>
        <div className="pl-meter big" role="progressbar" aria-label="People finished" aria-valuemin={0} aria-valuemax={100} aria-valuenow={p === null ? undefined : Math.round(p)}>
          <span style={{ width: (p || 0) + "%" }} />
        </div>
        <p className="pl-sub">{isNum(inProg) ? inProg + " in progress" : ""}{isNum(failed) && failed ? " · " + failed + " failed" : ""}</p>
      </div>
      <div className="pl-current" aria-live="polite">
        <span className="pl-k">Now collecting</span>
        {cur ? (
          <>
            <button type="button" className="pl-curname" onClick={() => onOpen(cur.key)}>{cur.name || cur.key}</button>
            <span className="pl-sub">{[cur.party, isNum(cur.docs) ? cur.docs + " documents so far" : "", cur.since ? "since " + fmtAgo(cur.since) : ""].filter(Boolean).join(" · ")}</span>
            <Steps stage={cur.stage} />
          </>
        ) : <span className="pl-sub">No one right now.</span>}
      </div>
      <div className="pl-lists">
        <div><span className="pl-k">Recently finished</span>
          {Array.isArray(recent) && recent.length ? recent.slice(0, 6).map((r) => (
            <div className="pl-rec" key={r.key}><Person p={r} onOpen={onOpen} /><small>{isNum(r.docs) ? r.docs + " docs" : ""} {fmtAgo(r.finished_at)}</small></div>
          )) : <span className="pl-sub">None yet.</span>}
        </div>
        <div><span className="pl-k">Up next</span>
          {Array.isArray(queue) && queue.length ? queue.slice(0, 6).map((r) => <div className="pl-rec" key={r.key}><Person p={r} onOpen={onOpen} /></div>) : <span className="pl-sub">Queue is empty.</span>}
        </div>
      </div>
    </div>
  );
}

export function Roster({ ownerKey, onOpen, refreshKey }) {
  const [q, setQ] = useState(""), [st, setSt] = useState(""), [page, setPage] = useState(0);
  const [qNow, setQNow] = useState("");
  const [data, setData] = useState(null), [err, setErr] = useState(""), [loading, setLoading] = useState(false);
  useEffect(() => { const t = setTimeout(() => { setQNow(q.trim()); setPage(0); }, 300); return () => clearTimeout(t); }, [q]);
  useEffect(() => {
    const ab = new AbortController();
    setLoading(true);
    const qs = "?status=" + encodeURIComponent(st) + "&q=" + encodeURIComponent(qNow) + "&limit=" + PAGE + "&offset=" + page * PAGE;
    collectCall(ownerKey, "/people" + qs, { signal: ab.signal })
      .then((d) => { setData(d); setErr(""); setLoading(false); })
      .catch((e) => { if (e.name !== "AbortError") { setErr(e.message); setLoading(false); } });
    return () => ab.abort();
  }, [ownerKey, st, qNow, page, refreshKey]);
  const items = data && Array.isArray(data.items) ? data.items : [];
  const total = data && isNum(data.total) ? data.total : 0;
  const pages = Math.max(1, Math.ceil(total / PAGE));
  return (
    <div className="pl-roster">
      <div className="pl-filters">
        <label className="fld">Search name<input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. Karunanidhi" /></label>
        <label className="fld">Status
          <select value={st} onChange={(e) => { setSt(e.target.value); setPage(0); }}>
            {STATUSES.map((s) => <option key={s} value={s}>{s ? s.replace("_", " ") : "all"}</option>)}
          </select>
        </label>
        <span className="pl-sub" role="status">{err ? err : loading ? "Loading…" : fmtNum(total) + " people"}</span>
      </div>
      <div className="pl-tablewrap">
        <table className="pl-table">
          <caption className="sr-only">Politicians, collected one by one</caption>
          <thead><tr><th scope="col">Person</th><th scope="col">Years</th><th scope="col">Status</th><th scope="col" className="num">Docs</th><th scope="col" className="num">Tokens</th><th scope="col" className="num">Corruption docs</th></tr></thead>
          <tbody>
            {items.map((p) => (
              <tr key={p.key} tabIndex={0} onClick={() => onOpen(p.key)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onOpen(p.key); } }}>
                <th scope="row"><span className="pl-pname">{p.name || p.key}</span><small>{p.party || ""}</small></th>
                <td>{p.first_year || ""}{p.last_year && p.last_year !== p.first_year ? "-" + p.last_year : ""}</td>
                <td><span className={"pl-pill s-" + p.status}>{(p.status || "").replace("_", " ")}</span></td>
                <td className="num">{fmtNum(p.docs)}</td><td className="num">{fmtNum(p.tokens)}</td><td className="num"><Corrupt n={p.corruption_docs} /></td>
              </tr>
            ))}
            {!items.length && !loading && <tr><td colSpan={6} className="pl-na">{err ? "Roster " + NA : "No people match."}</td></tr>}
          </tbody>
        </table>
      </div>
      <nav className="pl-pager" aria-label="Roster pages">
        <button type="button" className="pl-btn" disabled={page <= 0} onClick={() => setPage(page - 1)}>Previous</button>
        <span>Page {page + 1} of {pages}</span>
        <button type="button" className="pl-btn" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>Next</button>
      </nav>
    </div>
  );
}

const TOPIC_LABEL = { person: "Person", structured: "Structured", corruption: "Corruption", court_record: "Court records", agency_record: "Agency records" };
const TOPIC_ORDER = ["person", "structured", "corruption", "court_record", "agency_record"];
function groupByTopic(docs) {
  const m = {};
  docs.forEach((d) => { const t = d.topic || "other"; (m[t] = m[t] || []).push(d); });
  const keys = Object.keys(m).sort((x, y) => (TOPIC_ORDER.indexOf(x) + 99) % 99 - (TOPIC_ORDER.indexOf(y) + 99) % 99);
  return keys.map((k) => [k, m[k]]);
}

export function PersonDrawer({ ownerKey, personKey, onClose }) {
  const [d, setD] = useState(null), [err, setErr] = useState("");
  const closeRef = useRef(null);
  useEffect(() => {
    if (!personKey) return undefined;
    setD(null); setErr("");
    const ab = new AbortController();
    collectCall(ownerKey, "/person/" + encodeURIComponent(personKey), { signal: ab.signal }).then(setD).catch((e) => { if (e.name !== "AbortError") setErr(e.message); });
    if (closeRef.current) closeRef.current.focus();
    const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); onClose(); } };
    document.addEventListener("keydown", onKey, true);
    return () => { ab.abort(); document.removeEventListener("keydown", onKey, true); };
  }, [ownerKey, personKey, onClose]);
  if (!personKey) return null;
  const p = d && d.person, docs = d && Array.isArray(d.documents) ? d.documents : [];
  return (
    <>
      <div className="pl-scrim" onClick={onClose} />
      <aside className="pl-drawer" role="dialog" aria-modal="true" aria-label={p && p.name ? "Documents of " + p.name : "Person details"}>
        <button type="button" className="pl-btn pl-x" ref={closeRef} onClick={onClose} aria-label="Close details">Close</button>
        {err ? <p className="pl-na">{err}</p> : !d ? <p className="pl-na">Loading…</p> : (
          <>
            <h3>{(p && p.name) || personKey}</h3>
            <p className="pl-sub">{p ? [p.party, p.status && p.status.replace("_", " "), p.first_year ? p.first_year + (p.last_year ? "-" + p.last_year : "") : ""].filter(Boolean).join(" · ") : ""}</p>
            {p && <div className="pl-stats"><span><b>{fmtNum(p.docs)}</b> docs</span><span><b>{fmtNum(p.tokens)}</b> tokens</span><span><b><Corrupt n={p.corruption_docs} /></b> corruption</span></div>}
            {p && Array.isArray(p.positions) && p.positions.length > 0 && <p className="pl-sub">{p.positions.join(", ")}</p>}
            <h4>Documents ({docs.length})</h4>
            {docs.length === 0 && <p className="pl-na">No documents yet.</p>}
            {groupByTopic(docs).map(([topic, list]) => (
              <section key={topic} className="pl-tgroup" aria-label={topic + " documents"}>
                <h5><span className={"pl-topic t-" + topic}>{TOPIC_LABEL[topic] || topic}</span> {list.length}</h5>
                <ul className="pl-docs">
                  {list.map((x) => (
                    <li key={x.doc_id}>
                      {x.url ? <a href={x.url} target="_blank" rel="noopener noreferrer">{x.title || x.doc_id}</a> : <span>{x.title || x.doc_id}</span>}
                      <small>{x.source || ""} · {fmtNum(x.bytes)} B · {fmtAgo(x.collected_at)}{x.status === "in_kb" && <em className="pl-tag">already in the knowledge base</em>}</small>
                    </li>
                  ))}
                </ul>
              </section>
            ))}
          </>
        )}
      </aside>
    </>
  );
}
