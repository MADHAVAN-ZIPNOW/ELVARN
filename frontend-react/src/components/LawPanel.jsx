// "Law judgments": every court judgment in India (Supreme Court and all High Courts), not only Tamil Nadu politicians' cases.
// The settings are sent with Start; the numbers are the real ones from the status (documents stored, disk used against the budget, busiest courts).
import { useState } from "react";
import { NA, fmtBytes, fmtNum, isNum, pct } from "../pipeline/format";

const SCOPES = [["all", "All of India"], ["sc", "Supreme Court only"], ["hc", "High Courts only"]];
const TOPICS = [["election", "Election, MLA, MP"], ["constitutional", "Constitutional"], ["criminal", "Criminal"], ["corruption", "Corruption"], ["tax", "Tax"], ["service", "Service"],
  ["property", "Property"], ["family", "Family"], ["company", "Company"], ["labour", "Labour"], ["motor", "Motor accident"], ["environment", "Environment"]];

export function LawPanel({ status, running, offline, busy, onStart, onStop }) {
  const [scope, setScope] = useState("all");
  const [gb, setGb] = useState(40);
  const [from, setFrom] = useState(1950), [to, setTo] = useState(2026);
  const [topics, setTopics] = useState([]);
  const law = status && status.law;
  const mine = !!law && law.running;                                                    // this kind of collection is the one running
  const other = running && !mine;                                                       // the politician collector is running instead
  const used = law && isNum(law.bytes) ? law.bytes : null;
  const budget = law && isNum(law.budget_bytes) ? law.budget_bytes : gb * 1024 ** 3;
  const p = used === null ? null : pct(used, budget);
  const toggle = (id) => setTopics((t) => (t.includes(id) ? t.filter((x) => x !== id) : [...t, id]));
  const disabled = busy || offline || running;
  const go = () => onStart({ mode: "judgments", scope, max_gb: Number(gb) || 40, from_year: Number(from) || 1950, to_year: Number(to) || 2026, topics });
  const courts = law && Array.isArray(law.courts) ? law.courts : [];
  return (
    <div className="pl-lawpage">
      <section className="pl-panel pl-lawset" aria-label="What to collect">
        <h3 className="pl-h">All-India judgments <small>every court, every case type</small></h3>
        <div className="pl-chips" role="group" aria-label="Which courts">
          {SCOPES.map(([id, label]) => <button key={id} type="button" className={"pl-srcchip" + (scope === id ? " on" : "")} aria-pressed={scope === id} disabled={disabled} onClick={() => setScope(id)}>{label}</button>)}
        </div>
        <div className="pl-lawrow">
          <label className="fld">From year<input type="number" min="1950" max="2026" value={from} disabled={disabled} onChange={(e) => setFrom(e.target.value)} /></label>
          <label className="fld">To year<input type="number" min="1950" max="2026" value={to} disabled={disabled} onChange={(e) => setTo(e.target.value)} /></label>
          <label className="fld">Disk budget (GB)<input type="number" min="1" max="500" value={gb} disabled={disabled} onChange={(e) => setGb(e.target.value)} /></label>
        </div>
        <div className="pl-chips" role="group" aria-label="Only these subjects (none = everything)">
          {TOPICS.map(([id, label]) => <button key={id} type="button" className={"pl-srcchip" + (topics.includes(id) ? " on" : "")} aria-pressed={topics.includes(id)} disabled={disabled} onClick={() => toggle(id)}>{label}</button>)}
        </div>
        <p className="pl-sub">{topics.length ? "Only judgments about the chosen subjects are kept." : "No subject chosen: every judgment is kept."} Each one is saved with its case name, number, court, year, judges, outcome and the Acts it cites.</p>
        <div className="pl-lawgo">
          {mine ? <button type="button" className="pl-go stop" disabled={busy} onClick={onStop}>Stop judgments</button>
            : <button type="button" className="pl-go start" disabled={disabled} onClick={go}>Start judgments</button>}
          {other && <span className="pl-msg warn">The politician collection is running. Stop it first.</span>}
          {offline && <span className="pl-msg warn">Your collection computer is offline.</span>}
        </div>
      </section>
      <section className="pl-panel pl-lawlive" aria-label="Judgments collected">
        <h3 className="pl-h">Collected so far <small>{mine ? "running now" : "stopped"}</small></h3>
        <dl className="pl-summary pl-lawsum">
          <div><dt>Judgments</dt><dd>{law ? fmtNum(law.documents) : NA}</dd></div>
          <div><dt>Disk used</dt><dd>{used === null ? NA : fmtBytes(used)}</dd></div>
          <div><dt>Per minute</dt><dd>{law && isNum(law.per_minute) ? law.per_minute.toFixed(1) : NA}</dd></div>
        </dl>
        <div className="pl-bar">
          <div className="pl-barhead"><span>Disk budget</span><b>{p === null ? NA : fmtBytes(used) + " of " + fmtBytes(budget)}</b></div>
          <div className="pl-meter" role="progressbar" aria-label="Disk budget" aria-valuemin={0} aria-valuemax={100} aria-valuenow={p === null ? undefined : Math.round(p)}><span style={{ width: (p || 0) + "%" }} /></div>
        </div>
        <p className="pl-nowmsg">{mine && status ? status.message : "Pick the courts, years and subjects, then press Start judgments."}</p>
        <div className="pl-lawcourts" aria-label="Busiest courts">
          {courts.length ? courts.map((c) => <div key={c.court}><span>{c.court}</span><b>{fmtNum(c.documents)}</b></div>) : <span className="pl-sub">No judgments stored yet.</span>}
        </div>
      </section>
    </div>
  );
}
