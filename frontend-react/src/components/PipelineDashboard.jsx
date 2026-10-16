// Owner-only data collection dashboard: one Start / Stop control and the whole pipeline on one page.
import { useCallback, useState } from "react";
import { usePipelineStatus } from "../hooks/usePipelineStatus";
import { NA, fmtBytes, fmtDuration, fmtNum, isNum, pct, runState } from "../pipeline/format";
import { PipelineFlow, MovementFeed } from "./PipelineFlow";
import { TargetedPanel } from "./TargetedPanel";
import { Overview, PersonDrawer, Roster } from "./PeoplePanel";

const LABEL = { idle: "Idle", running: "Collecting", stopping: "Stopping", error: "Error" };

function Summary({ status }) {
  const s = (status && status.stages) || {}, c = s.collector, q = s.corpus, r = s.raw;
  const tp = q ? pct(q.clean_tokens, isNum(q.target_tokens) ? q.target_tokens : 1e9) : null;
  const cells = [
    ["Documents / min", c && isNum(c.per_minute) ? c.per_minute.toFixed(1) : NA],
    ["Collected", c ? fmtBytes(c.bytes) : NA],
    ["Failures", c ? fmtNum(c.failed) : NA],
    ["Elapsed", status ? fmtDuration(status.elapsed_s) : NA],
    ["Tokens vs 1 B target", q ? fmtNum(q.clean_tokens) + (tp === null ? "" : " (" + tp.toFixed(2) + "%)") : NA],
    ["Free disk", r ? fmtBytes(r.free_disk_bytes) : NA],
  ];
  return (
    <dl className="pl-summary" aria-label="Summary">
      {cells.map(([k, v]) => <div key={k}><dt>{k}</dt><dd className={v === NA ? "na" : ""}>{v}</dd></div>)}
    </dl>
  );
}

export function PipelineDashboard({ ownerKey, onClose }) {
  const { status, rates, error, busy, start, stop, targeted } = usePipelineStatus(ownerKey);
  const [person, setPerson] = useState(null);
  const [rosterOpen, setRosterOpen] = useState(false);
  const closeDrawer = useCallback(() => setPerson(null), []);
  const state = runState(status);
  const running = state === "running", stopping = state === "stopping";
  const scope = status && status.scope;
  const msg = stopping ? "Stopping, finishing the current document" : (status && status.message) || (status ? "" : "Connecting to the collector…");
  const refreshKey = status && status.people ? (status.people.done || 0) + "/" + (status.people.failed || 0) : "";
  return (
    <section className={"pl-page st-" + state} id="pipelinePage" aria-label="Data collection dashboard">
      <header className="pl-top">
        <div>
          <h2>Data collection</h2>
          <p className="pl-sub">{scope ? "Tamil Nadu politicians, " + scope.from_year + " to " + scope.to_year + ". One person at a time, then the next." : "Tamil Nadu politicians, one person at a time."}</p>
        </div>
        <div className="pl-ctrl">
          <span className={"pl-state " + state}><i aria-hidden="true" />{LABEL[state]}</span>
          {running || stopping ? (
            <button type="button" className="pl-go stop" onClick={stop} disabled={!!busy || stopping} aria-busy={busy === "stop"}>
              {busy === "stop" || stopping ? "Stopping…" : "Stop collecting"}
            </button>
          ) : (
            <button type="button" className="pl-go start" onClick={start} disabled={!!busy || !status} aria-busy={busy === "start"}>
              {busy === "start" ? "Starting…" : "Start collecting"}
            </button>
          )}
          <button type="button" className="pl-btn" onClick={onClose}>Back to chat</button>
        </div>
      </header>
      <div className="sr-only" role="status" aria-live="polite">{LABEL[state]}. {msg}</div>
      {msg && <p className={"pl-msg" + (stopping ? " warn" : "")}>{msg}</p>}
      {error && <p className="pl-msg err" role="alert">{error}{status ? " (showing the last known numbers)" : ""}</p>}
      <TargetedPanel running={!!status && (status.running === true || running || stopping)} busy={!!busy} onSubmit={targeted} />
      <Summary status={status} />
      <PipelineFlow status={status} rates={rates} />
      <div className="pl-grid">
        <MovementFeed items={status ? status.movement : undefined} />
        <section className="pl-panel" aria-label="Politician by politician">
          <h3 className="pl-h">Politician by politician</h3>
          <Overview people={status ? status.people : null} onOpen={setPerson} />
        </section>
      </div>
      <details className="pl-panel pl-rosterbox" open={rosterOpen} onToggle={(e) => setRosterOpen(e.currentTarget.open)}>
        <summary className="pl-h">Everyone, person-wise and corruption-wise</summary>
        {rosterOpen && <Roster ownerKey={ownerKey} onOpen={setPerson} refreshKey={refreshKey} />}
      </details>
      <PersonDrawer ownerKey={ownerKey} personKey={person} onClose={closeDrawer} />
    </section>
  );
}
