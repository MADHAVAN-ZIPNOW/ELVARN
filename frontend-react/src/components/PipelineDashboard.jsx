// Owner-only data collection dashboard. Four pages (tabs), each of which fits on one screen: nothing on this dashboard needs a scrollbar.
//   Pipeline  : who is being collected right now, the seven stages in one horizontal row, the totals
//   Collect one person : type "collect C. Joseph Vijay ED cases"
//   People    : progress, recently finished, up next, and the roster in pages
//   On the move : documents travelling through the stages, in pages
import { useCallback, useEffect, useState } from "react";
import { usePipelineStatus } from "../hooks/usePipelineStatus";
import { NA as NA_TEXT, fmtBytes, fmtDuration, fmtNum, isNum, pct, runState } from "../pipeline/format";
import { PipelineFlow, MovementFeed } from "./PipelineFlow";
import { TargetedPanel } from "./TargetedPanel";
import { Lists, NowStrip, PersonDrawer, Progress, Roster } from "./PeoplePanel";

const LABEL = { idle: "Idle", running: "Collecting", stopping: "Stopping", error: "Error", offline: "Computer offline" };
const TABS = [["pipeline", "Pipeline"], ["person", "Collect one person"], ["people", "People"], ["move", "On the move"]];
const TAB_KEY = "elvarn.collectTab";

function readTab() {
  try { const t = localStorage.getItem(TAB_KEY); return TABS.some(([id]) => id === t) ? t : "pipeline"; } catch (e) { return "pipeline"; }
}

function Totals({ status }) {
  const s = (status && status.stages) || {}, c = s.collector, q = s.corpus, r = s.raw;
  const NA = status ? NA_TEXT : "…";                                                // the first numbers are on their way
  const cells = [
    ["Documents per minute", c && isNum(c.per_minute) ? c.per_minute.toFixed(1) : NA],
    ["Collected", c ? fmtBytes(c.bytes) : NA],
    ["Failures", c ? fmtNum(c.failed) : NA],
    ["Elapsed", status && isNum(status.elapsed_s) ? fmtDuration(status.elapsed_s) : NA],
    ["Tokens vs 1B target", q && isNum(q.clean_tokens) ? fmtNum(q.clean_tokens) + " (" + (pct(q.clean_tokens, isNum(q.target_tokens) ? q.target_tokens : 1e9) || 0).toFixed(2) + "%)" : NA],
    ["Free disk", r ? fmtBytes(r.free_disk_bytes) : NA],
  ];
  return (
    <dl className="pl-summary" aria-label="Totals">
      {cells.map(([k, v]) => <div key={k}><dt>{k}</dt><dd className={v === NA ? "na" : ""}>{v}</dd></div>)}
    </dl>
  );
}

export function PipelineDashboard({ ownerKey, onClose }) {
  const { status, rates, error, busy, start, stop, targeted } = usePipelineStatus(ownerKey);
  const [tab, setTabState] = useState(readTab);
  const [person, setPerson] = useState(null);
  const setTab = useCallback((t) => { setTabState(t); try { localStorage.setItem(TAB_KEY, t); } catch (e) { /* private mode: the tab is simply not remembered */ } }, []);
  const closeDrawer = useCallback(() => setPerson(null), []);
  const state = runState(status);
  const running = state === "running", stopping = state === "stopping", offline = state === "offline";      // offline: the deployed site cannot reach the collection computer
  const scope = status && status.scope;
  const people = status ? status.people : null;
  const msg = stopping ? "Stopping, finishing the current document" : (status && status.message) || (status ? "" : "Connecting to the collector…");
  const refreshKey = people ? (people.done || 0) + "/" + (people.failed || 0) : "";
  useEffect(() => { if (running && tab === "person") setTab("pipeline"); }, [running]);          // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section className={"pl-page st-" + state} id="pipelinePage" aria-label="Data collection dashboard">
      <div className="pl-top">
        <div>
          <h2>Data collection</h2>
          <p className="pl-sub">{scope ? "Tamil Nadu politicians, " + scope.from_year + " to " + scope.to_year + ". One person at a time." : "Tamil Nadu politicians, one person at a time."}</p>
        </div>
        <div className="pl-ctrl">
          <span className={"pl-state " + state}><i aria-hidden="true" />{LABEL[state]}</span>
          {running || stopping ? (
            <button type="button" className="pl-go stop" onClick={stop} disabled={!!busy || stopping} aria-busy={busy === "stop"}>
              {busy === "stop" || stopping ? "Stopping…" : "Stop collecting"}
            </button>
          ) : (
            <button type="button" className="pl-go start" onClick={start} disabled={!!busy || !status || offline} aria-busy={busy === "start"}>
              {busy === "start" ? "Starting…" : "Start collecting"}
            </button>
          )}
          <button type="button" className="pl-btn" onClick={onClose}>Back to chat</button>
        </div>
      </div>
      <div className="sr-only" role="status" aria-live="polite">{LABEL[state]}. {msg}</div>
      {error && <p className="pl-msg err" role="alert">{error}{status ? " (showing the last known numbers)" : ""}</p>}

      <div className="pl-tabs" role="tablist" aria-label="Dashboard pages">
        {TABS.map(([id, label]) => (
          <button key={id} type="button" role="tab" id={"pltab-" + id} aria-selected={tab === id} aria-controls="pltabbody" className={"pl-tab" + (tab === id ? " on" : "")}
            onClick={() => setTab(id)}>{label}</button>
        ))}
      </div>

      <div className="pl-tabbody" id="pltabbody" role="tabpanel" aria-labelledby={"pltab-" + tab}>
        {tab === "pipeline" && (
          <>
            <NowStrip current={people ? people.current : null} state={state} message={msg} onOpen={setPerson} />
            <PipelineFlow status={status} rates={rates} />
            <Totals status={status} />
            <MovementFeed items={status ? status.movement : undefined} compact />
          </>
        )}
        {tab === "person" && <TargetedPanel running={!!status && (status.running === true || running || stopping)} offline={offline} busy={!!busy} onSubmit={targeted} />}
        {tab === "people" && (
          <div className="pl-people">
            <aside className="pl-panel pl-side" aria-label="Progress">
              <Progress people={people} />
              <Lists people={people} onOpen={setPerson} />
            </aside>
            <section className="pl-panel pl-rosterpanel" aria-label="Everyone, person-wise and corruption-wise">
              <Roster ownerKey={ownerKey} onOpen={setPerson} refreshKey={refreshKey} />
            </section>
          </div>
        )}
        {tab === "move" && <MovementFeed items={status ? status.movement : undefined} />}
      </div>
      <PersonDrawer ownerKey={ownerKey} personKey={person} onClose={closeDrawer} />
    </section>
  );
}
