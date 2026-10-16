// The pipeline picture: stage cards joined by animated connectors, plus the movement feed.
import { NA, PIPE_LABEL, buildStages, flowLook, fmtAgo, isNum } from "../pipeline/format";

const GLYPH = {
  collector: <><circle cx="12" cy="12" r="8" fill="url(#g3Navy)" /><path d="M12 7v6l4 2" stroke="url(#g3Ivory)" strokeWidth="2" fill="none" strokeLinecap="round" /></>,
  raw: <><path d="M3 8l9-4 9 4v9l-9 4-9-4z" fill="url(#g3Terra)" /><path d="M3 8l9 4 9-4M12 12v9" stroke="url(#g3Ivory)" strokeWidth="1.4" fill="none" /></>,
  kafka: <><circle cx="6" cy="12" r="3" fill="url(#g3Emerald)" /><circle cx="18" cy="6" r="3" fill="url(#g3Emerald)" /><circle cx="18" cy="18" r="3" fill="url(#g3Emerald)" /><path d="M8.6 10.6L15.4 7.4M8.6 13.4l6.8 3.2" stroke="url(#g3Ivory)" strokeWidth="1.6" fill="none" /></>,
  spark: <path d="M13 2L4 14h6l-1 8 9-12h-6z" fill="url(#g3Terra)" />,
  corpus: <><rect x="4" y="3" width="16" height="5" rx="2" fill="url(#g3Navy)" /><rect x="4" y="9.5" width="16" height="5" rx="2" fill="url(#g3Navy)" /><rect x="4" y="16" width="16" height="5" rx="2" fill="url(#g3Navy)" /></>,
  graph: <><circle cx="12" cy="5" r="3" fill="url(#g3Emerald)" /><circle cx="5" cy="18" r="3" fill="url(#g3Emerald)" /><circle cx="19" cy="18" r="3" fill="url(#g3Emerald)" /><path d="M11 7.6L6 15.4M13 7.6l5 7.8M8 18h8" stroke="url(#g3Ivory)" strokeWidth="1.4" fill="none" /></>,
  vector: <><path d="M12 3l9 5v8l-9 5-9-5V8z" fill="url(#g3Terra)" /><circle cx="12" cy="12" r="3" fill="url(#g3Ivory)" /></>,
};

function Connector({ rate, active, label }) {
  const { dots, dur } = flowLook(rate, active);
  return (
    <div className={"pl-conn" + (dots ? " live" : " still")} aria-hidden="true" title={label}>
      <span className="pl-track" />
      {Array.from({ length: dots }, (_, i) => (
        <i key={i} className="pl-dot" style={{ animationDuration: dur + "s", animationDelay: -(dur * i) / dots + "s" }} />
      ))}
      <span className="pl-arrow" />
    </div>
  );
}

function Stage({ st, rate, n }) {
  const down = st.up === false, unknown = st.up === null, idle = st.up === "idle";        // idle: the collector exists and is simply not running (not a fault)
  const state = unknown ? "unavailable" : idle ? "idle" : down ? "down" : "up";
  return (
    <article className={"pl-card tone-" + st.tone + " " + state} aria-label={st.title + ": " + state}>
      <div className="pl-chead">
        <svg viewBox="0 0 24 24" className="pl-glyph" aria-hidden="true">{GLYPH[st.icon]}</svg>
        <div className="pl-ctitle"><h3><span className="pl-step" aria-hidden="true">{n}</span>{st.title}</h3><small>{st.sub}</small></div>
        <span className={"pl-led " + state} role="img" aria-label={state === "up" ? "up" : state === "down" ? "down" : state === "idle" ? "idle" : NA} title={state} />
      </div>
      {st.rows ? (
        <dl className="pl-rows">
          {st.rows.map(([k, v]) => <div key={k}><dt>{k}</dt><dd className={v === NA ? "na" : ""}>{v}</dd></div>)}
        </dl>
      ) : <p className="pl-na">Numbers {NA}</p>}
      {st.bar && (
        <div className="pl-bar">
          <div className="pl-barhead"><span>{st.bar.label}</span>
            <b>{st.bar.pct === null ? NA : st.bar.value !== null ? st.bar.pct.toFixed(st.bar.pct < 10 ? 2 : 1) + "%" : st.bar.pct.toFixed(0) + "%"}</b></div>
          <div className="pl-meter" role="progressbar" aria-label={st.bar.label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={st.bar.pct === null ? undefined : Math.round(st.bar.pct)}>
            <span style={{ width: (st.bar.pct || 0) + "%" }} />
          </div>
        </div>
      )}
      {st.note ? <p className="pl-note" title={st.note}>{st.note}</p> : null}
      {isNum(rate) && <p className="pl-rate">{(rate < 10 ? rate.toFixed(1) : Math.round(rate)) + " / min"}</p>}
    </article>
  );
}

export function PipelineFlow({ status, rates }) {
  const stages = buildStages(status);
  const live = !!status && (status.state === "running" || status.state === "stopping");
  return (
    <section className="pl-flow" aria-label="Pipeline stages">
      {stages.map((st, i) => {
        const prev = stages[i - 1];
        const ok = live && st.up !== false && (!prev || prev.up !== false);
        return (
          <div className="pl-cell" key={st.id}>
            {prev && <Connector rate={rates[st.id]} active={ok} label={prev.title + " to " + st.title} />}
            <Stage st={st} rate={rates[st.id]} n={i + 1} />
          </div>
        );
      })}
    </section>
  );
}

export function MovementFeed({ items }) {
  const list = Array.isArray(items) ? items.slice(0, 20) : null;
  return (
    <section className="pl-panel pl-move" aria-label="Documents on the move">
      <h3 className="pl-h">On the move <small>newest first</small></h3>
      {list === null ? <p className="pl-na">Movement feed {NA}</p>
        : !list.length ? <p className="pl-na">Nothing has moved yet.</p>
        : (
          <ol className="pl-feed">
            {list.map((m, i) => (
              <li key={(m.doc_id || "") + (m.stage || "") + (m.at || "") + i} className={"pl-item st-" + m.stage}>
                <span className="pl-chip">{PIPE_LABEL[m.stage] || m.stage || "?"}</span>
                <span className="pl-ititle" title={m.title || m.doc_id}>{m.title || m.doc_id || "document"}</span>
                <span className="pl-iwho">{m.person || ""}</span>
                <time className="pl-itime">{fmtAgo(m.at)}</time>
              </li>
            ))}
          </ol>
        )}
    </section>
  );
}
