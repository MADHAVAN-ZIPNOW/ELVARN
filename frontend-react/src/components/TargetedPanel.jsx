// "Collect for one person": type a request, pick the sources, and only that person is collected from only those sources.
import { useState } from "react";

const CHIPS = [["all", "All"], ["wikipedia", "Wikipedia"], ["wikidata", "Wikidata"], ["articles", "Case articles"], ["courts", "Courts"], ["ed", "ED"], ["cbi", "CBI"], ["dvac", "DVAC"], ["news", "News"]];
const EXAMPLES = ["collect C. Joseph Vijay ED cases", "collect Karunanidhi Wikipedia", "collect J. Jayalalithaa CBI and court cases"];

export function TargetedPanel({ running, busy, onSubmit }) {
  const [text, setText] = useState("");
  const [picked, setPicked] = useState([]);
  const [result, setResult] = useState(null);
  const [wait, setWait] = useState(false);
  const disabled = running || busy || wait;

  const send = async (body) => {
    setWait(true);
    const rq = await onSubmit(body);
    setResult(rq);
    setWait(false);
  };
  const submit = (e) => {
    if (e) e.preventDefault();
    if (disabled || !text.trim()) return;
    send(picked.length ? { text: text.trim(), sources: picked } : { text: text.trim() });
  };
  const toggle = (id) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));
  const choose = (c) => send({ person: c.key, sources: (result && result.sources) || picked });

  const cands = result && result.needs_choice && Array.isArray(result.candidates) ? result.candidates : [];
  let line = "";
  if (result) {
    if (result.ok) {
      const src = Array.isArray(result.sources) && result.sources.length ? result.sources.map((s) => String(s).toUpperCase()).join(", ") + " records" : "records";
      line = "Collecting " + src + " for " + ((result.person && result.person.name) || "the chosen person");
    } else line = result.message || "Nothing to collect for this request.";
  }
  return (
    <section className="pl-panel pl-target" aria-label="Collect for one person">
      <h3 className="pl-h">Collect for one person <small>only that person, only those sources</small></h3>
      <form className="pl-tform" onSubmit={submit}>
        <input type="text" className="pl-tinput" value={text} disabled={disabled} onChange={(e) => setText(e.target.value)}
          placeholder="collect C. Joseph Vijay ED cases" aria-label="Collection request" maxLength={200} />
        <button type="submit" className="pl-go start pl-tgo" disabled={disabled || !text.trim()} aria-busy={wait}>{wait ? "Working…" : "Collect"}</button>
      </form>
      <div className="pl-chips" role="group" aria-label="Sources (optional, overrides the sentence)">
        {CHIPS.map(([id, label]) => (
          <button key={id} type="button" className={"pl-srcchip" + (picked.includes(id) ? " on" : "")} aria-pressed={picked.includes(id)} disabled={disabled} onClick={() => toggle(id)}>{label}</button>
        ))}
      </div>
      <div className="pl-ex"><span className="pl-k">Try</span>
        {EXAMPLES.map((x) => <button key={x} type="button" className="pl-example" disabled={disabled} onClick={() => setText(x)}>{x}</button>)}
      </div>
      {running && <p className="pl-msg warn">A collection is running. Stop it first.</p>}
      <div className={"pl-tmsg" + (result && !result.ok && !result.needs_choice ? " bad" : "")} role="status" aria-live="polite">{line}</div>
      {cands.length > 0 && (
        <ul className="pl-cands" aria-label="Which person do you mean?">
          {cands.map((c) => (
            <li key={c.key}>
              <button type="button" className="pl-cand" disabled={disabled} onClick={() => choose(c)}>
                <b>{c.name || c.key}</b>
                <span>{[c.party, c.first_year ? c.first_year + (c.last_year && c.last_year !== c.first_year ? "-" + c.last_year : "") : "", c.description].filter(Boolean).join(" · ")}</span>
                {c.new && <em className="pl-tag">from Wikidata</em>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
