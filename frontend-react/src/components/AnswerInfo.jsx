// Under each answer: where it came from, the stored facts (if any), and a note when no sources could be shown.
import { originInfo, safeHttpUrl, sourceList } from "../utils";

function Facts({ info }) {
  return (
    <section className="facts" aria-label="Verified facts from the knowledge graph">
      <div className="fhead">Facts from the knowledge graph</div>
      <div className="fnote">Read these first: they are stored facts with their source and status. The model's wording above may differ and can be wrong.</div>
      <ul>
        {info.facts.slice(0, 6).map((f, i) => {
          const lv = String(f.level || "");
          const href = safeHttpUrl(f.source && f.source.url);
          return (
            <li key={i}>
              <span className={"lvl " + lv.replace(/[^a-z_]/g, "")}>{lv.replace("_", " ")}</span>
              {String(f.text || "").slice(0, 300)}
              {href && (
                <a href={href} target="_blank" rel="noopener noreferrer">
                  {"[" + String((f.source && f.source.label) || "source").slice(0, 40) + "]"}
                </a>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

export function AnswerInfo({ meta }) {
  if (!meta) return null;
  const info = meta.search;
  const o = originInfo(meta);
  const showSearch = info && info.status && info.status !== "skipped";
  const hasSourceCards = showSearch && info.searched && sourceList(info).length > 0;     // those live in the "N sources" button
  return (
    <div>
      {o && (
        <div className={"origin " + o.cls}>
          <b>{o.label}</b>
          {o.text}
        </div>
      )}
      {showSearch && info.kb && (info.facts || []).length > 0 && <Facts info={info} />}
      {showSearch && !hasSourceCards && (
        <section className="sources warn" aria-label="Web sources">
          <div className="snote">{info.message || "This answer was not checked against the web."}</div>
        </section>
      )}
    </div>
  );
}
