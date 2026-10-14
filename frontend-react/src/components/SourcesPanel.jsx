// The "N sources" button content (stacked site icons) and the right-side drawer with one card per source.
import { useEffect, useRef, useState } from "react";
import { Icon } from "../icons";
import { sourceList } from "../utils";

/** A site's own favicon, or its first letter when the icon cannot be loaded. */
export function Favicon({ item }) {
  const [failed, setFailed] = useState(false);
  const dom = String(item.s.domain || new URL(item.href).hostname).replace(/^www\./, "");
  let src = null;
  try { src = new URL(item.href).origin + "/favicon.ico"; } catch (e) { /* letter fallback */ }
  return (
    <span className="fav">
      {src && !failed ? (
        <img src={src} alt="" referrerPolicy="no-referrer" loading="lazy" onError={() => setFailed(true)} />
      ) : (
        (dom[0] || "?").toUpperCase()
      )}
    </span>
  );
}

export function SourcesPanel({ view, onClose }) {
  const ref = useRef(null);
  const [flash, setFlash] = useState(null);
  const list = view ? sourceList(view.info) : [];
  const open = !!view && list.length > 0;

  useEffect(() => {
    if (!open || !view.focus) return undefined;
    const c = ref.current && ref.current.querySelector("#scard" + view.focus);
    if (!c) return undefined;
    c.scrollIntoView({ block: "center" });
    setFlash(view.focus);
    const t = setTimeout(() => setFlash(null), 500);
    return () => clearTimeout(t);
  }, [open, view]);

  return (
    <aside className={"srcpanel" + (open ? " show" : "")} id="srcPanel" aria-label="Sources" aria-hidden={!open} ref={ref}>
      {open && (
        <>
          <h3>
            {list.length + (list.length === 1 ? " source" : " sources")}
            <button className="x" type="button" aria-label="Close sources" onClick={onClose}>
              <Icon name="x" />
            </button>
          </h3>
          <div className="sub">
            {view.info.kb
              ? "Stored documents used for this answer. Open one to read the original. The small model's wording can still be wrong."
              : "Pages found by the web search. Open one to read it. The small model's wording can still be wrong."}
          </div>
          {list.map((it, i) => (
            <a
              key={i}
              className={"scard" + (flash === i + 1 ? " flash" : "")}
              id={"scard" + (i + 1)}
              href={it.href}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={"Source " + (i + 1) + ": " + (it.s.title || it.s.domain) + ", opens in a new tab"}
            >
              <div className="top">
                <Favicon item={it} />
                <span className="dom">{String(it.s.domain || "").replace(/^www\./, "").slice(0, 60)}</span>
                <span className="num">{i + 1}</span>
              </div>
              <div className="ttl">{String(it.s.title || it.s.domain || it.href).slice(0, 160)}</div>
              {it.s.snippet && <div className="snip">{String(it.s.snippet).slice(0, 300)}</div>}
              <div className="go">
                <Icon name="link" />
                Open source
              </div>
            </a>
          ))}
        </>
      )}
    </aside>
  );
}
