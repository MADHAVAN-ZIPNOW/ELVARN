// Small menu to pick PDF / Word / Text, placed next to the button that opened it.
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Icon } from "../icons";

const FORMATS = [["pdf", "PDF", ".pdf"], ["docx", "Word", ".docx"], ["txt", "Text", ".txt"]];

export function DownloadMenu({ menu, onPick, onClose }) {
  const ref = useRef(null);
  const [pos, setPos] = useState({ left: 0, top: 0 });

  useLayoutEffect(() => {
    if (!menu || !ref.current) return;
    const r = menu.anchor.getBoundingClientRect(), mw = ref.current.offsetWidth, mh = ref.current.offsetHeight;
    setPos({
      left: Math.max(8, Math.min(r.left, innerWidth - mw - 8)),
      top: r.bottom + mh + 12 > innerHeight ? Math.max(8, r.top - mh - 8) : r.bottom + 8,
    });
    ref.current.querySelector("button").focus();
  }, [menu]);

  useEffect(() => {
    if (!menu) return undefined;
    const away = (e) => { if (!e.target.closest(".dlmenu") && !e.target.closest("[data-dl]")) onClose(); };
    const esc = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("click", away);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("click", away); document.removeEventListener("keydown", esc); };
  }, [menu, onClose]);

  if (!menu) return null;
  return (
    <div className="dlmenu" role="menu" ref={ref} style={{ left: pos.left, top: pos.top }}>
      {FORMATS.map(([fmt, label, ext]) => (
        <button key={fmt} type="button" role="menuitem" onClick={(e) => { e.stopPropagation(); onPick(fmt); }}>
          <Icon name="download" />
          {label}
          <small>{ext}</small>
        </button>
      ))}
    </div>
  );
}
