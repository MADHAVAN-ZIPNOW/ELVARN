// Readable answer text: paragraphs, lists, bold, code, links and [n] citation marks.
// Built from React elements, never from HTML strings, so answer text can never inject markup.
import { safeHttpUrl } from "../utils";

const INLINE = /(\*\*[^*\n]+\*\*|`[^`\n]+`|https?:\/\/[^\s<>)\]]+|\[\d{1,2}\])/g;

function inline(text, onCite, keyBase) {
  const out = [];
  let last = 0, m, n = 0;
  INLINE.lastIndex = 0;
  while ((m = INLINE.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0], key = keyBase + "-" + n++;
    if (t.startsWith("**")) {
      out.push(<strong key={key}>{t.slice(2, -2)}</strong>);
    } else if (t[0] === "`") {
      out.push(<code key={key}>{t.slice(1, -1)}</code>);
    } else if (t[0] === "[") {
      const num = +t.slice(1, -1);
      out.push(
        <sup
          key={key}
          className="cite"
          role="button"
          tabIndex={0}
          title={"Show source " + num}
          onClick={onCite ? () => onCite(num) : undefined}
          onKeyDown={onCite ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onCite(num); } } : undefined}
        >
          {num}
        </sup>
      );
    } else {
      const tail = t.match(/[.,;:!?]+$/);
      const url = tail ? t.slice(0, -tail[0].length) : t;
      const href = safeHttpUrl(url);
      if (href) {
        out.push(<a key={key} href={href} target="_blank" rel="noopener noreferrer">{url}</a>);
        if (tail) out.push(tail[0]);
      } else {
        out.push(t);
      }
    }
    last = m.index + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

/** Split text into blocks: { type: "p" | "h" | "ul" | "ol", text | items }. */
export function parseBlocks(text) {
  const blocks = [];
  let para = [], list = null;
  const flushPara = () => { if (para.length) { blocks.push({ type: "p", text: para.join(" ") }); para = []; } };
  String(text || "").replace(/\r/g, "").split("\n").forEach((raw) => {
    const line = raw.trim();
    let mm;
    if (!line) { flushPara(); list = null; return; }
    if ((mm = line.match(/^#{1,4}\s+(.*)$/))) { flushPara(); list = null; blocks.push({ type: "h", text: mm[1] }); return; }
    if ((mm = line.match(/^[-*•]\s+(.*)$/))) {
      flushPara();
      if (!list || list.type !== "ul") { list = { type: "ul", items: [] }; blocks.push(list); }
      list.items.push(mm[1]);
      return;
    }
    if ((mm = line.match(/^\d{1,2}[.)]\s+(.*)$/))) {
      flushPara();
      if (!list || list.type !== "ol") { list = { type: "ol", items: [] }; blocks.push(list); }
      list.items.push(mm[1]);
      return;
    }
    list = null;
    para.push(line);
  });
  flushPara();
  return blocks;
}

export function RichText({ text, onCite }) {
  const blocks = parseBlocks(text);
  if (!blocks.length) return <>{String(text || "")}</>;
  return (
    <>
      {blocks.map((b, i) => {
        const k = "b" + i;
        if (b.type === "p") return <p key={k}>{inline(b.text, onCite, k)}</p>;
        if (b.type === "h") return <h4 key={k}>{inline(b.text, onCite, k)}</h4>;
        const Tag = b.type;
        return (
          <Tag key={k}>
            {b.items.map((it, j) => (
              <li key={j}>{inline(it, onCite, k + "-" + j)}</li>
            ))}
          </Tag>
        );
      })}
    </>
  );
}
