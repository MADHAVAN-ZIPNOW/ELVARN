// The row of buttons under an answer (and the Copy button under your own question).
import { useEffect, useRef, useState } from "react";
import { Icon } from "../icons";
import { Favicon } from "./SourcesPanel";
import { copyPlain, sourceList } from "../utils";

/** A button whose label changes for a moment after it was used ("Copied"). */
function useFlash(ms = 1400) {
  const [flash, setFlash] = useState(null);
  const timer = useRef(null);
  useEffect(() => () => clearTimeout(timer.current), []);
  const show = (value) => {
    setFlash(value);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setFlash(null), ms);
  };
  return [flash, show];
}

export function CopyButton({ text }) {
  const [flash, show] = useFlash();
  const onClick = async () => show((await copyPlain(text)) ? "ok" : "fail");
  return (
    <button className="tool" type="button" aria-label="Copy the text" onClick={onClick}>
      <Icon name={flash === "ok" ? "check" : "copy"} />
      {flash === "ok" ? "Copied" : flash === "fail" ? "Copy failed" : "Copy"}
    </button>
  );
}

function ShareButton({ text }) {
  const [flash, show] = useFlash();
  const onClick = async () => {
    try {
      if (navigator.share) { await navigator.share({ text }); return; }
    } catch (e) {
      if (e && e.name === "AbortError") return;
    }
    if (await copyPlain(text)) show("ok");
  };
  return (
    <button className="tool" type="button" aria-label="Share this answer" onClick={onClick}>
      <Icon name={flash === "ok" ? "check" : "share"} />
      {flash === "ok" ? "Copied" : "Share"}
    </button>
  );
}

export function ActionBar({ text, meta, speaking, onSpeak, onDetails, onOpenSources, onFeedback, onDownload }) {
  const info = meta && meta.search;
  const list = sourceList(info);
  const vote = (meta && meta.feedback) || "";
  return (
    <div className="tools">
      {list.length > 0 && (
        <button className="tool srcchip" type="button" aria-label={"Show " + list.length + " sources"} onClick={() => onOpenSources(info)}>
          <span className="favstack">
            {list.slice(0, 3).map((x, i) => <Favicon key={i} item={x} />)}
          </span>
          {list.length + (list.length === 1 ? " source" : " sources")}
        </button>
      )}
      <CopyButton text={text} />
      <button className={"tool" + (vote === "up" ? " on" : "")} type="button" aria-label="Good answer" title="Good answer" aria-pressed={vote === "up"}
        onClick={() => onFeedback(vote === "up" ? "" : "up")}>
        <Icon name="thumbup" />
      </button>
      <button className={"tool" + (vote === "down" ? " on" : "")} type="button" aria-label="Bad answer" title="Bad answer" aria-pressed={vote === "down"}
        onClick={() => onFeedback(vote === "down" ? "" : "down")}>
        <Icon name="thumbdown" />
      </button>
      <ShareButton text={text} />
      <button className="tool" type="button" data-dl="1" aria-label="Download" title="Download as PDF, Word or text" aria-haspopup="menu"
        onClick={(e) => { e.stopPropagation(); onDownload(e.currentTarget); }}>
        <Icon name="download" />
        Download
      </button>
      <button className={"tool" + (speaking ? " speaking-btn" : "")} type="button" onClick={onSpeak}>
        <Icon name={speaking ? "stop" : "speaker"} />
        {speaking ? "Stop" : "Listen"}
      </button>
      <button className="tool" type="button" onClick={onDetails}>
        <Icon name="info" />
        Details
      </button>
      {meta && meta.generated_tokens != null && <span className="meta">{`${meta.generated_tokens} tokens · ${meta.seconds}s`}</span>}
    </div>
  );
}
