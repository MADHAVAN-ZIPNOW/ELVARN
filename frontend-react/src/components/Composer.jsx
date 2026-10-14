// The message box: attach button, text area, mic and the rocket Send button, plus the attached-file bar and the live voice caption.
import { useEffect, useRef } from "react";
import { Icon, SpanIcon } from "../icons";
import { FILE_LABEL } from "../constants";
import { FileChip } from "./FileParts";

const DEFAULT_PLACEHOLDER = "Type, or tap the mic and speak…";

function FileBar({ pending, onRemove, onCancelUpload, onOpen, onConvert }) {
  if (!pending) return <div className="filebar" id="fileBar" hidden />;
  if (pending.uploading) {
    return (
      <div className="filebar" id="fileBar">
        <span className="fchip busy">
          <span className="fname">{pending.name}</span>
          <span className="fmeta">{"Uploading " + (pending.pct || 0) + "%"}</span>
          <button type="button" aria-label="Cancel upload" onClick={onCancelUpload}><Icon name="x" /></button>
        </span>
      </div>
    );
  }
  const targets = pending.supported_targets || [];
  return (
    <div className="filebar" id="fileBar">
      <FileChip file={{ name: pending.name, size: pending.size, type: pending.detected_type }} onRemove={onRemove}
        onOpen={(opener) => onOpen({
          id: pending.file_id, name: pending.name, size: pending.size, downloadId: null,
          label: FILE_LABEL[pending.detected_type] || String(pending.detected_type || "").toUpperCase(),
        }, opener)} />
      {targets.length > 0 && <span className="qlabel">Convert to:</span>}
      {targets.map((k) => (
        <button key={k} className="qchip" type="button" aria-label={"Convert " + pending.name + " to " + (FILE_LABEL[k] || k)} onClick={() => onConvert(k)}>
          {FILE_LABEL[k] || k}
        </button>
      ))}
    </div>
  );
}

export function Composer({
  hidden, input, setInput, busy, pending, accept, live, voice, micSupported,
  onSend, onStop, onMic, onPickFile, onRemoveFile, onCancelUpload, onOpenFile, onConvert, inputRef,
}) {
  const fileRef = useRef(null);
  const ready = !busy && input.trim().length > 0;

  // the text area grows with what you type, up to 150 px
  useEffect(() => {
    const t = inputRef.current;
    if (!t) return;
    t.style.height = "auto";
    t.style.height = Math.min(t.scrollHeight, 150) + "px";
    t.style.overflowY = t.scrollHeight > 150 ? "auto" : "hidden";
  }, [input, inputRef]);

  const placeholder = pending && !pending.uploading && pending.file_id ? "What should I do with it? e.g. convert to PDF" : DEFAULT_PLACEHOLDER;
  const sendTitle = busy ? "Stop generating" : "Send";
  return (
    <>
      <div className="notice" hidden={hidden}>
        ELVARN is a small model trained on a limited dataset. Replies are generated text and can be wrong or incomplete — they are not verified facts, even when web sources are shown.
      </div>
      <div className="composer" hidden={hidden}>
        <div className="livecap" id="liveCap" hidden={!live.show} aria-live="polite">
          <span className="wave"><i /><i /><i /><i /></span>
          <span id="liveText" className={live.idle ? "idle" : undefined}>{live.text || "Listening… start speaking"}</span>
        </div>
        <input type="file" id="fileInput" hidden tabIndex={-1} aria-label="Choose a file to upload" accept={accept} ref={fileRef}
          onChange={(e) => { const f = e.target.files && e.target.files[0]; e.target.value = ""; if (f) onPickFile(f); }} />
        <div className="box">
          <FileBar pending={pending} onRemove={onRemoveFile} onCancelUpload={onCancelUpload} onOpen={onOpenFile} onConvert={onConvert} />
          <div className="boxrow">
            <button className={"attach" + (pending && !pending.uploading ? " has" : "")} id="attachBtn" type="button"
              title="Attach a file (txt, md, csv, json, html, docx)" aria-label="Attach a file" onClick={() => fileRef.current.click()}>
              <SpanIcon name="upload" />
            </button>
            <textarea id="input" rows={1} placeholder={placeholder} aria-label="Message" ref={inputRef} value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); onSend(); } }} />
            <button className={"mic" + (voice.on ? " on" : "")} id="micBtn" type="button" onClick={onMic}
              title={micSupported ? "Speak your message" : "Voice input is not supported in this browser (use Chrome or Edge)"}
              aria-label={voice.on ? "Stop voice input" : "Start voice input"} style={micSupported ? undefined : { opacity: 0.45 }}>
              <SpanIcon name="mic" />
            </button>
            <button className={"send" + (busy ? " stopping" : "") + (ready ? " ready" : "")} id="sendBtn" type="button" title={sendTitle}
              aria-label={sendTitle} onClick={busy ? onStop : onSend}>
              <Icon name={busy ? "stop" : "rocket"} />
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
