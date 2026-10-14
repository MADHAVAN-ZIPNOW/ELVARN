// File chip (attached / sent file), the converted-file card under an answer, and the right-side file viewer.
import { useEffect, useRef, useState } from "react";
import { Icon } from "../icons";
import { FILE_LABEL } from "../constants";
import { copyPlain, fmtSize } from "../utils";
import { getPreview, runDownload } from "../files";
import { useUi } from "../UiContext";

export function FileChip({ file, onRemove, onOpen }) {
  const body = (
    <>
      <Icon name="file" />
      <span className="fname">{file.name || "file"}</span>
      <span className="fmeta">{fmtSize(file.size || 0) + (file.type ? " · " + String(file.type).toUpperCase() : "")}</span>
    </>
  );
  return (
    <span className="fchip">
      {onOpen ? (
        <button className="fopen" type="button" aria-label={"View contents of " + (file.name || "file")} title="Click to view the file"
          onClick={(e) => onOpen(e.currentTarget)}>
          {body}
        </button>
      ) : (
        <span className="fopen">{body}</span>
      )}
      {onRemove && (
        <button type="button" aria-label={"Remove attached file " + (file.name || "")} onClick={onRemove}>
          <Icon name="x" />
        </button>
      )}
    </span>
  );
}

/** Copy the text of a stored file; the label shows "Copied" for a moment. */
function useCopyPreview(id, onGone) {
  const { announce } = useUi();
  const [label, setLabel] = useState(null);
  const [busy, setBusy] = useState(false);
  const timer = useRef(null);
  useEffect(() => () => clearTimeout(timer.current), []);
  const copy = async () => {
    setBusy(true);
    let msg = "";
    try {
      const d = await getPreview(id);
      const ok = await copyPlain(d.text);
      setLabel(ok ? "Copied" : "Copy failed");
      msg = ok
        ? (d.truncated ? "Copied the first part of the text. The file is longer than the preview limit." : "Copied the text to the clipboard.")
        : "Copy failed. Select the text and copy it by hand.";
    } catch (e) {
      setLabel("Unavailable");
      if (e && e.status === 404) { msg = "This file has expired. Upload it again."; if (onGone) onGone(); } else msg = "The text could not be loaded. Try again.";
    }
    announce(msg);
    timer.current = setTimeout(() => { setLabel(null); setBusy(false); }, 1800);
    return msg;
  };
  return { label, busy, copy };
}

export function DownloadCard({ file: f }) {
  const { announce, openFilePanel } = useUi();
  const cardRef = useRef(null);
  const [expired, setExpired] = useState(!!(f.expiresAt && Date.now() >= f.expiresAt));
  const [note, setNote] = useState(null);
  const [downloading, setDownloading] = useState(false);
  const { label: copyLabel, busy: copyBusy, copy } = useCopyPreview(f.download_id, () => setExpired(true));

  const label = f.label || FILE_LABEL[f.format] || String(f.format || "").toUpperCase();
  const textOnly = f.format === "docx" || f.format === "pdf";
  const mins = f.expiresAt ? Math.max(0, Math.round((f.expiresAt - Date.now()) / 60000)) : null;
  const expiryNote = expired
    ? "This link has expired. Upload the file again and ask for the conversion again."
    : "Download link expires in about " + (mins == null ? "an hour" : Math.max(1, mins) + " min") + ".";

  const open = () => openFilePanel({ id: f.download_id, name: f.name, label, size: f.size, downloadId: f.download_id }, cardRef.current);
  const download = async (e) => {
    e.stopPropagation();
    setDownloading(true);
    announce("Downloading " + f.name);
    try {
      const r = await runDownload(f.download_id, f.name);
      if (!r.ok) {
        if (r.status === 404) { setExpired(true); announce("That download has expired."); } else setNote("Download failed (" + r.status + "). Try again.");
      }
    } catch (err) {
      setNote("Download failed. Check your connection and try again.");
    }
    setDownloading(false);
  };

  return (
    <div className={"dlcard" + (expired ? " expired" : "")} role="group" tabIndex={0} ref={cardRef}
      aria-label={"Converted file " + (f.name || "") + ". Press Enter to view its contents."}
      onClick={(e) => { if (!e.target.closest("button")) open(); }}
      onKeyDown={(e) => { if (e.target === cardRef.current && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); open(); } }}>
      <span className="dlicon"><Icon name="file" /></span>
      <div className="dlmain">
        <div className="dlname">{f.name || "file"}</div>
        <div className="dlmeta"><span className="badge">{label}</span>{fmtSize(f.size || 0)}</div>
        <div className="dlnote">{note || expiryNote}</div>
        <div className="dlnote">{(textOnly ? "Preview and Copy show the text only. " : "") + "Click the card to view the contents."}</div>
      </div>
      <div className="dlbtns">
        <button className="dlbtn alt" type="button" aria-label={"Copy the text of " + (f.name || "file")} disabled={expired || copyBusy}
          onClick={(e) => { e.stopPropagation(); copy(); }}>
          <Icon name="copy" />
          <span className="lbl">{copyLabel || "Copy"}</span>
        </button>
        <button className="dlbtn" type="button" aria-label={"Download " + (f.name || "file")} disabled={expired || downloading} onClick={download}>
          {expired ? "Expired" : "Download"}
        </button>
      </div>
    </div>
  );
}

/** The right-side viewer. `panel` is {info, opener} or null. Everything from the server is shown as text, never as HTML. */
export function FilePanel({ panel, onClose }) {
  const { announce } = useUi();
  const info = panel && panel.info;
  const nameRef = useRef(null);
  const [state, setState] = useState({ kind: "loading", text: "", notes: [], label: "", truncated: false, gone: false });
  const [copyLabel, setCopyLabel] = useState(null);
  const [dlDisabled, setDlDisabled] = useState(false);
  const [status, setStatus] = useState("");

  useEffect(() => {
    if (!info) return undefined;
    let alive = true;
    setState({ kind: "loading", text: "", notes: [], label: "", truncated: false, gone: false });
    setCopyLabel(null);
    setDlDisabled(false);
    setStatus("");
    if (nameRef.current) nameRef.current.focus();
    document.body.classList.add("fp-open");
    getPreview(info.id).then((d) => {
      if (!alive) return;
      const notes = [];
      if (d.preview_note) notes.push(d.preview_note);
      if (d.truncated) notes.push("Showing the first " + (d.text || "").length.toLocaleString() + " characters only. The file is longer. Copy copies this part.");
      setState({ kind: d.text ? "text" : "empty", text: d.text || "", notes, label: d.label || "", truncated: !!d.truncated, gone: false });
    }).catch((e) => {
      if (!alive) return;
      const gone = !!(e && e.status === 404);
      setState({ kind: "error", text: gone ? "This file has expired. Upload it again." : (e && e.message) || "The file could not be loaded.", notes: [], label: "", truncated: false, gone });
      setDlDisabled(gone);
    });
    return () => { alive = false; document.body.classList.remove("fp-open"); };
  }, [info]);

  if (!info) return <aside className="fpanel" id="filePanel" role="complementary" aria-label="File viewer" hidden />;

  const badge = info.label || state.label;
  const copy = async () => {
    try {
      const d = await getPreview(info.id);
      const ok = await copyPlain(d.text);
      setCopyLabel(ok ? "Copied" : "Copy failed");
      const m = ok ? (d.truncated ? "Copied the first part of the text. The file is longer than the preview limit." : "Copied the text to the clipboard.") : "Copy failed. Select the text and copy it by hand.";
      setStatus(m); announce(m);
    } catch (e) {
      setCopyLabel("Unavailable");
      const m = e && e.status === 404 ? "This file has expired. Upload it again." : "The text could not be loaded. Try again.";
      setStatus(m); announce(m);
    }
    setTimeout(() => setCopyLabel(null), 1800);
  };
  const download = async () => {
    if (!info.downloadId) return;
    setDlDisabled(true);
    try {
      const r = await runDownload(info.downloadId, info.name);
      if (!r.ok) {
        const m = r.status === 404 ? "This file has expired. Upload it again." : "Download failed (" + r.status + "). Try again.";
        setStatus(m); announce(m);
        setDlDisabled(r.status === 404);
        return;
      }
      announce("Downloading " + info.name);
    } catch (e) {
      setStatus("Download failed. Check your connection.");
    }
    setDlDisabled(false);
  };

  return (
    <aside className="fpanel" id="filePanel" role="complementary" aria-label="File viewer">
      <div className="fphead">
        <div className="fprow">
          <span className="fpicon"><Icon name="file" /></span>
          <div className="fptitle">
            <div className="fpname" tabIndex={-1} ref={nameRef} title={info.name || ""}>{info.name || "file"}</div>
            <div className="fpmeta">
              <span className="badge" hidden={!badge}>{badge}</span>
              <span>{fmtSize(info.size || 0)}</span>
            </div>
          </div>
          <button className="fpbtn round" type="button" aria-label="Close file viewer" title="Close" onClick={onClose}><Icon name="x" /></button>
        </div>
        <div className="fpacts">
          <button className="fpbtn" type="button" aria-label="Copy the file text" title="Copy the text" disabled={state.kind !== "text" || copyLabel === "Copied"} onClick={copy}>
            <Icon name="copy" />
            <span className="lbl">{copyLabel || "Copy"}</span>
          </button>
          <button className="fpbtn" type="button" aria-label="Download the file" title="Download" hidden={!info.downloadId} disabled={dlDisabled} onClick={download}>
            <Icon name="download" />
            <span className="lbl">Download</span>
          </button>
        </div>
      </div>
      <div className="fpnote" hidden={!state.notes.length}>{state.notes.join(" ")}</div>
      <div className="fpbody" tabIndex={0} aria-label="File contents">
        <pre className={"fptext" + (state.kind === "loading" || state.kind === "empty" ? " msg" : state.kind === "error" ? " err" : "")}>
          {state.kind === "loading" ? "Loading…" : state.kind === "empty" ? "(This file has no text.)" : state.text}
        </pre>
      </div>
      <div className="sr-only" role="status" aria-live="polite">{status}</div>
    </aside>
  );
}
