// The file attached to the next message: validate, upload with progress, remove, and drag-and-drop onto the page.
import { useCallback, useRef, useState } from "react";
import { FILE_EXT } from "../constants";
import { deleteFile, fetchLimits, forgetPreview, uploadRaw } from "../files";
import { fmtSize } from "../utils";

export function useAttachment({ isBusy, showBanner, announce, focusInput }) {
  const [pending, setPending] = useState(null);       // null | {name,size,uploading,pct} | {file_id,name,size,detected_type,supported_targets}
  const [dragOver, setDragOver] = useState(false);
  const pendingRef = useRef(null);
  pendingRef.current = pending;
  const upXhr = useRef(null), maxUpload = useRef(5 * 1024 * 1024), depth = useRef(0), limitsLoaded = useRef(false);

  const loadLimits = useCallback(() => {
    if (limitsLoaded.current) return;
    limitsLoaded.current = true;
    fetchLimits().then((d) => { if (d && d.max_upload_bytes) maxUpload.current = d.max_upload_bytes; });
  }, []);

  const remove = useCallback((silent) => {
    const p = pendingRef.current;
    setPending(null);
    if (p && p.file_id) { forgetPreview(p.file_id); deleteFile(p.file_id); }
    if (!silent) announce("Attachment removed.");
  }, [announce]);

  const upload = useCallback(async (file) => {
    if (!file || isBusy()) return;
    showBanner("");
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    if (!FILE_EXT.includes(ext)) { showBanner("That file type is not supported. Allowed: txt, md, csv, json, html, docx. (Programs, archives and macro files are never accepted.)"); return; }
    if (file.size > maxUpload.current) { showBanner("That file is too large. The limit is " + fmtSize(maxUpload.current) + "."); return; }
    if (file.size === 0) { showBanner("That file is empty."); return; }
    if (pendingRef.current && pendingRef.current.file_id) remove(true);
    setPending({ name: file.name, size: file.size, uploading: true, pct: 0 });
    announce("Uploading " + file.name);
    const { xhr, done } = uploadRaw(file, (pct) => setPending((p) => (p && p.uploading ? { ...p, pct } : p)));
    upXhr.current = xhr;
    const r = await done;
    upXhr.current = null;
    if (r.aborted) { setPending(null); announce("Upload cancelled."); return; }
    if (!r.ok) { setPending(null); showBanner(r.message); announce("Upload failed. " + r.message); return; }
    const d = r.data;
    setPending({ file_id: d.file_id, name: d.name, size: d.size, detected_type: d.detected_type, supported_targets: d.supported_targets });
    announce(d.name + " is ready. Type a request such as convert to PDF, or press one of the Convert buttons.");
    focusInput();
  }, [isBusy, showBanner, announce, focusInput, remove]);

  const hasFiles = (e) => e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
  const dropHandlers = {
    onDragEnter: (e) => { if (hasFiles(e)) { e.preventDefault(); depth.current++; setDragOver(true); } },
    onDragOver: (e) => { if (hasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; } },
    onDragLeave: () => { depth.current = Math.max(0, depth.current - 1); if (!depth.current) setDragOver(false); },
    onDrop: (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth.current = 0;
      setDragOver(false);
      const f = e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) upload(f);
    },
  };

  return {
    pending, setPending, dragOver, dropHandlers, loadLimits, upload, remove,
    cancelUpload: () => { if (upXhr.current) upXhr.current.abort(); },
  };
}
