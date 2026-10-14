// Uploaded / converted files: previews, downloads, upload progress.
import { API_BASE } from "./api";

const PREVIEW_CACHE = new Map();

export function forgetPreview(id) {
  PREVIEW_CACHE.delete(id);
}

/** The text of a stored file (cached for a moment). Rejects with an Error that has .status. */
export function getPreview(id) {
  if (PREVIEW_CACHE.has(id)) return Promise.resolve(PREVIEW_CACHE.get(id));
  return fetch(API_BASE + "/api/files/" + encodeURIComponent(id) + "/preview", { cache: "no-store" }).then(async (r) => {
    let d = null;
    try { d = await r.json(); } catch (e) { /* no body */ }
    if (!r.ok) {
      const err = new Error(d && typeof d.detail === "string" ? d.detail : "The file could not be loaded (" + r.status + ").");
      err.status = r.status;
      throw err;
    }
    if (PREVIEW_CACHE.size >= 12) PREVIEW_CACHE.delete(PREVIEW_CACHE.keys().next().value);
    PREVIEW_CACHE.set(id, d);
    return d;
  });
}

export async function runDownload(id, name) {
  const r = await fetch(API_BASE + "/api/files/" + encodeURIComponent(id) + "/download");
  if (!r.ok) return { ok: false, status: r.status };
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name || "download";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  return { ok: true };
}

export function deleteFile(id) {
  return fetch(API_BASE + "/api/files/" + encodeURIComponent(id), { method: "DELETE" }).catch(() => {});
}

export function fetchLimits() {
  return fetch(API_BASE + "/api/files/limits").then((r) => (r.ok ? r.json() : null)).catch(() => null);
}

/** Upload a file as the raw request body. Returns {xhr, done: Promise<{ok, data?, message?, aborted?}>}; onProgress(percent). */
export function uploadRaw(file, onProgress) {
  const xhr = new XMLHttpRequest();
  const done = new Promise((resolve) => {
    xhr.open("POST", API_BASE + "/api/files");
    xhr.setRequestHeader("X-File-Name", encodeURIComponent(file.name));
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(Math.round((100 * e.loaded) / e.total)); };
    xhr.onerror = () => resolve({ ok: false, message: "Upload failed. Check that the ELVARN server is running." });
    xhr.onabort = () => resolve({ ok: false, aborted: true });
    xhr.onload = () => {
      let d = null;
      try { d = JSON.parse(xhr.responseText); } catch (e) { /* not json */ }
      if (xhr.status < 200 || xhr.status >= 300) {
        resolve({ ok: false, message: d && typeof d.detail === "string" ? d.detail : "Upload failed (" + xhr.status + ")." });
      } else {
        resolve({ ok: true, data: d });
      }
    };
    xhr.send(file);
  });
  return { xhr, done };
}
