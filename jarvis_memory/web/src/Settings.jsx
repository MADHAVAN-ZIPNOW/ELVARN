import { useEffect, useState } from "react";
import { deleteAll, exportAll, forgetKey, stats } from "./api";
import { Banner } from "./components";

export default function Settings({ onSignOut }) {
  const [s, setS] = useState(null);
  const [msg, setMsg] = useState({ kind: "", text: "" });
  const [typed, setTyped] = useState("");
  useEffect(() => { stats().then(setS).catch((e) => setMsg({ kind: "error", text: e.message })); }, []);

  const doExport = async () => {
    try {
      const data = await exportAll();
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
      const a = Object.assign(document.createElement("a"), { href: url, download: "jarvis-memory-export.json" });
      a.click(); URL.revokeObjectURL(url);
      setMsg({ kind: "ok", text: `Exported ${data.count} memories.` });
    } catch (e) { setMsg({ kind: "error", text: e.message }); }
  };
  const wipe = async () => {
    try { const r = await deleteAll(); setTyped(""); setMsg({ kind: "ok", text: `Deleted ${r.deleted} memories.` }); setS(await stats()); }
    catch (e) { setMsg({ kind: "error", text: e.message }); }
  };

  return (
    <section>
      <h2>Settings</h2>
      <Banner kind={msg.kind || "error"}>{msg.text}</Banner>
      <h3 className="section">Storage</h3>
      {s && (
        <>
          <p>{s.memories.toLocaleString()} memories · about {Math.round(s.avg_bytes_per_memory / 100) / 10} KB each</p>
          <div className="bar" role="progressbar" aria-valuenow={s.used_percent} aria-valuemin={0} aria-valuemax={100}>
            <div className={"fill" + (s.used_percent > 80 ? " hot" : "")} style={{ width: Math.max(1, Math.min(100, s.used_percent)) + "%" }} />
          </div>
          <p className="muted">{s.database_mb} MB of {s.limit_mb} MB limit used ({s.used_percent}%). New memories are refused near the limit so the free database never fills up.</p>
          {s.ram_status !== "unknown" && (
            <p className={s.ram_status === "ok" ? "muted" : "warn"} role="status">
              Server memory: {s.process_mb} MB of {s.ram_limit_mb} MB safe limit{s.ram_status === "close" ? " — getting close to the limit" : s.ram_status === "over" ? " — OVER the safe limit" : ""} (the free host stops the service at 512 MB).
            </p>
          )}
        </>
      )}
      <h3 className="section">Your data</h3>
      <button className="btn" onClick={doExport}>Download everything (JSON)</button>
      <h3 className="section">Delete everything</h3>
      <p className="muted">Permanently deletes every memory. Download a copy first. Type <strong>DELETE</strong> to enable the button.</p>
      <div className="task-row">
        <input value={typed} onChange={(e) => setTyped(e.target.value)} placeholder="DELETE" aria-label="Type DELETE to confirm" />
        <button className="btn danger" disabled={typed !== "DELETE"} onClick={wipe}>Delete all memories</button>
      </div>
      <h3 className="section">This browser</h3>
      <button className="btn" onClick={() => { forgetKey(); onSignOut(); }}>Forget my owner key</button>
    </section>
  );
}
