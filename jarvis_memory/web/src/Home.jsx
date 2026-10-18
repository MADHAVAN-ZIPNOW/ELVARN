import { useCallback, useEffect, useRef, useState } from "react";
import { createMemory, listMemories } from "./api";
import { Banner, MemoryCard, SOURCE_ICON, SOURCE_LABEL } from "./components";
import { useSpeech } from "./speech";

const FILTERS = [["", "All"], ["manual", "Typed"], ["voice", "Voice"], ["email", "Email"], ["daily_review", "Daily review"]];

export default function Home({ onOpen }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(null);
  const [recent, setRecent] = useState(null);
  const [filter, setFilter] = useState("");
  const baseRef = useRef("");

  const speech = useSpeech({
    onText: ({ final, interim }) => setText((baseRef.current ? baseRef.current + " " : "") + [final, interim].filter(Boolean).join(" ")),
  });

  const load = useCallback(async () => {
    try { setRecent(await listMemories(filter)); } catch (e) { setError(e.message); setRecent([]); }
  }, [filter]);
  useEffect(() => { load(); }, [load]);

  const toggleMic = () => {
    if (speech.listening) { speech.stop(); return; }
    baseRef.current = text.trim();
    speech.reset();
    speech.start();
  };

  const save = async () => {
    const content = text;
    if (!content.trim() || busy) return;
    speech.stop();
    setBusy(true); setError(""); setSaved(null);
    const raw = speech.raw();
    try {
      const memory = await createMemory(raw ? { content, source: "voice", raw_transcript: raw } : { content, source: "manual" });
      setSaved(memory); setText(""); baseRef.current = ""; speech.reset();
      setRecent((r) => (filter && filter !== memory.source ? r : [memory, ...(r || [])]));
    } catch (e) {
      setError(e.message);  // the text stays in the box so nothing is lost
    } finally { setBusy(false); }
  };

  return (
    <section>
      <h2>What did you do since the last update?</h2>
      <div className="composer">
        <textarea value={text} onChange={(e) => { setText(e.target.value); baseRef.current = e.target.value; }}
          placeholder="Type it, or tap the microphone and talk. You can edit before saving." rows={5} maxLength={20000}
          aria-label="Your update" disabled={busy} />
        <div className="composer-bar">
          {speech.supported
            ? <button className={"mic" + (speech.listening ? " on" : "")} onClick={toggleMic} aria-pressed={speech.listening}
                aria-label={speech.listening ? "Stop recording" : "Start voice input"} disabled={busy}>
                {speech.listening ? "■ Stop" : "🎙️ Speak"}
              </button>
            : <small className="muted">Voice input isn't supported in this browser — use your keyboard's microphone button.</small>}
          {speech.listening && <span className="live" role="status">Listening… your words appear above</span>}
          <span className="grow" />
          <small className="muted">{text.length.toLocaleString()} / 20,000</small>
          <button className="btn primary" onClick={save} disabled={busy || !text.trim()}>{busy ? "Saving… AI is reading it" : "Save memory"}</button>
        </div>
        {speech.error && <small className="warn">{speech.error}</small>}
      </div>
      <Banner onClose={() => setError("")}>{error}</Banner>
      {saved && (
        <div className="banner ok" role="status">
          <strong>Saved ✓</strong> {SOURCE_ICON[saved.source]} {saved.title}
          {saved.metadata_json?.extraction?.status === "failed" && " — AI summary unavailable, saved exactly as written."}
          {saved.metadata_json?.extraction?.status === "skipped" && " — saved without AI (no key configured)."}
        </div>
      )}

      <div className="filters" role="tablist" aria-label="Quick filters">
        {FILTERS.map(([id, label]) => (
          <button key={id} role="tab" aria-selected={filter === id} className={"tab" + (filter === id ? " active" : "")} onClick={() => setFilter(id)}>
            {id ? SOURCE_ICON[id] + " " : ""}{label || SOURCE_LABEL[id]}
          </button>
        ))}
      </div>
      <h3 className="section">Recent memories</h3>
      {recent === null && <p className="muted">Loading…</p>}
      {recent && recent.length === 0 && <p className="empty">Nothing here yet. Your first update will show up above.</p>}
      <div className="list">{(recent || []).map((m) => <MemoryCard key={m.id} memory={m} onOpen={onOpen} fresh={saved && saved.id === m.id} />)}</div>
    </section>
  );
}
