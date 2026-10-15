// The "Model lab" drawer: generation settings, web search, voice, the loaded model and the last reply's numbers.
import { useState } from "react";
import { Icon } from "../icons";
import { pct } from "../utils";
import { getOwnerKey } from "../ownerKey";

function Range({ label, id, value, min, max, step, format, onChange }) {
  return (
    <div className="field">
      <label>{label} <output>{format ? format(value) : value}</output></label>
      <input type="range" id={id} min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </div>
  );
}

function Switch({ id, title, note, checked, disabled, onChange }) {
  return (
    <div className="switch">
      <span>{title}{note && <div className="note">{note}</div>}</span>
      <input type="checkbox" id={id} aria-label={title} checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
    </div>
  );
}

function Stats({ rows }) {
  return (
    <>
      {rows.map(([k, v]) => (
        <div className="stat" key={k}><small>{k}</small><b>{v == null ? "—" : String(v)}</b></div>
      ))}
    </>
  );
}

export function LabPanel({ open, onClose, settings, setSetting, voices, speechSupported, voiceNote, status, model, last, onReload }) {
  const [rawOpen, setRawOpen] = useState(false);
  const [reloadNote, setReloadNote] = useState("Loads the newest completed checkpoint after a training run.");
  const reload = async () => {
    setReloadNote("Reloading…");
    setReloadNote(await onReload());
  };
  const modelRows = model ? [
    ["Checkpoint", model.checkpoint], ["Trained step", model.weights_from_step],
    ["Val loss", model.best_val_loss != null ? Number(model.best_val_loss).toFixed(3) : "—"],
    ["Parameters", Number(model.parameters).toLocaleString()], ["Vocabulary", model.vocab_size], ["Context", model.sequence_length + " tokens"],
    ["Attention heads", model.num_heads != null ? `${model.num_heads} × ${model.head_size}` : "—"],
    ["Tokenizer", model.tokenizer_sha256], ["Finished", model.completed_at || "—"],
  ] : [];
  const lastRows = last ? [
    ["Prompt tokens", last.prompt_tokens], ["Generated tokens", last.generated_tokens], ["Time", last.seconds != null ? last.seconds + " s" : "—"],
    ["Stopped by", last.stop_reason], ["Speed", last.tokens_per_second != null ? last.tokens_per_second + " tok/s" : "—"],
    ["Avg confidence", pct(last.mean_token_probability)], ["Checkpoint", last.checkpoint], ["Step", last.weights_from_step],
  ] : null;

  return (
    <aside className={"panel" + (open ? " show" : "")} id="panel" aria-label="Model lab and settings">
      <h3>Model lab <button className="icon-btn" type="button" aria-label="Close" onClick={onClose}><span style={{ display: "inline-grid" }}><Icon name="x" /></span></button></h3>

      <h4>Generation</h4>
      <Switch id="greedy" title="Greedy decoding" note="Off = random sampling" checked={settings.greedy} onChange={(v) => setSetting("greedy", v)} />
      <Range label="Temperature" id="temp" min={0.1} max={1.6} step={0.05} value={settings.temp} format={(v) => (+v).toFixed(2)} onChange={(v) => setSetting("temp", v)} />
      <Range label="Max new tokens" id="maxTok" min={10} max={200} step={5} value={settings.maxTok} onChange={(v) => setSetting("maxTok", v)} />
      <Range label="Top-k" id="topk" min={0} max={200} step={5} value={settings.topk} onChange={(v) => setSetting("topk", v)} />

      <h4>Web search</h4>
      <Switch id="webSearch" title="Search the web when needed" note="Off = answers come from the model only"
        checked={settings.webSearch !== false} onChange={(v) => setSetting("webSearch", v)} />

      <h4>Voice</h4>
      <Switch id="autoSpeak" title="Read replies aloud" checked={!!settings.autoSpeak} disabled={!speechSupported} onChange={(v) => setSetting("autoSpeak", v)} />
      <div className="field">
        <label>Voice</label>
        <select id="voiceSel" value={settings.voice || ""} onChange={(e) => setSetting("voice", e.target.value)}>
          <option value="">Browser default</option>
          {voices.map((v) => <option key={v.name + v.lang} value={v.name}>{`${v.name} (${v.lang})`}</option>)}
        </select>
      </div>
      <Range label="Speaking speed" id="rate" min={0.6} max={1.6} step={0.1} value={settings.rate} format={(v) => (+v).toFixed(1)} onChange={(v) => setSetting("rate", v)} />
      {getOwnerKey() && <Switch id="emailOnNew" title="Email me a colourful greeting for every new chat" note="Owner mode: sent to your own address by the server"
        checked={settings.emailOnNew !== false} onChange={(v) => setSetting("emailOnNew", v)} />}
      <Switch id="voiceSounds" title="Sounds in the voice conversation" note="A chime when you tap Talk, a beep when ELVARN starts listening"
        checked={settings.voiceSounds !== false} onChange={(v) => setSetting("voiceSounds", v)} />
      <Switch id="voiceOnNew" title="Start a voice conversation when I open a new chat" note="ELVARN greets you out loud and listens. Needs Chrome or Edge and microphone permission."
        checked={settings.voiceOnNew !== false} onChange={(v) => setSetting("voiceOnNew", v)} />
      <Switch id="autoSend" title="Send automatically when I stop talking" note="Off = your words wait in the box until you press Send"
        checked={!!settings.autoSend} onChange={(v) => setSetting("autoSend", v)} />
      <div className="field">
        <label>Listening language</label>
        <select id="recLang" value={settings.recLang} onChange={(e) => setSetting("recLang", e.target.value)}>
          <option value="en-IN">English (India)</option>
          <option value="ta-IN">Tamil (தமிழ்)</option>
          <option value="en-US">English (US)</option>
          <option value="en-GB">English (UK)</option>
        </select>
      </div>
      <div className="note" id="voiceNote">{voiceNote}</div>

      <h4>Loaded model</h4>
      <div className="statusline"><span className={"dot " + status.dot} id="dot" /><span id="statusText">{status.text}</span></div>
      <div className="stats" id="modelStats"><Stats rows={modelRows} /></div>
      <button className="btn gloss c-green" id="reloadBtn" type="button" onClick={reload}>Reload latest model</button>
      <div className="note" id="reloadNote">{reloadNote}</div>

      <h4>Last reply (measured)</h4>
      <div className="stats" id="lastStats">
        {lastRows ? <Stats rows={lastRows} /> : <div className="stat"><small>Nothing yet</small><b>—</b></div>}
      </div>
      <button className="btn gloss c-teal" id="rawBtn" type="button" onClick={() => setRawOpen(!rawOpen)}>Show raw model output</button>
      <pre id="rawOut" className="note" style={{ whiteSpace: "pre-wrap", display: rawOpen ? "block" : "none" }}>{(last && last.raw_text) || "(no reply yet)"}</pre>
    </aside>
  );
}
