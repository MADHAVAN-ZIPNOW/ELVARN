import { useEffect, useState } from "react";
import { AuthError, api, captureKey, getKey, setKey } from "./api";
import Detail from "./Detail";
import Home from "./Home";
import Search from "./Search";
import Settings from "./Settings";
import { Banner } from "./components";

captureKey();

function KeyGate({ onDone, message }) {
  const [k, setK] = useState("");
  return (
    <main className="gate">
      <h1>Jarvis Memory</h1>
      <p className="muted">Enter your owner key to open your memories. It is stored only in this browser.</p>
      <form onSubmit={(e) => { e.preventDefault(); if (k.trim()) { setKey(k); onDone(); } }}>
        <input type="password" value={k} onChange={(e) => setK(e.target.value)} placeholder="Owner key" aria-label="Owner key" autoFocus autoComplete="off" />
        <button className="btn primary" disabled={!k.trim()}>Open</button>
      </form>
      <Banner>{message}</Banner>
    </main>
  );
}

export default function App() {
  const [authed, setAuthed] = useState(!!getKey());
  const [authMsg, setAuthMsg] = useState("");
  const [tab, setTab] = useState("home");
  const [detail, setDetail] = useState(null);
  const [reload, setReload] = useState(0);

  // verify the stored key once; a wrong/expired key goes back to the gate instead of showing empty pages
  useEffect(() => {
    if (!authed) return;
    api("/stats").catch((e) => { if (e instanceof AuthError) { setAuthed(false); setAuthMsg(e.message); } });
  }, [authed]);

  if (!authed) return <KeyGate message={authMsg} onDone={() => { setAuthMsg(""); setAuthed(true); }} />;

  const open = (id) => setDetail(id);
  const back = (changed) => { setDetail(null); if (changed) setReload((n) => n + 1); };

  return (
    <div className="app">
      <header className="top">
        <strong className="brand">JARVIS <span>Memory</span></strong>
        <nav aria-label="Main">
          {[["home", "Memory"], ["search", "Search"], ["settings", "Settings"]].map(([id, label]) => (
            <button key={id} className={"nav" + (tab === id && !detail ? " active" : "")} aria-current={tab === id && !detail ? "page" : undefined}
              onClick={() => { setTab(id); setDetail(null); }}>{label}</button>
          ))}
        </nav>
      </header>
      <main>
        {detail ? <Detail id={detail} onBack={back} />
          : tab === "home" ? <Home key={reload} onOpen={open} />
          : tab === "search" ? <Search onOpen={open} />
          : <Settings onSignOut={() => setAuthed(false)} />}
      </main>
    </div>
  );
}
