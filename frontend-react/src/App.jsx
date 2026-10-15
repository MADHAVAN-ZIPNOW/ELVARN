// ELVARN chat: owns the state (chats, settings, the answer being streamed) and wires the components together.
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { GradientDefs } from "./icons";
import { API_BASE, api, streamChat } from "./api";
import { saveState, sessionId } from "./storage";
import { compact, searchInfo } from "./utils";
import { downloadAs } from "./export";
import { initialState, launchRocket, newConv, viewOf } from "./appHelpers";
import { UiContext } from "./UiContext";
import { useSpeech } from "./hooks/useSpeech";
import { useVoiceInput } from "./hooks/useVoiceInput";
import { useAttachment } from "./hooks/useAttachment";
import { Banner, Header, Sidebar } from "./components/Chrome";
import { Welcome, UserRow, BotRow } from "./components/Messages";
import { SessionsView } from "./components/SessionsView";
import { Composer } from "./components/Composer";
import { LabPanel } from "./components/LabPanel";
import { SourcesPanel } from "./components/SourcesPanel";
import { DownloadMenu } from "./components/DownloadMenu";
import { FilePanel } from "./components/FileParts";
import { TalkBar } from "./components/TalkBar";
import { continuing, greeting, rememberName } from "./talkHelpers";
import { cue } from "./sounds";
import { notifyNewChat } from "./ownerKey";
import { LeadCard, shouldAskForEmail } from "./components/LeadCard";

export default function App() {
  const [boot] = useState(initialState);
  const [settings, setSettings] = useState(boot.settings);
  const [mode, setMode] = useState(boot.mode);
  const [convs, setConvs] = useState(boot.convs);
  const [currentId, setCurrentId] = useState(boot.currentId);
  const [view, setView] = useState("chat");
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState(null);                  // the answer being streamed: {convId, index, userIndex, text, proc}
  const [banner, setBanner] = useState({ msg: "", kind: "" });
  const [sideOpen, setSideOpen] = useState(false);
  const [labOpen, setLabOpen] = useState(false);
  const [sourcesView, setSourcesView] = useState(null);
  const [dlMenu, setDlMenu] = useState(null);
  const [filePanel, setFilePanel] = useState(null);
  const [model, setModel] = useState(null);
  const [status, setStatus] = useState({ dot: "", text: "Connecting…" });
  const [lastReply, setLastReply] = useState(null);
  const [srText, setSrText] = useState("");
  const [lead, setLead] = useState(null);                // the question shown with the optional email card
  const [talk, setTalk] = useState(null);                // voice conversation: null | greeting | listening | thinking | speaking
  const [epoch, setEpoch] = useState(0);                   // bumped when the thread must jump to the bottom (switching chats)

  const chatRef = useRef(null), threadRef = useRef(null), spacerRef = useRef(null), inputRef = useRef(null);
  const abortRef = useRef(null), busyRef = useRef(false);
  const pinRequest = useRef(false), autoSendOnce = useRef(false), talkRef = useRef(null), silent = useRef(0);
  const latest = useRef({});

  const conv = useMemo(() => convs.find((c) => c.id === currentId) || convs[0], [convs, currentId]);
  const sv = view === "sessions";

  // ---------------------------------------------------------------- small helpers
  const showBanner = useCallback((msg, kind) => setBanner({ msg: msg || "", kind: kind || "" }), []);
  const announce = useCallback((msg) => { setSrText(""); setTimeout(() => setSrText(msg), 40); }, []);
  const focusInput = () => { if (inputRef.current) inputRef.current.focus(); };
  const updateConv = useCallback((id, fn) => setConvs((cs) => cs.map((c) => (c.id === id ? fn(c) : c))), []);
  const setSetting = useCallback((k, v) => setSettings((s) => ({ ...s, [k]: v })), []);

  const speech = useSpeech({
    voiceName: settings.voice, rate: settings.rate,
    onUnsupported: () => showBanner("Your browser does not support reading aloud.", "info"),
  });

  const attach = useAttachment({ isBusy: () => busyRef.current, showBanner, announce, focusInput });
  const { pending, setPending } = attach;
  latest.current = { convs, currentId, mode, settings, pending, input, busy, filePanel };   // the newest values, for code that runs later (sending, voice)

  // ---------------------------------------------------------------- persistence and page-level effects
  useEffect(() => { saveState({ settings, mode, currentId, convs }); }, [settings, mode, currentId, convs]);
  useEffect(() => {
    document.documentElement.dataset.theme = settings.theme === "dusk" ? "dusk" : "";
    const meta = document.querySelector("meta[name=theme-color]");
    if (meta) meta.content = settings.theme === "dusk" ? "#43558c" : "#e6eaf3";
  }, [settings.theme]);
  useEffect(() => { document.body.classList.toggle("side-collapsed", !!settings.collapsed); }, [settings.collapsed]);

  // the chat jumps to the bottom when you switch chats
  useEffect(() => {
    if (spacerRef.current) spacerRef.current.style.height = "0px";
    const id = requestAnimationFrame(() => { if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight; });
    return () => cancelAnimationFrame(id);
  }, [currentId, epoch, view]);

  // keep the newest question at the top of the view while the answer grows
  const fitSpacer = useCallback((row) => {
    const chat = chatRef.current, thread = threadRef.current;
    if (!chat || !thread || !row) return;
    const kids = thread.children, last = kids[kids.length - 2];
    if (!last) return;
    const content = last.getBoundingClientRect().bottom - row.getBoundingClientRect().top;
    spacerRef.current.style.height = Math.max(0, chat.clientHeight - content - 24) + "px";
  }, []);
  const userRow = useCallback((index) => threadRef.current && threadRef.current.querySelector(`[data-user-row="${index}"]`), []);
  useLayoutEffect(() => {
    if (!live) return;
    const row = userRow(live.userIndex);
    if (!row) return;
    if (pinRequest.current) {
      pinRequest.current = false;
      fitSpacer(row);
      chatRef.current.scrollTo({ top: row.offsetTop - 10, behavior: "smooth" });
    } else fitSpacer(row);
  }, [live, fitSpacer, userRow]);

  // ---------------------------------------------------------------- model status
  const refreshStatus = useCallback(async () => {
    try {
      const h = await api("/api/health");
      if (h.model_ready) {
        const m = await api("/api/model");
        setStatus({ dot: "ok", text: `${m.checkpoint} · step ${m.weights_from_step}${m.smoke_run ? " · SMOKE TEST" : ""}` });
        showBanner(m.smoke_run ? "This is a tiny smoke-test model, not a fully trained one." : "", "info");
        setModel(m);
        return m;
      }
      setStatus({ dot: "bad", text: "Model not loaded" });
      showBanner(h.error || "No trained model found. Train ELVARN first.");
    } catch (e) {
      setStatus({ dot: "bad", text: "Server unreachable" });
      showBanner("Cannot reach the ELVARN backend at " + (API_BASE || location.origin) + ". Start it with:  python -m uvicorn backend.api.server:app --port 8000");
    }
    return null;
  }, [showBanner]);
  useEffect(() => {
    refreshStatus();
    const t = setInterval(() => { if (!busyRef.current) refreshStatus(); }, 30000);
    attach.loadLimits();
    return () => clearInterval(t);
  }, [refreshStatus]); // eslint-disable-line react-hooks/exhaustive-deps
  const reloadModel = async () => {
    try {
      const r = await api("/api/reload", { method: "POST" });
      await refreshStatus();
      return "Loaded " + r.model.checkpoint + " (step " + r.model.weights_from_step + ").";
    } catch (e) {
      refreshStatus();
      return "Reload failed: " + e.message;
    }
  };

  // ---------------------------------------------------------------- navigation
  const closeAll = useCallback(() => { setSideOpen(false); setLabOpen(false); setSourcesView(null); }, []);
  const closeFilePanel = useCallback(() => {
    const o = latest.current.filePanel && latest.current.filePanel.opener;
    setFilePanel(null);
    setTimeout(() => { if (o && document.contains(o) && o.focus) o.focus(); else if (inputRef.current) inputRef.current.focus(); }, 0);
  }, []);
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") { closeAll(); if (filePanel) closeFilePanel(); } };
    const onClick = (e) => {
      if (sourcesView && !e.target.closest("#srcPanel") && !e.target.closest(".srcchip") && !e.target.closest("sup.cite")) setSourcesView(null);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("click", onClick);
    return () => { document.removeEventListener("keydown", onKey); document.removeEventListener("click", onClick); };
  }, [closeAll, closeFilePanel, filePanel, sourcesView]);

  const toggleSide = () => {
    if (matchMedia("(max-width:860px)").matches) setSideOpen((o) => !o);
    else setSettings((s) => ({ ...s, collapsed: !s.collapsed }));
  };
  const switchTo = (id) => {
    speech.stop();
    const c = latest.current.convs.find((x) => x.id === id);
    setCurrentId(id);
    if (c && c.mode) setMode(c.mode);
    setView("chat");
    setEpoch((e) => e + 1);
    closeAll();
  };
  const startNew = () => {
    speech.stop();
    const c = latest.current.convs.find((x) => x.id === latest.current.currentId);
    if (!c || c.messages.length) {
      const n = newConv();
      setConvs((cs) => [n, ...cs]);
      setCurrentId(n.id);
    }
    setMode("chat");
    setView("chat");
    setEpoch((e) => e + 1);
    closeAll();
    setTimeout(focusInput, 0);
    if (latest.current.settings.emailOnNew !== false) notifyNewChat();          // owner only: a colourful greeting email
    if (latest.current.settings.voiceOnNew && voice.supported && speech.supported) { stopTalk(); setTimeout(startTalk, 250); }
  };
  const openMode = (id) => {
    speech.stop();
    setMode(id);
    const c = latest.current.convs.find((x) => x.id === latest.current.currentId);
    if (!c || c.messages.length) {
      const n = { ...newConv(), mode: id };
      setConvs((cs) => [n, ...cs]);
      setCurrentId(n.id);
      if (latest.current.settings.emailOnNew !== false) notifyNewChat();          // a new chat: the owner's greeting email
    } else updateConv(c.id, (x) => ({ ...x, mode: id }));
    setView("chat");
    setEpoch((e) => e + 1);
    closeAll();
    setTimeout(focusInput, 0);
  };
  const deleteConv = (id) => {
    const rest = latest.current.convs.filter((c) => c.id !== id);
    let next = rest, nextId = latest.current.currentId;
    if (nextId === id) {
      const n = rest.find((c) => c.messages.length);
      if (n) nextId = n.id;
      else { const fresh = newConv(); next = [fresh, ...rest]; nextId = fresh.id; }
    }
    setConvs(next);
    setCurrentId(nextId);
    setEpoch((e) => e + 1);
  };
  const clearChat = () => {
    speech.stop();
    if (conv) updateConv(conv.id, (c) => ({ ...c, messages: [], title: "New chat" }));
    setEpoch((e) => e + 1);
    showBanner("");
  };

  const openFilePanel = useCallback((info, opener) => setFilePanel({ info, opener: opener || null }), []);

  // ---------------------------------------------------------------- sending a message
  const send = async (textArg) => {
    const L = latest.current;
    const text = (textArg != null ? textArg : L.input).trim();
    if (!text || busyRef.current) return;
    speech.stop();
    const c = L.convs.find((x) => x.id === L.currentId);
    if (!c) return;
    launchRocket(document.getElementById("sendBtn"));
    busyRef.current = true;
    setBusy(true);
    setInput("");
    const attached = L.pending && L.pending.file_id ? L.pending : null;
    const files = attached ? (c.files || []).concat([{ file_id: attached.file_id, name: attached.name }]).slice(-5) : c.files;
    if (attached) setPending(null);
    const userMsg = { role: "user", text };
    if (talkRef.current) userMsg.voice = true;                  // said aloud in a voice conversation
    rememberName(text);
    if (attached) userMsg.file = { name: attached.name, size: attached.size, type: attached.detected_type };

    // the last two finished exchanges of this chat, for chat-style modes
    const hist = [], ms = c.messages;
    for (let i = ms.length - 2; i >= 0 && hist.length < 2; i--) {
      if (ms[i].role === "user" && ms[i + 1] && ms[i + 1].role === "bot" && !ms[i + 1].error && !(ms[i + 1].meta && ms[i + 1].meta.file_action) && ms[i + 1].mode === L.mode) {
        hist.unshift({ user: ms[i].text, assistant: ms[i + 1].text });
      }
    }
    const userIndex = c.messages.length, index = userIndex + 1;
    const title = c.title === "New chat" ? (text.length > 42 ? text.slice(0, 42) + "…" : text) : c.title;
    updateConv(c.id, (x) => ({ ...x, mode: L.mode, updated: Date.now(), files, title, messages: [...x.messages, userMsg] }));

    const S = L.settings;
    const body = {
      text, mode: L.mode, max_new_tokens: +S.maxTok, temperature: +S.temp, top_k: +S.topk, greedy: S.greedy,
      history: (L.mode === "chat" || L.mode === "humour") && hist.length ? hist : null,
      web_search: S.webSearch !== false ? "auto" : "off",
    };
    const sid = sessionId();
    if (sid) body.session_id = sid;
    if (files && files.length) body.file_ids = files.map((f) => f.file_id);

    const proc = { sentAt: Date.now(), start: null, tokens: 0, last: null, done: null, searching: false, search: null };
    let replyText = "";
    const publish = () => setLive({ convId: c.id, index, userIndex, text: replyText, proc: { ...proc } });
    pinRequest.current = true;
    publish();
    const tick = setInterval(() => { if (!proc.start) publish(); }, 250);
    abortRef.current = new AbortController();

    let finalEv = null, errorMsg = null, aborted = false;
    try {
      await streamChat(body, (ev) => {
        if (ev.event === "status" && ev.status === "searching") { proc.searching = true; announce("Searching the web"); }
        else if (ev.event === "sources") {
          proc.searching = false;
          proc.search = searchInfo(ev);
          announce(ev.searched ? "Web search finished, " + (ev.sources || []).length + " sources found" : "Web search finished without usable results");
        } else if (ev.event === "start") proc.start = ev;
        else if (ev.event === "token") { proc.tokens = ev.index; proc.last = ev; if (ev.text) replyText += ev.text; }
        else if (ev.event === "done") { finalEv = ev; proc.done = ev; if (ev.search_status && ev.search_status !== "skipped") proc.search = searchInfo(ev); }
        else if (ev.event === "error") errorMsg = ev.message;
        publish();
      }, abortRef.current.signal);
    } catch (e) {
      if (e.name === "AbortError") aborted = true; else errorMsg = e.message;
    }
    clearInterval(tick);
    proc.searching = false;

    let botMsg, shownMeta = null;
    if (errorMsg) {
      botMsg = { role: "bot", text: "Could not get a reply: " + errorMsg, error: true };
      refreshStatus();
    } else if (finalEv && finalEv.file_action) {
      // a file request answered by plain server code (the model was not used)
      const stamped = (finalEv.files || []).map((f) => ({ ...f, expiresAt: Date.now() + (+f.expires_in || 0) * 1000 }));
      botMsg = { role: "bot", text: finalEv.text, mode: L.mode, meta: { file_action: finalEv.file_action, files: stamped } };
      announce(stamped.length ? "File converted. A download button is available." : "File request finished.");
    } else {
      let meta;
      if (finalEv) { replyText = finalEv.text; meta = compact(finalEv); proc.done = finalEv; }
      else {                                  // stopped by the user: keep what was generated so far
        replyText = replyText.trim();
        const secs = ((Date.now() - proc.sentAt) / 1000).toFixed(2);
        proc.done = { generated_tokens: proc.tokens, seconds: +secs, tokens_per_second: proc.last ? proc.last.tokens_per_second : 0, stop_reason: "stopped", mean_token_probability: null };
        meta = { ...proc.done, prompt_tokens: proc.start ? proc.start.prompt_tokens : null, raw_text: replyText };
      }
      meta.proc = { start: proc.start, done: compact(proc.done), search: proc.search || null };
      meta.search = proc.search || null;
      botMsg = { role: "bot", text: replyText, mode: L.mode, meta };
      shownMeta = meta;
    }
    updateConv(c.id, (x) => ({ ...x, updated: Date.now(), messages: [...x.messages, botMsg] }));
    setLive(null);
    if (userIndex === 0 && shownMeta) shouldAskForEmail().then((yes) => { if (yes) setLead(text); });          // after a visitor's first answer
    if (shownMeta) {
      setLastReply(shownMeta);
      if (talkRef.current && replyText && !aborted) {
        setTalk("speaking"); talkRef.current = "speaking";
        speech.speak(speakable(replyText), c.id + ":" + index, () => { silent.current = 0; listenAgain(); });
      } else if (settings.autoSpeak && replyText && !aborted) speech.speak(replyText, c.id + ":" + index);
    }
    if (talkRef.current && !shownMeta) stopTalk("Sorry, I could not get an answer. Voice conversation stopped.");
    busyRef.current = false;
    abortRef.current = null;
    setBusy(false);
    focusInput();
  };

  const voice = useVoiceInput({
    lang: settings.recLang,
    getText: () => latest.current.input,
    setText: setInput,
    onBanner: showBanner,
    beforeStart: speech.stop,
    autoStopMs: () => (talkRef.current === "listening" ? 1500 : 0),          // in a voice conversation: send when you stop speaking
    onFinished: (valid, heard) => {
      focusInput();
      if (talkRef.current === "listening") {                // voice conversation: what you said is sent at once; silence listens again, twice
        if (valid && heard && heard.trim() && !busyRef.current) { setTalk("thinking"); talkRef.current = "thinking"; send(heard); }
        else if (++silent.current <= 2) setTimeout(listenAgain, 300);
        else stopTalk("I did not hear anything, so I stopped listening. Tap Talk to ELVARN to start again.");
        return;
      }
      if (valid && latest.current.settings.autoSend && latest.current.input.trim() && !busyRef.current) send();
    },
  });

  // ---------------------------------------------------------------- voice conversation: ELVARN greets you out loud, you speak, it answers on the page and aloud
  const speakable = (t) => String(t).replace(/\[\d+\]/g, "").replace(/https?:\/\/\S+/g, "").replace(/\s+/g, " ").trim().slice(0, 420);
  const listenAgain = () => {
    if (!talkRef.current || busyRef.current) return;
    setTalk("listening"); talkRef.current = "listening";
    setInput("");
    cue("listen", latest.current.settings.voiceSounds !== false);
    voice.start();
  };
  const interrupt = () => {                                 // while ELVARN speaks: stop it and listen now
    if (!talkRef.current) return;
    speech.stop(); silent.current = 0; talkRef.current = "speaking";
    listenAgain();
  };
  const stopTalk = (message) => {
    if (talkRef.current) cue("stop", latest.current.settings.voiceSounds !== false);
    talkRef.current = null; silent.current = 0;
    setTalk(null); speech.stop(); voice.stop();
    if (message) showBanner(message, "info");
  };
  const startTalk = () => {
    if (!voice.supported || !speech.supported) { showBanner("Voice conversation needs a browser with speech recognition and speech, such as Chrome or Edge (on Android or computer).", "info"); return; }
    if (talkRef.current) return;
    showBanner("");
    cue("start", latest.current.settings.voiceSounds !== false);          // the chime on the tap itself
    silent.current = 0; talkRef.current = "greeting"; setTalk("greeting");
    const c = latest.current.convs.find((x) => x.id === latest.current.currentId);
    speech.speak(c && c.messages.length ? continuing() : greeting(), "greeting", listenAgain);          // inside an existing chat: no long hello, just carry on
  };

  const onSendClick = () => {
    if (busyRef.current) return;
    if (voice.on) {
      autoSendOnce.current = true;
      voice.stop();
      setTimeout(() => { if (autoSendOnce.current) { autoSendOnce.current = false; send(); } }, 250);
    } else send();
  };
  const onMic = () => {
    if (!voice.supported) { showBanner("Voice input needs a browser with speech recognition, such as Chrome or Edge.", "info"); return; }
    if (voice.on) voice.stop(); else voice.start();
  };

  // ---------------------------------------------------------------- per-message actions
  const messageActions = (c, i, m) => {
    const key = c.id + ":" + i;
    return {
      speaking: speech.speakingKey === key,
      onSpeak: () => speech.speak(m.text, key),
      onDetails: () => { setLastReply((m.meta) || {}); setLabOpen(true); },
      onOpenSources: (info, focus) => setSourcesView({ info, focus }),
      onFeedback: (value) => updateConv(c.id, (x) => ({
        ...x, messages: x.messages.map((mm, j) => (j === i ? { ...mm, meta: { ...(mm.meta || {}), feedback: value } } : mm)),
      })),
      onDownload: (anchor) => setDlMenu({
        anchor, title: c.title, getMsgs: () => (i > 0 && c.messages[i - 1].role === "user" ? [c.messages[i - 1], m] : [m]),
      }),
    };
  };

  const rows = [];
  if (conv) {
    conv.messages.forEach((m, i) => {
      if (m.role === "user") rows.push(<UserRow key={conv.id + "-" + i} message={m} index={i} />);
      else rows.push(<BotRow key={conv.id + "-" + i} view={viewOf(m)} {...messageActions(conv, i, m)} />);
    });
    if (live && live.convId === conv.id) {
      const noop = () => {};
      rows.push(
        <BotRow key={conv.id + "-" + live.index} view={{ plain: true, text: live.text, proc: live.proc, procLive: true }}
          speaking={false} onSpeak={noop} onDetails={noop} onOpenSources={noop} onFeedback={noop} onDownload={noop} />
      );
    }
  }
  const showWelcome = !!conv && !conv.messages.length && !live;

  const sessionCount = convs.filter((c) => c.messages.length).length;
  const voiceNote = voice.supported
    ? "Voice input uses your browser's speech service (Chrome/Edge may send audio to their provider)."
    : "Voice input: not supported in this browser.";

  return (
    <UiContext.Provider value={{ announce, openFilePanel }}>
      <GradientDefs />
      <div className="shell">
        <Sidebar open={sideOpen} mode={mode} view={view} sessionCount={sessionCount} onToggle={toggleSide} onNewChat={startNew}
          onOpenSessions={() => { setView("sessions"); closeAll(); }} onOpenMode={openMode} />

        <div className={"main" + (attach.dragOver ? " dragover" : "")} {...attach.dropHandlers}>
          <Header view={view} mode={mode} convTitle={conv ? conv.title : ""} hasMessages={!!(conv && conv.messages.length)}
            talking={!!talk} onTalk={() => (talkRef.current ? stopTalk() : startTalk())} onToggleSide={toggleSide} onTheme={() => setSetting("theme", settings.theme === "dusk" ? "aurora" : "dusk")}
            onClear={clearChat} onLab={() => setLabOpen(true)} />
          <Banner banner={banner} />

          <SessionsView hidden={!sv} convs={convs} onOpen={switchTo} onNew={startNew}
            onPin={(id) => updateConv(id, (c) => ({ ...c, pinned: !c.pinned }))}
            onRename={(id, title) => updateConv(id, (c) => ({ ...c, title }))}
            onDownload={(anchor, c) => setDlMenu({ anchor, title: c.title, getMsgs: () => c.messages })}
            onDelete={deleteConv} />

          <div className="sr-only" id="srStatus" role="status" aria-live="polite">{srText}</div>
          <main className="chat" id="chat" aria-live="polite" ref={chatRef} hidden={sv}>
            <div className="thread" id="thread" ref={threadRef}>
              {showWelcome ? <Welcome mode={mode} onPick={(t) => send(t)} onTalk={startTalk} /> : rows}
              {lead && !showWelcome && <LeadCard question={lead} onDone={() => setLead(null)} />}
              <div id="spacer" ref={spacerRef} />
            </div>
          </main>

          <Composer hidden={sv} input={input} setInput={setInput} busy={busy} pending={pending}
            accept=".txt,.md,.markdown,.csv,.json,.html,.htm,.docx" live={voice.live} voice={voice} micSupported={voice.supported}
            inputRef={inputRef} onSend={onSendClick} onStop={() => { if (abortRef.current) abortRef.current.abort(); }} onMic={onMic}
            onPickFile={attach.upload} onRemoveFile={() => attach.remove(false)} onCancelUpload={attach.cancelUpload}
            onOpenFile={openFilePanel} onConvert={(k) => send("Convert this to " + k)} />
        </div>

        <FilePanel panel={filePanel} onClose={closeFilePanel} />
      </div>

      <div className={"scrim" + (sideOpen || labOpen ? " show" : "")} id="scrim" onClick={closeAll} />
      <TalkBar state={talk} heard={voice.live.text} idle={voice.live.idle} answer={talk === "speaking" && conv ? speakable(((([...conv.messages].reverse().find((m) => m.role === "bot" && !m.error)) || {}).text) || "").slice(0, 220) : ""}
        onStop={() => stopTalk()} onInterrupt={interrupt} onSettings={() => setLabOpen(true)} />
      <SourcesPanel view={sourcesView} onClose={() => setSourcesView(null)} />
      <DownloadMenu menu={dlMenu} onClose={() => setDlMenu(null)}
        onPick={async (fmt) => {
          const m = dlMenu;
          setDlMenu(null);
          const err = await downloadAs(m.title, m.getMsgs(), fmt);
          if (err) showBanner(err, "info");
        }} />
      <LabPanel open={labOpen} onClose={closeAll} settings={settings} setSetting={setSetting} voices={speech.voices}
        speechSupported={speech.supported} voiceNote={voiceNote} status={status} model={model} last={lastReply} onReload={reloadModel} />
    </UiContext.Provider>
  );
}
