// The frame around the chat: sidebar, top bar and the error / info banner.
import { Icon, SpanIcon } from "../icons";
import { MODES, MODE_PAGES } from "../constants";
import { useTilt } from "../hooks/useTilt";
import { Logo } from "./Logo";

function ModeKey({ mode, active, onOpen }) {
  const tilt = useTilt(10, 14);
  return (
    <button className={"key k-" + mode.col + (active ? " active" : "")} type="button" role="tab" aria-selected={active}
      title={mode.label + " - " + mode.hint} onClick={() => onOpen(mode.id)} {...tilt}>
      <span className="kicon"><Icon name={mode.icon} /></span>
      <span className="klabel">{mode.label}</span>
    </button>
  );
}

export function Sidebar({ open, mode, view, sessionCount, onToggle, onNewChat, onOpenSessions, onOpenMode }) {
  const tiltKey = useTilt(8, 12);
  const tiltKey2 = useTilt(8, 12);
  const tiltWord = useTilt(16, 26);
  const tiltOrb = useTilt(20, 26);
  return (
    <aside className={"side" + (open ? " show" : "")} id="side" aria-label="Navigation">
      <div className="brand">
        <button className="orbbtn" id="sideMenu" type="button" aria-label="Collapse sidebar" title="Collapse sidebar" onClick={onToggle} {...tiltOrb}>
          <SpanIcon name="menu" />
        </button>
        <span className="brandlogo"><Logo /></span>
        <h1 className="wordmark3d" id="sideWordmark" {...tiltWord}>ELVARN</h1>
      </div>
      <div className="sidelabel">Modes</div>
      <div className="modes">
        <button className="key solid k-blue" id="newChat" type="button" title="New chat" onClick={onNewChat} {...tiltKey}>
          <span className="kicon"><SpanIcon name="plus" /></span>
          <span className="klabel">New chat</span>
        </button>
        <div className="modelist" id="modes" role="tablist" aria-label="Prompt style">
          {MODES.filter((m) => !m.hidden).map((m) => (
            <ModeKey key={m.id} mode={m} active={view !== "sessions" && m.id === mode} onOpen={onOpenMode} />
          ))}
        </div>
      </div>
      <div className="sidelabel">Library</div>
      <button className={"key k-yellow" + (view === "sessions" ? " active" : "")} id="sessionsNav" type="button" title="Sessions" onClick={onOpenSessions} {...tiltKey2}>
        <span className="kicon"><SpanIcon name="sessions" /></span>
        <span className="klabel">Sessions</span>
        <span className="kcount" id="sessCount">{sessionCount}</span>
      </button>
    </aside>
  );
}

export function Header({ view, mode, convTitle, hasMessages, talking, onToggleSide, onTalk, onTheme, onClear, onLab }) {
  const tiltWord = useTilt(16, 26);
  const tiltOrb = useTilt(20, 26);
  const sv = view === "sessions";
  return (
    <header>
      <button className="orbbtn menu-btn" id="menuBtn" type="button" aria-label="Toggle sidebar" title="Toggle sidebar" onClick={onToggleSide} {...tiltOrb}>
        <SpanIcon name="menu" />
      </button>
      <div className="hdrbrandbox">
        <span className="hdrlogo"><Logo /></span>
        <h1 className="wordmark3d" id="wordmark" {...tiltWord}>ELVARN</h1>
      </div>
      <div className="hdrmid">
        <div className="ttl" id="chatTitle">{sv ? "" : (MODE_PAGES[mode] || MODE_PAGES.chat).h}</div>
        <div className="ttl-sub" id="chatSub">{!sv && hasMessages ? convTitle : ""}</div>
      </div>
      <div className="hdricons">
        <button className={"icon-btn talkhdr" + (talking ? " on" : "")} id="talkBtn" type="button" title={talking ? "Stop the voice conversation" : "Talk to ELVARN (works on every page)"}
          aria-label={talking ? "Stop the voice conversation" : "Talk to ELVARN"} aria-pressed={!!talking} onClick={onTalk}><SpanIcon name="headphones" /></button>
        <button className="icon-btn" id="themeBtn" type="button" title="Switch theme" aria-label="Switch theme" onClick={onTheme}><SpanIcon name="theme" /></button>
        <button className="icon-btn" id="clearBtn" type="button" title="Clear this chat" aria-label="Clear this chat" hidden={sv} onClick={onClear}><SpanIcon name="trash" /></button>
        <button className="icon-btn" id="labBtn" type="button" title="Model lab and settings" aria-label="Open model lab and settings" onClick={onLab}><SpanIcon name="sliders" /></button>
      </div>
    </header>
  );
}

export function Banner({ banner }) {
  const { msg, kind } = banner;
  return (
    <div className={"banner" + (msg ? " show" : "") + (kind === "info" ? " info" : "")} id="banner" role="alert">{msg}</div>
  );
}
