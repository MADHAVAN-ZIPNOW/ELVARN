// The conversation: your messages, the model's answers (with the process card, facts, source button and action bar) and the welcome page.
import { Icon } from "../icons";
import { MODE_PAGES } from "../constants";
import { useTilt } from "../hooks/useTilt";
import { Logo } from "./Logo";
import { CopyButton, ActionBar } from "./ActionBar";
import { AnswerInfo } from "./AnswerInfo";
import { FileChip, DownloadCard } from "./FileParts";
import { ProcessCard } from "./ProcessCard";
import { RichText } from "./RichText";

export function Welcome({ mode, onPick, onTalk }) {
  const pg = MODE_PAGES[mode] || MODE_PAGES.chat;
  return (
    <div className="welcome">
      <div className="homelogo"><Logo /></div>
      <h2>{mode === "chat" ? <>Hi, I'm <em>ELVARN</em>.</> : <em>{pg.h}</em>}</h2>
      <p>{pg.blurb || "A small language model running on your own computer. Type, or tap the mic and talk — then tap Listen to hear the reply."}</p>
      {onTalk && (
        <button className="talkbtn gloss c-blue" type="button" onClick={onTalk} aria-label="Talk to ELVARN with your voice">
          <Icon name="mic" />
          Talk to ELVARN
        </button>
      )}
      <div className="cards">
        {pg.tips.map((s) => (
          <button key={s.t} className="card" type="button" onClick={() => onPick(s.t)}>
            <span className={"tile c-" + s.col}><Icon name={s.i} /></span>
            <span><b>{s.t}</b><small>{s.s}</small></span>
          </button>
        ))}
      </div>
    </div>
  );
}

export function UserRow({ message, index }) {
  const tilt = useTilt(8, 14);
  return (
    <div className="row user" data-user-row={index}>
      <div className="msg">
        <div className="bubble" {...tilt}>
          {message.file && <FileChip file={message.file} />}
          {message.voice && <span className="voicemark" title="Said aloud" aria-label="Said aloud"><Icon name="mic" /></span>}
          {message.text}
        </div>
        <div className="tools utools"><CopyButton text={message.text} /></div>
      </div>
    </div>
  );
}

/**
 * One answer. `view` is {error, text, plain, proc, procLive, meta, files, fileAction}:
 *  - stored message: rich text, process summary, answer info, action bar
 *  - live message (plain=true): the text as it streams, with the live process card
 */
export function BotRow({ view, speaking, onSpeak, onDetails, onOpenSources, onFeedback, onDownload }) {
  const { error, text, plain, proc, procLive, meta, files, fileAction } = view;
  const finished = !error && !fileAction && !plain;
  return (
    <div className={"row bot" + (error ? " err" : "")}>
      <div className="msg">
        <div>{proc && !error && !fileAction && <ProcessCard proc={proc} live={procLive} />}</div>
        <div className={"bubble" + (finished && text ? " rich" : "") + (speaking ? " speaking" : "")}>
          {error || fileAction || plain ? (
            text
          ) : text ? (
            <RichText text={text} onCite={(n) => onOpenSources(meta && meta.search, n)} />
          ) : (
            <span className="empty-out">(the model produced no text)</span>
          )}
        </div>
        <div>{(files || []).map((f, i) => <DownloadCard key={(f.download_id || "") + i} file={f} />)}</div>
        <div>{finished && meta && <AnswerInfo meta={meta} />}</div>
        <div>
          {finished && (
            <ActionBar
              text={text}
              meta={meta}
              speaking={speaking}
              onSpeak={onSpeak}
              onDetails={onDetails}
              onOpenSources={(info) => onOpenSources(info)}
              onFeedback={onFeedback}
              onDownload={onDownload}
            />
          )}
        </div>
      </div>
    </div>
  );
}
