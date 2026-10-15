// The voice panel shown during a voice conversation: what ELVARN is doing, the words you are saying, a settings button and Close.
import { Icon, SpanIcon } from "../icons";

const SUGGESTIONS = [
  "Tell me about the DVAC in Tamil Nadu.",
  "What is the Prevention of Corruption Act?",
  "What was the amount in the DVAC raid on the Pollution Control Board?",
];

const HEAD = {
  greeting: "ELVARN is greeting you…",
  listening: "Say something…",
  thinking: "Searching the records…",
  speaking: "ELVARN is answering…",
};

export function TalkBar({ state, heard, idle, answer, onStop, onInterrupt, onSettings }) {
  if (!state) return null;
  const spoken = state === "listening" && heard && !idle ? heard : "";
  const sub = spoken || (state === "speaking" && answer ? answer : state === "listening" ? SUGGESTIONS[new Date().getMinutes() % SUGGESTIONS.length] : "");
  return (
    <div className={"talkpanel " + state} role="dialog" aria-label="Voice conversation" aria-live="polite">
      <div className="tp-top">
        <button className="tp-btn" type="button" onClick={onSettings} aria-label="Voice settings"><SpanIcon name="sliders" /></button>
        <span className="wave"><i /><i /><i /><i /><i /></span>
        <button className="tp-btn" type="button" onClick={onStop} aria-label="Close voice conversation"><SpanIcon name="x" /></button>
      </div>
      <div className="tp-main">
        <div className="tp-head">{spoken ? "I'm listening" : HEAD[state]}</div>
        <div className={"tp-sub" + (spoken ? " heard" : "")}>{sub}</div>
      </div>
      <div className="tp-actions">
        {state === "speaking" && (
          <button className="talkstop" type="button" onClick={onInterrupt} aria-label="Interrupt ELVARN and speak">
            <Icon name="mic" />
            Interrupt
          </button>
        )}
        <button className="talkstop" type="button" onClick={onStop} aria-label="Stop the voice conversation">
          <Icon name="stop" />
          Stop
        </button>
      </div>
    </div>
  );
}
