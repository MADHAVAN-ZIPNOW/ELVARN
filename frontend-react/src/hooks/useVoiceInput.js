// Voice to text with the browser's speech recognition (Chrome / Edge).
import { useCallback, useEffect, useRef, useState } from "react";

const SR = typeof window !== "undefined" ? window.SpeechRecognition || window.webkitSpeechRecognition : null;

const ERRORS = {
  "not-allowed": "Microphone permission was blocked. Allow it in the browser address bar, then try again.",
  "service-not-allowed": "Microphone permission was blocked. Allow it in the browser address bar, then try again.",
  "no-speech": "I didn't hear anything. Tap the mic and try again.",
  "audio-capture": "No microphone found.",
  network: "Speech service unreachable (this browser needs internet for voice input).",
};

/**
 * getText / setText read and write the message box. onFinished(text) is called when listening ends after real words
 * (the caller decides whether to send it). onBanner(message) shows an error.
 */
export function useVoiceInput({ lang, getText, setText, onFinished, onBanner, beforeStart }) {
  const [on, setOn] = useState(false);
  const [live, setLive] = useState({ show: false, text: "", idle: false });
  const r = useRef({ rec: null, on: false, base: "", gotWords: false });
  const cb = useRef({});
  cb.current = { lang, getText, setText, onFinished, onBanner, beforeStart };

  const stop = useCallback(() => {
    try { r.current.rec && r.current.rec.stop(); } catch (e) { /* already stopped */ }
  }, []);

  const start = useCallback(() => {
    const c = cb.current;
    if (c.beforeStart) c.beforeStart();
    const rec = new SR();
    rec.lang = c.lang;
    rec.interimResults = true;
    rec.continuous = true;
    rec.maxAlternatives = 1;
    const cur = c.getText();
    r.current = { rec, on: false, base: cur ? cur.trim() + " " : "", gotWords: false, lastText: "" };
    rec.onstart = () => {
      r.current.on = true;
      setOn(true);
      cb.current.onBanner("");
      setLive({ show: true, text: "Listening… start speaking", idle: true });
    };
    rec.onresult = (ev) => {
      let fin = "", interim = "";
      for (let i = 0; i < ev.results.length; i++) {
        const res = ev.results[i];
        if (res.isFinal) fin += res[0].transcript + " "; else interim += res[0].transcript;
      }
      r.current.gotWords = true;
      const spoken = (fin + interim).trim();
      r.current.lastText = (r.current.base + spoken).trim();
      cb.current.setText(r.current.lastText);
      setLive({ show: true, text: spoken || "Listening… start speaking", idle: !spoken });
    };
    rec.onerror = (ev) => {
      if (ev.error === "aborted") return;
      cb.current.onBanner(ERRORS[ev.error] || "Voice input error: " + ev.error);
    };
    rec.onend = () => {
      const wasOn = r.current.on;
      r.current.on = false;
      setOn(false);
      setLive({ show: false, text: "", idle: false });
      cb.current.onFinished(wasOn && r.current.gotWords, r.current.lastText);          // the heard text is passed along, not read back from screen state
    };
    try { rec.start(); } catch (e) { c.onBanner("Could not start the microphone."); }
  }, []);

  useEffect(() => () => { try { r.current.rec && r.current.rec.abort(); } catch (e) { /* ignore */ } }, []);

  return { supported: !!SR, on, live, start, stop };
}
