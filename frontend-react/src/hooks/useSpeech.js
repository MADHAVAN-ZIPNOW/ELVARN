// Text to speech (reading answers aloud).
import { useCallback, useEffect, useRef, useState } from "react";

const synth = typeof window !== "undefined" ? window.speechSynthesis || null : null;

export function useSpeech({ voiceName, rate, onUnsupported }) {
  const [voices, setVoices] = useState([]);
  const [speakingKey, setSpeakingKey] = useState(null);
  const keyRef = useRef(null);
  const generation = useRef(0);            // changes every time speech is stopped, so an old "finished" callback is ignored
  const settings = useRef({ voiceName, rate });
  settings.current = { voiceName, rate };

  useEffect(() => {
    if (!synth) return undefined;
    const load = () => setVoices(synth.getVoices());
    load();
    synth.onvoiceschanged = load;
    return () => { synth.onvoiceschanged = null; };
  }, []);

  const stop = useCallback(() => {
    generation.current += 1;
    if (synth) synth.cancel();
    keyRef.current = null;
    setSpeakingKey(null);
  }, []);

  /** Read `text` aloud. onDone() runs only when it was read to the end (not when it was stopped). */
  const speak = useCallback((text, key, onDone) => {
    if (!synth) { if (onUnsupported) onUnsupported(); return; }
    if (keyRef.current === key && !onDone) { stop(); return; }      // tap again to stop
    stop();
    const mine = generation.current;
    const u = new SpeechSynthesisUtterance(text);
    const v = synth.getVoices().find((x) => x.name === settings.current.voiceName);
    if (v) { u.voice = v; u.lang = v.lang; }
    u.rate = +settings.current.rate;
    const clear = () => { if (keyRef.current === key) { keyRef.current = null; setSpeakingKey(null); } };
    u.onend = () => { clear(); if (onDone && generation.current === mine) onDone(); };
    u.onerror = clear;
    keyRef.current = key;
    setSpeakingKey(key);
    synth.speak(u);
  }, [stop, onUnsupported]);

  return { supported: !!synth, voices, speakingKey, speak, stop };
}
