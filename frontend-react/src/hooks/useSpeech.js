// Text to speech (reading answers aloud).
import { useCallback, useEffect, useRef, useState } from "react";

const synth = typeof window !== "undefined" ? window.speechSynthesis || null : null;

export function useSpeech({ voiceName, rate, onUnsupported }) {
  const [voices, setVoices] = useState([]);
  const [speakingKey, setSpeakingKey] = useState(null);
  const keyRef = useRef(null);
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
    if (synth) synth.cancel();
    keyRef.current = null;
    setSpeakingKey(null);
  }, []);

  const speak = useCallback((text, key) => {
    if (!synth) { if (onUnsupported) onUnsupported(); return; }
    if (keyRef.current === key) { stop(); return; }      // tap again to stop
    stop();
    const u = new SpeechSynthesisUtterance(text);
    const v = synth.getVoices().find((x) => x.name === settings.current.voiceName);
    if (v) { u.voice = v; u.lang = v.lang; }
    u.rate = +settings.current.rate;
    const done = () => { if (keyRef.current === key) { keyRef.current = null; setSpeakingKey(null); } };
    u.onend = done;
    u.onerror = done;
    keyRef.current = key;
    setSpeakingKey(key);
    synth.speak(u);
  }, [stop, onUnsupported]);

  return { supported: !!synth, voices, speakingKey, speak, stop };
}
