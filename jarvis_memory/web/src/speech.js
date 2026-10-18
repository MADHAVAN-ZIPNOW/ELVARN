import { useCallback, useEffect, useRef, useState } from "react";

const Recognition = typeof window !== "undefined" ? window.SpeechRecognition || window.webkitSpeechRecognition : null;

/**
 * Live speech-to-text with the browser's Web Speech API (feature-detected: `supported` is false where it is missing,
 * e.g. Firefox; the page then tells the user to use the keyboard's own microphone).
 * `raw()` returns exactly what the recognizer produced (kept unedited so it can be stored as raw_transcript).
 */
export function useSpeech({ onText }) {
  const [listening, setListening] = useState(false);
  const [error, setError] = useState("");
  const recRef = useRef(null);
  const rawRef = useRef("");        // final results only, exactly as recognised
  const onTextRef = useRef(onText);
  onTextRef.current = onText;

  const stop = useCallback(() => { try { recRef.current && recRef.current.stop(); } catch (e) { /* already stopped */ } }, []);

  const start = useCallback(() => {
    if (!Recognition) return;
    setError("");
    const rec = new Recognition();
    rec.lang = "en-IN";
    rec.continuous = true;
    rec.interimResults = true;
    rec.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const piece = event.results[i][0].transcript;
        if (event.results[i].isFinal) rawRef.current = (rawRef.current + " " + piece).trim();
        else interim += piece;
      }
      onTextRef.current({ final: rawRef.current, interim: interim.trim() });
    };
    rec.onerror = (e) => {
      const map = { "not-allowed": "Microphone access was blocked. Allow it in the browser's site settings.",
        "service-not-allowed": "Speech recognition is not allowed here.", "no-speech": "I didn't hear anything.",
        "audio-capture": "No microphone was found.", network: "Speech recognition needs an internet connection." };
      setError(map[e.error] || "Speech recognition stopped (" + e.error + ").");
    };
    rec.onend = () => setListening(false);
    recRef.current = rec;
    try { rec.start(); setListening(true); } catch (e) { setError("Could not start the microphone."); }
  }, []);

  useEffect(() => () => stop(), [stop]);

  return { supported: !!Recognition, listening, error, start, stop, raw: () => rawRef.current, reset: () => { rawRef.current = ""; } };
}
