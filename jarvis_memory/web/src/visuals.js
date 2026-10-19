import { useEffect, useState } from "react";

// "Animated visuals" preference. Persisted in localStorage["jarvis.visuals"] (default on); storage may be unavailable.
const KEY = "jarvis.visuals";
const EVT = "jarvis-visuals";

export function readVisuals() {
  try { return localStorage.getItem(KEY) !== "off"; } catch (e) { return true; }
}
function apply(on) {
  try { document.documentElement.dataset.visuals = on ? "on" : "off"; } catch (e) { /* ignore */ }
}
export function setVisuals(on) {
  try { localStorage.setItem(KEY, on ? "on" : "off"); } catch (e) { /* ignore */ }
  apply(on);
  window.dispatchEvent(new CustomEvent(EVT, { detail: on }));
}
apply(readVisuals());

export function useVisuals() {
  const [on, setOn] = useState(readVisuals);
  useEffect(() => {
    const h = (e) => setOn(!!e.detail);
    window.addEventListener(EVT, h);
    return () => window.removeEventListener(EVT, h);
  }, []);
  return [on, setVisuals];
}

export function useReducedMotion() {
  const [r, setR] = useState(() => { try { return window.matchMedia("(prefers-reduced-motion: reduce)").matches; } catch (e) { return false; } });
  useEffect(() => {
    let mq;
    try { mq = window.matchMedia("(prefers-reduced-motion: reduce)"); } catch (e) { return undefined; }
    const h = () => setR(mq.matches);
    mq.addEventListener ? mq.addEventListener("change", h) : mq.addListener(h);
    return () => { mq.removeEventListener ? mq.removeEventListener("change", h) : mq.removeListener(h); };
  }, []);
  return r;
}
