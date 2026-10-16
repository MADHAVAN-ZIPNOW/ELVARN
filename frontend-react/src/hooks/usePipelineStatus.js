// Polls GET /api/collect/status: every 1.5 s while collecting, every 3 s when idle, paused while the tab is hidden, aborted on unmount.
import { useCallback, useEffect, useRef, useState } from "react";
import { collectCall } from "../pipeline/collectApi";
import { measureRates } from "../pipeline/format";

export function usePipelineStatus(ownerKey) {
  const [status, setStatus] = useState(null);
  const [rates, setRates] = useState({});
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");                      // "" | "start" | "stop": a control request is in flight
  const ctl = useRef({ abort: null, timer: 0, prev: null, prevAt: 0, rates: {}, alive: true });

  const take = useCallback((data) => {
    const c = ctl.current, now = Date.now();
    if (!data || typeof data !== "object") return;
    c.rates = measureRates(c.prev, data, (now - c.prevAt) / 1000, c.rates);
    c.prev = data; c.prevAt = now;
    setRates(c.rates);
    setStatus(data);
    setError("");
  }, []);

  const poll = useCallback(async () => {
    const c = ctl.current;
    clearTimeout(c.timer);
    if (!c.alive) return;
    let next = 3000;
    if (!document.hidden) {
      if (c.abort) c.abort.abort();
      const ab = new AbortController();
      c.abort = ab;
      try { take(await collectCall(ownerKey, "/status", { signal: ab.signal })); }
      catch (e) { if (e.name === "AbortError" || !c.alive) return; setError(e.message || "The collector did not answer."); }
      const st = c.prev && c.prev.state;
      next = st === "running" || st === "stopping" ? 1500 : 3000;
    }
    if (c.alive) c.timer = setTimeout(poll, next);
  }, [ownerKey, take]);

  useEffect(() => {
    const c = ctl.current;
    c.alive = true;
    poll();
    const onVis = () => { if (!document.hidden) poll(); else clearTimeout(c.timer); };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      c.alive = false; clearTimeout(c.timer);
      if (c.abort) c.abort.abort();
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [poll]);

  // after a request that changes the state, check again soon (fast interval) instead of waiting for the idle timer
  const soon = useCallback(() => { const c = ctl.current; clearTimeout(c.timer); if (c.alive) c.timer = setTimeout(poll, 700); }, [poll]);

  const control = useCallback(async (what) => {
    if (busy) return;
    setBusy(what);
    try { take(await collectCall(ownerKey, "/" + what, { method: "POST", body: "{}" })); soon(); }
    catch (e) { setError(e.message || "Request failed."); }
    finally { setBusy(""); }
  }, [busy, ownerKey, take, soon]);

  /** POST /targeted. Returns the "request" object (always an object, so the caller never has to guard). */
  const targeted = useCallback(async (body) => {
    if (busy) return { ok: false, message: "Another request is in progress." };
    setBusy("targeted");
    try {
      const data = await collectCall(ownerKey, "/targeted", { method: "POST", body: JSON.stringify(body) });
      take(data);
      const rq = data && data.request && typeof data.request === "object" ? data.request : { ok: false, message: "The server sent no answer for this request." };
      if (rq.ok) soon();
      return rq;
    } catch (e) { return { ok: false, message: e.message || "Request failed." }; }
    finally { setBusy(""); }
  }, [busy, ownerKey, take, soon]);

  return { status, rates, error, busy, start: () => control("start"), stop: () => control("stop"), targeted };
}
