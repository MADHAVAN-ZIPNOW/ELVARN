// After a visitor's first answer: an optional box to leave an email (the server sends a welcome mail; needs SMTP set up on the server).
import { useState } from "react";
import { api } from "../api";

const KEY = "elvarn.lead";
export const leadAsked = () => { try { return !!localStorage.getItem(KEY); } catch (e) { return true; } };
const remember = (v) => { try { localStorage.setItem(KEY, v); } catch (e) { /* private mode: it may ask again */ } };

/** True when the server can send mail and this browser has not been asked yet. */
export async function shouldAskForEmail() {
  if (leadAsked()) return false;
  try { return !!(await api("/api/lead")).enabled; } catch (e) { return false; }
}

export function LeadCard({ question, onDone }) {
  const [email, setEmail] = useState("");
  const [agree, setAgree] = useState(false);
  const [state, setState] = useState({ busy: false, error: "", sent: false });

  const submit = async (e) => {
    e.preventDefault();
    setState({ busy: true, error: "", sent: false });
    try {
      await api("/api/lead", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, consent: agree, question }) });
      remember("given");
      setState({ busy: false, error: "", sent: true });
      setTimeout(onDone, 3500);
    } catch (err) {
      setState({ busy: false, error: err.message, sent: false });
    }
  };
  const skip = () => { remember("skipped"); onDone(); };

  return (
    <form className="leadcard" onSubmit={submit} aria-label="Leave your email">
      {state.sent ? (
        <div className="lead-note">Thank you! Please check your inbox for a welcome mail.</div>
      ) : (
        <>
          <div className="lead-title">Want to hear from ELVARN by email? <span>(optional)</span></div>
          <div className="lead-row">
            <input type="email" required placeholder="your email address" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" aria-label="Your email address" />
            <button type="submit" className="lead-send" disabled={state.busy || !agree}>{state.busy ? "Sending…" : "Send"}</button>
            <button type="button" className="lead-skip" onClick={skip}>No thanks</button>
          </div>
          <label className="lead-agree">
            <input type="checkbox" checked={agree} onChange={(e) => setAgree(e.target.checked)} />
            I agree ELVARN may keep my email and write to me. I can ask to delete it any time.
          </label>
          {state.error && <div className="lead-error" role="alert">{state.error}</div>}
        </>
      )}
    </form>
  );
}
