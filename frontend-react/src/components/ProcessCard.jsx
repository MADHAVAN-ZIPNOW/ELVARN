// "How this reply was made": only real events from the backend (live while the reply is generated, then a summary).
import { Icon } from "../icons";
import { WHY } from "../constants";
import { pct } from "../utils";

function Step({ state, title, detail }) {
  return (
    <div className={"step " + state}>
      <span className="mark">{state === "ok" && <Icon name="check" />}</span>
      <div>
        <b>{title}</b>
        <span className="d">{detail || ""}</span>
      </div>
    </div>
  );
}

export function ProcessCard({ proc, live }) {
  const S = proc.start, L = proc.last, dn = proc.done, max = S ? S.max_new_tokens : 0;
  let word = null;
  if (live) {
    // the word is the real stage: waiting for the model -> tokenized -> generating
    word = !S && proc.searching ? "Searching the web" : !S ? "Reading your message" : !L ? "Tokenizing" : `Generating · ${L.index} tokens`;
  }
  const waited = proc.sentAt ? ((Date.now() - proc.sentAt) / 1000).toFixed(1) : "0.0";

  let tok = "not started";
  if (S) {
    tok = `${S.prompt_tokens} tokens (the model can see ${S.context_limit})`;
    if (S.prompt_tokens_before_trim > S.prompt_tokens) tok += ` · oldest ${S.prompt_tokens_before_trim - S.prompt_tokens} dropped to fit`;
    if (S.history_turns) tok += ` · ${S.history_turns} earlier turn${S.history_turns > 1 ? "s" : ""} included`;
  }
  let gen = "waiting";
  if (dn) gen = `${dn.generated_tokens} tokens in ${dn.seconds}s · ${dn.tokens_per_second} tokens/s`;
  else if (L) gen = `token ${L.index} of up to ${max} · ${L.tokens_per_second} tokens/s · last piece chosen with ${pct(L.prob)} confidence`;
  else if (S) gen = "running the model…";

  const barClass = live && !S ? "indet" : undefined;
  const barWidth = live && !S ? undefined : dn ? "100%" : max ? Math.min(100, ((proc.tokens || 0) / max) * 100) + "%" : "0";
  const webSearched = proc.search && proc.search.searched;

  return (
    <details className={"proc" + (live ? " live" : "")}>
      <summary>
        {live ? (
          <>
            <span className="runner"><i /><i /><i /><i /></span>
            <span className="shimmer">{word + "…"}</span>
          </>
        ) : (
          <>
            <span>How this reply was made</span>
            {dn && <span className="sum-meta">{`${dn.generated_tokens} tokens · ${dn.seconds}s`}</span>}
          </>
        )}
      </summary>
      <div className="bar"><i className={barClass} style={barWidth ? { width: barWidth } : undefined} /></div>
      <div className="steps">
        <Step state={S ? "ok" : "run"} title="Message sent to the local model"
          detail={S ? `${S.message_chars} characters · mode: ${S.mode}` : `waiting for the model… ${waited}s`} />
        <Step state={S ? "ok" : "pending"} title="Prompt built and tokenized" detail={tok} />
        <Step state={dn ? "ok" : S ? "run" : "pending"} title="Generating, one token at a time" detail={gen} />
        <Step state={dn ? "ok" : "pending"} title="Finished"
          detail={dn ? `${WHY[dn.stop_reason] || dn.stop_reason} · average confidence ${pct(dn.mean_token_probability)}` : ""} />
      </div>
      <div className="honest">
        {(webSearched
          ? "The server searched the web and showed the model a few short snippets as notes. The model itself does not browse or check facts: it predicts text one token at a time. "
          : "No web search was used. The model does not search or research: it predicts text one token at a time. ") +
          "Confidence = the probability it gave the piece it picked."}
      </div>
    </details>
  );
}
