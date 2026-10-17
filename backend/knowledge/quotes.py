"""Quote selection for the explanation layer: which sentences of the retrieved records answer the question?

Done by CODE, not by the model: the cross-encoder (reranker.py) scores every sentence of the final records against the question, and the best
ones (at most MAX_QUOTES, none below QUOTE_MIN_LOGIT) are the quotes. A quote is always an exact piece of a record's text (nothing is rewritten),
and it carries the record it came from. The small model only ever sees a few of these quotes; it never picks them.

The quotes are computed where the reranker lives (the knowledge service on Render, or in-process on localhost) and travel with the context.
"""
import re

MAX_QUOTES = 8
QUOTE_MIN_LOGIT = 0.0                      # the cross-encoder's neutral point: below it a sentence is not about the question
MIN_WORDS = 6
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(\[])")


def sentences(text):
    """The sentences of a record, each an exact slice of `text` (stripped), at least MIN_WORDS words long."""
    out = []
    for part in _SENTENCE_END.split(text or ""):
        part = part.strip()
        if len(part.split()) >= MIN_WORDS:
            out.append(part)
    return out


def select_quotes(question, items, reranker, k=MAX_QUOTES, min_logit=QUOTE_MIN_LOGIT):
    """[{item, item_text, quote, score}] best first: `item` is the index of the record in `items`, `item_text` its text (so a caller that
    drops or renumbers records can still find it). Empty when there is no reranker or no sentence scores above `min_logit`."""
    if reranker is None or not items:
        return []
    cands = [(idx, s) for idx, item in enumerate(items) for s in sentences(item.get("text", ""))]
    if not cands:
        return []
    logits = reranker.scores(question, [s for _, s in cands])
    out, seen = [], set()
    for (idx, sentence), logit in sorted(zip(cands, logits), key=lambda pair: -pair[1]):
        if logit < min_logit or len(out) >= k:
            break
        if sentence in seen:
            continue
        seen.add(sentence)
        out.append({"item": idx, "item_text": items[idx]["text"], "quote": sentence, "score": round(float(logit), 3)})
    return out
