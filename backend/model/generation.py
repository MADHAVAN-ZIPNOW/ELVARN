"""Autoregressive text generation for ELVARN (shared by the CLI and the API)."""
import time

import torch

# Prompt formats match how documents were written in the training data.
MODES = {
    # <CASUAL_CONVERSATION>\nUser: ...\nAssistant:
    "chat": ("<CASUAL_CONVERSATION>\nUser: {q}\nAssistant:", "chat"),
    # <DOMAIN_KNOWLEDGE>\nTopic: ...\nQuestion: ...\nAnswer:
    "domain": ("<DOMAIN_KNOWLEDGE>\nTopic: General\nQuestion: {q}\nAnswer:", "qa"),
    "english": ("<GENERAL_ENGLISH>\nQuestion: {q}\nAnswer:", "qa"),
    "humour": ("<HUMOUR>\nUser: {q}\nAssistant:", "chat"),
    # Raw continuation of the text exactly as typed.
    "raw": ("{q}", "raw"),
}

# Text that means the model has started a new turn / block; cut there.
STOP_STRINGS = ["\nUser:", "\nQuestion:", "\nTopic:", "<DOMAIN_KNOWLEDGE>",
                "<GENERAL_ENGLISH>", "<CASUAL_CONVERSATION>", "<HUMOUR>"]


def build_prompt(text, mode, history=None):
    """history: optional list of (user, assistant) pairs, chat-style modes only."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {', '.join(MODES)}")
    template, kind = MODES[mode]
    if kind == "chat" and history:
        tag = template.split("\n", 1)[0]
        turns = "".join(f"User: {u.strip()}\nAssistant: {a.strip()}\n" for u, a in history)
        return f"{tag}\n{turns}User: {text.strip()}\nAssistant:"
    return template.format(q=text.strip())


def sample_next(logits, temperature, top_k, greedy):
    if greedy or temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < values[-1], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def _holdback(text):
    """Characters at the end of `text` that could still become a stop string.

    While streaming, these are not shown yet, so a marker like "
User:" never
    flashes on screen before being cut off.
    """
    longest = 0
    for stop in STOP_STRINGS:
        for k in range(min(len(stop) - 1, len(text)), 0, -1):
            if text.endswith(stop[:k]):
                longest = max(longest, k)
                break
    return longest


def _trim(text):
    for stop in STOP_STRINGS:
        cut = text.find(stop)
        if cut != -1:
            text = text[:cut]
    return text


@torch.inference_mode()
def generate_stream(model, tokenizer, text, mode="chat", max_new_tokens=80,
                    temperature=0.8, top_k=40, greedy=False, history=None):
    """Generate a reply and yield real events while it happens.

    Events (dicts):
      start  - prompt was built and tokenized (real counts)
      token  - one generated token: its text, its probability, running speed
      done   - the final result (same fields generate() returns)
    """
    model.eval()
    prompt = build_prompt(text, mode, history)
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    if not prompt_ids:
        raise ValueError("Empty prompt.")

    context_limit = model.sequence_length
    # Keep room for at least one new token; drop the oldest tokens if needed.
    ids = prompt_ids[-(context_limit - 1):]
    prompt_token_count = len(ids)
    generated, probs_chosen, stop_reason = [], [], "max_new_tokens"
    started = time.perf_counter()

    yield {
        "event": "start", "mode": mode, "message_chars": len(text.strip()),
        "prompt_tokens": prompt_token_count,
        "prompt_tokens_before_trim": len(prompt_ids),
        "context_limit": context_limit, "max_new_tokens": max_new_tokens,
        "temperature": temperature, "top_k": top_k, "greedy": greedy,
        "history_turns": len(history) if history else 0,
    }

    shown = 0
    for step in range(max_new_tokens):
        window = torch.tensor([(ids + generated)[-context_limit:]], dtype=torch.long)
        next_logits = model(window)[0, -1, :]
        # Never emit padding / BOS / UNK as an answer token.
        next_logits[[tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id]] = float("-inf")
        next_id = sample_next(next_logits, temperature, top_k, greedy)
        if not 0 <= next_id < tokenizer.vocab_size:
            raise RuntimeError(f"Generated invalid token id {next_id}.")
        if next_id == tokenizer.eos_id:
            stop_reason = "eos"
            break
        # Confidence the model gave this token (plain softmax, temperature 1).
        probs_chosen.append(float(torch.softmax(next_logits, dim=-1)[next_id]))
        generated.append(next_id)

        full = tokenizer.decode(generated, skip_special_tokens=False)
        if any(stop in full for stop in STOP_STRINGS):
            stop_reason = "new_block"
            break

        safe = len(full) - _holdback(full)
        delta = full[shown:safe] if safe > shown else ""
        shown = max(shown, safe)
        elapsed = time.perf_counter() - started
        yield {"event": "token", "index": len(generated), "text": delta,
               "prob": round(probs_chosen[-1], 4),
               "seconds": round(elapsed, 3),
               "tokens_per_second": round(len(generated) / elapsed, 1) if elapsed else 0.0}

    raw_text = tokenizer.decode(generated, skip_special_tokens=False)
    elapsed = time.perf_counter() - started
    yield {
        "event": "done",
        "text": _trim(raw_text).strip(),
        "raw_text": raw_text,
        "prompt": prompt,
        "prompt_tokens": prompt_token_count,
        "generated_tokens": len(generated),
        "seconds": round(elapsed, 3),
        "tokens_per_second": round(len(generated) / elapsed, 1) if elapsed else 0.0,
        "mean_token_probability": round(sum(probs_chosen) / len(probs_chosen), 4) if probs_chosen else None,
        "stop_reason": stop_reason,
    }


def generate(model, tokenizer, text, **kwargs):
    """Generate a reply (no streaming). Returns the final result dict."""
    result = None
    for event in generate_stream(model, tokenizer, text, **kwargs):
        if event["event"] == "done":
            result = event
    result.pop("event", None)
    return result
