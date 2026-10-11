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


@torch.inference_mode()
def generate(model, tokenizer, text, mode="chat", max_new_tokens=80,
             temperature=0.8, top_k=40, greedy=False, history=None):
    """Generate a reply. Returns the raw decoded model output plus metrics."""
    model.eval()
    prompt = build_prompt(text, mode, history)
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    if not prompt_ids:
        raise ValueError("Empty prompt.")

    context_limit = model.sequence_length
    # Keep room for at least one new token; drop the oldest tokens if needed.
    ids = prompt_ids[-(context_limit - 1):]
    prompt_token_count = len(ids)
    generated, stop_reason = [], "max_new_tokens"
    started = time.perf_counter()

    for _ in range(max_new_tokens):
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
        generated.append(next_id)

        # Stop when the model starts a new turn or a new training block.
        tail = tokenizer.decode(generated[-12:], skip_special_tokens=False)
        if any(s in tail for s in STOP_STRINGS):
            stop_reason = "new_block"
            break

    text_out = tokenizer.decode(generated, skip_special_tokens=False)
    raw_text = text_out
    for s in STOP_STRINGS:  # trim the marker that triggered the stop
        cut = text_out.find(s)
        if cut != -1:
            text_out = text_out[:cut]
    elapsed = time.perf_counter() - started

    return {
        "text": text_out.strip(),
        "raw_text": raw_text,
        "prompt": prompt,
        "prompt_tokens": prompt_token_count,
        "generated_tokens": len(generated),
        "seconds": round(elapsed, 3),
        "stop_reason": stop_reason,
    }
