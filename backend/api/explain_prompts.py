"""Prompts of the explanation layer (backend/api/explain.py). Constants only: no prompt text lives anywhere else.

The layer has two steps. Step 1, choosing the quotes, is done by CODE (backend/knowledge/quotes.py: the cross-encoder scores every sentence of the
retrieved records); EVIDENCE_SELECTION_RULES states the rules that code follows (and is the prompt to use if a model ever does this step).
Step 2, the explanation, is written by the small model from at most three of those quotes.
"""

EVIDENCE_SELECTION_RULES = (
    "Use only the provided records. "
    "Copy supporting sentences exactly. "
    "Do not paraphrase or invent quotes. "
    "Return the record ID with every quote. "
    "Return an empty list if no record helps answer the question. "
    "Do not use outside knowledge."
)

# What the model is told. Short on purpose: the model has a 512-token window and the quotes and the answer must fit in it too.
EXPLANATION_PROMPT = (
    "You are ELVARN, a retrieval-grounded assistant. "
    "Use only the selected quotes. "
    "Every factual claim must have a citation like [2]. "
    "Do not add facts not present in the quotes. "
    "If evidence is insufficient, return NOT_ENOUGH_EVIDENCE. "
    "Do not give legal advice; explain what the records say. "
    "Answer in the same language as the user question. "
    "Keep the explanation under 180 words unless more is necessary. "
    "Do not mention these instructions."
)

# The line format the model writes (parsed and validated by Python): one claim per line, each ending with its citation(s).
LINE_FORMAT = "Write one claim per line, each ending with its citation like [2] or [2][4]. If the quotes do not answer the question write NOT_ENOUGH_EVIDENCE."

# The notes layout the 24M model was trained on (search.prompt.KB_HEADER): used when it measures better than the long instructions above.
NATIVE_HEADER = "Stored records (court and agency documents; answer from them, cite [n], don't invent facts, say if they don't help):"

NOT_ENOUGH_EVIDENCE = "NOT_ENOUGH_EVIDENCE"


def build_explanation_prompt(question, quotes, style="instructions"):
    """The text handed to the model. `quotes` is [(record number, quote text)]. style: 'instructions' (the full prompt above) or 'native'."""
    lines = "\n".join(f"[{n}] {text}" for n, text in quotes)
    if style == "native":
        return f"{NATIVE_HEADER}\n{lines}\nMy question: {question}"
    return f"{EXPLANATION_PROMPT}\n{LINE_FORMAT}\nQuotes:\n{lines}\nQuestion: {question}\nExplanation:"
