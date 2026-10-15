// Small helpers for the voice conversation: a changing greeting, and the visitor's name when they have told it.
const NAME_KEY = "elvarn.name";
const NAME_RE = /\b(?:my name is|call me|i am|i'm|this is)\s+([A-Za-z][A-Za-z.'-]{1,20})\b/i;
const NOT_NAMES = new Set(["a", "an", "the", "not", "so", "very", "good", "fine", "great", "ok", "okay", "sad", "happy", "tired", "bored", "here", "back", "new", "sure", "just", "in", "from", "looking", "trying", "asking", "wondering", "interested", "curious"]);

export function rememberName(text) {
  const m = NAME_RE.exec(text || "");
  if (!m || NOT_NAMES.has(m[1].toLowerCase())) return;
  try { localStorage.setItem(NAME_KEY, m[1][0].toUpperCase() + m[1].slice(1)); } catch (e) { /* private mode */ }
}

export function knownName() {
  try { return localStorage.getItem(NAME_KEY) || ""; } catch (e) { return ""; }
}

const PLAIN = [
  "Hi, I am ELVARN. What would you like to know?",
  "Hello! ELVARN here. What shall we talk about?",
  "Hey there! I am ELVARN. Ask me anything about the records I know.",
  "Hi! Good to hear you. What are you curious about today?",
];
const NAMED = [
  "Welcome back, {n}! What would you like to know?",
  "Hi {n}! ELVARN here. What shall we look into today?",
  "Hello {n}! What are you curious about today?",
];

export function greeting() {
  const n = knownName();
  const list = n ? NAMED : PLAIN;
  return list[Math.floor(Math.random() * list.length)].replace("{n}", n);
}
