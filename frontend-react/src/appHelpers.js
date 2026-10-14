// Pure helpers for App: the first state, how a stored message becomes an answer row, the rocket that flies off when you send.
import { loadState } from "./storage";
import { iconSvg } from "./icons";
import { uid } from "./utils";

export const newConv = () => ({ id: uid(), title: "New chat", created: Date.now(), messages: [] });

export function initialState() {
  const s = loadState();
  let { convs, currentId } = s;
  if (!convs.find((c) => c.id === currentId)) {
    const last = convs.find((c) => c.messages.length);
    if (last) currentId = last.id;
    else { const c = newConv(); convs = [c, ...convs]; currentId = c.id; }
  }
  return { ...s, convs, currentId };
}

/** What one stored message looks like as an answer row. */
export function viewOf(m) {
  if (m.error) return { error: true, text: m.text };
  const meta = m.meta;
  if (meta && meta.file_action) return { fileAction: true, text: m.text, files: meta.files, meta };
  return { text: m.text, proc: meta && meta.proc ? meta.proc : null, procLive: false, meta };
}

export function launchRocket(button) {
  if (!button || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const r = button.getBoundingClientRect(), g = document.createElement("span");
  g.className = "rocket-ghost";
  g.appendChild(iconSvg("rocket"));
  g.style.left = r.left + r.width / 2 - 13 + "px";
  g.style.top = r.top + r.height / 2 - 13 + "px";
  document.body.appendChild(g);
  setTimeout(() => g.remove(), 950);
}
