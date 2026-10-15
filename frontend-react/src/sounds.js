// Short sound cues for the voice conversation, made with the browser's audio (no sound files).
// "start": a two-note chime when you tap Talk. "listen": a soft beep when ELVARN starts listening. "stop": a lower tone when it ends.
let ctx = null;

function audio() {
  try {
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      ctx = new AC();
    }
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  } catch (e) {
    return null;
  }
}

function tone(c, freq, start, length, volume = 0.18) {
  const o = c.createOscillator();
  const g = c.createGain();
  o.type = "sine";
  o.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, start);
  g.gain.exponentialRampToValueAtTime(volume, start + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, start + length);
  o.connect(g);
  g.connect(c.destination);
  o.start(start);
  o.stop(start + length + 0.05);
}

const CUES = {
  start: [[659.25, 0, 0.16], [880, 0.14, 0.28]],        // E5 then A5: rising
  listen: [[988, 0, 0.12]],                             // one soft high beep
  stop: [[740, 0, 0.14], [523.25, 0.12, 0.24]],         // falling
};

/** Play a cue. Call the first one from a click (browsers allow sound only after a tap). */
export function cue(name, enabled = true) {
  if (!enabled) return;
  const c = audio();
  if (!c || !CUES[name]) return;
  const t = c.currentTime + 0.01;
  CUES[name].forEach(([f, at, len]) => tone(c, f, t + at, len));
}
