// The ELVARN logo: a canvas redrawn about 30 times a second (bobbing, glowing, a light sweep, a spinning star).
// Every <Logo/> on the page registers its canvas with one shared animation loop.
import { useEffect, useRef } from "react";
import brainUrl from "../assets/logo-brain.svg";
import starUrl from "../assets/logo-star.svg";

export const LOGO_META = { ratio: 0.9872, star: { x: 0.3626943005181347, y: 0.01278772378516624, w: 0.27202072538860106, h: 0.2506393861892583 } };
if (typeof document !== "undefined") document.documentElement.style.setProperty("--lg-ratio", String(LOGO_META.ratio));

const canvases = new Set();
const imgs = { brain: new Image(), star: new Image(), ok: 0 };
imgs.brain.onload = imgs.star.onload = () => { imgs.ok++; };
imgs.brain.src = brainUrl;
imgs.star.src = starUrl;

function drawLogo(c, t) {
  const r = c.getBoundingClientRect();
  if (!r.width || !r.height) return;
  const dpr = Math.min(3, Math.max(1.5, window.devicePixelRatio || 1));
  const W = Math.round(r.width * dpr), H = Math.round(r.height * dpr);
  if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
  const x = c.getContext("2d"), bi = imgs.brain, si = imgs.star;
  x.clearRect(0, 0, W, H);
  x.imageSmoothingEnabled = true;
  x.imageSmoothingQuality = "high";
  const ratio = bi.naturalWidth / bi.naturalHeight, w = Math.min(W, H * ratio) * 0.94, h = w / ratio, ox = (W - w) / 2;
  const oy = (H - h) / 2 + Math.sin((t * 2 * Math.PI) / 4) * H * 0.03 - H * 0.01;          // gentle bob
  x.drawImage(bi, ox, oy, w, h);
  x.save();
  x.globalCompositeOperation = "source-atop";
  x.fillStyle = "rgba(255,255,255," + (0.07 * (1 + Math.sin((t * 2 * Math.PI) / 3.2))).toFixed(3) + ")";
  x.fillRect(0, 0, W, H);
  x.restore();                                                                              // glow pulse
  const phase = (t % 3.8) / 3.8;                                                            // light sweep, left to right, then a pause
  if (phase > 0.3) {
    const q = (phase - 0.3) / 0.7, gx = ox - w * 0.5 + q * w * 2.0, g = x.createLinearGradient(gx, oy, gx + w * 0.55, oy + h * 0.35);
    g.addColorStop(0, "rgba(255,255,255,0)");
    g.addColorStop(0.5, "rgba(255,255,255,.85)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    x.save();
    x.globalCompositeOperation = "source-atop";
    x.fillStyle = g;
    x.fillRect(0, 0, W, H);
    x.restore();
  }
  const b = LOGO_META.star, sw = b.w * w, sh = b.h * h, sx = ox + (b.x + b.w / 2) * w, sy = oy + (b.y + b.h / 2) * h;
  const sc = 1 + 0.15 * Math.sin((t * 2 * Math.PI) / 1.7);
  x.save();
  x.translate(sx, sy);
  x.rotate((-t * 2 * Math.PI) / 8);
  x.scale(sc, sc);
  x.drawImage(si, -sw / 2, -sh / 2, sw, sh);
  x.restore();                                                                              // spinning, pulsing star
}

let loopStarted = false;
function startLoop() {
  if (loopStarted) return;
  loopStarted = true;
  let last = 0;
  const tick = (ts) => {
    requestAnimationFrame(tick);
    if (ts - last < 33 || imgs.ok < 2) return;
    last = ts;
    canvases.forEach((c) => {
      if (!c.isConnected) { canvases.delete(c); return; }
      if (c.checkVisibility && !c.checkVisibility({ visibilityProperty: true })) return;    // not on screen
      drawLogo(c, ts / 1000);
    });
  };
  requestAnimationFrame(tick);
}

export function Logo() {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    canvases.add(c);
    startLoop();
    return () => { canvases.delete(c); };
  }, []);
  return (
    <span className="logo-wrap">
      <canvas className="lg-canvas" role="img" aria-label="ELVARN logo" ref={ref} />
    </span>
  );
}

/** The browser-tab icon moves too: pre-rendered frames (star turning, light sweeping) swapped every 200 ms. */
export function startAnimatedTabIcon() {
  let link = document.querySelector('link[rel="icon"]');
  if (!link) { link = document.createElement("link"); link.rel = "icon"; document.head.appendChild(link); }
  const bi = new Image(), si = new Image();
  let loaded = 0;
  const go = () => { if (++loaded < 2) return; build(); };
  bi.onload = go;
  si.onload = go;
  bi.src = brainUrl;
  si.src = starUrl;
  function build() {
    const S = 64, N = 12, frames = [], r = bi.naturalWidth / bi.naturalHeight;
    const w = r >= 1 ? S : S * r, h = r >= 1 ? S / r : S, ox = (S - w) / 2;
    for (let i = 0; i < N; i++) {
      const c = document.createElement("canvas");
      c.width = c.height = S;
      const x = c.getContext("2d"), t = i / N;
      const oy = (S - h) / 2 + Math.sin(t * 2 * Math.PI) * 1.2;
      x.drawImage(bi, ox, oy, w, h);
      x.save();
      x.globalCompositeOperation = "source-atop";
      const gx = -S * 0.6 + t * S * 2.0, g = x.createLinearGradient(gx, 0, gx + S * 0.5, S * 0.35);
      g.addColorStop(0, "rgba(255,255,255,0)");
      g.addColorStop(0.5, "rgba(255,255,255,.75)");
      g.addColorStop(1, "rgba(255,255,255,0)");
      x.fillStyle = g;
      x.fillRect(0, 0, S, S);
      x.restore();
      const b = LOGO_META.star, sw = b.w * w, sh = b.h * h, sx = ox + (b.x + b.w / 2) * w, sy = oy + (b.y + b.h / 2) * h;
      const sc = 1 + 0.14 * Math.sin(t * 4 * Math.PI);
      x.save();
      x.translate(sx, sy);
      x.rotate(-t * 2 * Math.PI);
      x.scale(sc, sc);
      x.drawImage(si, -sw / 2, -sh / 2, sw, sh);
      x.restore();
      frames.push(c.toDataURL("image/png"));
    }
    link.type = "image/png";
    let k = 0;
    link.href = frames[0];
    setInterval(() => { if (!document.hidden) { k = (k + 1) % N; link.href = frames[k]; } }, 200);
  }
}
