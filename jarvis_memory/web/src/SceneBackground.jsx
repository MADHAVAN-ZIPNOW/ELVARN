import { useEffect, useState } from "react";
import "./scene.css";
import { useReducedMotion, useVisuals } from "./visuals";

// Deterministic pseudo-random points for stars / circuit nodes (no images, no randomness at render time).
const rnd = (i, k) => { const x = Math.sin(i * 12.9898 + k * 78.233) * 43758.5453; return x - Math.floor(x); };
const STARS = Array.from({ length: 70 }, (_, i) => ({ x: rnd(i, 1) * 100, y: rnd(i, 2) * 100, r: 0.15 + rnd(i, 3) * 0.35, o: 0.3 + rnd(i, 4) * 0.7 }));
const NODES = Array.from({ length: 16 }, (_, i) => ({ x: 6 + (i % 4) * 29 + rnd(i, 5) * 8, y: 8 + Math.floor(i / 4) * 27 + rnd(i, 6) * 8 }));
const LINES = NODES.slice(0, -1).map((n, i) => {
  const m = NODES[i + 1]; const mx = (n.x + m.x) / 2;
  return `M${n.x},${n.y} L${mx},${n.y} L${mx},${m.y} L${m.x},${m.y}`;
});

function Scene({ id }) {
  switch (id) {
    case 0: return <div className="sc sc-nebula"><i className="n1" /><i className="n2" /><i className="n3" /></div>;
    case 1: return <div className="sc sc-grid"><i className="sun" /><div className="plane" /></div>;
    case 2: return <div className="sc sc-aurora"><i className="a1" /><i className="a2" /><i className="a3" /></div>;
    case 3: return (
      <div className="sc sc-stars"><i className="core" />
        <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          {STARS.map((s, i) => <circle key={i} cx={s.x} cy={s.y} r={s.r} fill="#cfe0ff" opacity={s.o} />)}
        </svg>
      </div>
    );
    default: return (
      <div className="sc sc-circuit">
        <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          {LINES.map((d, i) => <path key={i} d={d} fill="none" stroke="#2F7DFF" strokeWidth="0.25" opacity="0.6" />)}
          {NODES.map((n, i) => <circle key={i} cx={n.x} cy={n.y} r="0.9" fill="#47A6FF" />)}
        </svg>
      </div>
    );
  }
}

const COUNT = 5;

/** Two stacked layers that cross-fade (opacity, 900 ms) to the next scene every 3000 ms. */
export default function SceneBackground() {
  const [visuals] = useVisuals();
  const reduced = useReducedMotion();
  const [hidden, setHidden] = useState(typeof document !== "undefined" && document.hidden);
  const [st, setSt] = useState({ a: 0, b: 1, front: "a" });
  const run = visuals && !reduced && !hidden;

  useEffect(() => {
    const h = () => setHidden(document.hidden);
    document.addEventListener("visibilitychange", h);
    return () => document.removeEventListener("visibilitychange", h);
  }, []);

  useEffect(() => {
    if (!run) return undefined;
    const t = setInterval(() => {
      setSt((s) => (s.front === "a" ? { a: s.a, b: (s.a + 1) % COUNT, front: "b" } : { a: (s.b + 1) % COUNT, b: s.b, front: "a" }));
    }, 3000);
    return () => clearInterval(t);
  }, [run]);

  return (
    <div className="scene" aria-hidden="true">
      <div className={"scene-layer" + (st.front === "a" ? " on" : "")}><Scene id={st.a} /></div>
      <div className={"scene-layer" + (st.front === "b" ? " on" : "")}><Scene id={st.b} /></div>
      <div className="scene-scrim" />
    </div>
  );
}
