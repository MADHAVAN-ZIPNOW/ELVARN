import "./orb.css";

// Animated JARVIS orb: core sphere, three tilted orbit rings with particles, outer halo. CSS only; animates transform/opacity.
const RINGS = [
  { cls: "r1", n: 6 },
  { cls: "r2", n: 5 },
  { cls: "r3", n: 5 },
];

export default function Orb({ state = "idle", size = 220, className = "" }) {
  return (
    <div className={"orb " + className} data-state={state} style={{ "--orb-size": size + "px" }} role="img" aria-label={"Jarvis orb: " + state}>
      <div className="orb-shake">
        <span className="orb-halo" />
        <span className="orb-wave w1" />
        <span className="orb-wave w2" />
        <div className="orb-stage">
          {RINGS.map((r) => (
            <div className={"orb-tilt " + r.cls} key={r.cls}>
              <div className="orb-spin">
                {Array.from({ length: r.n }, (_, i) => (
                  <i key={i} className="orb-p" style={{ "--a": (360 / r.n) * i + "deg", "--s": 0.7 + ((i * 37) % 5) * 0.15 }} />
                ))}
              </div>
            </div>
          ))}
          <div className="orb-breathe"><span className="orb-core" /></div>
        </div>
      </div>
    </div>
  );
}
