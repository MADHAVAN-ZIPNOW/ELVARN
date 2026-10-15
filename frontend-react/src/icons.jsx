// Vector icons (they follow the text colour or use the gradient fills defined in <GradientDefs/>).
const PATHS = {
  plus: '<path d="M12 5v14M5 12h14"/>',
  plusb: '<path d="M12 5v14M5 12h14"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 002 2h6a2 2 0 002-2l1-12M9 7V4h6v3"/>',
  sliders: '<circle cx="12" cy="12" r="3"/><path d="M10.4 3.5h3.2l.5 2.3 1.4.8 2.2-.8 1.6 2.8-1.7 1.6v1.6l1.7 1.6-1.6 2.8-2.2-.8-1.4.8-.5 2.3h-3.2l-.5-2.3-1.4-.8-2.2.8-1.6-2.8 1.7-1.6v-1.6L4.7 8.3l1.6-2.8 2.2.8 1.4-.8z"/>',
  theme: '<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 010 16z" fill="currentColor"/>',
  upload: '<path d="M6.4 10.8h6.4l3.8 3.8v5.7a1.4 1.4 0 01-1.4 1.4H6.4A1.4 1.4 0 015 20.3V12.2a1.4 1.4 0 011.4-1.4z"/><path d="M12.8 10.8v3.8h3.8"/><path class="arrow" d="M10.7 17.2V4.6M10.7 4.6L8 7.3M10.7 4.6l2.7 2.7"/><path class="spark" d="M18.8 3.2v3M17.3 4.7h3"/><path class="spark s2" d="M4.5 3.6v2.2M3.4 4.7h2.2"/>',
  clip: '<path d="M20 11.5l-8 8a5 5 0 01-7-7l8.6-8.6a3.4 3.4 0 014.8 4.8l-8.6 8.6a1.8 1.8 0 01-2.6-2.6l7.6-7.6"/>',
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0013 0M12 17.5V21M8.5 21h7"/>',
  send: '<path d="M7.5 5v14l11.5-7z"/>',
  speaker: '<path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5z"/><path d="M15.5 9a4 4 0 010 6M18 6.5a7.5 7.5 0 010 11"/>',
  stop: '<rect x="6.5" y="6.5" width="11" height="11" rx="2.5"/>',
  copy: '<rect x="8.5" y="8.5" width="11" height="11" rx="2"/><path d="M15.5 8.5v-2a2 2 0 00-2-2h-7a2 2 0 00-2 2v7a2 2 0 002 2h2"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  file: '<path d="M7 3.5h6.5L18 8v11.5a1 1 0 01-1 1H7a1 1 0 01-1-1v-15a1 1 0 011-1z"/><path d="M13.5 3.5V8H18M9 13h6M9 16.5h6"/>',
  download: '<path d="M12 4v11M7.5 10.5L12 15l4.5-4.5M5 19.5h14"/>',
  info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5M12 8v.01"/>',
  chat: '<path d="M5 5h14a2 2 0 012 2v8a2 2 0 01-2 2h-7l-4.5 3.5V17H5a2 2 0 01-2-2V7a2 2 0 012-2z"/>',
  bank: '<path d="M3 9.5L12 4l9 5.5M5 10v7M9.5 10v7M14.5 10v7M19 10v7M3.5 20h17"/>',
  book: '<path d="M5 4.5h8a3 3 0 013 3V20a2.5 2.5 0 00-2.5-2.5H5zM19 4.5h-3"/><path d="M5 4.5v13"/>',
  smile: '<circle cx="12" cy="12" r="8.5"/><path d="M8.5 14a4 4 0 007 0M9 9.5v.01M15 9.5v.01"/>',
  pen: '<path d="M4 20l1-4L16.5 4.5a2 2 0 013 3L8 19z"/><path d="M14.5 6.5l3 3"/>',
  spark: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM18.5 15l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z"/>',
  bulb: '<path d="M9 18h6M10 21h4M12 3a6 6 0 00-3.5 10.9c.6.5 1 1.2 1 2V16h5v-.1c0-.8.4-1.5 1-2A6 6 0 0012 3z"/>',
  sessions: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3.2 2"/>',
  rocket: '<path d="M12 2.4c3.1 2 4.6 5.6 4.6 9.6l-1.9 3.6H9.3L7.4 12c0-4 1.5-7.6 4.6-9.6z"/><circle cx="12" cy="9.2" r="1.7"/><path d="M7.5 12.2L4.4 14.9l1 3.2 3.2-1.6M16.5 12.2l3.1 2.7-1 3.2-3.2-1.6"/><path class="flame" d="M10.4 18.6c.4 1.7.9 2.9 1.6 3.9.7-1 1.2-2.2 1.6-3.9"/>',
  pin: '<path d="M9 4h6l-1 5 3 3v1.5H7V12l3-3zM12 13.5V20"/>',
  headphones: '<path d="M4 15v-3a8 8 0 0116 0v3"/><rect x="3" y="14" width="4" height="7" rx="1.6"/><rect x="17" y="14" width="4" height="7" rx="1.6"/>',
  thumbup: '<path d="M7 11v8H4.5a1 1 0 01-1-1v-6a1 1 0 011-1zM7 11l3.5-6.5a1.8 1.8 0 013.2 1.4L13 9.5h5a2 2 0 012 2.4l-1.2 5.8a2.5 2.5 0 01-2.4 2H7"/>',
  thumbdown: '<path d="M7 13V5H4.5a1 1 0 00-1 1v6a1 1 0 001 1zM7 13l3.5 6.5a1.8 1.8 0 003.2-1.4L13 14.5h5a2 2 0 002-2.4l-1.2-5.8a2.5 2.5 0 00-2.4-2H7"/>',
  share: '<circle cx="6.5" cy="12" r="2.3"/><circle cx="17.5" cy="6" r="2.3"/><circle cx="17.5" cy="18" r="2.3"/><path d="M8.5 10.9l7-3.8M8.5 13.1l7 3.8"/>',
  link: '<path d="M10 14a4 4 0 005.7 0l3-3a4 4 0 00-5.7-5.7l-1 1M14 10a4 4 0 00-5.7 0l-3 3a4 4 0 005.7 5.7l1-1"/>',
  hand: '<path d="M8 12V6.5a1.5 1.5 0 013 0V11M11 10V4.5a1.5 1.5 0 013 0V11M14 10V6a1.5 1.5 0 013 0v7a6 6 0 01-6 6h-.5a5 5 0 01-4.2-2.3L4.5 13.5a1.5 1.5 0 012.5-1.7L8 13"/>',
};

const COLOR = {
  upload: "logo", plusb: "blue", file: "blue", download: "blue", rocket: "blue", menu: "blue", plus: "white", x: "pink", search: "blue",
  trash: "pink", pen: "blue", pin: "blue", headphones: "blue", sliders: "orange", theme: "blue", mic: "logo", clip: "blue", send: "cyan", speaker: "blue",
  stop: "orange", copy: "blue", thumbup: "blue", thumbdown: "blue", share: "blue", link: "blue", info: "blue", check: "white",
};
const FILLED = new Set(["send", "stop"]);

export function Icon({ name }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={"i g-" + (COLOR[name] || "white") + (FILLED.has(name) ? " fill" : "")}
      aria-hidden="true"
      dangerouslySetInnerHTML={{ __html: PATHS[name] || "" }}
    />
  );
}

/** An icon inside the small inline-grid wrapper the layout CSS expects. */
export function SpanIcon({ name }) {
  return (
    <span style={{ display: "inline-grid" }}>
      <Icon name={name} />
    </span>
  );
}

/** The gradient fills the icons refer to (url(#gBlue) ...). Rendered once at the top of the page. */
export function GradientDefs() {
  const stops = (a, b, mid) => (
    <>
      <stop offset="0" stopColor={a} />
      {mid && <stop offset=".55" stopColor={mid} />}
      <stop offset="1" stopColor={b} />
    </>
  );
  const g = (id, a, b, mid) => (
    <linearGradient id={id} gradientUnits="userSpaceOnUse" x1="3" y1="3" x2="21" y2="21">
      {stops(a, b, mid)}
    </linearGradient>
  );
  return (
    <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden="true">
      <defs>
        {g("gPink", "#ff8fb1", "#f0366b")}
        {g("gBlue", "#5fb4ff", "#2a6cf0")}
        {g("gCyan", "#19d3f0", "#2ee59d")}
        {g("gLogo", "#7a2fd0", "#ff8a3d", "#d0338f")}
        {g("gOrange", "#ffc062", "#ff6f45")}
        {g("gGreen", "#5be37d", "#1da34f")}
      </defs>
    </svg>
  );
}

/** The same icon as a real SVG element, for effects that are drawn outside React (the rocket that flies off when you send). */
export function iconSvg(name) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("class", "i g-" + (COLOR[name] || "white") + (FILLED.has(name) ? " fill" : ""));
  s.setAttribute("aria-hidden", "true");
  s.innerHTML = PATHS[name] || "";
  return s;
}
