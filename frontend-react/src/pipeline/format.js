// Pure helpers for the collection dashboard: number formatting, stage cards, throughput. No React in here.
export const NA = "unavailable";
export const isNum = (v) => typeof v === "number" && isFinite(v);

export function fmtNum(v) {
  if (!isNum(v)) return NA;
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(2) + " B";
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 1 : 2) + " M";
  if (a >= 1e4) return (v / 1e3).toFixed(1) + " k";
  return Math.round(v).toLocaleString("en-US");
}

export function fmtBytes(v) {
  if (!isNum(v)) return NA;
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, x = Math.max(0, v);
  while (x >= 1024 && i < u.length - 1) { x /= 1024; i++; }
  return (i === 0 ? String(Math.round(x)) : x.toFixed(x >= 100 ? 0 : 1)) + " " + u[i];
}

export function fmtDuration(sec) {
  if (!isNum(sec) || sec < 0) return NA;
  const s = Math.floor(sec), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  if (h) return h + "h " + String(m).padStart(2, "0") + "m";
  if (m) return m + "m " + String(s % 60).padStart(2, "0") + "s";
  return s + "s";
}

export function fmtAgo(iso, now) {
  const t = Date.parse(iso || "");
  if (!isFinite(t)) return "";
  const d = Math.max(0, ((now || Date.now()) - t) / 1000);
  if (d < 5) return "just now";
  if (d < 60) return Math.floor(d) + "s ago";
  if (d < 3600) return Math.floor(d / 60) + "m ago";
  if (d < 86400) return Math.floor(d / 3600) + "h ago";
  return Math.floor(d / 86400) + "d ago";
}

export const pct = (a, b) => (isNum(a) && isNum(b) && b > 0 ? Math.max(0, Math.min(100, (a / b) * 100)) : null);

export const STAGE_LABEL = {
  wikipedia_en: "Wikipedia (English)", wikipedia_ta: "Wikipedia (Tamil)", wikidata: "Wikidata record",
  corruption_articles: "Corruption articles", finishing: "Finishing",
};
export const PIPE_STAGES = ["collected", "raw", "kafka", "spark", "corpus", "neo4j", "chroma"];
export const PIPE_LABEL = { collected: "Collected", raw: "Raw", kafka: "Kafka", spark: "Spark", corpus: "Corpus", neo4j: "Neo4j", chroma: "Chroma" };

/** The counter that grows when a stage receives one more document / passage. Used to measure real throughput between two polls. */
const COUNTERS = {
  collector: (s) => s.collector && s.collector.documents,
  raw: (s) => s.raw && s.raw.files,
  kafka: (s) => s.kafka && s.kafka.messages,
  spark: (s) => s.spark && s.spark.processed_documents,
  corpus: (s) => s.corpus && s.corpus.passages,
  neo4j: (s) => s.neo4j && s.neo4j.passages,
  chroma: (s) => s.chroma && s.chroma.passages,
};
export const STAGE_IDS = Object.keys(COUNTERS);

/** items per minute per stage, from two snapshots (null when it cannot be measured). The collector reports its own per_minute. */
export function measureRates(prev, next, dtSec, old) {
  const out = {};
  const ps = (prev && prev.stages) || {}, ns = (next && next.stages) || {};
  for (const id of STAGE_IDS) {
    let r = null;
    if (id === "collector" && ns.collector && isNum(ns.collector.per_minute)) r = ns.collector.per_minute;
    else {
      const a = COUNTERS[id](ps), b = COUNTERS[id](ns);
      if (isNum(a) && isNum(b) && dtSec > 0.5) r = Math.max(0, ((b - a) / dtSec) * 60);
    }
    const o = old && isNum(old[id]) ? old[id] : null;
    out[id] = r === null ? o : o === null ? r : o * 0.5 + r * 0.5;     // light smoothing so the dots do not jitter
  }
  return out;
}

/** Dot speed and count for a connector from the real throughput (items per minute). 0 dots means still. */
export function flowLook(rate, active) {
  if (!active || !isNum(rate) || rate <= 0) return { dots: 0, dur: 0 };
  const l = Math.log10(1 + rate);
  return { dots: Math.min(4, 1 + Math.floor(l)), dur: Math.max(0.8, 4.2 - l * 1.1) };
}

/** Everything the flow diagram needs, one entry per stage; every number may be null. */
export function buildStages(status) {
  const s = (status && status.stages) || {};
  const run = !!status && status.state === "running";
  const c = s.collector, r = s.raw, k = s.kafka, p = s.spark, q = s.corpus, n = s.neo4j, h = s.chroma;
  const upOf = (o) => (o ? (typeof o.up === "boolean" ? o.up : true) : null);
  const bySrc = r && r.by_source ? Object.entries(r.by_source).filter(([, v]) => isNum(v)).map(([a, v]) => a + " " + fmtNum(v)).join(" · ") : null;
  const tokPct = q ? pct(q.clean_tokens, isNum(q.target_tokens) ? q.target_tokens : 1e9) : null;
  return [
    { id: "collector", title: "Collector", sub: "person by person", icon: "collector", up: c ? run || status.state === "stopping" : null, tone: "navy",
      rows: c ? [["Documents", fmtNum(c.documents)], ["Downloaded", fmtBytes(c.bytes)], ["Failed", fmtNum(c.failed)], ["Retries", fmtNum(c.retries)]] : null,
      note: c && c.last && c.last.title ? "Last: " + c.last.title : "" },
    { id: "raw", title: "Raw storage", sub: "kb_raw/", icon: "raw", up: r ? true : null, tone: "terra",
      rows: r ? [["Files", fmtNum(r.files)], ["Size", fmtBytes(r.bytes)], ["Free disk", fmtBytes(r.free_disk_bytes)]] : null, note: bySrc || "" },
    { id: "kafka", title: "Kafka", sub: k && k.topic ? k.topic : "kb.raw_documents", icon: "kafka", up: upOf(k), tone: "emerald",
      rows: k ? [["Messages", fmtNum(k.messages)], ["Consumer lag", fmtNum(k.consumer_lag)]] : null, note: "" },
    { id: "spark", title: "Spark", sub: "clean, split, dedupe", icon: "spark", up: upOf(p), tone: "terra",
      rows: p ? [["Processed", fmtNum(p.processed_documents)], ["Skipped", fmtNum(p.skipped)], ["Failed", fmtNum(p.failed)], ["Dead letters", fmtNum(p.dead_letters)]] : null, note: "" },
    { id: "corpus", title: "Corpus", sub: "Parquet passages", icon: "corpus", up: q ? true : null, tone: "navy",
      rows: q ? [["Passages", fmtNum(q.passages)], ["Duplicates removed", fmtNum(q.duplicates_removed)], ["Files", fmtNum(q.files)]] : null,
      bar: q ? { label: "Tokens", value: q.clean_tokens, target: isNum(q.target_tokens) ? q.target_tokens : 1e9, pct: tokPct } : null, note: "" },
    { id: "neo4j", title: "Neo4j", sub: "knowledge base", icon: "graph", up: upOf(n), tone: "emerald",
      rows: n ? [["Documents", fmtNum(n.documents)], ["Passages", fmtNum(n.passages)], ["People", fmtNum(n.people)], ["Entities", fmtNum(n.entities)], ["Relationships", fmtNum(n.relationships)]] : null, note: "" },
    { id: "chroma", title: "Chroma", sub: "vector search", icon: "vector", up: upOf(h), tone: "terra",
      rows: h ? [["Passages", fmtNum(h.passages)]] : null,
      bar: h ? { label: "Embedded", value: null, target: null, pct: isNum(h.embedded_pct) ? Math.max(0, Math.min(100, h.embedded_pct)) : null } : null, note: "" },
  ];
}

export function runState(status) {
  const st = status && status.state;
  return st === "running" || st === "stopping" || st === "error" ? st : "idle";
}
