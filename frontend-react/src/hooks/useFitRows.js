// How many rows of `rowHeight` pixels fit in an element, kept up to date as the window is resized: lists are cut into pages of exactly that many
// rows, so a dashboard page never needs a scrollbar. Attach `ref` to the element that holds ONLY the rows (it must have a height of its own).
import { useCallback, useEffect, useRef, useState } from "react";

export function useFitRows(rowHeight, { min = 3, max = 40, start = 8, reserve = 0 } = {}) {
  const [rows, setRows] = useState(start);
  const node = useRef(null);
  const watcher = useRef(null);
  const measure = useCallback(() => {
    const el = node.current;
    if (!el) return;
    setRows(Math.max(min, Math.min(max, Math.floor((el.clientHeight - reserve) / rowHeight))));
  }, [rowHeight, min, max, reserve]);
  const ref = useCallback((el) => {
    if (watcher.current) { watcher.current.disconnect(); watcher.current = null; }
    node.current = el;
    if (el) {
      measure();
      if (typeof ResizeObserver !== "undefined") { watcher.current = new ResizeObserver(measure); watcher.current.observe(el); }
    }
  }, [measure]);
  useEffect(() => () => { if (watcher.current) watcher.current.disconnect(); }, []);
  return [ref, rows];
}
