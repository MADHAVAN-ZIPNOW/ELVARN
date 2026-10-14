// Cards tilt toward the cursor and a light sheen follows it. Spread the result on any element: <button {...useTilt(10, 14)}>.
import { useMemo } from "react";

export function useTilt(maxX, maxY) {
  return useMemo(
    () => ({
      onPointerMove(e) {
        if (e.pointerType === "touch") return;
        const node = e.currentTarget;
        const r = node.getBoundingClientRect();
        const x = (e.clientX - r.left) / r.width;
        const y = (e.clientY - r.top) / r.height;
        node.style.setProperty("--ry", (x - 0.5) * maxY + "deg");
        node.style.setProperty("--rx", -(y - 0.5) * maxX + "deg");
        node.style.setProperty("--mx", x * 100 + "%");
        node.style.setProperty("--my", y * 100 + "%");
      },
      onPointerLeave(e) {
        e.currentTarget.style.setProperty("--rx", "0deg");
        e.currentTarget.style.setProperty("--ry", "0deg");
      },
    }),
    [maxX, maxY]
  );
}
