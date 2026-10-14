import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The ELVARN server serves the built files under /static (index.html at /), so assets are built with that prefix.
// `npm run dev` proxies /api to the local backend on port 8000.
export default defineConfig({
  base: "/static/",
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { outDir: "dist", emptyOutDir: true, assetsInlineLimit: 0 },
});
