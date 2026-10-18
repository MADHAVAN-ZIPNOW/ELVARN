import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run dev` proxies the API to a locally running backend (uvicorn app.main:app --port 8100).
export default defineConfig({
  plugins: [react()],
  server: { port: 5180, proxy: { "/api": "http://127.0.0.1:8100" } },
  build: { outDir: "dist", emptyOutDir: true },
});
