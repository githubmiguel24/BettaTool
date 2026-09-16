import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies API paths to the FastAPI backend so the frontend can
// use relative URLs and avoid CORS entirely during development. Override the
// target with VITE_API_BASE in .env.local when the backend lives elsewhere
// (e.g. behind a Kaggle/Colab tunnel).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/analyze": { target: "http://localhost:8000", changeOrigin: true },
      "/reports": { target: "http://localhost:8000", changeOrigin: true },
      "/export": { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
