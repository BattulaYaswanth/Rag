import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev-only proxy so `npm run dev` can talk to a local backend.
// Production uses the nginx /api/ proxy (see nginx.conf.template).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_URL || "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
