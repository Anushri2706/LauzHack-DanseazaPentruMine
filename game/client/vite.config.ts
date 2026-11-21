import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The backend runs on :8000; the dev server proxies /api → backend (no CORS issues).
export default defineConfig({
  plugins: [react()],
  // Pre-bundle everything up front: otherwise opening the phone (lazy chunk with recharts) makes Vite
  // re-optimize and reload the page mid-game.
  optimizeDeps: { include: ['phaser', 'recharts', 'easystarjs', 'mitt', 'react', 'react-dom', 'react-dom/client'] },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, '') },
    },
  },
})
