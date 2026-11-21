import '@fontsource/pixelify-sans/400.css'
import '@fontsource/pixelify-sans/600.css'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>)

// Dev only: expose the event bus so the headless smoke test (scripts/smoke.mjs) can drive the town.
if (import.meta.env.DEV) import('./bridge').then(({ bus }) => { (window as unknown as { __bus: unknown }).__bus = bus })
