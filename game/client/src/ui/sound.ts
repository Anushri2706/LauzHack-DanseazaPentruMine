// Interface sounds (Kenney "Interface Sounds", CC0). Muted state lives in localStorage.
const FILES = {
  click: 'click_002', confirm: 'confirmation_001', error: 'error_004', open: 'open_001', close: 'close_001',
  phoneUp: 'maximize_006', phoneDown: 'minimize_006', letter: 'bong_001', coin: 'drop_002', sleep: 'glass_002',
} as const
export type Sfx = keyof typeof FILES
const cache: Partial<Record<Sfx, HTMLAudioElement>> = {}
let muted = (() => { try { return localStorage.getItem('ml-muted') === '1' } catch { return false } })()

export function sfx(name: Sfx) {
  if (muted) return
  try {
    const a = (cache[name] ??= new Audio(`assets/interface-sounds/${FILES[name]}.ogg`))
    a.volume = 0.35
    a.currentTime = 0
    void a.play().catch(() => {})
  } catch { /* audio unavailable */ }
}
export const isMuted = () => muted
export function setMuted(m: boolean) {
  muted = m
  try { localStorage.setItem('ml-muted', m ? '1' : '0') } catch { /* ignore */ }
}
