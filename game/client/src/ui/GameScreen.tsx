import { lazy, Suspense, useEffect } from 'react'
import type { Place } from '../api/types'
import { bus } from '../bridge'
import { PhaserHost } from '../game/PhaserHost'
import { getStore, setStore, useStore } from '../store/game'
import { HudBar, Toasts } from './HudBar'
import { PlaceDialog } from './PlaceDialog'
import { SeasonOverlay } from './SeasonOverlay'
import { Timeline } from './Timeline'
import { sfx } from './sound'
import { useGameState } from './useGame'

const Phone = lazy(() => import('./Phone').then((m) => ({ default: m.Phone })))
const EndScreen = lazy(() => import('./EndScreen').then((m) => ({ default: m.EndScreen })))

const NAMES: Record<string, string> = {
  home: 'Home', university: 'University', office: 'Office', cafe: 'Café', shop: 'Shop', bank: 'Bank',
  realestate: 'Real-estate agency', station: 'Train station',
}

export function GameScreen() {
  const place = useStore((s) => s.place) as Place | null
  const phone = useStore((s) => s.phone)
  const endSeen = useStore((s) => s.endSeen)
  const hintSeen = useStore((s) => s.hintSeen)
  const game = useGameState()
  const st = game.data
  const near = useNear()

  useEffect(() => { bus.emit('freeze', !!place || !!phone || (!!st?.ended && !endSeen)) }, [place, phone, st?.ended, endSeen])
  useEffect(() => {
    const onDoor = ({ place }: { place: string }) => {
      if (getStore().place || getStore().phone) return
      sfx('open'); setStore({ place })
    }
    bus.on('door', onDoor)
    return () => bus.off('door', onDoor)
  }, [])
  useEffect(() => { if (st) { bus.emit('abroad', st.state.abroad) } }, [st?.state.abroad]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (st && getStore().horizon !== st.horizon) setStore({ horizon: st.horizon }) }, [st?.horizon])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = (e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'SELECT'
      if (e.key === 'Escape') {
        const s = getStore()
        if (s.phone) { sfx('phoneDown'); setStore({ phone: null }) } else if (s.place) { sfx('close'); setStore({ place: null }) }
      } else if (!typing && (e.key === 'p' || e.key === 'P')) {
        const s = getStore()
        if (s.place) return
        sfx(s.phone ? 'phoneDown' : 'phoneUp')
        setStore({ phone: s.phone ? null : 'money' })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <div className="game-screen">
      <PhaserHost />
      <SeasonOverlay month={st?.month ?? 0} abroad={!!st?.state.abroad} />
      <HudBar st={st} />
      {game.error && <div className="engine-error px-box">⚠️ {game.error}
        {getStore().queue.length > 0 && <button className="link" onClick={() => setStore({ queue: [] })}> Clear this month's plans</button>}</div>}
      {near && !place && !phone && (
        <button className="door-hint px-box" onClick={() => bus.emit('door', { place: near })}>
          Press <kbd>E</kbd> or click to enter the <b>{NAMES[near]}</b>
        </button>
      )}
      {!place && !phone && st && st.month === 0 && !st.ended && !hintSeen && (
        <div className="help-hint px-box tiny">Walk with <kbd>WASD</kbd>/arrows or click. Enter a building with <kbd>E</kbd> or a click. <kbd>P</kbd> phone · <kbd>Esc</kbd> close. Point at ▲ Life timeline at the bottom to move through your life (<kbd>T</kbd> pins it open).
          <button className="icon-btn hint-close" title="Got it" aria-label="Close the instructions" onClick={() => setStore({ hintSeen: true })}>✕</button></div>
      )}
      {st && !phone && <Timeline st={st} />}
      {place && st && <PlaceDialog place={place} st={st} />}
      <Suspense fallback={null}>
        {phone && st && <Phone st={st} />}
        {st?.ended && !endSeen && !phone && <EndScreen st={st} />}
      </Suspense>
      <Toasts />
    </div>
  )
}

function useNear() {
  const near = useStore((s) => s.nearPlace)
  useEffect(() => {
    const f = ({ place }: { place: string | null }) => setStore({ nearPlace: place })
    bus.on('near', f)
    return () => bus.off('near', f)
  }, [])
  return near
}
