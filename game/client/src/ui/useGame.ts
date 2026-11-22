import { api } from '../api/client'
import { useApi } from '../api/useApi'
import type { GameState } from '../api/types'
import { requestFor, useStore, type Store } from '../store/game'

/** The engine's view of the cursor reality at the current month (with this month's queue). */
export function useGameState() {
  const s = useStore((x) => x)
  const req = requestFor(s)
  return useApi(JSON.stringify(req), () => api.state(req))
}

/** Projection (no-surprises path, P10–P90 bands, ghosts) for the cursor reality, without the queue. */
export function useProjection(s: Store, n = 60) {
  const req = requestFor(s, undefined, false)
  return useApi(JSON.stringify(req) + n, () => api.projection(req, n))
}

/** The learned twin from the player's state now (HMM life states x learned engine); null key = off. */
export function useTwin(s: Store, on = true) {
  const req = requestFor(s, undefined, false)
  return useApi(on && !s.busy ? JSON.stringify(req) + 'twin' : null, () => api.twinFan(req))
}

/** Display conversion: a value after `i` months, in today's francs when the 💱 toggle is on.
 *  The divisor is the engine's own price index (GameState.price_level), so no money maths happens here. */
export function useDeflate(st: GameState | undefined) {
  const real = useStore((x) => x.real)
  const pl = st?.price_level
  return (v: number, i: number) => {
    if (!real || !pl || !pl.length) return v
    return v / (pl[Math.max(0, Math.min(pl.length - 1, Math.round(i)))] ?? 1)
  }
}
