import { api } from '../api/client'
import { useApi } from '../api/useApi'
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
