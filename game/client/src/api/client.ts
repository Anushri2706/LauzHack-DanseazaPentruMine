import type {
  AssumptionRow, Catalogue, Check, CompareResult, GameRequest, GameState, HmmBranch, HmmFan, HmmRestore,
  HmmSnapshot, PersonaCard, Projection, Roll,
} from './types'

const BASE = '/api'
const cache = new Map<string, Promise<unknown>>()

/** Thrown for 4xx answers so callers can show the engine's reason ("You need 12 months since…"). */
export class EngineError extends Error {}

async function call<T>(method: 'GET' | 'POST', path: string, body?: unknown): Promise<T> {
  const key = method + path + (body ? JSON.stringify(body) : '')
  const hit = cache.get(key)
  if (hit) return hit as Promise<T>
  const p = fetch(BASE + path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  }).then(async (r) => {
    if (!r.ok) {
      const detail = await r.json().catch(() => ({}))
      const msg = typeof detail.detail === 'string' ? detail.detail : `${r.status} ${r.statusText}`
      throw r.status < 500 ? new EngineError(msg) : new Error(msg)
    }
    return r.json() as Promise<T>
  })
  p.catch(() => cache.delete(key))
  cache.set(key, p)
  if (cache.size > 400) cache.delete(cache.keys().next().value as string)
  return p
}

export const api = {
  personas: (q: { status?: string; random?: boolean; seed?: number; limit?: number }) => {
    const qs = new URLSearchParams(
      Object.entries(q).filter(([, v]) => v !== undefined && v !== '').map(([k, v]) => [k, String(v)]),
    )
    return call<PersonaCard[]>('GET', `/personas?${qs}`)
  },
  persona: (id: string) => call<PersonaCard>('GET', `/personas/${id}`),
  assumptions: (id: string) => call<AssumptionRow[]>('GET', `/assumptions/defaults/${id}`),
  catalogue: (id: string) => call<Catalogue>('GET', `/catalogue/${id}`),
  state: (req: GameRequest) => call<GameState>('POST', '/game/state', req),
  projection: (req: GameRequest, n = 500) => call<Projection>('POST', '/game/projection', { ...req, n }),
  roll: (req: GameRequest, roll: number) => call<Roll>('POST', '/game/roll', { ...req, roll }),
  compare: (a: GameRequest, b: GameRequest, name_a: string, name_b: string) =>
    call<CompareResult>('POST', '/compare', { a, b, name_a, name_b }),
  validation: () => call<Check[]>('GET', '/validation'),
  hmmStart: (persona_id: string) => call<{ snapshot: HmmSnapshot; decisions: { decision: string; label: string }[] }>('POST', '/hmm/start', { persona_id }),
  hmmBranch: (snapshot: HmmSnapshot, decisions: string[]) =>
    call<HmmBranch>('POST', '/hmm/branch', { snapshot, decisions, n_paths: 40 }),
  hmmCommit: (body: { snapshot: HmmSnapshot; flows: number[]; decision: string; label: string; cash: number; debt: number; investments: number }) =>
    call<HmmSnapshot>('POST', '/hmm/commit', body),
  hmmRestore: (snapshot: HmmSnapshot) => call<HmmRestore>('POST', '/hmm/restore', { snapshot }),
  hmmDistribution: (snapshot: HmmSnapshot) =>
    call<HmmFan>('POST', '/hmm/distribution', { snapshot, horizon_months: 60, n_paths: 40 }),
}
