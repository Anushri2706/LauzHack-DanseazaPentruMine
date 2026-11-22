// Game state held by the client: persona, look, assumptions, the multiverse tree of turns and this
// month's queue. The backend is stateless and replays a reality from month 0 on every request, so this
// store holds only choices, never money.
import { useSyncExternalStore } from 'react'
import type { ActionIn, Goal, GameRequest, PersonaCard, TurnIn } from '../api/types'

export interface Look { skin: number; hair: number; hairColor: number; top: number; legs: number }
export const DEFAULT_LOOK: Look = { skin: 0, hair: 2, hairColor: 0, top: 2, legs: 0 }

export type Dim = 'money' | 'safety' | 'goals' | 'freedom' | 'experience' | 'joy'
export type AppId = 'money' | 'matters' | 'multiverse' | 'twin' | 'ask' | 'goals' | 'settings' | 'checks' | 'mail' | 'learn'

export interface TNode {
  id: string
  parent: string | null
  turn: TurnIn | null // null for the root (start of the game)
  children: string[]
  color: number
  label: string
  reached: number // furthest month lived on this node when it is the tip of a reality
}

export interface Toast { id: number; icon: string; title: string; text?: string; tone?: 'good' | 'warn' | 'bad' | 'info' }

export interface Store {
  screen: 'pick' | 'create' | 'play'
  persona: PersonaCard | null
  look: Look
  assumptions: Record<string, number>
  seed: number
  world: number
  goalSel: string[] | null // null = the persona's goals from the data
  customGoals: Goal[]
  nodes: Record<string, TNode>
  root: string
  cursor: string
  /** The reality being viewed: the cursor lies on its path. Its choices after `current` are future plans
   *  (redo): moving forward re-lives them, a different choice branches off. */
  tip: string
  horizon: number
  current: number
  queue: ActionIn[]
  nextColor: number
  matters: Dim[]
  place: string | null
  phone: AppId | null
  pinA: string | null
  pinB: string | null
  toasts: Toast[]
  busy: boolean
  endSeen: boolean
  nearPlace: string | null
  hintSeen: boolean
  studyPlanAsked: boolean // the start-of-life study-plan question was answered, skipped or passed by (plan later at the University)
  futureView: 'rules' | 'twin'
  /** Show amounts in today's (Jan 2025) francs instead of nominal francs. Display only. */
  real: boolean
}

export const REALITY_COLORS = ['#9a7fd8', '#ef8aa0', '#4fb38a', '#e8a64a', '#5aa9d6', '#c77dc9', '#7c9a3d', '#d9785b']

const KEY = 'many-lives-town-v2'
const fresh = (): Store => ({
  screen: 'pick', persona: null, look: DEFAULT_LOOK, assumptions: {}, seed: 42, world: 0,
  goalSel: null, customGoals: [],
  nodes: { root: { id: 'root', parent: null, turn: null, children: [], color: 0, label: 'Start', reached: 0 } },
  root: 'root', cursor: 'root', tip: 'root', horizon: 1200, current: 0, queue: [], nextColor: 1, matters: [], place: null, phone: null,
  pinA: null, pinB: null, toasts: [], busy: false, endSeen: false, nearPlace: null, hintSeen: false, studyPlanAsked: false, futureView: 'rules', real: false,
})

function load(): Store {
  try {
    const raw = localStorage.getItem(KEY)
    if (raw) return { ...fresh(), ...JSON.parse(raw), toasts: [], place: null, phone: null, busy: false }
  } catch { /* storage unavailable: start fresh */ }
  return fresh()
}

let state: Store = load()
const listeners = new Set<() => void>()

export function getStore(): Store { return state }
export function setStore(patch: Partial<Store> | ((s: Store) => Partial<Store>)) {
  const p = typeof patch === 'function' ? patch(state) : patch
  state = { ...state, ...p }
  try {
    const { toasts: _t, busy: _b, ...persist } = state
    localStorage.setItem(KEY, JSON.stringify(persist))
  } catch { /* ignore */ }
  listeners.forEach((l) => l())
}
export function useStore<T>(sel: (s: Store) => T): T {
  return useSyncExternalStore((cb) => { listeners.add(cb); return () => listeners.delete(cb) }, () => sel(state))
}
export function resetGame() { setStore({ ...fresh(), screen: 'pick' }) }

// ------------------------------------------------------------------------- tree helpers
export function pathIds(s: Pick<Store, 'nodes'>, id: string): string[] {
  const out: string[] = []
  let n: TNode | undefined = s.nodes[id]
  while (n) { out.unshift(n.id); n = n.parent ? s.nodes[n.parent] : undefined }
  return out
}
export function turnsOf(s: Pick<Store, 'nodes'>, id: string): TurnIn[] {
  return pathIds(s, id).map((p) => s.nodes[p].turn).filter((t): t is TurnIn => t !== null)
}
export function colorOf(s: Pick<Store, 'nodes'>, id: string): string {
  return REALITY_COLORS[s.nodes[id].color % REALITY_COLORS.length]
}

const LABELS: Record<string, string> = {
  start_work: 'Work', master_ft: "Master's", master_pt: "PT master's", move_abroad: 'Abroad', return_home: 'Back home',
  drop_out: 'Drop out', graduate: 'Degree', finish_study: 'Finished', extend_study: 'Keep studying', quit_school: 'Drop out (no degree)', quit_job: 'Quit', retire_early: 'Retire early', reduce_hours: '80%', full_hours: '100%',
  buy_home: 'Buy home', sell_home: 'Sell home', change_flat: 'Move flat', take_loan: 'Loan', repay_debt: 'Repay', sell_investments: 'Sell investments',
  set_savings_rate: 'Savings', ask_raise: 'Raise?', set_habit: 'Habit', buy_item: 'Buy', plan_study: 'Study plan',
}
export const ICONS: Record<string, string> = {
  life_choice: '⭐', set_habit: '☕', buy_item: '🛍️', change_flat: '📦', buy_home: '🏠', sell_home: '🔑',
  take_loan: '🏦', repay_debt: '✅', sell_investments: '📤', set_savings_rate: '🐷', ask_raise: '📈',
  start_work: '💼', master_ft: '🎓', master_pt: '🎓', move_abroad: '✈️', return_home: '🚆', drop_out: '🚪', graduate: '🎓', quit_school: '🚪', finish_study: '🎓', extend_study: '📚',
  quit_job: '🚪', retire_early: '🌅', plan_study: '📅', reduce_hours: '🌿', full_hours: '⏱️', weekend_trip: '🧳', flight_home: '✈️',
}
export function actionIcon(a: ActionIn): string { return ICONS[a.id ?? ''] ?? ICONS[a.type] ?? '•' }
export function actionLabel(a: ActionIn): string {
  if (a.type === 'life_choice') return LABELS[a.id ?? ''] ?? a.id ?? ''
  if (a.type === 'sell_investments') return a.id === 'debt' ? 'Invest. → debt' : 'Invest. → cash'
  if (a.type === 'buy_item') return a.id === 'weekend_trip' ? 'Trip' : 'Buy ' + (a.id ?? '').replace('_', ' ')
  return LABELS[a.type] ?? a.type
}
function turnLabel(t: TurnIn): string {
  const big = t.actions.find((a) => a.type === 'life_choice' || a.type === 'buy_home' || a.type === 'sell_home')
  const main = big ? actionLabel(big) : actionLabel(t.actions[0])
  return t.actions.length > 1 ? `${main} +${t.actions.length - 1}` : main
}
export function realityName(s: Pick<Store, 'nodes'>, id: string): string {
  const parts = pathIds(s, id).map((p) => s.nodes[p].turn)
    .flatMap((t) => (t ? t.actions : []))
    .filter((a) => a.type === 'life_choice' || a.type === 'buy_home')
    .map(actionLabel)
  const collapsed = parts.filter((p, i) => p !== parts[i - 1])
  return collapsed.length ? collapsed.slice(-3).join(' → ') : 'As things are'
}

/** Realities = leaves of the tree, plus the cursor when the player went back to an inner node. */
export function realities(s: Pick<Store, 'nodes' | 'cursor'>): string[] {
  const leaves = Object.values(s.nodes).filter((n) => n.children.length === 0).map((n) => n.id)
  return leaves.includes(s.cursor) ? leaves : [...leaves, s.cursor]
}

// ------------------------------------------------------------------------- requests
export function goalsOf(s: Store): Goal[] | null {
  if (s.goalSel === null && s.customGoals.length === 0) return null
  const sel = s.goalSel ?? s.persona?.goals.map((g) => g.goal_id) ?? []
  return [...(s.persona?.goals ?? []).filter((g) => sel.includes(g.goal_id)), ...s.customGoals]
}

/** Request for the reality at `node` (defaults: the cursor, with this month's queue). */
export function requestFor(s: Store, node?: string, withQueue = true): GameRequest {
  const id = node ?? s.cursor
  const isCursor = id === s.cursor
  const month = isCursor ? s.current : s.nodes[id].reached
  const turns = turnsOf(s, id)
  if (isCursor && withQueue && s.queue.length) turns.push({ month, actions: s.queue })
  return {
    persona_id: s.persona?.profile_id ?? 'P00001', look: { ...s.look }, assumptions: s.assumptions,
    seed: s.seed, world: s.world, turns, current_month: month, goals: goalsOf(s),
  }
}

const same = (a: TurnIn, b: TurnIn) => a.month === b.month && JSON.stringify(a.actions) === JSON.stringify(b.actions)

/** Commit this month's queue as a turn (a new node, or an identical existing child). */
function commitQueue(s: Store): Pick<Store, 'nodes' | 'cursor' | 'nextColor' | 'tip'> {
  const nodes = { ...s.nodes }
  let cursor = s.cursor
  let nextColor = s.nextColor
  let tip = tipOf(s)
  if (s.queue.length) {
    const turn: TurnIn = { month: s.current, actions: s.queue }
    const cur = nodes[cursor]
    const existing = cur.children.map((c) => nodes[c]).find((c) => c.turn && same(c.turn, turn))
    if (existing) {
      cursor = existing.id
      if (!pathIds({ nodes }, tip).includes(cursor)) tip = cursor
    } else {
      const id = newId(nodes)
      const forks = cur.children.length > 0
      nodes[id] = { id, parent: cur.id, turn, children: [], color: forks ? nextColor : cur.color, label: turnLabel(turn), reached: s.current }
      nodes[cur.id] = { ...cur, children: [...cur.children, id] }
      if (forks) nextColor++
      cursor = id
      tip = id
    }
  }
  return { nodes, cursor, nextColor, tip }
}

const newId = (nodes: Record<string, TNode>) => `n${Object.keys(nodes).length}_${Math.random().toString(36).slice(2, 6)}`

/** The viewed reality (falls back to the cursor when the stored tip is stale). */
export function tipOf(s: Pick<Store, 'nodes' | 'cursor' | 'tip'>): string {
  return s.tip && s.nodes[s.tip] && pathIds(s, s.tip).includes(s.cursor) ? s.tip : s.cursor
}

/** Deepest node on `tip`'s path whose turn happened before month `m` (the root if none). */
function lastBefore(s: Pick<Store, 'nodes'>, tip: string, m: number): string {
  let cursor = pathIds(s, tip)[0]
  for (const id of pathIds(s, tip)) {
    const t = s.nodes[id].turn
    if (!t || t.month < m) cursor = id
    else break
  }
  return cursor
}

/** Move the playhead to month `m` along the viewed reality. Moving forward first commits this month's
 *  plans; moving back keeps later choices as future plans (redo) until you choose differently. */
export function travel(s: Store, m: number): Partial<Store> {
  const month = Math.max(0, Math.min(s.horizon, Math.round(m)))
  const base: Store = month > s.current && s.queue.length ? { ...s, ...commitQueue(s) } : s
  const tip = tipOf(base)
  const nodes = { ...base.nodes }
  nodes[tip] = { ...nodes[tip], reached: Math.max(nodes[tip].reached, month) }
  return { nodes, tip, cursor: lastBefore(base, tip, month), current: month, queue: [], nextColor: base.nextColor, endSeen: false }
}

/** Sleep `months` months (kept for the Home buttons): the same as moving the playhead forward. */
export function commitSleep(s: Store, months: number): Partial<Store> {
  return travel(s, s.current + months)
}

/** A new reality without one action of node `id` (later choices are kept and replayed). */
export function withoutAction(s: Store, id: string, index: number): Partial<Store> | null {
  const tip = tipOf(s)
  const path = pathIds(s, tip)
  const at = path.indexOf(id)
  const node = s.nodes[id]
  if (at < 0 || !node.turn || !node.parent) return null
  const rest = node.turn.actions.filter((_, j) => j !== index)
  const turns: TurnIn[] = [...(rest.length ? [{ month: node.turn.month, actions: rest }] : []),
    ...path.slice(at + 1).map((p) => s.nodes[p].turn!)]
  const nodes = { ...s.nodes }
  let nextColor = s.nextColor
  const color = nextColor++
  let parent = node.parent
  for (const turn of turns) {
    const cur = nodes[parent]
    const existing = cur.children.map((c) => nodes[c]).find((c) => c.turn && same(c.turn, turn))
    if (existing) { parent = existing.id; continue }
    const nid = newId(nodes)
    nodes[nid] = { id: nid, parent: cur.id, turn, children: [], color, label: turnLabel(turn), reached: turn.month }
    nodes[cur.id] = { ...cur, children: [...cur.children, nid] }
    parent = nid
  }
  nodes[parent] = { ...nodes[parent], reached: Math.max(nodes[parent].reached, s.nodes[tip].reached, s.current) }
  return { nodes, nextColor, tip: parent, cursor: lastBefore({ nodes }, parent, s.current), queue: [] }
}

/** Go back to just before node `id` was chosen (root: the start), with an empty queue. */
export function rollbackTo(s: Store, id: string): Partial<Store> {
  const n = s.nodes[id]
  if (!n.parent || !n.turn) {
    return { cursor: s.root, tip: tipOf(s), current: 0, queue: [], endSeen: false }
  }
  return { ...travel({ ...s, tip: id, cursor: n.parent, queue: [] }, n.turn.month), tip: id }
}

/** Continue a reality from its tip. */
export function resumeAt(s: Store, id: string): Partial<Store> {
  return { cursor: id, tip: id, current: s.nodes[id].reached, queue: [], endSeen: false }
}

let toastId = 1
export function toast(t: Omit<Toast, 'id'>) {
  const id = toastId++
  setStore((s) => ({ toasts: [...s.toasts.slice(-4), { ...t, id }] }))
  setTimeout(() => setStore((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) })), 5200)
}
