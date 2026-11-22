// Player intents. Every money effect is computed by the engine; these only edit the list of choices.
import { api, EngineError } from '../api/client'
import type { ActionIn, GameState, Letter } from '../api/types'
import { bus } from '../bridge'
import { sfx } from '../ui/sound'
import { actionIcon, actionLabel, getStore, requestFor, resumeAt, rollbackTo, setStore, toast, travel, withoutAction } from './game'

const sameSlot = (a: ActionIn, b: ActionIn) =>
  a.type === b.type && (a.type === 'set_savings_rate' || a.type === 'plan_study' || (a.type === 'set_habit' && a.id === b.id))

/** Add an action to this month's queue, after the engine accepted it (masking + affordability). */
export async function queueAction(a: ActionIn): Promise<boolean> {
  const s = getStore()
  const queue = [...s.queue.filter((q) => !sameSlot(q, a)), a]
  try {
    await api.state(requestFor({ ...s, queue }))
    setStore({ queue, studyPlanAsked: true })  // any choice moves past the start-of-life study question
    bus.emit('queue', queue.map(actionIcon))
    sfx('confirm')
    return true
  } catch (e) {
    sfx('error')
    toast({ icon: '🚫', title: 'Not possible', text: e instanceof EngineError ? e.message.replace(/^.*?: /, '') : String(e), tone: 'bad' })
    return false
  }
}

export function unqueue(i: number) {
  const s = getStore()
  const queue = s.queue.filter((_, j) => j !== i)
  setStore({ queue })
  bus.emit('queue', queue.map(actionIcon))
  sfx('click')
}

const BIG = ['retired', 'degree', 'home_bought', 'income_loss', 'found_job', 'financing_needed', 'home_sold']
function letterToasts(all: Letter[]) {
  // Long jumps: show the life events first, then the rest in time order.
  const letters = all.length > 4 ? [...all.filter((l) => BIG.includes(l.kind)), ...all.filter((l) => !BIG.includes(l.kind))] : all
  letters.slice(0, 4).forEach((l, i) => setTimeout(() => {
    sfx('letter')
    const tone = ['financing_needed', 'income_loss', 'market_drop', 'rent_rise'].includes(l.kind) ? 'warn'
      : ['degree', 'home_bought', 'loan_paid', 'mortgage_paid', 'raise'].includes(l.kind) ? 'good' : 'info'
    toast({ icon: l.icon, title: l.title, text: l.text, tone })
  }, 300 + i * 700))
  if (letters.length > 4) toast({ icon: '📬', title: `${letters.length - 4} more letters`, text: 'Open the Mail app on your phone.', tone: 'info' })
}

const ageAt = (m: number) => {
  const p = getStore().persona
  return Math.floor((p?.age ?? 20) + m / 12)
}

/** Move along the timeline to month `month`: forward lives the months (letters arrive), back rewinds. */
export async function travelTo(month: number) {
  const s = getStore()
  if (s.busy) return
  let target = Math.max(0, Math.min(s.horizon, Math.round(month)))
  if (target === s.current) return
  const before = s.current
  const forward = target > before
  if (forward) {
    // Time stops at the date you expect to finish your master's: you'll be asked how it went.
    try {
      const now = await api.state(requestFor(s))
      const end = now.pending_state.program ? now.pending_state.program_end : null
      if (end !== null && end > before && end < target) {
        target = end
        toast({ icon: '🎓', title: 'Your planned end date', text: 'Time stops here: did you finish your master’s?', tone: 'info' })
      }
    } catch { /* the move below reports engine errors */ }
    if (getStore().busy) return
  }
  const patch = travel(s, target)
  setStore({ ...patch, busy: true, place: null })
  bus.emit('queue', [])
  sfx('sleep')
  try {
    const st: GameState = await api.state(requestFor(getStore()))
    bus.emit('wake')
    bus.emit('season', { month: st.month })
    bus.emit('abroad', st.state.abroad)
    if (forward) {
      letterToasts(st.letters.filter((l) => l.month >= before))
      if (st.state.cash < 0 && !st.letters.some((l) => l.month >= before && l.kind === 'financing_needed')) {
        toast({ icon: '⚠️', title: 'Cash below zero', text: 'Financing needed: see the Bank.', tone: 'warn' })
      }
    } else {
      const ahead = getStore()
      const plans = ahead.tip !== ahead.cursor
      toast({ icon: '⏪', title: `Back to age ${ageAt(target)}`, tone: 'info',
        text: plans ? 'Your later choices wait ahead on the timeline. Choose something else to branch off.' : undefined })
    }
  } catch (e) {
    setStore({ nodes: s.nodes, cursor: s.cursor, tip: s.tip, current: s.current, queue: s.queue, nextColor: s.nextColor })
    toast({ icon: '🚫', title: 'Could not move in time', text: String(e instanceof Error ? e.message : e), tone: 'bad' })
  } finally {
    setStore({ busy: false })
  }
}

/** Sleep `months` months (1 = next month). */
export function sleep(months: number) {
  return travelTo(getStore().current + months)
}

export function sleepUntilBirthday() {
  const s = getStore()
  const left = 12 - (s.current % 12)
  return sleep(left === 0 ? 12 : left)
}

/** Undo one choice: a new reality without it, later choices replayed (the old one stays in the Multiverse). */
export async function undoChoice(nodeId: string, index: number) {
  const s = getStore()
  if (s.busy) return
  const patch = withoutAction(s, nodeId, index)
  if (!patch) return
  const label = actionLabel(s.nodes[nodeId].turn!.actions[index])
  const next = { ...s, ...patch }
  setStore({ busy: true })
  try {
    // The engine replays the whole new reality to its furthest month: later choices must still be possible.
    await api.state(requestFor(next, patch.tip, false))
    await api.state(requestFor(next))
    setStore({ ...patch, phone: null })
    bus.emit('queue', [])
    bus.emit('wake')
    sfx('confirm')
    toast({ icon: '↩️', title: `Undid “${label}”`, text: 'A new reality; the old one is kept in the Multiverse.', tone: 'info' })
  } catch (e) {
    sfx('error')
    toast({ icon: '🚫', title: `Can't undo “${label}”`, tone: 'bad',
      text: 'A later choice depends on it: ' + (e instanceof EngineError ? e.message.replace(/^.*?: /, '') : String(e)) })
  } finally {
    setStore({ busy: false })
  }
}

export function goBack(nodeId: string) {
  setStore((s) => ({ ...rollbackTo(s, nodeId), phone: null }))
  afterJump()
}
export function continueReality(nodeId: string) {
  setStore((s) => ({ ...resumeAt(s, nodeId), phone: null }))
  afterJump()
}
function afterJump() {
  const s = getStore()
  bus.emit('queue', [])
  bus.emit('wake')
  bus.emit('season', { month: s.current })
  sfx('sleep')
  api.state(requestFor(s)).then((st) => bus.emit('abroad', st.state.abroad)).catch(() => {})
  toast({ icon: '🌀', title: 'Another reality', text: `You are at age ${ageAt(s.current)}, ${new Date(2025, s.current, 1).toLocaleString('en', { month: 'long', year: 'numeric' })}.`, tone: 'info' })
}
