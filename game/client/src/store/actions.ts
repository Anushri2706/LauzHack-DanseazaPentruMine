// Player intents. Every money effect is computed by the engine; these only edit the list of choices.
import { api, EngineError } from '../api/client'
import type { ActionIn, GameState, Letter } from '../api/types'
import { bus } from '../bridge'
import { sfx } from '../ui/sound'
import { actionIcon, commitSleep, getStore, requestFor, resumeAt, rollbackTo, setStore, toast } from './game'

const sameSlot = (a: ActionIn, b: ActionIn) =>
  a.type === b.type && (a.type === 'set_savings_rate' || (a.type === 'set_habit' && a.id === b.id))

/** Add an action to this month's queue, after the engine accepted it (masking + affordability). */
export async function queueAction(a: ActionIn): Promise<boolean> {
  const s = getStore()
  const queue = [...s.queue.filter((q) => !sameSlot(q, a)), a]
  try {
    await api.state(requestFor({ ...s, queue }))
    setStore({ queue })
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

function letterToasts(letters: Letter[]) {
  letters.slice(0, 4).forEach((l, i) => setTimeout(() => {
    sfx('letter')
    const tone = ['financing_needed', 'income_loss', 'market_drop', 'rent_rise'].includes(l.kind) ? 'warn'
      : ['degree', 'home_bought', 'loan_paid', 'mortgage_paid', 'raise'].includes(l.kind) ? 'good' : 'info'
    toast({ icon: l.icon, title: l.title, text: l.text, tone })
  }, 300 + i * 700))
  if (letters.length > 4) toast({ icon: '📬', title: `${letters.length - 4} more letters`, text: 'Open the Mail app on your phone.', tone: 'info' })
}

/** Sleep `months` months (1 = next month). Letters from those months arrive as toasts. */
export async function sleep(months: number) {
  const s = getStore()
  if (s.busy) return
  const horizon = 120
  const before = s.current
  const patch = commitSleep(s, months, horizon)
  setStore({ ...patch, busy: true, place: null })
  bus.emit('queue', [])
  sfx('sleep')
  try {
    const st: GameState = await api.state(requestFor(getStore()))
    bus.emit('wake')
    bus.emit('season', { month: st.month })
    bus.emit('abroad', st.state.abroad)
    letterToasts(st.letters.filter((l) => l.month >= before))
    if (st.state.cash < 0 && !st.letters.some((l) => l.month >= before && l.kind === 'financing_needed')) {
      toast({ icon: '⚠️', title: 'Cash below zero', text: 'Financing needed: see the Bank.', tone: 'warn' })
    }
  } catch (e) {
    // Should not happen (the queue was validated); roll the sleep back so the player loses nothing.
    setStore({ nodes: s.nodes, cursor: s.cursor, current: s.current, queue: s.queue, nextColor: s.nextColor })
    toast({ icon: '🚫', title: 'Could not sleep', text: String(e instanceof Error ? e.message : e), tone: 'bad' })
  } finally {
    setStore({ busy: false })
  }
}

export function sleepUntilBirthday() {
  const s = getStore()
  const left = 12 - (s.current % 12)
  return sleep(left === 0 ? 12 : left)
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
  toast({ icon: '🌀', title: 'Another reality', text: `You are back in ${new Date(2025, s.current, 1).toLocaleString('en', { month: 'long', year: 'numeric' })}.`, tone: 'info' })
}
