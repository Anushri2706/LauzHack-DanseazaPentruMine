import { useState } from 'react'
import type { ActionIn, GameAction, GameState, Place } from '../api/types'
import { queueAction, sleep, sleepUntilBirthday, unqueue } from '../store/actions'
import { setStore, useStore } from '../store/game'
import { chf } from './format'
import { sfx } from './sound'
import { Avatar, Npc } from './Sprite'

const PLACE: Record<Place, { title: string; npc: number | null; who: string }> = {
  home: { title: 'Home', npc: null, who: 'You' },
  university: { title: 'University', npc: 6, who: 'Student advisor' },
  office: { title: 'Office', npc: 4, who: 'HR desk' },
  cafe: { title: 'Café du Lac', npc: 1, who: 'Barista' },
  shop: { title: 'Shop', npc: 5, who: 'Shopkeeper' },
  bank: { title: 'Bank', npc: 7, who: 'Bank clerk' },
  realestate: { title: 'Homes 4 Sale', npc: 3, who: 'Estate agent' },
  station: { title: 'Train station', npc: 0, who: 'Ticket office' },
}

function greeting(place: Place, st: GameState): string {
  const s = st.state
  switch (place) {
    case 'home':
      return s.home_id ? `Your own home. Time passes only when you sleep: plans you made this month happen then.`
        : `Home. Time passes only when you sleep: plans you made this month happen then.`
    case 'university':
      return s.program ? `You plan to finish your master's in ${monthName(s.program_end ?? 0)}. Hang in there!`
        : s.has_master ? 'Congratulations again on your degree!' : 'Thinking about a master’s? Here is what it would mean, in numbers.'
    case 'office':
      return s.status === 'working' ? `You earn ${chf(s.salary)}/month${s.work_share < 1 ? ` at ${Math.round(s.work_share * 100)}%` : ''}.`
        : s.status === 'abroad' ? 'You work abroad at the moment.' : 'Looking for work? Here are your options.'
    case 'cafe': return 'Coffee? Your eating-out habit is set here. Habits repeat every month until you change them.'
    case 'shop': return 'One-off buys are paid once; a shopping budget repeats monthly. Purchases are consumption, not investments.'
    case 'bank': return `Cash ${chf(s.cash)} · invested ${chf(s.investments)}${s.debt > 0 ? ` · loan ${chf(s.debt)}` : ''}${s.mortgage > 0 ? ` · mortgage ${chf(s.mortgage)}` : ''}.`
    case 'realestate': return s.home_id ? `Your home is worth about ${chf(s.home_value)} under our growth assumption.`
      : 'Buying needs a 20% down payment plus fees. The rest is a mortgage.'
    case 'station': return s.abroad ? 'Homesick? Trains back to Switzerland leave every hour.' : 'Where to? A weekend away, or a whole new country?'
  }
}
const monthName = (m: number) => new Date(2025, m, 1).toLocaleString('en', { month: 'short', year: 'numeric' })

// A master's is planned with the player's own length (value), so match it by type and id only.
const isQueued = (q: ActionIn[], a: GameAction) =>
  q.findIndex((x) => x.type === a.type && (x.id ?? null) === (a.id ?? null) && (a.duration || (x.value ?? null) === (a.value ?? null)))

/** "When do you expect to finish?": pick a planned length before the master's is queued. */
function StudyPlan({ a, month, onDone }: { a: GameAction; month: number; onDone: () => void }) {
  const d = a.duration!
  const [months, setMonths] = useState(d.default)
  const steps = Array.from(new Set([d.min, ...Array.from({ length: 20 }, (_, i) => 6 * (i + 1)), d.default, d.max]))
    .filter((m) => m >= d.min && m <= d.max).sort((x, y) => x - y)
  return (
    <div className="study-plan">
      <label className="small"><b>When do you expect to finish?</b>{' '}
        <select value={months} onChange={(e) => setMonths(Number(e.target.value))}>
          {steps.map((m) => <option key={m} value={m}>in {m} months ({monthName(month + m)}){m === d.default ? ' · usual length' : ''}</option>)}
        </select>
      </label>
      <p className="tiny muted">🟡 Tuition about {chf(d.tuition_per_month * months)} in total ({chf(d.tuition_per_month)} per month of study).
        On {monthName(month + months)} time stops and you'll be asked: finished, keep studying, or drop out?</p>
      <div className="row-end">
        <button className="btn ghost" onClick={onDone}>Cancel</button>
        <button className="btn" onClick={() => { void queueAction({ type: a.type, id: a.id, value: months }).then((ok) => ok && onDone()) }}>
          🎓 Plan it: finish {monthName(month + months)}</button>
      </div>
    </div>
  )
}

const DOT: Record<string, string> = { data: '🟢', assumption: '🟡', simulated: '🔵' }

function ActionRow({ a, queue, month }: { a: GameAction; queue: ActionIn[]; month: number }) {
  const qi = isQueued(queue, a)
  const queued = qi >= 0
  const disabled = !a.enabled && !queued
  const [planning, setPlanning] = useState(false)
  const plannedEnd = queued && a.duration ? queue[qi].value : null
  const click = () => {
    if (queued) unqueue(qi)
    else if (a.duration) setPlanning(!planning)
    else void queueAction({ type: a.type, id: a.id, value: a.value })
  }
  return (
    <li className={`act ${disabled ? 'off' : ''} ${queued ? 'queued' : ''} ${a.current ? 'current' : ''}`}>
      <button disabled={disabled} onClick={click}
        title={disabled ? a.reason ?? '' : queued ? 'Planned for this month: click to cancel' : 'Plan this for this month'}>
        <span className="act-label">{queued ? '✓ ' : ''}{a.label}{a.current && <em> · now</em>}</span>
        <span className="act-tags">{a.tags.map((t) => <span key={t} title={t}>{DOT[t] ?? ''}</span>)}</span>
      </button>
      <div className="act-preview">{a.preview.text}</div>
      {plannedEnd != null && <div className="act-preview">📅 You expect to finish in {plannedEnd} months ({monthName(month + plannedEnd)}).</div>}
      {planning && !queued && <StudyPlan a={a} month={month} onDone={() => setPlanning(false)} />}
      {a.preview.warning && <div className="act-warn">{a.preview.warning}</div>}
      {disabled && a.reason && !a.current && <div className="act-reason">🔒 {a.reason}</div>}
    </li>
  )
}

function Group({ title, items, queue, note }: { title?: string; items: GameAction[]; queue: ActionIn[]; note?: string }) {
  const month = useStore((s) => s.current)
  if (!items.length) return null
  return (
    <section className="act-group">
      {title && <h4>{title}</h4>}
      {note && <p className="tiny muted">{note}</p>}
      <ul className="acts">{items.map((a, i) => <ActionRow key={`${a.type}-${a.id}-${a.value}-${i}`} a={a} queue={queue} month={month} />)}</ul>
    </section>
  )
}

export function PlaceDialog({ place, st }: { place: Place; st: GameState }) {
  const queue = useStore((s) => s.queue)
  const busy = useStore((s) => s.busy)
  const look = useStore((s) => s.look)
  const meta = PLACE[place]
  const acts = st.actions[place] ?? []
  const by = (f: (a: GameAction) => boolean) => acts.filter(f)
  const close = () => { sfx('close'); setStore({ place: null }) }
  const title = place === 'station' && st.state.abroad ? 'Station · Back to Switzerland' : meta.title

  return (
    <div className="dialog-wrap" onClick={close}>
      <div className="dialog px-box" role="dialog" aria-label={title} onClick={(e) => e.stopPropagation()}>
        <header className="dialog-head">
          <div className="portrait">
            <Avatar look={look} scale={4} />
            {meta.npc !== null && <Npc char={meta.npc} scale={4} />}
          </div>
          <div>
            <h2>{title}</h2>
            <p className="greet"><b>{meta.who}:</b> “{greeting(place, st)}”</p>
          </div>
        </header>
        <div className="dialog-body">
          {st.ended && <p className="note">You reached the end of the timeline (age 120). Go back on the timeline, or open the phone to explore another reality.</p>}
          {place === 'home' && !st.ended && (
            <section className="act-group sleep-group">
              <button className="btn big" disabled={busy} onClick={() => void sleep(1)}>😴 Sleep → next month</button>
              <button className="btn big ghost" disabled={busy} onClick={() => void sleepUntilBirthday()}>
                🎂 Sleep until next birthday ({st.months_to_birthday === 12 ? 12 : st.months_to_birthday} months)
              </button>
              {queue.length > 0 && <p className="tiny">Your {queue.length} plan{queue.length > 1 ? 's' : ''} for {st.date.label} will happen when you sleep.</p>}
            </section>
          )}
          {place === 'home' && (
            <section className="act-group">
              <h4>📬 Mailbox</h4>
              {st.letters.length === 0 ? <p className="tiny muted">No letters yet.</p> : (
                <ul className="letters">{st.letters.slice(-4).reverse().map((l, i) => (
                  <li key={i}><span>{l.icon}</span><div><b>{l.title}</b> <span className="tiny muted">{monthName(l.month)}</span><div className="small">{l.text}</div></div></li>
                ))}</ul>
              )}
              <button className="link" onClick={() => setStore({ place: null, phone: 'mail' })}>All letters in the Mail app →</button>
            </section>
          )}
          {place === 'home' && <Group title="📦 Change flat" items={by((a) => a.type === 'change_flat')} queue={queue}
            note="🟡 Rent factors and moving cost are our assumptions." />}
          {place === 'university' && <Group items={acts} queue={queue} />}
          {place === 'office' && <Group items={acts} queue={queue} />}
          {place === 'cafe' && <>
            <Group title="☕ Eating out (habit, every month)" items={by((a) => a.type === 'set_habit')} queue={queue} />
            <Group title="One-off" items={by((a) => a.type === 'buy_item')} queue={queue} />
          </>}
          {place === 'shop' && <>
            <Group title="🛍️ One-off purchases" items={by((a) => a.type === 'buy_item')} queue={queue}
              note="Consumption, not assets: your net worth falls by the price." />
            <Group title="Shopping budget (habit)" items={by((a) => a.type === 'set_habit' && a.id === 'shopping')} queue={queue} />
            <Group title="Gym (habit)" items={by((a) => a.type === 'set_habit' && a.id === 'gym')} queue={queue} />
          </>}
          {place === 'bank' && <>
            <Statement st={st} />
            <Group title="🐷 Savings rule" items={by((a) => a.type === 'set_savings_rate')} queue={queue} />
            <Group title="🏦 Borrow" items={by((a) => a.type === 'take_loan')} queue={queue}
              note="A loan raises cash and debt equally (net worth unchanged); interest is a cost." />
            <Group title="Repay" items={by((a) => a.type === 'repay_debt')} queue={queue} />
            <Group title="📤 Take money out of investments" items={by((a) => a.type === 'sell_investments' && a.id === 'cash')} queue={queue}
              note="Moves money from investments to cash: net worth unchanged, but it stops earning the assumed return." />
            {st.state.debt > 0 && <Group title="📤 Pay the loan from investments" items={by((a) => a.type === 'sell_investments' && a.id === 'debt')} queue={queue}
              note="Sells investments and uses the money to cut your debt. Less debt means less interest from next month." />}
          </>}
          {place === 'realestate' && <Group items={acts} queue={queue}
            note="🟢 Mortgage rate (SNB) and home value growth (BIS) come from public data; 🟡 prices, amortization and maintenance are our assumptions. All editable in Settings." />}
          {place === 'station' && <>
            <Group title="Moves" items={by((a) => a.type === 'life_choice')} queue={queue} />
            <Group title="Trips" items={by((a) => a.type === 'buy_item')} queue={queue} />
          </>}
        </div>
        <footer className="dialog-foot">
          <span className="tiny muted">🟢 data · 🟡 assumption · 🔵 simulated · plans happen as you move forward on the timeline</span>
          <button className="btn ghost" onClick={close}>Leave</button>
        </footer>
      </div>
    </div>
  )
}

function Statement({ st }: { st: GameState }) {
  const s = st.statement
  if (!s) return <p className="tiny muted">Your first monthly statement arrives after you sleep.</p>
  return (
    <section className="act-group">
      <h4>🧾 Statement for {s.label} 🔵</h4>
      <table className="statement">
        <tbody>
          {s.lines.map((l) => (
            <tr key={l.label} className={l.kind}>
              <td>{l.label}</td><td className="num">{l.amount > 0 ? '+' : ''}{chf(l.amount)}</td>
            </tr>
          ))}
          <tr className="total"><td>Cash at month end</td><td className="num">{chf(s.cash_after)}</td></tr>
          <tr className="total"><td>Net worth at month end</td><td className="num">{chf(s.net_worth_after)}</td></tr>
        </tbody>
      </table>
      {s.labels.length > 0 && <p className="tiny muted">Included: {s.labels.join(', ')}</p>}
      <p className="tiny muted">Transfers move money between your own pockets: they never change net worth.</p>
    </section>
  )
}
