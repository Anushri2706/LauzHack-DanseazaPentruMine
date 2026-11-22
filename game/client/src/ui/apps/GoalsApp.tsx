import { useState } from 'react'
import type { CheckedGoal, GameState, Goal, Projection } from '../../api/types'
import { setStore, useStore } from '../../store/game'
import { chf, pct } from '../format'

/** Adapted from the old GoalsPanel: choose which data goals count, add your own. */
export function GoalsApp({ st, proj }: { st: GameState; proj?: Projection }) {
  const persona = st.persona
  const goalSel = useStore((s) => s.goalSel)
  const custom = useStore((s) => s.customGoals)
  const selected = new Set(goalSel ?? persona.goals.map((g) => g.goal_id))
  const startAge = persona.age
  const maxAge = Math.floor(startAge + st.horizon / 12)
  const [name, setName] = useState('Travel fund')
  const [amount, setAmount] = useState(10000)
  const [age, setAge] = useState(Math.round(startAge + 5))
  const [retireAge, setRetireAge] = useState(55)
  const [untilAge, setUntilAge] = useState(90)
  const checked: CheckedGoal[] = proj?.goals ?? []
  const share = proj?.goal_met_share
  const toggle = (id: string) => {
    const next = new Set(selected)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    setStore({ goalSel: [...next] })
  }
  const Badge = ({ id }: { id: string }) => {
    const g = checked.find((x) => x.goal_id === id)
    if (!g) return <span className="tiny muted">…</span>
    if (g.status === 'beyond_horizon') return <span className="tag neutral">after {maxAge}</span>
    if (g.status === 'not_possible') return <span className="tag bad" title={g.reason ?? ''}>not possible</span>
    return <span className={`tag ${g.status === 'met' ? 'good' : 'bad'}`} title="On the path with no surprises; % = share of simulated worlds">
      {g.status === 'met' ? 'met' : 'not met'}{share?.[id] !== undefined && ` · ${pct(share[id])} of worlds`}</span>
  }
  return (
    <div>
      <h3>🎯 Goals that matter to you</h3>
      <p className="tiny muted">Checked in priority order against cash + investments at the target date, on the path “if nothing surprising happens from now”. Money already set aside for one goal is never counted twice.</p>
      <ul className="goal-list">
        {persona.goals.map((g) => (
          <li key={g.goal_id}>
            <label><input type="checkbox" checked={selected.has(g.goal_id)} onChange={() => toggle(g.goal_id)} /> 🟢 {g.name}</label>
            <span className="tiny muted">{chf(g.target)} by {g.target_date?.slice(0, 7)}</span>
            {selected.has(g.goal_id) && <Badge id={g.goal_id} />}
          </li>
        ))}
        {custom.map((g) => (
          <li key={g.goal_id}>
            <span>{g.kind === 'retire_early' ? '🌅' : '✏️'} {g.name}</span>
            {g.kind === 'retire_early'
              ? <span className="tiny muted">stop working at {Math.round(startAge + g.month / 12)} · money lasts to {g.until_age ?? 90}</span>
              : <span className="tiny muted">{chf(g.target)} by age {Math.round(startAge + (g.month + 1) / 12)}</span>}
            <Badge id={g.goal_id} />
            {g.kind === 'retire_early' && <RetireDetail g={checked.find((x) => x.goal_id === g.goal_id)} />}
            <button className="icon-btn" title="Remove" onClick={() => setStore((s) => ({ customGoals: s.customGoals.filter((x) => x.goal_id !== g.goal_id) }))}>🗑️</button>
          </li>
        ))}
      </ul>
      <form className="goal-form" onSubmit={(e) => {
        e.preventDefault()
        const a = Math.min(maxAge, Math.max(Math.ceil(startAge + 1 / 12), age))
        const goal: Goal = { goal_id: `custom-${Date.now()}`, name: name || 'My goal', target: Math.max(0, amount),
          month: Math.max(0, Math.round((a - startAge) * 12) - 1), priority: 'high', custom: true }
        setStore((s) => ({ customGoals: [...s.customGoals, goal] }))
      }}>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Goal name" />
        <label>CHF <input type="number" min={0} step={500} value={amount} onChange={(e) => setAmount(Number(e.target.value))} /></label>
        <label>by age <input type="number" min={Math.ceil(startAge)} max={maxAge} value={age} onChange={(e) => setAge(Number(e.target.value))} /></label>
        <button className="btn" type="submit">＋ Add</button>
      </form>
      <form className="goal-form" onSubmit={(e) => {
        e.preventDefault()
        // Keep the chosen age even if it is already lived: the engine then checks what actually happened.
        const a = Math.min(75, Math.max(Math.ceil(startAge), Math.round(retireAge)))
        const u = Math.min(120, Math.max(a + 1, 40, Math.round(untilAge)))
        const goal: Goal = { goal_id: `retire-${Date.now()}`, name: `Retire at ${a}`, target: 0,
          month: Math.max(0, Math.round((a - startAge) * 12)), priority: 'high', custom: true,
          kind: 'retire_early', until_age: u }
        setStore((s) => ({ customGoals: [...s.customGoals, goal] }))
      }}>
        <span>🌅 Retire early</span>
        <label>stop working at <input type="number" min={Math.ceil(startAge)} max={75} value={retireAge} onChange={(e) => setRetireAge(Number(e.target.value))} /></label>
        <label>money lasts to <input type="number" min={40} max={120} value={untilAge} onChange={(e) => setUntilAge(Number(e.target.value))} /></label>
        <button className="btn" type="submit">＋ Add</button>
      </form>
      <p className="tiny muted">🌅 A retirement goal is a check, not a choice: the engine replays your life as it is now, with “retire early” added at that age (no income until the pension at 65, which is lower for each year you stop early, 🟡), and checks that savings and investments last. 90 by default: at 65, Swiss men live to about 85 and women to about 88 on average (BFS). To actually do it, go to the 💼 Office.</p>
    </div>
  )
}

function RetireDetail({ g }: { g?: CheckedGoal }) {
  if (!g) return null
  const lasts = (g.status === 'met' ? `✔ Money lasts to ${g.until_age ?? 90}` : g.reason ?? '').replace(/\.$/, '')
  const earliest = g.earliest_age === undefined ? null
    : g.earliest_age === null ? 'Not before the pension age under these assumptions.'
      : `Earliest age that works: ${g.earliest_age}.`
  const lived = !g.what_if && g.status === 'not_met' && (g.reason ?? '').startsWith('Still')
  return <span className="tiny" style={{ flexBasis: '100%' }}>
    {lasts}{g.what_if ? ' (if you stop then, nothing else changed)' : ''}
    {lived ? ': that age is already lived in this reality. Go back on the timeline to before it to see the what-if' : ''}. {earliest}
  </span>
}
