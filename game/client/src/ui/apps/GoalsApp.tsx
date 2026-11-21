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
            <span>✏️ {g.name}</span>
            <span className="tiny muted">{chf(g.target)} by age {Math.round(startAge + (g.month + 1) / 12)}</span>
            <Badge id={g.goal_id} />
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
    </div>
  )
}
