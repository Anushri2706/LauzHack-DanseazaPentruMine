import { useState } from 'react'
import type { GameState } from '../api/types'
import { queueAction } from '../store/actions'
import { setStore, useStore } from '../store/game'
import { Avatar, Npc } from './Sprite'

const monthName = (m: number) => new Date(2025, m, 1).toLocaleString('en', { month: 'long', year: 'numeric' })

/** A student without a planned end: when do they expect to finish? The future (and the check-in) follow the answer. */
export function StudyPlan({ st }: { st: GameState }) {
  const plan = st.study_plan!
  const look = useStore((s) => s.look)
  const busy = useStore((s) => s.busy)
  const m = st.month
  const [months, setMonths] = useState(12)
  const preview = (st.actions.university ?? []).find((a) => a.type === 'plan_study')?.preview.text
  const choose = (n: number) => void queueAction({ type: 'plan_study', id: null, value: n })

  return (
    <div className="dialog-wrap">
      <div className="dialog px-box" role="dialog" aria-label="When do you expect to finish your studies?">
        <header className="dialog-head">
          <div className="portrait"><Avatar look={look} scale={4} /><Npc char={6} scale={4} /></div>
          <div>
            <h2>📅 When do you expect to finish your studies?</h2>
            <p className="greet"><b>Student advisor:</b> “Tell me when you think you’ll have your degree. I’ll check in with
              you then: finished, a bit more time, or something else.”</p>
          </div>
        </header>
        <div className="dialog-body">
          <ul className="acts">
            {plan.presets.map((n) => (
              <li key={n} className="act">
                <button disabled={busy} onClick={() => choose(n)}>
                  <span className="act-label">🎓 In {n} months ({monthName(m + n)})</span>
                </button>
              </li>
            ))}
            <li className="act">
              <div className="study-plan">
                <label className="small"><b>Another time:</b>{' '}
                  <select value={months} onChange={(e) => setMonths(Number(e.target.value))}>
                    {Array.from({ length: plan.max - plan.min + 1 }, (_, i) => plan.min + i).map((n) =>
                      <option key={n} value={n}>{n} months ({monthName(m + n)})</option>)}
                  </select>
                </label>
                <div className="row-end">
                  <button className="btn" disabled={busy} onClick={() => choose(months)}>Plan to finish in {months} months</button>
                </div>
              </div>
            </li>
          </ul>
          {preview && <p className="tiny muted">{preview.replace(/^You (change your )?plan to finish in \d+ months\. /, '')}</p>}
        </div>
        <footer className="dialog-foot">
          <button className="btn ghost" onClick={() => setStore({ studyPlanAsked: true })}>Not sure yet</button>
          <span className="tiny muted">🟡 Without a plan the future assumes you keep studying. You can plan later at the 🎓 University.</span>
        </footer>
      </div>
    </div>
  )
}
