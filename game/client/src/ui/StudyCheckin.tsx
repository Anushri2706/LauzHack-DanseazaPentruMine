import { useState } from 'react'
import type { GameAction, GameState } from '../api/types'
import { queueAction } from '../store/actions'
import { useStore } from '../store/game'
import { chf } from './format'
import { Avatar, Npc } from './Sprite'

const monthName = (m: number) => new Date(2025, m, 1).toLocaleString('en', { month: 'long', year: 'numeric' })

/** The planned end of a master's has come: the player says how it went before time moves on. */
export function StudyCheckin({ st }: { st: GameState }) {
  const c = st.study_checkin!
  const look = useStore((s) => s.look)
  const busy = useStore((s) => s.busy)
  const uni = st.actions.university ?? []
  const find = (id: string) => uni.find((a) => a.id === id)
  const finish = find('finish_study')
  const quit = find('quit_school')
  const presets = uni.filter((a) => a.id === 'extend_study')
  const perMonth = (presets[0]?.preview.one_off ?? 0) / (presets[0]?.value || 1)
  const [custom, setCustom] = useState(9)
  const kind = c.program === 'master_pt' ? 'part-time master’s' : c.program === 'bachelor' ? 'degree' : 'master’s'

  const Choice = ({ a, children }: { a: GameAction | undefined; children: React.ReactNode }) => (
    <li className={`act ${a?.enabled ? '' : 'off'}`}>
      <button disabled={busy || !a?.enabled} onClick={() => a && void queueAction({ type: a.type, id: a.id, value: a.value })}
        title={a?.enabled ? '' : a?.reason ?? ''}>
        <span className="act-label">{children}</span>
      </button>
      {a && <div className="act-preview">{a.preview.text}</div>}
      {a && !a.enabled && a.reason && <div className="act-reason">🔒 {a.reason}</div>}
    </li>
  )

  return (
    <div className="dialog-wrap">
      <div className="dialog px-box" role="dialog" aria-label="Did you finish your studies?">
        <header className="dialog-head">
          <div className="portrait"><Avatar look={look} scale={4} /><Npc char={6} scale={4} /></div>
          <div>
            <h2>🎓 {monthName(c.planned_end)}: your planned end date</h2>
            <p className="greet"><b>Student advisor:</b> “You expected to finish your {kind} now{c.program === 'bachelor' ? '' : `, after ${c.months_studied} months`}
              {c.default_months && c.months_studied !== c.default_months ? ` (the usual length is ${c.default_months})` : ''}. How did it go?”</p>
          </div>
        </header>
        <div className="dialog-body">
          <ul className="acts">
            <Choice a={finish}>🎓 Yes, I’ve finished with a degree</Choice>
            {presets.map((a) => <Choice key={a.value} a={a}>📚 Not yet: keep studying {a.value} more months</Choice>)}
            <li className="act">
              <div className="study-plan">
                <label className="small"><b>📚 Keep studying another length:</b>{' '}
                  <select value={custom} onChange={(e) => setCustom(Number(e.target.value))}>
                    {Array.from({ length: 24 }, (_, i) => i + 1).map((m) => <option key={m} value={m}>{m} more months ({monthName(c.planned_end + m)})</option>)}
                  </select>
                </label>
                {perMonth > 0 && <p className="tiny muted">🟡 Extra tuition about {chf(perMonth * custom)}.</p>}
                <div className="row-end">
                  <button className="btn ghost" disabled={busy || !presets[0]?.enabled}
                    onClick={() => void queueAction({ type: 'life_choice', id: 'extend_study', value: custom })}>Keep studying {custom} months</button>
                </div>
              </div>
            </li>
            <Choice a={quit}>🚪 I’m dropping out{c.program === 'bachelor' ? ' (without a degree)' : ''}</Choice>
          </ul>
        </div>
        <footer className="dialog-foot">
          <span className="tiny muted">🟡 assumption · your answer happens when you move forward on the timeline</span>
        </footer>
      </div>
    </div>
  )
}
