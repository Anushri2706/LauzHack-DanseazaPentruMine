import type { GameState } from '../../api/types'
import { monthLabel } from '../format'

export function MailApp({ st }: { st: GameState }) {
  const letters = [...st.letters].reverse()
  return (
    <div>
      <h3>📬 Mail</h3>
      <p className="tiny muted">Letters about what happened in your world (#{st.world}). Market drops, job losses and rent rises come from that seeded world, the same one for all your realities.</p>
      {letters.length === 0 ? <p className="muted">No letters yet. Sleep at home to let time pass.</p> : (
        <ul className="letters">
          {letters.map((l, i) => (
            <li key={i}><span style={{ fontSize: 22 }}>{l.icon}</span>
              <div><b>{l.title}</b> <span className="tiny muted">{monthLabel(l.month)}</span><div className="small">{l.text}</div></div></li>
          ))}
        </ul>
      )}
    </div>
  )
}
