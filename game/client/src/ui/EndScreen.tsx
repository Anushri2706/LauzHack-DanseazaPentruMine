import type { GameState } from '../api/types'
import { realityName, setStore, useStore } from '../store/game'
import { ScoreTiles } from './apps/MoneyApp'
import { useProjection } from './useGame'
import { Avatar } from './Sprite'

/** Postcard at the end of the horizon: this reality's six scores, honestly framed. */
export function EndScreen({ st }: { st: GameState }) {
  const s = useStore((x) => x)
  const proj = useProjection(s)
  const done = (phone: 'multiverse' | null) => setStore({ endSeen: true, phone })
  return (
    <div className="dialog-wrap">
      <div className="postcard px-box" role="dialog" aria-label="Ten years later">
        <div className="stamp">🏔️</div>
        <div style={{ display: 'flex', gap: 14, alignItems: 'center', marginBottom: 8 }}>
          <Avatar look={s.look} scale={5} />
          <div>
            <h2>Greetings from age {st.age_years}!</h2>
            <p className="small" style={{ margin: 0 }}>{st.persona.name} lived ten years of “<b>{realityName(s, s.cursor)}</b>” in world #{st.world}.</p>
          </div>
        </div>
        <ScoreTiles st={st} proj={proj.data} />
        <p className="note small" style={{ marginTop: 14 }}>This is <b>one of many possible futures</b>: a simulation under stated assumptions, not a prediction or advice.
          A higher net worth is not a better life. What would a different choice have changed?</p>
        <div className="row-end">
          <button className="btn ghost" onClick={() => done(null)}>Keep looking around</button>
          <button className="btn ghost" onClick={() => { setStore({ pinB: s.cursor }); done('multiverse') }}>⚖️ Compare</button>
          <button className="btn" onClick={() => done('multiverse')}>🌀 Explore another reality</button>
        </div>
      </div>
    </div>
  )
}
