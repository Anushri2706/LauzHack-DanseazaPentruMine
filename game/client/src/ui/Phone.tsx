import type { GameState } from '../api/types'
import { setStore, useStore, type AppId } from '../store/game'
import { BranchApp } from './apps/BranchApp'
import { ChecksApp } from './apps/ChecksApp'
import { DecideApp } from './apps/DecideApp'
import { GoalsApp } from './apps/GoalsApp'
import { LifeApp } from './apps/LifeApp'
import { MailApp } from './apps/MailApp'
import { MoneyApp } from './apps/MoneyApp'
import { MultiverseApp } from './apps/MultiverseApp'
import { SettingsApp } from './apps/SettingsApp'
import { sfx } from './sound'
import { useProjection } from './useGame'

const APPS: { id: AppId; icon: string; label: string }[] = [
  { id: 'branches', icon: '🌿', label: 'Paths' }, { id: 'decide', icon: '🔀', label: 'Decide' },
  { id: 'life', icon: '📈', label: 'Life' },
  { id: 'money', icon: '💰', label: 'Money' }, { id: 'multiverse', icon: '🌀', label: 'Multiverse' },
  { id: 'goals', icon: '🎯', label: 'Goals' }, { id: 'settings', icon: '⚙️', label: 'Settings' },
  { id: 'checks', icon: '✅', label: 'Checks' }, { id: 'mail', icon: '📬', label: 'Mail' },
]

export function Phone({ st }: { st: GameState }) {
  const app = useStore((s) => s.phone) ?? 'money'
  const s = useStore((x) => x)
  const proj = useProjection(s)
  const close = () => { sfx('phoneDown'); setStore({ phone: null }) }
  return (
    <div className="phone-wrap" onClick={close}>
      <div className="phone" role="dialog" aria-label="Phone" onClick={(e) => e.stopPropagation()}>
        <div className="phone-top">
          <span>📶 {st.date.label} · age {st.age_years}</span>
          <span>🌍 World #{st.world} · <span className="badge">Simulation · not a prediction or advice</span></span>
          <button className="icon-btn" style={{ color: '#fff6e9' }} onClick={close} title="Close (Esc)">✕</button>
        </div>
        <div className="phone-screen">
          {app === 'branches' && <BranchApp personaId={st.persona.profile_id} />}
          {app === 'decide' && <DecideApp />}
          {app === 'life' && <LifeApp startAge={st.persona.age} />}
          {app === 'money' && <MoneyApp st={st} />}
          {app === 'multiverse' && <MultiverseApp horizon={st.horizon} startAge={st.persona.age} />}
          {app === 'goals' && <GoalsApp st={st} proj={proj.data} />}
          {app === 'settings' && <SettingsApp personaId={st.persona.profile_id} />}
          {app === 'checks' && <ChecksApp />}
          {app === 'mail' && <MailApp st={st} />}
        </div>
        <nav className="phone-nav">
          {APPS.map((a) => (
            <button key={a.id} className={app === a.id ? 'on' : ''} onClick={() => { sfx('click'); setStore({ phone: a.id }) }}>
              <span className="ico">{a.icon}</span>{a.label}
            </button>
          ))}
        </nav>
      </div>
    </div>
  )
}
