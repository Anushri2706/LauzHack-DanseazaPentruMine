import { api } from '../../api/client'
import type { AssumptionRow } from '../../api/types'
import { useApi } from '../../api/useApi'
import { setStore, useStore } from '../../store/game'
import { TAGS } from '../format'

const isPct = (unit: string) => unit.startsWith('%')
const UNCERTAINTY: { label: string; keys: string[] }[] = [
  { label: 'Market swings and drops', keys: ['return_volatility', 'crash_prob_annual'] },
  { label: 'Income-loss spells', keys: ['job_loss_prob_annual'] },
  { label: 'Rent rises', keys: ['rent_shock_prob_annual'] },
]

/** The old Assumptions drawer, inside the phone: every number (incl. the game catalogue), its tag, editable. */
export function SettingsApp({ personaId }: { personaId: string }) {
  const rows = useApi(`assume-${personaId}`, () => api.assumptions(personaId))
  const cat = useApi(`cat-${personaId}`, () => api.catalogue(personaId))
  const overrides = useStore((s) => s.assumptions)
  const set = (o: Record<string, number>) => setStore({ assumptions: o })
  const groups = [...new Set(rows.data?.map((r) => r.group))]
  const val = (r: AssumptionRow) => overrides[r.key] ?? r.value
  const on = (keys: string[]) => keys.every((k) => overrides[k] !== 0)
  const toggle = (keys: string[]) => {
    const o = { ...overrides }
    if (on(keys)) keys.forEach((k) => { o[k] = 0 })
    else keys.forEach((k) => { delete o[k] })
    set(o)
  }
  return (
    <div>
      <h3>⚙️ Settings & assumptions</h3>
      <p className="tiny muted">Every number the engine uses. Change one and every reality is recomputed from month 0 (your past choices stay lived; if one is no longer affordable it shows up as negative cash).
        {Object.keys(overrides).length > 0 && <> <button className="link" onClick={() => set({})}>↺ reset all ({Object.keys(overrides).length} edited)</button></>}</p>
      <div className="toggles">
        <b>Uncertainty in the simulated worlds</b>
        {UNCERTAINTY.map((u) => (
          <label key={u.label}><input type="checkbox" checked={on(u.keys)} onChange={() => toggle(u.keys)} /> {u.label}</label>
        ))}
        <label><input type="checkbox" checked={overrides.joy_enabled !== 0} onChange={() => {
          const o = { ...overrides }
          if (o.joy_enabled === 0) delete o.joy_enabled
          else o.joy_enabled = 0
          set(o)
        }} /> Count joy points (a game preference score: switch it off if you prefer)</label>
      </div>
      {cat.data && <p className="tiny muted">🏠 Home prices are Lausanne prices × {cat.data.home_price_level.level} for canton {cat.data.home_price_level.canton} (rough level, our assumption). {cat.data.note}</p>}
      <p className="tiny">{Object.entries(TAGS).filter(([k]) => ['profile', 'starter_kit', 'scenario_csv', 'ours', 'peer'].includes(k)).map(([k, t]) => <span key={k} style={{ marginRight: 12 }}>{t.dot} {t.label}</span>)}</p>
      {rows.error && <p className="error">{rows.error}</p>}
      {groups.map((g) => (
        <details key={g} className="assume-group" open={g === 'Saving & investing'}>
          <summary>{g}</summary>
          {rows.data!.filter((r) => r.group === g).map((r) => {
            const v = val(r)
            const shown = isPct(r.unit) ? +(v * 100).toFixed(2) : +v.toFixed(2)
            const edited = r.key in overrides
            return (
              <label key={r.key} className={`assume-row ${edited ? 'edited' : ''}`} title={r.note || undefined}>
                <span>{TAGS[edited ? 'ours' : r.tag]?.dot} {r.label}</span>
                <input type="number" value={shown} step={isPct(r.unit) ? 0.5 : r.unit.includes('CHF') ? 100 : 1}
                  onChange={(e) => {
                    const raw = Number(e.target.value)
                    if (Number.isNaN(raw) || e.target.value === '') return
                    let x = isPct(r.unit) ? raw / 100 : raw
                    if (r.min !== null) x = Math.max(r.min, x)
                    if (r.max !== null) x = Math.min(r.max, x)
                    set({ ...overrides, [r.key]: x })
                  }} />
                <span className="unit">{r.unit}</span>
              </label>
            )
          })}
        </details>
      ))}
    </div>
  )
}
