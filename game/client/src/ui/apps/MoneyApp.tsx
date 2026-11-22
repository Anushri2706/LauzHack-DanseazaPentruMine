import { useState, type ReactNode } from 'react'
import { api } from '../../api/client'
import type { GameState, Projection, Roll } from '../../api/types'
import { actionLabel, colorOf, getStore, pathIds, requestFor, setStore, useStore, type Dim } from '../../store/game'
import { chf, EDU_LABEL, k, pct } from '../format'
import { useProjection } from '../useGame'
import { MoneyChart, type Metric } from './MoneyChart'

/** P10–P90 range bar with a P50 tick (from the old Hud). */
function RangeBar({ lo, mid, hi }: { lo: number; mid: number; hi: number }) {
  const min = Math.min(0, lo), max = Math.max(hi * 1.1, 1)
  const x = (v: number) => `${((v - min) / (max - min)) * 100}%`
  return (
    <div className="rangebar" title="80% CI under our assumptions (P10–P90 of simulated worlds)">
      <div className="rangebar-fill" style={{ left: x(lo), width: `calc(${x(hi)} - ${x(lo)})` }} />
      <div className="rangebar-tick" style={{ left: x(mid) }} />
    </div>
  )
}

export function Tile({ dim, title, main, sub, tone, children }: {
  dim: Dim; title: string; main: ReactNode; sub?: ReactNode; tone?: string; children?: ReactNode
}) {
  const matters = useStore((s) => s.matters.includes(dim))
  const toggle = () => setStore((s) => ({ matters: s.matters.includes(dim) ? s.matters.filter((d) => d !== dim) : [...s.matters, dim] }))
  return (
    <div className={`tile ${matters ? 'matters' : ''} ${tone ?? ''}`}>
      <div className="t-title"><span>{title}</span>
        <button className={`heart ${matters ? 'on' : ''}`} onClick={toggle} title="This matters to me">❤️</button></div>
      <div className="t-main">{main}</div>
      {sub && <div className="t-sub">{sub}</div>}
      {children}
    </div>
  )
}

// 🟡 Our thresholds: high ≥ 20% of worlds below zero within 12 months, some ≥ 5% (or ≥ 20% at any point ahead).
const RISK_LABEL = { low: 'Low risk', some: 'Some risk', high: 'High risk' } as const
const RISK_TONE = { low: 'good', some: 'warn', high: 'bad' } as const

export function ScoreTiles({ st, proj }: { st: GameState; proj?: Projection }) {
  const s = st.scores
  const f = proj?.final_net_worth
  const fs = proj?.final_scores
  const endAge = Math.round(st.persona.age + st.horizon / 12)
  const shares = proj ? Object.values(proj.goal_met_share) : []
  const tone = { steady: 'good', tight: 'warn', shortfall: 'bad' }[s.safety.level]
  const risk = proj?.risk
  return (
    <div className="tiles">
      <Tile dim="money" title={`💰 Money at ${endAge}`}
        main={f ? <>≈ {k(f.p10)} – {k(f.p90)}</> : k(s.money.net_worth)}
        sub={proj?.ended
          ? <>🔵 lived: {k(s.money.net_worth)} · range = same choices in {proj.n} worlds (other luck)</>
          : <>🔵 net worth · middle {f ? k(f.p50) : '…'} · now {k(s.money.net_worth)}</>}>
        {f && <RangeBar lo={f.p10} mid={f.p50} hi={f.p90} />}
      </Tile>
      <Tile dim="safety" title="🛡️ Cash risk" tone={risk ? RISK_TONE[risk.level] : tone}
        main={risk && !proj?.ended ? RISK_LABEL[risk.level] : { steady: 'Steady', tight: 'Tight', shortfall: 'Shortfall' }[s.safety.level]}
        sub={<>
          {risk && !proj?.ended
            ? <>🔵 cash below zero in {pct(risk.below_zero_12m)} of worlds in the next {risk.months} months ({pct(risk.below_zero_ahead)} at any point ahead)</>
            : proj?.ended ? <>🔵 cash below zero in {pct(proj.share_worlds_shortfall)} of worlds with the same choices</> : '…'}
          <br />now: {{ steady: 'above your buffer', tight: 'below your buffer', shortfall: 'cash below zero, financing needed' }[s.safety.level]}
          {s.safety.months_negative_cash > 0 && s.safety.recovered && <> · dipped below zero for {s.safety.months_negative_cash} months earlier, recovered</>}
        </>} />
      <Tile dim="goals" title="🎯 Goals" main={fs ? `${fs.goals.met} / ${fs.goals.total}` : `${s.goals.met} / ${s.goals.total}`}
        sub={shares.length ? <>{proj?.ended ? 'met on the path you lived' : 'met if nothing surprising happens'} · in {shares.map(pct).join(', ')} of worlds</>
          : proj && fs?.goals.total === 0 ? 'no goals chosen' : '…'} />
      <Tile dim="freedom" title="🕊️ Freedom"
        main={s.freedom.runway_months == null ? '—' : `${Math.max(0, Math.round(s.freedom.runway_months))} months`}
        sub={<>of costs covered by savings now{fs?.freedom.runway_months != null ? ` · ${Math.max(0, Math.round(fs.freedom.runway_months))} at ${endAge}` : ''}{s.freedom.debt_free ? ' · debt-free' : ''}</>} />
      <Tile dim="experience" title="🎓 Experience"
        main={EDU_LABEL[s.experience.education]}
        sub={[`${s.experience.months_studying} months studying so far`,
          s.experience.months_abroad > 0 && `${(s.experience.months_abroad / 12).toFixed(1)} years abroad`,
          fs && !proj?.ended && fs.experience.education !== s.experience.education && `${EDU_LABEL[fs.experience.education]} by ${endAge} if nothing changes`,
        ].filter(Boolean).join(' · ')} />
      <Tile dim="joy" title="🌸 Joy (not money)" main={`${Math.round(s.joy.total)} pts`}
        sub={<>{fs && !proj?.ended ? `≈ ${Math.round(fs.joy.total)} by ${endAge} if nothing changes · ` : ''}🟡 a game preference score, not a measure of a good life</>} />
    </div>
  )
}

const BRIDGE_LABELS: Record<string, string> = {
  income: 'Income', expenses: 'Rent & living costs', lifestyle: 'Habits', purchases: 'Purchases & trips',
  one_offs: 'Tuition & moves', interest: 'Loan interest', mortgage_interest: 'Mortgage interest',
  maintenance: 'Home maintenance', home_fees: 'Home fees', investment_return: 'Investment returns',
  home_appreciation: 'Home value growth',
}

export function MoneyApp({ st }: { st: GameState }) {
  const s = useStore((x) => x)
  const proj = useProjection(s)
  const [metric, setMetric] = useState<Metric>('net_worth')
  const [band, setBand] = useState(true)
  const [ghosts, setGhosts] = useState(false)
  const [roll, setRoll] = useState<Roll | undefined>()
  const color = colorOf(s, s.cursor)
  const marks = pathIds(s, s.cursor).map((id) => s.nodes[id].turn).filter((t) => t !== null)
    .map((t) => ({ month: t!.month, label: actionLabel(t!.actions.find((a) => a.type === 'life_choice' || a.type === 'buy_home') ?? t!.actions[0]) }))
  const doRoll = async () => {
    const w = Math.floor(Math.random() * 500)
    setRoll(await api.roll(requestFor(getStore(), undefined, false), w))
  }
  const p = proj.data
  return (
    <div>
      <h3>Money & life scores</h3>
      <p className="tiny muted">Ranges are where 80% of 500 simulated worlds land under our assumptions: not probabilities, not a prediction. Tap ❤️ on what matters to you.</p>
      <ScoreTiles st={st} proj={p} />
      <div className="chart-tools">
        <button className={`chip ${metric === 'net_worth' ? 'on' : ''}`} onClick={() => setMetric('net_worth')}>Net worth</button>
        <button className={`chip ${metric === 'cash' ? 'on' : ''}`} onClick={() => setMetric('cash')}>Cash</button>
        <label><input type="checkbox" checked={band} onChange={(e) => setBand(e.target.checked)} /> 80% CI (P10–P90)</label>
        <label><input type="checkbox" checked={ghosts} onChange={(e) => setGhosts(e.target.checked)} /> Stress tests</label>
        <button className="btn" onClick={() => void doRoll()} disabled={!p || st.ended}>🎲 Roll another world</button>
        {proj.loading && <span className="tiny muted">computing 500 worlds…</span>}
      </div>
      <div className="chart-box">
        {p ? <MoneyChart proj={p} startAge={st.persona.age} opening={{ net_worth: st.persona.net_worth, cash: st.persona.opening_cash }}
          metric={metric} color={color} showBand={band} showGhosts={ghosts} roll={roll} goals={p.goals} marks={marks} />
          : <p className="muted small" style={{ padding: 20 }}>{proj.error ?? 'Simulating…'}</p>}
      </div>
      <p className="tiny muted">Solid: lived so far in world #{st.world}. Dotted: if nothing surprising happens from now on. Band: other worlds from now on, same choices.
        {ghosts && p && <> Ghosts: {p.ghosts.map((g) => g.text).join('; ')}.</>}</p>
      {roll && (
        <div className="note small">🎲 World #{roll.world}: same choices, different luck from now on.
          {roll.events.length ? <ul className="events">{roll.events.map((e, i) => <li key={i}>{e.text}</li>)}</ul> : ' Nothing unusual happens in it.'}
          <button className="link" onClick={() => setRoll(undefined)}>hide</button></div>
      )}
      {p && (
        <>
          <h4>Why net worth ends where it does (if nothing surprising happens) 🔵</h4>
          <table className="statement">
            <tbody>
              <tr><td>Starting net worth 🟢</td><td className="num">{chf(p.bridge.opening)}</td></tr>
              {Object.entries(BRIDGE_LABELS).filter(([key]) => Math.abs(p.bridge[key] ?? 0) > 0.5).map(([key, label]) => (
                <tr key={key}><td>{label}</td><td className="num">{p.bridge[key] > 0 ? '+' : ''}{chf(p.bridge[key])}</td></tr>
              ))}
              <tr className="total"><td>Net worth at {Math.round(st.persona.age + st.horizon / 12)}</td><td className="num">{chf(p.bridge.closing)}</td></tr>
            </tbody>
          </table>
          <p className="tiny muted">An exact bridge: the lines add up to the closing value. Saving, loans, the down payment and amortization are transfers and never appear.</p>
        </>
      )}
      {st.statement && (
        <>
          <h4>Last month ({st.statement.label})</h4>
          <table className="statement"><tbody>
            {st.statement.lines.map((l) => <tr key={l.label} className={l.kind}><td>{l.label}</td><td className="num">{l.amount > 0 ? '+' : ''}{chf(l.amount)}</td></tr>)}
          </tbody></table>
        </>
      )}
    </div>
  )
}
