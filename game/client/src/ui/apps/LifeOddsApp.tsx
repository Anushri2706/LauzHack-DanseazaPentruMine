import { useState } from 'react'
import {
  Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api } from '../../api/client'
import type { HmmFan, HmmFanDecision } from '../../api/types'
import { useApi } from '../../api/useApi'
import { chf, k, pct } from '../format'

const COLORS = ['#7a6fd0', '#e8a64a', '#4fb38a', '#e0707a', '#4a90c2', '#a0784a']
const STATE_META: Record<string, { label: string; color: string }> = {
  employed: { label: 'Employed', color: '#4fb38a' },
  part_time: { label: 'Part-time', color: '#9ad3b8' },
  job_search: { label: 'Job search', color: '#e0707a' },
  student: { label: 'Student', color: '#7a6fd0' },
  retired: { label: 'Retired', color: '#b9a98f' },
}
const HORIZONS = [60, 120, 240]

/** Cash fans (P10–P90 band and median) for every chosen decision, from the regime HMM. */
function FanChart({ fan, startAge }: { fan: HmmFan; startAge: number }) {
  const H = fan.horizon
  const step = H > 120 ? 3 : 1
  const data = []
  for (let i = 0; i <= H; i += step) {
    const d: Record<string, number | number[]> = { age: +(startAge + i / 12).toFixed(3) }
    fan.decisions.forEach((x, j) => {
      if (i === 0) {
        d[`b${j}`] = [fan.opening_cash, fan.opening_cash]
        d[`m${j}`] = fan.opening_cash
      } else {
        d[`b${j}`] = [x.cash.p10[i - 1], x.cash.p90[i - 1]]
        d[`m${j}`] = x.cash.p50[i - 1]
      }
    })
    data.push(d)
  }
  const names: Record<string, string> = {}
  fan.decisions.forEach((x, j) => { names[`b${j}`] = `${x.label} (P10–P90)`; names[`m${j}`] = `${x.label} (median)` })
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 12, right: 16, bottom: 4, left: 4 }}>
        <CartesianGrid stroke="#eee3d3" strokeDasharray="2 4" />
        <XAxis dataKey="age" type="number" domain={[startAge, startAge + H / 12]} tickCount={Math.min(11, H / 12 + 1)}
          tickFormatter={(a) => `${Math.round(a)}`} stroke="#7a7093" fontSize={12} />
        <YAxis tickFormatter={k} stroke="#7a7093" width={52} fontSize={12} />
        <Tooltip
          contentStyle={{ background: '#fff6e9', border: '2px solid #4a4063', borderRadius: 0, fontFamily: 'inherit' }}
          labelFormatter={(a) => `Age ${Number(a).toFixed(1)}`}
          formatter={(v, name) => [Array.isArray(v) ? `${k(Number(v[0]))} – ${k(Number(v[1]))}` : chf(Number(v)), names[String(name)] ?? String(name)]}
        />
        <ReferenceLine y={0} stroke="#ff8a8a" strokeDasharray="4 4" />
        {fan.decisions.map((_, j) => (
          <Area key={`b${j}`} dataKey={`b${j}`} stroke="none" fill={COLORS[j % COLORS.length]} fillOpacity={0.16} isAnimationActive={false} />
        ))}
        {fan.decisions.map((_, j) => (
          <Line key={`m${j}`} dataKey={`m${j}`} stroke={COLORS[j % COLORS.length]} strokeWidth={2.5} dot={false} isAnimationActive={false} />
        ))}
      </ComposedChart>
    </ResponsiveContainer>
  )
}

/** Share of simulated lives in each regime, over time (one strip per decision). */
function RegimeStrip({ d, states, horizon, startAge }: { d: HmmFanDecision; states: string[]; horizon: number; startAge: number }) {
  const W = 600, H = 34, cols = Math.min(horizon, 120)
  const per = horizon / cols
  return (
    <svg viewBox={`0 0 ${W} ${H + 14}`} style={{ width: '100%', display: 'block' }} role="img" aria-label={`Regime mix for ${d.label}`}>
      {Array.from({ length: cols }, (_, c) => {
        const t = Math.min(horizon - 1, Math.floor(c * per))
        let y = 0
        return states.map((s) => {
          const h = (d.occupancy[s]?.[t] ?? 0) * H
          const r = <rect key={`${c}-${s}`} x={(c * W) / cols} y={y} width={W / cols + 0.5} height={h} fill={STATE_META[s]?.color ?? '#ccc'}>
            <title>{`Age ${(startAge + t / 12).toFixed(1)} · ${STATE_META[s]?.label ?? s}: ${pct(d.occupancy[s]?.[t] ?? 0)}`}</title></rect>
          y += h
          return r
        })
      })}
      <text x={0} y={H + 12} fontSize={10} fill="#7a7093">{Math.round(startAge)}</text>
      <text x={W} y={H + 12} fontSize={10} fill="#7a7093" textAnchor="end">{Math.round(startAge + horizon / 12)}</text>
    </svg>
  )
}

export function LifeOddsApp({ personaId }: { personaId: string }) {
  const opts = useApi(personaId, () => api.hmmDecisions(personaId))
  const [picked, setPicked] = useState<string[] | null>(null)
  const [horizon, setHorizon] = useState(240)
  const available = opts.data?.decisions.filter((d) => d.available) ?? []
  const chosen = picked ?? (available.some((d) => d.id === 'masters_now') ? ['stay', 'masters_now'] : ['stay'])
  const valid = chosen.filter((id) => available.some((d) => d.id === id))
  const key = opts.data && valid.length ? JSON.stringify([personaId, valid, horizon]) : null
  const fan = useApi(key, () => api.hmmFan(personaId, valid, horizon))
  const toggle = (id: string) => {
    const has = chosen.includes(id)
    if (has && chosen.length === 1) return
    if (!has && chosen.length >= COLORS.length) return
    setPicked(has ? chosen.filter((x) => x !== id) : [...chosen, id])
  }
  const f = fan.data
  const startAge = f?.age ?? 0
  return (
    <div>
      <h3>🔮 Life odds</h3>
      <p className="tiny muted">
        Big life decisions, each simulated over {f ? f.n_paths.toLocaleString() : 'many'} possible lives by the regime
        model (employed, part-time, job search, student, retired, with transition odds from Swiss labour data). All
        decisions share the same random seed, so differences come from the decision, not luck. Starts Jan 2025.
      </p>
      {opts.error && <p className="error">{opts.error}</p>}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, margin: '8px 0' }}>
        {available.map((d) => {
          const j = valid.indexOf(d.id)
          return (
            <button key={d.id} className={`chip ${j >= 0 ? 'on' : ''}`} title={d.text} onClick={() => toggle(d.id)}
              style={j >= 0 ? { boxShadow: `0 0 0 2px ${COLORS[j % COLORS.length]}` } : undefined}>
              {d.icon} {d.label}
            </button>
          )
        })}
      </div>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6 }}>
        <span className="tiny muted">Horizon:</span>
        {HORIZONS.map((h) => (
          <button key={h} className={`chip ${horizon === h ? 'on' : ''}`} onClick={() => setHorizon(h)}>{h / 12} years</button>
        ))}
        {fan.loading && <span className="tiny muted">simulating lives…</span>}
      </div>
      {fan.error && <p className="error">{fan.error}</p>}
      {f && (
        <>
          <FanChart fan={f} startAge={startAge} />
          <table className="cmp-table">
            <thead>
              <tr><th>Decision</th><th>Cash at the end (P10 – median – P90)</th><th>Lives with cash below 0 at some point</th>
                {f.decisions[0]?.goals.map((g) => <th key={g.goal_id}>🎯 {g.name}</th>)}</tr>
            </thead>
            <tbody>
              {f.decisions.map((d, j) => (
                <tr key={d.id}>
                  <td><span className="dot" style={{ background: COLORS[j % COLORS.length], display: 'inline-block', width: 10, height: 10, marginRight: 6 }} />{d.icon} {d.label}</td>
                  <td>{k(d.closing.p10)} – <b>{k(d.closing.p50)}</b> – {k(d.closing.p90)}</td>
                  <td>{pct(d.p_ever_negative)}</td>
                  {d.goals.map((g) => (
                    <td key={g.goal_id} title={`${chf(g.target)} by ${g.target_date}${g.evaluated_at !== 'goal_month' ? ` (scored at ${g.evaluated_at.replace(/_/g, ' ')})` : ''}`}>
                      {pct(g.hit_rate)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <h4>What life looks like, month by month</h4>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 4 }}>
            {f.states.map((s) => (
              <span key={s} className="tiny"><span style={{ display: 'inline-block', width: 10, height: 10, background: STATE_META[s]?.color, marginRight: 4 }} />{STATE_META[s]?.label ?? s}</span>
            ))}
          </div>
          {f.decisions.map((d, j) => (
            <div key={d.id} style={{ marginBottom: 6 }}>
              <div className="tiny"><b style={{ color: COLORS[j % COLORS.length] }}>{d.icon} {d.label}</b>{' '}
                <span className="muted">· on average {Math.round(d.months_in_state.job_search ?? 0)} months job-searching, {Math.round(d.months_in_state.part_time ?? 0)} part-time
                  {d.one_offs.length > 0 && ` · one-offs: ${d.one_offs.map((o) => `${o.label} ${k(o.amount)}`).join(', ')}`}</span></div>
              <RegimeStrip d={d} states={f.states} horizon={f.horizon} startAge={startAge} />
            </div>
          ))}
          <p className="tiny muted">🔵 {f.note} Percentages are shares of simulated lives under the stated assumptions, not real-world probabilities. {f.disclaimer}</p>
        </>
      )}
    </div>
  )
}
