import { useState } from 'react'
import { api } from '../../api/client'
import type { CompareResult, FinalScores } from '../../api/types'
import { useApi } from '../../api/useApi'
import { continueReality, goBack } from '../../store/actions'
import {
  colorOf, pathIds, realities, realityName, requestFor, setStore, useStore, type Dim, type Store,
} from '../../store/game'
import { chf, EDU_RANK, k, monthLabel } from '../format'
import { SCORE_META } from '../HudBar'

/** The reality tree over age (adapted from the old Multiverse map; nodes are months with choices). */
function Tree({ s, horizon, startAge, selected, onSelect }: {
  s: Store; horizon: number; startAge: number; selected: string | null; onSelect: (id: string) => void
}) {
  const W = 1000
  const lanes: Record<string, number> = {}
  let next = 0
  const place = (id: string): number => {
    const n = s.nodes[id]
    if (n.children.length === 0) { lanes[id] = next++; return lanes[id] }
    const ys = n.children.map(place)
    lanes[id] = ys[0]
    return ys[0]
  }
  place(s.root)
  const H = Math.max(1, next) * 48 + 44
  const month = (id: string) => s.nodes[id].turn?.month ?? -4
  // Scale to the part of life you have explored (at least ten years), not the whole timeline to 120.
  const furthest = Math.max(s.current, ...Object.values(s.nodes).map((n) => Math.max(n.reached, n.turn?.month ?? 0)))
  const span = Math.min(horizon, Math.max(120, Math.ceil((furthest + 12) / 12) * 12))
  const x = (m: number) => 30 + ((Math.min(m, span) + 4) / (span + 4)) * (W - 190)
  const y = (id: string) => 40 + lanes[id] * 48
  const onPath = new Set(pathIds(s, s.cursor))
  const tips = new Set(realities(s))
  const every = span <= 180 ? 12 : span <= 480 ? 60 : 120
  const ticks = Array.from({ length: Math.floor(span / every) + 1 }, (_, i) => i * every)
  return (
    <svg className="multiverse" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMinYMin meet">
      {ticks.map((m) => (
        <g key={m}>
          <line x1={x(m)} x2={x(m)} y1={16} y2={H - 6} stroke="#efe5d4" />
          <text x={x(m)} y={12} textAnchor="middle" className="mv-tick">{Math.round(startAge + m / 12)}</text>
        </g>
      ))}
      {Object.values(s.nodes).filter((n) => n.parent).map((n) => {
        const p = n.parent!
        const x1 = x(month(p)), y1 = y(p), x2 = x(month(n.id)), y2 = y(n.id)
        const mx = (x1 + x2) / 2
        return <path key={n.id} d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`} fill="none"
          stroke={colorOf(s, n.id)} strokeWidth={onPath.has(n.id) ? 4 : 2} strokeOpacity={onPath.has(n.id) ? 1 : 0.45} />
      })}
      {[...tips].map((id) => {
        const reached = id === s.cursor ? s.current : s.nodes[id].reached
        if (reached <= Math.max(0, month(id))) return null
        return <line key={`tip-${id}`} x1={x(month(id))} x2={x(reached)} y1={y(id)} y2={y(id)} stroke={colorOf(s, id)}
          strokeWidth={id === s.cursor ? 4 : 2} strokeOpacity={id === s.cursor ? 1 : 0.45} strokeDasharray={id === s.cursor ? undefined : '4 3'} />
      })}
      {Object.values(s.nodes).map((n) => {
        const cx = x(month(n.id)), cy = y(n.id)
        const tip = tips.has(n.id)
        const reached = n.id === s.cursor ? s.current : n.reached
        return (
          <g key={n.id} className="mv-node" onClick={() => onSelect(n.id)} opacity={onPath.has(n.id) || tip ? 1 : 0.55}>
            <title>{n.turn ? `${n.label} · ${monthLabel(n.turn.month)}` : 'Start'}</title>
            {selected === n.id && <rect x={cx - 11} y={cy - 11} width={22} height={22} fill="none" stroke="#4a4063" strokeWidth={2} />}
            <rect x={cx - 6} y={cy - 6} width={12} height={12} fill={n.turn ? colorOf(s, n.id) : '#fff6e9'} stroke="#4a4063" strokeWidth={2} />
            {n.turn && <text x={cx} y={cy - 12} textAnchor="middle" className="mv-label">{n.label}</text>}
            {tip && (
              <g onClick={(e) => { e.stopPropagation(); onSelect(`tip:${n.id}`) }}>
                {n.id === s.cursor && <circle cx={x(reached)} cy={cy} r={10} fill="none" stroke="#4a4063" strokeWidth={2} className="pulse" />}
                <circle cx={x(reached)} cy={cy} r={5} fill={colorOf(s, n.id)} stroke="#4a4063" strokeWidth={2} />
                <text x={x(reached) + 10} y={cy + 4} className="mv-label" fill={colorOf(s, n.id)}>{realityName(s, n.id)}{n.id === s.cursor ? ' (you are here)' : ''}</text>
              </g>
            )}
          </g>
        )
      })}
    </svg>
  )
}

function lean(dim: Dim, a: FinalScores, b: FinalScores): number {
  switch (dim) {
    case 'money': return b.wealth.net_worth - a.wealth.net_worth
    case 'safety': return (a.safety.months_negative_cash - b.safety.months_negative_cash) * 100 + (a.safety.months_below_buffer - b.safety.months_below_buffer)
    case 'goals': return b.goals.met - a.goals.met
    case 'freedom': return (b.freedom.runway_months ?? 0) - (a.freedom.runway_months ?? 0)
    case 'experience': return (EDU_RANK[b.experience.education] - EDU_RANK[a.experience.education]) * 100 + (b.experience.months_abroad - a.experience.months_abroad)
    case 'joy': return b.joy.total - a.joy.total
  }
}

const MONEY_KEYS = new Set(['final_net_worth', 'final_cash', 'final_investments', 'final_debt', 'min_cash', 'final_home_equity'])
const LOWER_IS_SAFER = new Set(['final_debt', 'months_below_buffer', 'months_negative_cash'])

function Compare({ s, a, b }: { s: Store; a: string; b: string }) {
  const na = realityName(s, a), nb = realityName(s, b)
  const ra = requestFor(s, a, false), rb = requestFor(s, b, false)
  const startAge = s.persona?.age ?? 20
  const ages = [30, 40, 65].filter((x) => x > startAge + 0.5)
  // Default: 10 years on, the challenge's own horizon (Alexia: 20 → 30). "End" = the whole life.
  const [at, setAt] = useState<number | undefined>(ages[0])
  const cmp = useApi(JSON.stringify([ra, rb, at]), () => api.compare(ra, rb, `A: ${na}`, `B: ${nb}`, at))
  const ca = colorOf(s, a), cb = colorOf(s, b)
  const d: CompareResult | undefined = cmp.data
  return (
    <div>
      <h4>Compare <span style={{ color: ca }}>A: {na}</span> vs <span style={{ color: cb }}>B: {nb}</span></h4>
      <div className="chart-tools">
        <span className="tiny">Compare at age</span>
        {ages.map((x) => <button key={x} className={`chip ${at === x ? 'on' : ''}`} onClick={() => setAt(x)}>{x}</button>)}
        <button className={`chip ${at === undefined ? 'on' : ''}`} onClick={() => setAt(undefined)}>end</button>
        {cmp.loading && d && <span className="tiny muted">recomputing…</span>}
      </div>
      {cmp.loading && !d && <p className="muted small">Comparing in the same world…</p>}
      {cmp.error && <p className="error">{cmp.error}</p>}
      {d && (
        <>
          <table className="cmp-table">
            <thead><tr><th /><th style={{ color: ca }}>A</th><th style={{ color: cb }}>B</th></tr></thead>
            <tbody>
              <tr><td>🔵 Net worth at {d.at_age != null ? Math.round(d.at_age) : 'the end'}, 80% CI (P10–P90)</td>
                <td>{d.ranges.a ? `${k(d.ranges.a.p10)} – ${k(d.ranges.a.p90)}` : '—'}</td>
                <td>{d.ranges.b ? `${k(d.ranges.b.p10)} – ${k(d.ranges.b.p90)}` : '—'}</td></tr>
              {d.table.map((r) => {
                const fmt = (v: number) => (MONEY_KEYS.has(r.key) ? chf(v) : String(Math.round(v)))
                const saferB = LOWER_IS_SAFER.has(r.key) ? r.b < r.a : r.b > r.a
                return (
                  <tr key={r.key}>
                    <td>{r.key === 'total_joy' ? '🟡' : '🔵'} {r.label}</td>
                    <td className={r.diff !== 0 && !saferB ? 'lead' : ''}>{fmt(r.a)}</td>
                    <td className={r.diff !== 0 && saferB ? 'lead' : ''}>{fmt(r.b)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          <h4>Why they differ</h4>
          <ul className="explain">{d.explanation.map((e, i) => <li key={i}>{e}</li>)}</ul>
          <h4>Fit with what matters to you</h4>
          {s.matters.length === 0 && <p className="tiny muted">Tap ❤️ on the score tiles in the Money app to say what matters to you.</p>}
          <ul className="explain">
            {s.matters.map((dim) => {
              const l = lean(dim, d.scores.a, d.scores.b)
              return <li key={dim}><b>{SCORE_META[dim].icon} {SCORE_META[dim].label}:</b>{' '}
                {Math.abs(l) < 1e-6 ? 'about even' : <span style={{ color: l > 0 ? cb : ca }}>{l > 0 ? 'B' : 'A'} fits better</span>}</li>
            })}
          </ul>
          <p className="tiny muted">No reality is “best”: each trades money, safety, goals, freedom, experience and joy differently. {d.note} Amounts in nominal CHF; the 🌪️ Matters app shows which assumptions drive each reality.</p>
        </>
      )}
    </div>
  )
}

export function MultiverseApp({ horizon, startAge }: { horizon: number; startAge: number }) {
  const s = useStore((x) => x)
  const [sel, setSel] = useState<string | null>(null)
  const tipId = sel?.startsWith('tip:') ? sel.slice(4) : null
  const node = sel && !tipId ? s.nodes[sel] : null
  const ids = realities(s)
  const age = (m: number) => Math.floor(startAge + m / 12)
  return (
    <div>
      <h3>🌀 Multiverse</h3>
      <p className="tiny muted">Every square is a month where you chose something. Tap one to go back to just before it and choose differently. Tap the round end of a path to continue that reality. All realities live in world #{s.world}.</p>
      <div className="mv-wrap"><Tree s={s} horizon={horizon} startAge={startAge} selected={tipId ?? sel} onSelect={setSel} /></div>
      {(node || tipId) && (
        <div className="mv-pop px-box">
          {node && (node.turn
            ? <><span>Go back to <b>{monthLabel(node.turn.month)}</b> (age {age(node.turn.month)}), before “{node.label}”?</span>
              <button className="btn" onClick={() => goBack(node.id)}>↩ Go back</button></>
            : <><span>Go back to the very start (age {age(0)})?</span><button className="btn" onClick={() => goBack(node.id)}>↩ Start over</button></>)}
          {tipId && (
            <><span><b>{realityName(s, tipId)}</b>, lived until {monthLabel(tipId === s.cursor ? s.current : s.nodes[tipId].reached)}</span>
              {tipId !== s.cursor && <button className="btn" onClick={() => continueReality(tipId)}>▶ Continue this reality</button>}</>
          )}
          <button className="btn ghost" onClick={() => setSel(null)}>Cancel</button>
        </div>
      )}
      <h4>Your realities</h4>
      <ul className="reality-list">
        {ids.map((id) => (
          <li key={id}>
            <span className="dot" style={{ background: colorOf(s, id) }} />
            <b>{realityName(s, id)}</b>
            <span className="tiny muted">until {monthLabel(id === s.cursor ? s.current : s.nodes[id].reached)}{id === s.cursor ? ' · you are here' : ''}</span>
            <span className="spacer" />
            <button className={`pin ${s.pinA === id ? 'on' : ''}`} onClick={() => setStore({ pinA: s.pinA === id ? null : id })}>A</button>
            <button className={`pin ${s.pinB === id ? 'on' : ''}`} onClick={() => setStore({ pinB: s.pinB === id ? null : id })}>B</button>
          </li>
        ))}
      </ul>
      {ids.length < 2 && <p className="tiny muted">Make a second reality: go back to a square and choose differently. Then pin A and B to compare.</p>}
      {s.pinA && s.pinB && s.pinA !== s.pinB && s.nodes[s.pinA] && s.nodes[s.pinB] && <Compare s={s} a={s.pinA} b={s.pinB} />}
    </div>
  )
}
