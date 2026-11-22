import { useState } from 'react'
import { api } from '../../api/client'
import { useApi } from '../../api/useApi'
import type { GameState, SensitivityBar } from '../../api/types'
import { requestFor, setStore, useStore } from '../../store/game'
import { chf, k } from '../format'

/** "Which assumptions matter most?" A tornado chart from the engine: each assumption moved ±swing on its
 *  own, the reality's choices replayed with no luck. Sensitivity, not probability. */
const fmt = (v: number, unit: string) =>
  unit.includes('%') ? `${(v * 100).toFixed(unit.includes('year') ? 2 : 1).replace(/\.?0+$/, '')}%` : chf(v)

function Bar({ b, base, lo, hi, real }: { b: SensitivityBar; base: number; lo: number; hi: number; real: boolean }) {
  const a = real ? b.real_low : b.net_worth_low
  const c = real ? b.real_high : b.net_worth_high
  const x = (v: number) => ((v - lo) / (hi - lo)) * 100
  const seg = (from: number, to: number, up: boolean) => (
    <div className={`tor-seg ${up ? 'up' : 'down'}`}
      style={{ left: `${x(Math.min(from, to))}%`, width: `${Math.max(0.6, Math.abs(x(to) - x(from)))}%` }} />
  )
  return (
    <li className="tor-row" title={`${b.label}: ${fmt(b.low, b.unit)} → ${k(a)} · ${fmt(b.high, b.unit)} → ${k(c)}`}>
      <div className="tor-label"><b>{b.label}</b>
        <span className="tiny muted"> {fmt(b.low, b.unit)} / <u>{fmt(b.value, b.unit)}</u> / {fmt(b.high, b.unit)}</span></div>
      <div className="tor-track">
        {seg(base, a, a > base)}
        {seg(base, c, c > base)}
        <div className="tor-base" style={{ left: `${x(base)}%` }} />
        <span className="tor-v" style={{ left: `${x(Math.min(a, c))}%` }}>{k(Math.min(a, c))}</span>
        <span className="tor-v r" style={{ left: `${x(Math.max(a, c))}%` }}>{k(Math.max(a, c))}</span>
      </div>
      {(b.negative_months_low > 0 || b.negative_months_high > 0) && (
        <div className="tiny muted">months with cash below zero: {b.negative_months_low} (low) · {b.negative_months_high} (high)</div>
      )}
    </li>
  )
}

export function MattersApp({ st }: { st: GameState }) {
  const s = useStore((x) => x)
  const req = requestFor(s, undefined, false)
  const startAge = st.persona.age
  const ages = [30, 40, 65].filter((a) => a > startAge + 0.5)
  const [age, setAge] = useState<number | undefined>(undefined)
  const [swing, setSwing] = useState(0.2)
  const q = useApi(JSON.stringify(req) + age + swing, () => api.sensitivity(req, age, swing))
  const d = q.data
  const real = s.real
  const base = d ? (real ? d.base_real : d.base) : 0
  const top = d?.bars.slice(0, 10) ?? []
  const vals = top.flatMap((b) => (real ? [b.real_low, b.real_high] : [b.net_worth_low, b.net_worth_high]))
  let lo = Math.min(base, ...vals), hi = Math.max(base, ...vals)
  const pad = (hi - lo) * 0.12 || 1
  lo -= pad; hi += pad
  return (
    <div>
      <h3>🌪️ What matters most?</h3>
      <p className="tiny muted">For <b>this reality's choices</b>: each assumption moved ±{Math.round(swing * 100)}% on its own, everything
        else fixed, no luck (no market swings, job losses or rent rises). The longest bar is the guess your result leans on most.
        This is <b>sensitivity, not probability</b>.</p>
      <div className="chart-tools">
        <span className="tiny">Net worth at</span>
        {ages.map((a) => (
          <button key={a} className={`chip ${(age ?? d?.target_age) === a ? 'on' : ''}`} onClick={() => setAge(a)}>{a}</button>
        ))}
        <span className="tiny">· swing</span>
        {[0.1, 0.2, 0.3].map((w) => (
          <button key={w} className={`chip ${swing === w ? 'on' : ''}`} onClick={() => setSwing(w)}>±{w * 100}%</button>
        ))}
        <button className={`chip ${real ? 'on' : ''}`} onClick={() => setStore({ real: !real })}>💱 {real ? "today's CHF" : 'nominal CHF'}</button>
        {q.loading && <span className="tiny muted">recomputing…</span>}
      </div>
      {q.error && <p className="note small">{q.error}</p>}
      {d && (
        <>
          <p className="small">With every assumption at its default: <b>{chf(base)}</b> at {Math.round(d.target_age)}
            {d.base_negative_months > 0 && <> · cash below zero for {d.base_negative_months} months (financed at the overdraft rate)</>}.
            <span className="tiny muted"> 🟥 lower · 🟩 higher than that.</span></p>
          <ol className="tornado">
            {top.map((b) => <Bar key={b.key} b={b} base={base} lo={lo} hi={hi} real={real} />)}
          </ol>
          {top[0] && (
            <p className="note small">💡 Your result leans most on <b>{top[0].label.toLowerCase()}</b>
              {top[1] && <> and <b>{top[1].label.toLowerCase()}</b></>}. Change them in ⚙️ Settings to see the whole life recompute.</p>
          )}
          {d.no_effect.length > 0 && (
            <p className="tiny muted">✅ No effect on this reality (scenario isolation): {d.no_effect.join(', ')}.</p>
          )}
          <p className="tiny muted">{d.note} Engine: <code>POST /game/sensitivity</code>.</p>
        </>
      )}
    </div>
  )
}
