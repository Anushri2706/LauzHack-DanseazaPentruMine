// The life timeline along the bottom of the screen, from today to age 120.
//
// Left of the playhead is the life you lived: your net worth and, below it, your life status every month
// (working / reduced hours / between jobs / studying / retired) in your world.
// Right of the playhead is what the simulated worlds fan out from where you are now: the likely range (8 in 10 lives) of net-worth
// band over many simulated lives and, per month, the share of lives in each status.
// Your choices sit on the timeline as markers. Click or drag anywhere to move through time; click a
// marker to go back to it or to undo it. Later choices stay ahead as plans until you choose differently.
import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import type { GameState, Regime, TimelineInsights } from '../api/types'
import { travelTo, undoChoice, unqueue } from '../store/actions'
import { actionIcon, actionLabel, colorOf, getStore, pathIds, requestFor, setStore, tipOf, useStore } from '../store/game'
import { chf, k, monthLabel, pct } from './format'
import { useDeflate, useProjection, useTwin } from './useGame'

export const REGIME_META: Record<Regime, { label: string; color: string }> = {
  employed: { label: 'Working', color: '#4fb38a' },
  part_time: { label: 'Reduced hours', color: '#a9d98f' },
  job_search: { label: 'Job search', color: '#e8a64a' },
  student: { label: 'Studying', color: '#5aa9d6' },
  retired: { label: 'Retired', color: '#9a7fd8' },
}
const ORDER: Regime[] = ['employed', 'part_time', 'job_search', 'student', 'retired']
const EVENT_KINDS = new Set(['retired', 'degree', 'home_bought', 'home_sold', 'income_loss', 'found_job', 'back_to_work', 'financing_needed'])

const W = 1000
const NW_TOP = 6, NW_BOT = 84, STRIP_TOP = 92, STRIP_BOT = 112, SVG_H = 116

/** Signed log scale so a 20k start and a 2M retirement both stay readable. */

const f = (v: number) => Math.sign(v) * Math.log10(1 + Math.abs(v) / 1000)

export function Timeline({ st }: { st: GameState }) {
  const s = useStore((x) => x)
  const proj = useProjection(s)
  const D = useDeflate(st)
  const busy = s.busy
  const H = st.horizon
  const m = st.month
  const startAge = st.age - m / 12
  const ageAt = (t: number) => startAge + t / 12
  const X = (t: number) => (t / H) * W
  const [hover, setHover] = useState<number | null>(null)
  const [drag, setDrag] = useState(false)
  const [open, setOpen] = useState<string | null>(null)
  // Drawer: only a tab shows until you point at it (or pin it with the tab, 📌 or T).
  const [inside, setInside] = useState(false)
  const [pinned, setPinned] = useState(false)
  const shownDrawer = inside || pinned || drag || open !== null || busy
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
      if (e.key === 't' || e.key === 'T') setPinned((x) => !x)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  const area = useRef<HTMLDivElement>(null)

  // AI insights pinned to moments on this timeline. They are part of the workflow: whenever your life changes (a
  // choice, another month) and settles for a moment, the AI reads the engines' numbers and explains the key moments.
  // The code picks the moments; results are remembered per situation, so going back to one costs nothing.
  const reqKey = JSON.stringify(requestFor(s, undefined, false))
  const [ai, setAi] = useState<{ key: string; data?: TimelineInsights; busy: boolean; error?: string } | null>(null)
  const seen = useRef(new Map<string, TimelineInsights>())
  const explain = async (force = false) => {
    const req = requestFor(getStore(), undefined, false)
    const key = JSON.stringify(req)
    const hit = !force && seen.current.get(key)
    if (hit) { setAi({ key, data: hit, busy: false }); return }
    setAi((x) => ({ key, data: x?.data, busy: true }))
    try {
      const data = await api.timeline(req)
      seen.current.set(key, data)
      setAi({ key, data, busy: false })
    } catch (e) { setAi((x) => ({ key, data: x?.data, busy: false, error: (e as Error).message })) }
  }
  useEffect(() => {
    if (busy) return
    const t = setTimeout(() => void explain(), 1200)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reqKey, busy])
  const aiData = ai?.data
  const aiStale = !!aiData && ai?.key !== reqKey

  const p = proj.data && proj.data.month === m && !proj.data.ended ? proj.data : null
  // The future right of 📍 comes from one of two engines:
  //   twin  = the learned twin: HMM life states x money behaviour learned from data (liquid savings)
  //   rules = the town engine's seeded worlds (net worth, with investment returns)
  const view = s.futureView ?? 'twin'
  const twinQ = useTwin(s, view === 'twin' && m < H)
  const tw = twinQ.data && twinQ.data.from_month === m ? twinQ.data : null
  const fut0 = view === 'twin'
    ? (tw ? { band: tw.wealth, step: tw.step, shares: tw.states, n: tw.n_paths } : null)
    : (p?.bands && p.regimes ? { band: p.bands.net_worth, step: 1, shares: p.regimes.share, n: p.n } : null)
  // In today's francs, each fan point (after m + (j+1)·step months) is divided by that month's price index.
  const fut = useMemo(() => {
    if (!fut0) return null
    const dv = (a: number[]) => a.map((v, j) => D(v, m + (j + 1) * fut0.step))
    return { ...fut0, band: { p10: dv(fut0.band.p10), p50: dv(fut0.band.p50), p90: dv(fut0.band.p90) } }
  }, [fut0?.band, s.real, m]) // eslint-disable-line react-hooks/exhaustive-deps
  const loading = view === 'twin' ? twinQ.loading : proj.loading
  const valueOf = (r: { net_worth: number; cash: number; investments: number }, i: number) =>
    D(view === 'twin' ? r.cash + r.investments : r.net_worth, i)
  const metricName = (view === 'twin' ? 'Liquid savings' : 'Net worth') + (s.real ? " (today's CHF)" : '')

  // --------------------------------------------------------------- net worth (lived + fan)
  const nw = useMemo(() => {
    const lived = st.rows.map((r, j) => valueOf(r, j + 1))
    const band = fut?.band
    const fstep = fut?.step ?? 1
    const nowV = valueOf(st.state, m)
    const vals = [...lived, ...(band ? [...band.p10, ...band.p90] : []), 0, nowV]
    let lo = Math.min(...vals.map(f)), hi = Math.max(...vals.map(f))
    if (hi - lo < 1) hi = lo + 1
    const Y = (v: number) => NW_BOT - ((f(v) - lo) / (hi - lo)) * (NW_BOT - NW_TOP)
    const step = Math.max(1, Math.ceil(H / 500))
    const line = (arr: number[], off: number, k = 1) => arr
      .map((v, i) => (i % Math.max(1, Math.round(step / k)) === 0 || i === arr.length - 1
        ? `${X(Math.min(H, off + (i + 1) * k)).toFixed(1)},${Y(v).toFixed(1)}` : ''))
      .filter(Boolean)
    const open0 = view === 'twin' ? st.persona.opening_cash + st.persona.opening_investments : st.persona.net_worth
    const livedPts = [`0,${Y(open0).toFixed(1)}`, ...line(lived, 0)]
    let bandPath = '', midPath = ''
    if (band) {
      const top = line(band.p90, m, fstep)
      const bot = line(band.p10, m, fstep).reverse()
      const now = `${X(m).toFixed(1)},${Y(nowV).toFixed(1)}`
      bandPath = `M${now} L${top.join(' L')} L${bot.join(' L')} Z`
      midPath = `M${now} L${line(band.p50, m, fstep).join(' L')}`
    }
    const zero = Y(0)
    const ticks = [-1e5, 0, 1e4, 1e5, 1e6, 1e7].filter((v) => f(v) >= lo && f(v) <= hi).map((v) => ({ v, y: Y(v) }))
      .filter((t, _i, arr) => t.v === 0 || arr.every((o) => o === t || o.v === 0 || Math.abs(o.y - t.y) > 14 || Math.abs(o.v) < Math.abs(t.v)))
    return { lived: lived.length ? `M${livedPts.join(' L')}` : '', bandPath, midPath, zero, ticks }
  }, [st.rows, st.state, fut, view, H, m, s.real]) // eslint-disable-line react-hooks/exhaustive-deps

  // --------------------------------------------------------------- regime strip
  const strip = useMemo(() => {
    const out: { x: number; w: number; y: number; h: number; c: string }[] = []
    let i = 0
    while (i < st.rows.length) {
      const r = st.rows[i].regime ?? 'employed'
      let j = i
      while (j < st.rows.length && st.rows[j].regime === r) j++
      out.push({ x: X(i), w: X(j) - X(i), y: STRIP_TOP, h: STRIP_BOT - STRIP_TOP, c: REGIME_META[r].color })
      i = j
    }
    const share = fut?.shares
    if (share) {
      const n = share.employed.length
      const k = fut!.step
      const B = Math.max(1, Math.ceil(H / 200 / k))
      for (let b = 0; b < n; b += B) {
        let y = STRIP_TOP
        for (const r of ORDER) {
          const arr = share[r]
          let sum = 0, c = 0
          for (let t = b; t < Math.min(n, b + B); t++) { sum += arr[t]; c++ }
          const h = (sum / c) * (STRIP_BOT - STRIP_TOP)
          if (h > 0.05) out.push({ x: X(m + b * k), w: X(m + Math.min(n, b + B) * k) - X(m + b * k) + 0.3, y, h, c: REGIME_META[r].color })
          y += h
        }
      }
    }
    return out
  }, [st.rows, fut, H, m]) // eslint-disable-line react-hooks/exhaustive-deps

  // --------------------------------------------------------------- choices on the viewed reality
  const tip = tipOf(s)
  const markers = pathIds(s, tip).map((id) => s.nodes[id]).filter((n) => n.turn)
  const events = st.letters.filter((l) => EVENT_KINDS.has(l.kind))

  // --------------------------------------------------------------- pointer → month
  const monthAt = (clientX: number) => {
    const r = area.current!.getBoundingClientRect()
    return Math.max(0, Math.min(H, Math.round(((clientX - r.left) / r.width) * H)))
  }
  const onDown = (e: React.PointerEvent) => {
    if (busy) return
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
    setDrag(true); setOpen(null); setHover(monthAt(e.clientX))
  }
  const onMove = (e: React.PointerEvent) => setHover(monthAt(e.clientX))
  const onUp = (e: React.PointerEvent) => {
    if (!drag) return
    setDrag(false)
    void travelTo(monthAt(e.clientX))
  }

  const tipText = (t: number) => {
    const head = `Age ${Math.floor(ageAt(t))} · ${monthLabel(t)}`
    if (t < m && st.rows[t]) {
      const r = st.rows[t]
      return { head, lines: [`Lived: ${REGIME_META[r.regime]?.label ?? r.regime}`, `${metricName} ${chf(valueOf(r, t + 1))}`] }
    }
    if (t === m) return { head, lines: ['You are here', `${metricName} ${chf(valueOf(st.state, m))}`] }
    if (!fut) return { head, lines: [loading ? 'Fanning out possible lives…' : 'Future'] }
    const b = fut.band
    const i = Math.max(0, Math.min(Math.floor((t - m - 1) / fut.step), b.p50.length - 1))
    const regs = ORDER.map((r) => [r, fut.shares[r][i]] as const).filter(([, v]) => v >= 0.01)
      .sort((a, c) => c[1] - a[1]).map(([r, v]) => `${REGIME_META[r].label} ${pct(v)}`)
    return { head, lines: [`Typical ${metricName.toLowerCase()}: ${k(b.p50[i])} · 8 in 10 simulated lives land between ${k(b.p10[i])} and ${k(b.p90[i])}`,
      `Of ${fut.n} simulated lives (${view === 'twin' ? 'learned twin' : 'town engine'}): ${regs.join(' · ')}`] }
  }

  const retireMonth = Math.round((65 - startAge) * 12)
  const jumps: [string, number, string][] = [
    ['⏮', 0, 'Back to the start'], ['−1y', m - 12, 'One year back'], ['−1m', m - 1, 'One month back'],
    ['+1m', m + 1, 'Live one month'], ['+1y', m + 12, 'Live one year'], ['+10y', m + 120, 'Live ten years'],
    ...(retireMonth > m ? [['🌅 65', retireMonth, 'Live until retirement age'] as [string, number, string]] : []),
    ['🏁 120', H, 'Live to the end of the timeline'],
  ]
  const decades = [Math.ceil(startAge), ...Array.from({ length: 13 }, (_, i) => i * 10).filter((a) => a > startAge + 3 && a <= startAge + H / 12)]
  const shown = hover ?? null
  const info = shown !== null ? tipText(shown) : null

  return (
    <div className={`tl-dock ${shownDrawer ? 'shown' : ''}`}
      onMouseEnter={() => setInside(true)} onMouseLeave={() => { setInside(false); if (!drag) setHover(null) }}>
    <button className="tl-handle px-box" onClick={() => setPinned((x) => !x)} aria-expanded={shownDrawer}
      title={pinned ? 'Unpin the timeline (T)' : 'Point here to open the timeline · click to pin it (T)'}>
      {shownDrawer ? '▼' : '▲'} Life timeline · 📍 {st.date.label} · age {st.age_years}{pinned ? ' · 📌' : ''}
    </button>
    <div className="timeline px-box" aria-label="Life timeline" aria-hidden={!shownDrawer}>
      <div className="tl-controls">
        <button className={`chip tiny ${pinned ? 'on' : ''}`} title={pinned ? 'Unpin: hide when the mouse leaves (T)' : 'Pin open (T)'}
          onClick={() => setPinned((x) => !x)}>📌</button>
        {jumps.map(([label, to, title]) => (
          <button key={label} className="chip tiny" disabled={busy || to === m || to < 0 || (to > H && m >= H)}
            title={title} onClick={() => void travelTo(to)}>{label}</button>
        ))}
        <span className="tl-now">
          {busy ? '⏳ Living…' : <>📍 <b>{st.date.label}</b> · age {st.age_years} · {REGIME_META[(st.rows[m - 1]?.regime ?? (st.state.status === 'studying' ? 'student' : st.state.status === 'retired' ? 'retired' : 'employed')) as Regime].label}</>}
        </span>
        <button className="chip tiny tl-explain" disabled={busy || ai?.busy} onClick={() => void explain(true)}
          title="The AI explains the key moments automatically; press to have it read the timeline again">
          {ai?.busy ? '✨ Reading…' : '✨ Re-read'}
        </button>
        {s.queue.length > 0 && <span className="tl-plans">🗓️ {s.queue.length} plan{s.queue.length > 1 ? 's' : ''} this month · they happen as you move forward</span>}
        <span className="tl-legend">
          {ORDER.map((r) => <span key={r}><i style={{ background: REGIME_META[r].color }} />{REGIME_META[r].label}</span>)}
          <span className="tl-view" role="group" aria-label="Which engine draws the future">
            <button className={`chip tiny ${view === 'twin' ? 'on' : ''}`} onClick={() => setStore({ futureView: 'twin' })}
              title="HMM life states (sourced Swiss hazards) × money behaviour learned from 958 people, backtested on a locked test set. Shows liquid savings.">🧠 Learned twin</button>
            <button className={`chip tiny ${view === 'rules' ? 'on' : ''}`} onClick={() => setStore({ futureView: 'rules' })}
              title="The town engine's seeded worlds: stated assumptions, investment returns. Shows net worth.">📐 Rules</button>
          </span>
          <button className={`chip tiny ${s.real ? 'on' : ''}`} onClick={() => setStore({ real: !s.real })}
            title={`Amounts in today's (Jan 2025) francs: each value is divided by the engine's price index (inflation ${pct(st.inflation_annual)}/year). Off = nominal francs of that year.`}>
            💱 {s.real ? "today's CHF" : 'nominal CHF'}</button>
          <span className="muted" title={fut && fut.n < 200 ? 'Few simulated lives keep the whole-life fan fast; the edges of its likely range are rough. The Money app uses more worlds.' : undefined}>
            {fut ? `${fut.n} lives${fut.n < 200 ? ' (rough band)' : ''} · ${metricName.toLowerCase()}` : loading ? 'simulating…' : ''}</span>
        </span>
      </div>
      <div className="tl-area" ref={area} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}
        onPointerLeave={() => !drag && setHover(null)}>
        <svg viewBox={`0 0 ${W} ${SVG_H}`} preserveAspectRatio="none" className="tl-svg">
          <rect x={X(m)} y={0} width={W - X(m)} height={SVG_H} fill="var(--fog, #f4eefb)" />
          {nw.ticks.map((t) => <line key={t.v} x1={0} x2={W} y1={t.y} y2={t.y} stroke="#e9dfcf" strokeDasharray={t.v === 0 ? undefined : '3 4'} vectorEffect="non-scaling-stroke" />)}
          {nw.bandPath && <path d={nw.bandPath} fill="#9a7fd8" fillOpacity={0.32} stroke="none" />}
          {nw.midPath && <path d={nw.midPath} fill="none" stroke="#9a7fd8" strokeWidth={1.5} strokeDasharray="5 4" vectorEffect="non-scaling-stroke" />}
          {nw.lived && <path d={nw.lived} fill="none" stroke={colorOf(s, s.cursor)} strokeWidth={2.5} vectorEffect="non-scaling-stroke" />}
          {strip.map((r, i) => <rect key={i} x={r.x} y={r.y} width={Math.max(r.w, 0.4)} height={r.h} fill={r.c} />)}
          <line x1={X(m)} x2={X(m)} y1={0} y2={SVG_H} stroke="var(--ink, #3b2f4a)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
          {shown !== null && shown !== m && <line x1={X(shown)} x2={X(shown)} y1={0} y2={SVG_H} stroke="#3b2f4a" strokeOpacity={0.5} strokeDasharray="3 3" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />}
        </svg>
        {nw.ticks.filter((t) => t.v !== 0).map((t) => (
          <span key={t.v} className="tl-ytick" style={{ top: `${(t.y / SVG_H) * 100}%` }}>{k(t.v)}</span>
        ))}
        {events.map((l, i) => (
          <span key={i} className="tl-event" style={{ left: `${(l.month / H) * 100}%`, top: `calc(${(STRIP_TOP / SVG_H) * 100}% - 15px)` }} title={`${l.title} (age ${Math.floor(ageAt(l.month))}): ${l.text}`}
            onPointerDown={(e) => e.stopPropagation()}>{l.icon}</span>
        ))}
        {aiData?.insights.map((x, _i, all) => {
          const same = all.filter((y) => y.month === x.month)  // moments at the same age sit side by side
          const off = (same.indexOf(x) - (same.length - 1) / 2) * 20
          return (
          <button key={x.id} className={`tl-insight ${aiStale ? 'stale' : ''} ${open === 'ins:' + x.id ? 'open' : ''}`}
            style={{ left: `calc(${(x.month / H) * 100}% + ${off}px)` }} onPointerDown={(e) => e.stopPropagation()}
            onClick={() => setOpen(open === 'ins:' + x.id ? null : 'ins:' + x.id)} title={`${x.title}: ${x.text}`}>✨</button>
          )
        })}
        {markers.map((n) => {
          const t = n.turn!
          const future = t.month >= m
          return (
            <button key={n.id} className={`tl-marker ${future ? 'future' : ''} ${open === n.id ? 'open' : ''}`}
              style={{ left: `${(t.month / H) * 100}%` }} onPointerDown={(e) => e.stopPropagation()}
              onClick={() => setOpen(open === n.id ? null : n.id)}
              title={`${t.actions.map(actionLabel).join(', ')} at age ${Math.floor(ageAt(t.month))}${future ? ' (a plan ahead of you)' : ''}`}>
              {actionIcon(t.actions[0])}{t.actions.length > 1 && <small>+{t.actions.length - 1}</small>}
            </button>
          )
        })}
        {s.queue.length > 0 && (
          <button className={`tl-marker queued ${open === 'queue' ? 'open' : ''}`} style={{ left: `${(m / H) * 100}%` }}
            onPointerDown={(e) => e.stopPropagation()} onClick={() => setOpen(open === 'queue' ? null : 'queue')}
            title="This month's plans">{actionIcon(s.queue[0])}{s.queue.length > 1 && <small>+{s.queue.length - 1}</small>}</button>
        )}
        {info && shown !== null && (
          <div className="tl-tip" style={{ left: `${Math.min(82, Math.max(0, (shown / H) * 100 - 9))}%` }}>
            <b>{info.head}</b>{info.lines.map((x, i) => <div key={i}>{x}</div>)}
            {drag && <div className="muted">Release to {shown > m ? 'live until here' : shown < m ? 'go back here' : 'stay'}</div>}
          </div>
        )}
        {open?.startsWith('ins:') && aiData && (() => {
          const x = aiData.insights.find((i) => i.id === open.slice(4))
          return x ? (
            <div className="tl-pop px-box" style={{ left: `${Math.min(62, Math.max(0, (x.month / H) * 100 - 8))}%`, width: 300 }}
              onPointerDown={(e) => e.stopPropagation()}>
              <header><b>✨ Age {x.age} · {x.title}</b><button className="icon-btn" onClick={() => setOpen(null)}>✕</button></header>
              <p className="small" style={{ margin: '4px 0' }}>{x.text}</p>
              <p className="tiny muted">{x.by === 'llm' ? 'Written by the AI companion' : 'Told from the engine’s numbers'} · {aiData.disclaimer}</p>
            </div>
          ) : null
        })()}
        {open && !open.startsWith('ins:') && (
          <ChoicePopover id={open} st={st} startAge={startAge} H={H} onClose={() => setOpen(null)} />
        )}
      </div>
      <div className="tl-axis">
        {decades.map((a) => <span key={a} style={{ left: `${(((a - startAge) * 12) / H) * 100}%` }}>{a}</span>)}
      </div>
      {ai?.error && <p className="tiny error">{ai.error}</p>}
      {aiData && (
        <div className={`tl-ai ${aiStale ? 'stale' : ''}`}>
          <p className="tl-ai-head"><b>✨ {aiData.headline}</b>{aiStale && <span className="muted"> · from an earlier month: refresh to update</span>}</p>
          <div className="tl-ai-cards">
            {aiData.insights.map((x) => (
              <button key={x.id} className={`tl-ai-card ${open === 'ins:' + x.id ? 'open' : ''}`} onClick={() => setOpen(open === 'ins:' + x.id ? null : 'ins:' + x.id)}>
                <span className="tl-ai-age">Age {x.age}</span><b>{x.title}</b><span className="small">{x.text}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
    </div>
  )
}

function ChoicePopover({ id, st, startAge, H, onClose }: { id: string; st: GameState; startAge: number; H: number; onClose: () => void }) {
  const s = useStore((x) => x)
  const busy = s.busy
  if (id === 'queue') {
    return (
      <div className="tl-pop px-box" style={{ left: `${Math.min(70, (st.month / H) * 100)}%` }} onPointerDown={(e) => e.stopPropagation()}>
        <header><b>Plans for {st.date.label}</b><button className="icon-btn" onClick={onClose}>✕</button></header>
        <ul>{s.queue.map((a, i) => (
          <li key={i}><span>{actionIcon(a)} {actionLabel(a)}</span><button className="chip tiny" onClick={() => unqueue(i)}>✕ Remove</button></li>
        ))}</ul>
        <p className="tiny muted">They happen when you move forward on the timeline.</p>
      </div>
    )
  }
  const n = s.nodes[id]
  if (!n?.turn) return null
  const t = n.turn
  const lived = t.month < st.month
  const age = Math.floor(startAge + t.month / 12)
  return (
    <div className="tl-pop px-box" style={{ left: `${Math.min(70, (t.month / H) * 100)}%` }} onPointerDown={(e) => e.stopPropagation()}>
      <header><b>{monthLabel(t.month)} · age {age}</b><button className="icon-btn" onClick={onClose}>✕</button></header>
      <ul>{t.actions.map((a, i) => (
        <li key={i}><span>{actionIcon(a)} {actionLabel(a)}</span>
          <button className="chip tiny" disabled={busy} title="A new reality without this choice; later choices are replayed"
            onClick={() => { onClose(); void undoChoice(id, i) }}>↩ Undo</button></li>
      ))}</ul>
      <div className="row-end">
        <button className="btn" disabled={busy} onClick={() => { onClose(); void travelTo(t.month) }}>
          {lived ? '⏪ Go back to this moment' : '⏩ Live until this moment'}
        </button>
      </div>
      <p className="tiny muted">{lived
        ? 'Going back keeps this and later choices ahead of you as plans. Choose something else there to branch into a new reality.'
        : 'A plan ahead of you in this reality: moving past it lives it again.'}</p>
    </div>
  )
}
