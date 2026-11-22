import { useState } from 'react'
import type { GameState } from '../api/types'
import { setStore, useStore } from '../store/game'
import { chf, EDU_LABEL, EDU_SHORT, k, STATUS_LABEL } from './format'
import { isMuted, setMuted, sfx } from './sound'
import { Avatar } from './Sprite'

export const SCORE_META = {
  money: { icon: '💰', label: 'Money' },
  safety: { icon: '🛡️', label: 'Safety' },
  goals: { icon: '🎯', label: 'Goals' },
  freedom: { icon: '⛱️', label: 'Freedom' },
  experience: { icon: '🎓', label: 'Education' },
  joy: { icon: '😊', label: 'Joy' },
} as const

export function scoreLines(st: GameState) {
  const s = st.scores
  const safety = { steady: 'Steady', tight: 'Tight', shortfall: 'Shortfall' }[s.safety.level]
  return {
    money: { short: k(s.money.net_worth), tip: `🔵 Net worth now ${chf(s.money.net_worth)} (cash ${chf(s.money.cash)}, invested ${chf(s.money.investments)}${s.money.home_equity ? `, home equity ${chf(s.money.home_equity)}` : ''}${s.money.debt ? `, owed ${chf(s.money.debt)}` : ''}).`, tone: s.money.net_worth < 0 ? 'bad' : '' },
    safety: { short: safety, tip: `🔵 Now: ${{ steady: 'cash above your buffer', tight: 'cash below your buffer', shortfall: 'cash below zero, financing needed' }[s.safety.level]}.${s.safety.months_negative_cash ? ` Earlier: ${s.safety.months_negative_cash} months below zero${s.safety.recovered ? ', recovered' : ''}.` : ''} ${s.safety.months_below_buffer} months below buffer so far.`, tone: { steady: 'good', tight: 'warn', shortfall: 'bad' }[s.safety.level] },
    goals: { short: s.goals.due ? `${s.goals.met}/${s.goals.due}` : `0/${s.goals.total}`, tip: `🔵 ${s.goals.met} of ${s.goals.due} goals due so far were met (${s.goals.total} in total).`, tone: '' },
    freedom: { short: s.freedom.runway_months == null ? '—' : `${Math.max(0, Math.round(s.freedom.runway_months))} mo`, tip: `🔵 Savings cover ${s.freedom.runway_months == null ? '—' : Math.max(0, Math.round(s.freedom.runway_months))} months of costs${s.freedom.debt_free ? ' · debt-free' : ''}.`, tone: '' },
    experience: { short: EDU_SHORT[s.experience.education], tip: ['🔵 ' + EDU_LABEL[s.experience.education] + '.',
      s.experience.months_studying > 0 && `${s.experience.months_studying} months ${s.experience.studying ? 'studying so far' : 'studied'}.`].filter(Boolean).join(' '), tone: '' },
    joy: { short: `${Math.round(s.joy.total)}`, tip: '🟡 Joy points: a game preference score from your habits, trips and life situation. Not money, not a measure of a good life.', tone: '' },
  }
}

export function HudBar({ st }: { st: GameState | undefined }) {
  const look = useStore((s) => s.look)
  const queue = useStore((s) => s.queue)
  const [muted, setM] = useState(isMuted())
  if (!st) return <div className="hud-bar px-box"><span className="muted">Loading the town…</span></div>
  const lines = scoreLines(st)
  const cash = st.state.cash
  return (
    <div className="hud-bar px-box" role="status">
      <div className="hud-who">
        <Avatar look={look} scale={2} />
        <div>
          <b>{st.persona.name}</b>
          <div className="tiny">{STATUS_LABEL[st.state.status]}{st.state.program ? ` · master's until ${st.state.program_end ?? ''}` : ''}</div>
        </div>
      </div>
      <div className="hud-date" title="Move through time on the timeline at the bottom">📅 {st.date.label} · age {st.age_years}</div>
      <div className={`hud-cash ${cash < 0 ? 'neg' : ''}`} title={cash < 0 ? 'Financing needed: cash below zero' : '🔵 Cash in your account'}>
        💰 {chf(cash)}{cash < 0 && <span className="fin">⚠ financing</span>}
      </div>
      <div className="hud-scores">
        {(Object.keys(SCORE_META) as (keyof typeof SCORE_META)[]).map((d) => (
          <span key={d} className={`hud-score ${lines[d].tone}`} data-tip={`${SCORE_META[d].label}: ${lines[d].tip}`}>
            {SCORE_META[d].icon}<small>{lines[d].short}</small>
          </span>
        ))}
      </div>
      {queue.length > 0 && <span className="hud-queue" title="Plans for this month: they happen when you sleep">🗓️ {queue.length} planned</span>}
      <span className="hud-world" title="Your seeded world: market drops, job losses and rent rises come from it. Every reality uses the same world.">🌍 World #{st.world}</span>
      <span className="badge" title="All people are fictional. Outputs are conditional what-ifs under stated assumptions.">Simulation · not a prediction or advice</span>
      <button className="icon-btn" title={muted ? 'Sound off' : 'Sound on'} onClick={() => { setMuted(!muted); setM(!muted) }}>{muted ? '🔇' : '🔊'}</button>
      <button className="btn phone-btn" onClick={() => { sfx('phoneUp'); setStore({ phone: 'money' }) }} title="Open your phone (P)">📱 Phone</button>
    </div>
  )
}

export function Toasts() {
  const toasts = useStore((s) => s.toasts)
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className={`toast px-box ${t.tone ?? 'info'}`}>
          <span className="toast-icon">{t.icon}</span>
          <div><b>{t.title}</b>{t.text && <div className="small">{t.text}</div>}</div>
        </div>
      ))}
    </div>
  )
}
