import { useState } from 'react'
import { api, EngineError } from '../api/client'
import type { PersonaCard } from '../api/types'
import { STATUS_LABEL } from './format'

/** The same facts every synthetic person starts with (synthetic-profiles / -assets / -liabilities / -goals). */
export interface PersonaSpec {
  name: string
  age: number
  city: string
  canton: string
  employment_status: string
  household_type: string
  cash: number
  investments: number
  debt: number
  debt_rate: number
  income: number
  rent: number
  goals: { name: string; target: number; target_date: string; priority: string }[]
}

// The cities and cantons of the data (each canton has a home price level in the engine).
const CITIES: Record<string, string> = {
  Basel: 'BS', Bern: 'BE', Biel: 'BE', Fribourg: 'FR', Geneva: 'GE', Lausanne: 'VD', Lucerne: 'LU',
  Lugano: 'TI', Neuchatel: 'NE', 'St Gallen': 'SG', Winterthur: 'ZH', Zurich: 'ZH',
}
const CANTONS = ['BE', 'BS', 'FR', 'GE', 'LU', 'NE', 'SG', 'TI', 'VD', 'ZH']
const STATUSES = ['student', 'employed', 'part_time', 'self_employed', 'unemployed', 'retired']
const HOUSEHOLDS: Record<string, string> = { single: 'Single', shared_housing: 'Shared flat', couple: 'Couple', family: 'Family' }

/** Persona id that carries the spec: "custom-" + base64url(JSON), decoded and validated by the backend. */
export function customId(spec: PersonaSpec): string {
  const bytes = new TextEncoder().encode(JSON.stringify(spec))
  let bin = ''
  bytes.forEach((b) => { bin += String.fromCharCode(b) })
  return 'custom-' + btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

const START: PersonaSpec = {
  name: '', age: 25, city: 'Lausanne', canton: 'VD', employment_status: 'employed', household_type: 'single',
  cash: 10000, investments: 0, debt: 0, debt_rate: 0, income: 5000, rent: 1300, goals: [],
}

const incomeLabel = (status: string) =>
  status === 'student' ? 'Student income (side job, support)' : status === 'retired' ? 'Pension'
    : status === 'unemployed' ? 'Income now (benefits)' : 'Net salary'

export function PersonaBuilder({ onPick, onCancel }: { onPick: (p: PersonaCard) => void; onCancel: () => void }) {
  const [spec, setSpec] = useState<PersonaSpec>(START)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const set = <K extends keyof PersonaSpec>(k: K, v: PersonaSpec[K]) => setSpec((s) => ({ ...s, [k]: v }))
  const num = (k: keyof PersonaSpec) => ({
    type: 'number', value: spec[k] as number,
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => set(k, Number(e.target.value) as never),
  })
  const setGoal = (i: number, patch: Partial<PersonaSpec['goals'][number]>) =>
    set('goals', spec.goals.map((g, j) => (j === i ? { ...g, ...patch } : g)))

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setBusy(true)
    try {
      const clean = { ...spec, name: spec.name.trim(), city: spec.city.trim() }
      onPick(await api.persona(customId(clean)))   // the backend validates every field
    } catch (err) {
      setError(err instanceof EngineError ? err.message : `Could not reach the engine: ${(err as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className="persona-builder px-box" onSubmit={submit}>
      <h2>✏️ Build your own person</h2>
      <p className="tiny muted">The same facts each of the 1,200 synthetic people starts with, on 1 January 2025. They are
        tagged 🟢 <i>entered by you</i> in Settings; everything else (living costs, salary after a master's, …) starts
        from the same labelled assumptions as for other people, and can be edited there. Nothing leaves this game:
        the person exists only in this browser.</p>

      <fieldset>
        <legend>Who</legend>
        <label>Name <input required maxLength={40} value={spec.name} onChange={(e) => set('name', e.target.value)} placeholder="Your name" /></label>
        <label>Age <input {...num('age')} min={16} max={90} step={1} required /></label>
        <label>City <input list="pb-cities" required maxLength={40} value={spec.city}
          onChange={(e) => { const c = e.target.value; setSpec((s) => ({ ...s, city: c, canton: CITIES[c] ?? s.canton })) }} /></label>
        <datalist id="pb-cities">{Object.keys(CITIES).map((c) => <option key={c} value={c} />)}</datalist>
        <label>Canton <select value={spec.canton} onChange={(e) => set('canton', e.target.value)}>
          {CANTONS.map((c) => <option key={c}>{c}</option>)}</select></label>
        <label>Situation <select value={spec.employment_status} onChange={(e) => set('employment_status', e.target.value)}>
          {STATUSES.map((s) => <option key={s} value={s}>{STATUS_LABEL[s]}</option>)}</select></label>
        <label>Household <select value={spec.household_type} onChange={(e) => set('household_type', e.target.value)}>
          {Object.entries(HOUSEHOLDS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select></label>
      </fieldset>

      <fieldset>
        <legend>Money today (CHF)</legend>
        <label>Cash <input {...num('cash')} step={100} required /></label>
        <label>Investments <input {...num('investments')} min={0} step={100} required /></label>
        <label>Debt <input {...num('debt')} min={0} step={100} required /></label>
        <label>Debt interest (%/year) <input type="number" min={0} max={30} step={0.1} value={+(spec.debt_rate * 100).toFixed(2)}
          onChange={(e) => set('debt_rate', Number(e.target.value) / 100)} disabled={spec.debt <= 0} /></label>
        <label>{incomeLabel(spec.employment_status)} per month <input {...num('income')} min={0} step={50} required /></label>
        <label>Rent per month{spec.household_type === 'couple' || spec.household_type === 'family' ? ' (whole household)' : ''}
          <input {...num('rent')} min={0} step={50} required /></label>
      </fieldset>

      <fieldset>
        <legend>Goals (optional, up to 5)</legend>
        {spec.goals.map((g, i) => (
          <div key={i} className="pb-goal">
            <input required maxLength={40} value={g.name} onChange={(e) => setGoal(i, { name: e.target.value })} placeholder="Goal" />
            <label>CHF <input type="number" min={0} step={500} value={g.target} onChange={(e) => setGoal(i, { target: Number(e.target.value) })} /></label>
            <label>by <input type="month" min="2025-02" required value={g.target_date.slice(0, 7)}
              onChange={(e) => setGoal(i, { target_date: `${e.target.value}-28` })} /></label>
            <select value={g.priority} onChange={(e) => setGoal(i, { priority: e.target.value })}>
              <option value="high">high</option><option value="medium">medium</option><option value="low">low</option></select>
            <button type="button" className="icon-btn" title="Remove" onClick={() => set('goals', spec.goals.filter((_, j) => j !== i))}>🗑️</button>
          </div>
        ))}
        {spec.goals.length < 5 && <button type="button" className="chip" onClick={() => set('goals', [...spec.goals,
          { name: 'Emergency savings', target: 10000, target_date: '2029-12-28', priority: 'high' }])}>＋ Add a goal</button>}
      </fieldset>

      {error && <p className="error">{error}</p>}
      <div className="pb-actions">
        <button type="button" className="btn ghost" onClick={onCancel}>Back</button>
        <button type="submit" className="btn big" disabled={busy}>{busy ? 'Checking…' : 'Live this life →'}</button>
      </div>
    </form>
  )
}
