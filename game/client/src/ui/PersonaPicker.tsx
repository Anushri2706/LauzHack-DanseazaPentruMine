import { useState } from 'react'
import { api } from '../api/client'
import type { PersonaCard } from '../api/types'
import { useApi } from '../api/useApi'
import { chf, STATUS_LABEL } from './format'

const FILTERS = ['', 'student', 'employed', 'part_time', 'self_employed', 'unemployed', 'retired']

function Card({ p, onPick, featured }: { p: PersonaCard; onPick: (p: PersonaCard) => void; featured?: boolean }) {
  return (
    <button className={`persona-card px-box ${featured ? 'featured' : ''}`} onClick={() => onPick(p)}>
      {featured && <span className="ribbon">Featured</span>}
      <div className="persona-name">{p.name}, {p.age}</div>
      <div className="muted small">{STATUS_LABEL[p.employment_status]} · {p.city} ({p.canton}) · {p.household_type.replace('_', ' ')}</div>
      <div className="persona-stats">
        <span title="from the synthetic profile">🟢 Cash {chf(p.opening_cash)}</span>
        <span title="from the synthetic profile">🟢 Invested {chf(p.opening_investments)}</span>
        {p.opening_debt > 0 && <span title="from the synthetic profile">🟢 Debt {chf(p.opening_debt)}</span>}
        <span title="from the synthetic profile">🟢 Income {chf(p.reference_income)}/mo</span>
        <span title="from the synthetic profile">🟢 Rent {chf(p.reference_rent)}/mo</span>
      </div>
      {featured && <p className="question">“What if I start working, do a master's, or move abroad?”</p>}
    </button>
  )
}

export function PersonaPicker({ onPick }: { onPick: (p: PersonaCard) => void }) {
  const [status, setStatus] = useState('')
  const [seed, setSeed] = useState(0)
  const alexia = useApi('alexia', () => api.persona('P00001'))
  const list = useApi(`list-${status}-${seed}`, () => api.personas({ status, random: true, seed, limit: 9 }))
  const randomOne = async () => {
    const [p] = await api.personas({ random: true, seed: Math.floor(Math.random() * 1e6), limit: 1 })
    onPick(p)
  }
  return (
    <div className="screen pick-screen">
      <header className="title-block">
        <h1 className="logo">Many Lives</h1>
        <p className="lead">A lakeside town where you live the next ten years of someone's life, one month at a time. Go back in time, choose differently, and compare the realities you create.</p>
        <p className="badge-line"><span className="badge">Simulation · not a prediction or advice</span> All 1,200 people are fictional (synthetic data).</p>
      </header>
      {alexia.data && <Card p={alexia.data} onPick={onPick} featured />}
      <div className="filter-row">
        {FILTERS.map((f) => (
          <button key={f || 'all'} className={`chip ${status === f ? 'on' : ''}`} onClick={() => setStatus(f)}>
            {f ? STATUS_LABEL[f] : 'Everyone'}
          </button>
        ))}
        <button className="chip" onClick={() => setSeed(seed + 1)}>🔀 Shuffle</button>
        <button className="chip accent" onClick={randomOne}>🎲 Random person</button>
      </div>
      {list.error && <p className="error">Could not reach the engine: {list.error}. Is the backend running on :8000?</p>}
      <div className="persona-grid">
        {list.data?.filter((p) => p.profile_id !== 'P00001').map((p) => <Card key={p.profile_id} p={p} onPick={onPick} />)}
      </div>
    </div>
  )
}
