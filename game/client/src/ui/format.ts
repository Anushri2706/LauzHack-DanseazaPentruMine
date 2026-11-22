import type { Education } from '../api/types'
export const chf = (x: number) => `CHF ${Math.round(x).toLocaleString('de-CH').replace(/’/g, "'")}`.replace('CHF -', 'CHF −')
export const k = (x: number) => {
  const a = Math.abs(x)
  const s = a >= 1_000_000 ? `${(a / 1_000_000).toFixed(1)}M` : a >= 1000 ? `${Math.round(a / 1000)}k` : `${Math.round(a)}`
  return (x < 0 ? '−' : '') + s
}
export const pct = (x: number) => `${Math.round(x * 100)}%`
export const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
export const monthLabel = (m: number) => `${MONTHS[m % 12]} ${2025 + Math.floor(m / 12)}`

export const TAGS: Record<string, { dot: string; label: string }> = {
  profile: { dot: '🟢', label: 'from the synthetic profile' },
  data: { dot: '🟢', label: 'from the synthetic profile data' },
  user: { dot: '🟢', label: 'entered by you' },
  starter_kit: { dot: '🟡', label: 'assumption (starter kit)' },
  scenario_csv: { dot: '🟡', label: 'assumption (scenario CSV)' },
  ours: { dot: '🟡', label: 'our assumption' },
  official_est: { dot: '🟢', label: 'estimated from official Swiss statistics (BFS, SECO, Eurostat)' },
  public_data: { dot: '🟢', label: 'computed from public data (World Bank, SNB, BIS)' },
  assumption: { dot: '🟡', label: 'our assumption' },
  simulated: { dot: '🔵', label: 'simulated by the engine' },
  peer: { dot: '⚪', label: 'peer reference' },
}

export const STATUS_LABEL: Record<string, string> = {
  studying: 'Studying', working: 'Working', abroad: 'Abroad', unemployed: 'Between jobs', retired: 'Retired',
  student: 'Student', employed: 'Employed', part_time: 'Part-time', self_employed: 'Self-employed',
}

/** Highest education reached. No PhD in this engine. */
export const EDU_LABEL: Record<Education, string> = {
  none: 'No degree', bachelor_studying: "Bachelor's (studying)", bachelor: "Bachelor's",
  master_studying: "Master's (studying)", master: "Master's",
}
export const EDU_SHORT: Record<Education, string> = {
  none: 'No degree', bachelor_studying: 'BSc 📚', bachelor: 'BSc', master_studying: 'MSc 📚', master: 'MSc 🎓',
}
export const EDU_RANK: Record<Education, number> = { none: 0, bachelor_studying: 1, bachelor: 2, master_studying: 3, master: 4 }
