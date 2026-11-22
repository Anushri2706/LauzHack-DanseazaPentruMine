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
  starter_kit: { dot: '🟡', label: 'assumption (starter kit)' },
  scenario_csv: { dot: '🟡', label: 'assumption (scenario CSV)' },
  ours: { dot: '🟡', label: 'our assumption' },
  official_est: { dot: '🟢', label: 'estimated from official Swiss statistics (BFS, SECO, Eurostat)' },
  assumption: { dot: '🟡', label: 'our assumption' },
  simulated: { dot: '🔵', label: 'simulated by the engine' },
  peer: { dot: '⚪', label: 'peer reference' },
}

export const STATUS_LABEL: Record<string, string> = {
  studying: 'Studying', working: 'Working', abroad: 'Abroad', unemployed: 'Between jobs', retired: 'Retired',
  student: 'Student', employed: 'Employed', part_time: 'Part-time', self_employed: 'Self-employed',
}
