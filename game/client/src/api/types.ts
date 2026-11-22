// Mirrors game/backend/app/models.py + service.py. The client does no financial maths.
export type Status = 'studying' | 'working' | 'abroad' | 'unemployed' | 'retired'
export type Regime = 'employed' | 'part_time' | 'job_search' | 'student' | 'retired'
export type Place = 'home' | 'university' | 'office' | 'cafe' | 'shop' | 'bank' | 'realestate' | 'station'

export interface Goal {
  goal_id: string
  name: string
  target: number
  month: number
  priority: string
  target_date?: string
  custom?: boolean
}
export interface CheckedGoal extends Goal {
  status: 'met' | 'not_met' | 'beyond_horizon'
  available: number | null
}

export interface PersonaCard {
  profile_id: string
  name: string
  age: number
  city: string
  canton: string
  employment_status: string
  status: Status
  household_type: string
  opening_cash: number
  opening_investments: number
  opening_debt: number
  reference_income: number
  reference_rent: number
  net_worth: number
  goals: Goal[]
}

export interface ActionIn { type: string; id?: string | null; value?: number | null }
export interface TurnIn { month: number; actions: ActionIn[] }

export interface GameRequest {
  persona_id: string
  look: Record<string, number>
  assumptions: Record<string, number>
  seed: number
  world: number
  turns: TurnIn[]
  current_month: number
  goals: Goal[] | null
}

export interface Preview {
  text: string
  one_off: number | null
  monthly: number | null
  monthly_delta: number | null
  joy: number | null
  warning: string | null
}
export interface GameAction {
  place: Place
  type: string
  id: string | null
  value: number | null
  label: string
  enabled: boolean
  reason: string | null
  current: boolean
  preview: Preview
  tags: string[]
}
export interface Letter { month: number; kind: string; icon: string; title: string; text: string }

export interface GameStateDict {
  cash: number
  investments: number
  debt: number
  home_value: number
  mortgage: number
  home_id: string | null
  net_worth: number
  status: Status
  program: string | null
  program_end: number | null
  has_master: boolean
  no_degree: boolean
  enrolled: boolean
  salary: number
  work_share: number
  savings_rate: number
  rent_base: number
  flat_factor: number
  habits: Record<string, number>
  abroad: boolean
  last_switch: number | null
}

export interface ScoresNow {
  money: { net_worth: number; cash: number; investments: number; home_equity: number; debt: number }
  safety: { level: 'steady' | 'tight' | 'shortfall'; months_below_buffer: number; months_negative_cash: number }
  goals: { met: number; due: number; total: number }
  freedom: { runway_months: number | null; debt_free: boolean }
  experience: { degree: boolean; studying: boolean; months_abroad: number; months_studying: number }
  joy: { total: number }
}

export interface StatementLine { label: string; amount: number; kind: 'in' | 'cost' | 'transfer' | 'return' }
export interface Statement {
  month: number
  label: string
  lines: StatementLine[]
  labels: string[]
  cash_after: number
  net_worth_after: number
}

export interface LivedRow {
  t: number; age: number; status: Status; regime: Regime; cash: number; investments: number; net_worth: number
  home_value: number; mortgage: number; debt: number; joy: number; flags: string[]
}

export interface GameState {
  persona: PersonaCard
  world: number
  seed: number
  month: number
  horizon: number
  ended: boolean
  age: number
  age_years: number
  date: { month_name: string; year: number; label: string }
  months_to_birthday: number
  state: GameStateDict
  pending_state: GameStateDict
  queued: ActionIn[]
  queued_letters: Letter[]
  scores: ScoresNow
  actions: Partial<Record<Place, GameAction[]>>
  letters: Letter[]
  statement: Statement | null
  warnings: { kind: string; text: string }[]
  goals: CheckedGoal[]
  rows: LivedRow[]
  life_end_age: number
  disclaimer: string
}

export interface Band { p10: number[]; p50: number[]; p90: number[] }
export interface FinalScores {
  wealth: { net_worth: number; cash: number; investments: number; debt: number }
  safety: { level: 'steady' | 'tight' | 'shortfall'; months_below_buffer: number; months_negative_cash: number; min_cash: number }
  goals: { met: number; total: number; beyond_horizon: number }
  freedom: { runway_months: number | null; debt_free: boolean }
  experience: { degree: boolean; months_abroad: number; months_studying: number }
  joy: { total: number }
}
export interface Projection {
  month: number
  horizon: number
  n: number
  seed: number
  world: number
  lived: { net_worth: number[]; cash: number[] }
  no_surprises: { net_worth: number[]; cash: number[]; regime: Regime[]; joy_total: number } | null
  regimes: { states: Regime[]; from_month: number; share: Record<Regime, number[]> } | null
  bands: { net_worth: Band; cash: Band } | null
  ghosts: { kind: string; text: string; month: number; net_worth: number[]; cash: number[] }[]
  final_net_worth: { p10: number; p50: number; p90: number } | null
  share_worlds_shortfall: number
  goal_met_share: Record<string, number>
  bridge: Record<string, number>
  final_scores: FinalScores
  goals: CheckedGoal[]
  ended?: boolean
  note?: string
}
export interface WorldEvent { month: number; kind: string; text: string }
export interface Roll { world: number; from_month: number; net_worth: number[]; cash: number[]; events: WorldEvent[] }

export interface CompareResult {
  table: { key: string; label: string; a: number; b: number; diff: number }[]
  bridge: { key: string; a: number; b: number; diff: number }[]
  explanation: string[]
  note: string
  ranges: { a: { p10: number; p50: number; p90: number } | null; b: { p10: number; p50: number; p90: number } | null }
  scores: { a: FinalScores; b: FinalScores }
  net_worth: { a: number[]; b: number[] }
}

export interface AssumptionRow {
  key: string; label: string; value: number; unit: string; tag: string; group: string
  min: number | null; max: number | null; note: string
}
export interface Check { name: string; passed: boolean; detail: string }

export interface Catalogue {
  habits: { id: string; label: string; place: Place; levels: { level: number; label: string; cost: number; joy: number }[] }[]
  items: { id: string; label: string; place: Place; kind: string; price: number; joy: number; abroad_only: boolean }[]
  homes: { id: string; label: string; price: number }[]
  home_price_level: { canton: string; level: number }
  flats: { id: string; label: string; factor: number }[]
  rules: { key: string; value: number; label: string; unit: string; group: string }[]
  note: string
}
