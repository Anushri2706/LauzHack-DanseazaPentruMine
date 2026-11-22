// 🧠 Twin: your future from the learned twin, how it is built, and how accurate it was on people it never saw.
import { api } from '../../api/client'
import type { GameState, ModelScore } from '../../api/types'
import { useApi } from '../../api/useApi'
import { useStore } from '../../store/game'
import { chf, pct } from '../format'
import { useTwin } from '../useGame'
import { REGIME_META } from '../Timeline'

const MODELS: [string, string, string][] = [
  ['persistence', 'Keep your past average', 'Your own 2025 average carries on (± your monthly spread).'],
  ['hmm', 'HMM alone', "The teammates' regime HMM with its own spending model."],
  ['hmm+gmm', 'HMM + GMM', 'HMM life states and incomes, spending from the GMM, learned seasons and noise.'],
  ['hmm+gmm+history', 'Learned twin', 'HMM + GMM + how this person differs from people like them (shrunk).'],
]

function Row({ name, desc, m, best }: { name: string; desc: string; m: ModelScore; best: boolean }) {
  return (
    <tr className={best ? 'best' : ''}>
      <td title={desc}>{best ? '🏆 ' : ''}{name}</td>
      <td>{chf(m.mae_cash_end_chf)}</td>
      <td>{pct(m.range_hit_rate_p10_p90)}</td>
      <td>{m.r2_cash_end.toFixed(3)}</td>
    </tr>
  )
}

export function TwinApp({ st }: { st: GameState }) {
  const s = useStore((x) => x)
  const fan = useTwin(s)
  const card = useApi('twin-model', () => api.twinModel())
  const f = fan.data?.summary
  const c = card.data
  const best = c ? Object.entries(c.test).sort((a, b) => a[1].mae_cash_end_chf - b[1].mae_cash_end_chf)[0][0] : ''
  const learned = c?.learned as Record<string, Record<string, number>> | undefined
  return (
    <div className="twin-app">
      <h3>🧠 Your learned twin</h3>
      <p className="tiny muted">
        Life events from the teammates' <b>regime HMM</b> (sourced Swiss hazards, retirement at 65, pensions), money
        behaviour <b>learned</b> from 958 synthetic people (GMM spending per life state, December bonus, holiday months,
        month-to-month noise) and from how {st.persona.name}'s own data differs from people like them.
      </p>
      {fan.loading && !f && <p className="muted">Simulating lives from {st.date.label}…</p>}
      {fan.error && <p className="note">⚠️ {fan.error}</p>}
      {f && fan.data && (
        <div className="tiles">
          <div className="tile"><div className="t-title">Saved per month now (typical)</div>
            <div className="t-main">{chf(f.flow_now_p50)}</div>
            <div className="t-sub">starting as {REGIME_META[fan.data.start_state].label.toLowerCase()}
              {fan.data.offset_chf_per_month ? ` · your data adds ${chf(fan.data.offset_chf_per_month)}/month` : ''}</div></div>
          {f.wealth_at_65 && <div className="tile"><div className="t-title">Liquid savings at 65</div>
            <div className="t-main">{chf(f.wealth_at_65.p50)}</div>
            <div className="t-sub">P10–P90 {chf(f.wealth_at_65.p10)} – {chf(f.wealth_at_65.p90)}</div></div>}
          {f.share_lives_with_job_search_before_65 !== null && <div className="tile"><div className="t-title">A job search before 65</div>
            <div className="t-main">{pct(f.share_lives_with_job_search_before_65)}</div>
            <div className="t-sub">of lives · typically {f.months_in_job_search_before_65_p50} months in total</div></div>}
          <div className="tile"><div className="t-title">Retirement</div>
            <div className="t-main">{f.retirement_age_p50 ? `age ${f.retirement_age_p50.toFixed(0)}` : '—'}</div>
            <div className="t-sub">{pct(f.share_lives_ever_below_zero)} of lives ever go below zero</div></div>
        </div>
      )}
      {fan.data?.assumed_degree_after_months && (
        <p className="tiny note">🎓 You are enrolled without an end date: the twin assumes you finish your degree and start
          working in {fan.data.assumed_degree_after_months} months (in the HMM, studying ends only by a decision).</p>
      )}
      <p className="tiny muted">{fan.data?.basis}. The timeline's <b>🧠 Learned twin</b> view draws these lives; <b>📐 Rules</b> shows the town engine.</p>

      <h4>How accurate is it?</h4>
      {c && (
        <>
          <p className="tiny">Backtest on <b>{c.sizes.test} locked test people</b> never used for training or tuning: their
            first 12 months (2025) are known, their cash over the next 12 months (2026) is predicted. Settings were chosen on
            {' '}{c.sizes.validation} validation people; everything was learned from {c.sizes.train} training people.</p>
          <table className="cmp-table">
            <thead><tr><th>Model</th><th>Avg error (cash after a year)</th><th>True value in range (aim 80%)</th><th>R²</th></tr></thead>
            <tbody>{MODELS.map(([id, name, desc]) => c.test[id] && <Row key={id} name={name} desc={desc} m={c.test[id]} best={id === best} />)}</tbody>
          </table>
          <p className="tiny muted">Life states held as in the data (the synthetic people never change state: nobody retires
            or finds a job in two years). With the HMM's life events switched on the learned twin's error is
            {' '}{chf(c.test_life_events_on['hmm+gmm+history'].mae_cash_end_chf)}: events the data cannot reward.</p>
          {learned && (
            <details className="tiny"><summary>What it learned</summary>
              <ul>
                <li>December bonus (share of monthly income): employed {pct(learned.december_income_share.employed)},
                  self-employed {pct(learned.december_income_share.self_employed)}, retired {pct(learned.december_income_share.retired)}.</li>
                <li>Holiday months (extra spending as a share of rent): February {pct(-learned.february_rent_share.employed)},
                  August {pct(-learned.august_rent_share.employed)} for employed people.</li>
                <li>Spending follows rent, not income (in this synthetic data), so long-run savings are very high.</li>
                <li>Settings chosen on validation: {c.chosen_on_validation.n_components} GMM components, history weight
                  n/(n+{c.chosen_on_validation.kappa}), noise × {c.chosen_on_validation.noise_scale}.</li>
              </ul>
            </details>
          )}
        </>
      )}
      <p className="tiny muted">Simulated lives under stated assumptions and synthetic data. Not a prediction or financial advice.</p>
    </div>
  )
}
