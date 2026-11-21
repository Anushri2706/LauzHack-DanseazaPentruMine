import { api } from '../../api/client'
import { useApi } from '../../api/useApi'
import { useStore } from '../../store/game'
import { chf } from '../format'
import { LifeChart } from './LifeChart'

/** Final simulated-life distribution for the path you are on. */
export function LifeApp({ startAge }: { startAge: number }) {
  const node = useStore((s) => (s.hmmCursor ? s.hmmNodes[s.hmmCursor] : undefined))
  const key = node ? `${node.snapshot_id}:${node.month}:${node.observed_net_flows.length}` : null
  const dist = useApi(key, () => api.hmmDistribution(node!))
  const fan = dist.data
  const last = fan ? fan.p50.length - 1 : -1
  const age = node ? startAge + node.month / 12 : startAge
  return (
    <div>
      <h3>Life distribution</h3>
      <p className="tiny muted">
        The cash band is the simulated life from {node ? node.label : 'this path'} onward.
        Debt and investments are the running totals on that same roll, beside the cash.
      </p>
      {!node && <p className="muted small">Open Paths first.</p>}
      {dist.loading && !fan && <p className="muted small">Simulating the remaining life…</p>}
      {dist.error && <p className="error">{dist.error}</p>}
      {fan && last >= 0 && (
        <>
          <div className="tiles">
            <div className="tile"><div className="t-title"><span>Cash</span></div><div className="t-main">{chf(fan.p50[last])}</div><div className="t-sub">median at the end of this roll</div></div>
            <div className="tile"><div className="t-title"><span>Debt</span></div><div className="t-main">{chf(fan.debt[last])}</div><div className="t-sub">tuition-debt total</div></div>
            <div className="tile"><div className="t-title"><span>Investments</span></div><div className="t-main">{chf(fan.investments[last])}</div><div className="t-sub">running total of money set aside</div></div>
          </div>
          <div className="chart-box life-chart">
            <LifeChart fan={fan} startAge={age} />
          </div>
          <p className="tiny muted">{fan.disclaimer}</p>
        </>
      )}
    </div>
  )
}
