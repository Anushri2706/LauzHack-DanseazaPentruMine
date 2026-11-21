import { useState } from 'react'
import { api } from '../../api/client'
import type { HmmFork } from '../../api/types'
import { useApi } from '../../api/useApi'
import { setStore, useStore } from '../../store/game'
import { chf } from '../format'

/** Compare two HMM forks before keeping one. Nothing is committed until Keep. */
export function DecideApp() {
  const node = useStore((s) => (s.hmmCursor ? s.hmmNodes[s.hmmCursor] : undefined))
  const menu = useStore((s) => s.hmmMenu)
  const [pair, setPair] = useState<string[]>(['masters', 'start_working'])
  const [picked, setPicked] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [keeping, setKeeping] = useState(false)
  const key = node ? `${node.snapshot_id}:${pair.join(',')}` : null
  const branch = useApi(key, () => api.hmmBranch(node!, pair))
  const forks = branch.data?.forks ?? []
  const chosen = forks[picked] ?? forks[0]

  function show(decision: string) {
    const other = decision === 'start_working' ? 'masters' : 'start_working'
    setPair(decision === 'masters' ? ['masters', 'start_working'] : [decision, other])
    setPicked(0)
  }

  async function keep(fork: HmmFork) {
    if (!node || keeping) return
    setKeeping(true)
    setError(null)
    try {
      const child = await api.hmmCommit({
        snapshot: node,
        flows: fork.seeded_net_flows,
        decision: fork.decision,
        label: fork.label,
        cash: fork.cash,
        debt: fork.debt,
        investments: fork.investments,
      })
      setStore((s) => ({
        hmmNodes: { ...s.hmmNodes, [child.snapshot_id]: child },
        hmmCursor: child.snapshot_id,
        phone: 'branches',
      }))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setKeeping(false)
    }
  }

  return (
    <div>
      <h3>Decisions</h3>
      <p className="tiny muted">Two forks from this rollback point, one year each. Cash, debt, and investments are the path totals at the end of that year. Nothing is kept until you say so.</p>
      {!node && <p className="muted small">Open Paths first so there is a moment to branch from.</p>}
      {menu.length > 0 && (
        <div className="chart-tools">
          {menu.map((item) => (
            <button key={item.decision} className={pair[0] === item.decision ? 'btn' : 'btn ghost'} onClick={() => show(item.decision)}>
              {item.label}
            </button>
          ))}
        </div>
      )}
      {branch.loading && !forks.length && <p className="muted small">Simulating both forks…</p>}
      {branch.error && <p className="error">{branch.error}</p>}
      {error && <p className="error">{error}</p>}
      {forks.length > 0 && (
        <table className="cmp-table">
          <thead>
            <tr>
              <th />
              {forks.map((f, i) => (
                <th key={f.decision} style={{ fontWeight: i === picked ? 700 : 400 }}>{f.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Cash</td>
              {forks.map((f, i) => <td key={f.decision} className={i === picked ? 'lead' : ''}>{chf(f.cash)}</td>)}
            </tr>
            <tr>
              <td>Debt</td>
              {forks.map((f, i) => <td key={f.decision} className={i === picked ? 'lead' : ''}>{chf(f.debt)}</td>)}
            </tr>
            <tr>
              <td>Investments</td>
              {forks.map((f, i) => <td key={f.decision} className={i === picked ? 'lead' : ''}>{chf(f.investments)}</td>)}
            </tr>
          </tbody>
        </table>
      )}
      {forks.length > 1 && (
        <div className="chart-tools">
          {forks.map((f, i) => (
            <button key={f.decision} className={i === picked ? 'btn' : 'btn ghost'} onClick={() => setPicked(i)}>{f.label}</button>
          ))}
          <button className="btn" disabled={!chosen || keeping} onClick={() => chosen && keep(chosen)}>
            {keeping ? 'Keeping…' : 'Keep this path'}
          </button>
        </div>
      )}
    </div>
  )
}
