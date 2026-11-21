import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { HmmSnapshot } from '../../api/types'
import { getStore, setStore, useStore } from '../../store/game'
import { chf } from '../format'

function money(n: number | null) {
  return n == null ? '—' : chf(n)
}

/** Rollback snapshots from simulate-until-next-decision. A click jumps to that path. */
export function BranchApp({ personaId }: { personaId: string }) {
  const nodes = useStore((s) => s.hmmNodes)
  const cursor = useStore((s) => s.hmmCursor)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (getStore().hmmNodes.root) return
    let live = true
    setBusy(true)
    api.hmmStart(personaId)
      .then((res) => {
        if (!live) return
        setStore({
          hmmNodes: { [res.snapshot.snapshot_id]: res.snapshot },
          hmmCursor: res.snapshot.snapshot_id,
          hmmMenu: res.decisions,
        })
      })
      .catch((e: Error) => { if (live) setError(e.message) })
      .finally(() => { if (live) setBusy(false) })
    return () => { live = false }
  }, [personaId])

  const list = Object.values(nodes)
  const here = cursor ? nodes[cursor] : undefined

  async function jump(id: string) {
    const node = getStore().hmmNodes[id]
    if (!node) return
    setError(null)
    try {
      await api.hmmRestore(node)
      setStore({ hmmCursor: id })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div>
      <h3>Paths</h3>
      <p className="tiny muted">Each square is a rollback point from a simulated block. Tap one to jump to that path. The town's own multiverse is unchanged.</p>
      {busy && !list.length && <p className="muted small">Reading the ledger…</p>}
      {error && <p className="error">{error}</p>}
      {list.length > 0 && (
        <div className="mv-wrap hmm-tree">
          <Tree nodes={nodes} cursor={cursor} onJump={jump} />
        </div>
      )}
      {here && (
        <div className="mv-pop px-box">
          <span>You are here: <b>{here.label}</b> · month {here.month}</span>
          <span className="tiny">Cash {money(here.cash)} · debt {money(here.debt)} · investments {money(here.investments)}</span>
        </div>
      )}
    </div>
  )
}

function Tree({ nodes, cursor, onJump }: {
  nodes: Record<string, HmmSnapshot>
  cursor: string | null
  onJump: (id: string) => void
}) {
  const list = Object.values(nodes)
  const root = list.find((n) => n.parent_id == null) ?? list[0]
  const lanes: Record<string, number> = {}
  let next = 0
  const place = (id: string): number => {
    const kids = list.filter((n) => n.parent_id === id)
    if (kids.length === 0) { lanes[id] = next++; return lanes[id] }
    const ys = kids.map((k) => place(k.snapshot_id))
    lanes[id] = ys[0]
    return ys[0]
  }
  if (root) place(root.snapshot_id)
  const maxMonth = Math.max(12, ...list.map((n) => n.month))
  const W = 1000
  const H = Math.max(1, next) * 56 + 36
  const x = (m: number) => 48 + (m / (maxMonth + 12)) * (W - 200)
  const y = (id: string) => 28 + (lanes[id] ?? 0) * 56
  return (
    <svg className="multiverse" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMinYMin meet">
      {list.filter((n) => n.parent_id && nodes[n.parent_id]).map((n) => {
        const parent = n.parent_id as string
        const x1 = x(nodes[parent].month), y1 = y(parent), x2 = x(n.month), y2 = y(n.snapshot_id)
        const mx = (x1 + x2) / 2
        const on = n.snapshot_id === cursor || parent === cursor
        return <path key={n.snapshot_id} d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`} fill="none"
          stroke="#9a7fd8" strokeWidth={on ? 4 : 2} strokeOpacity={on ? 1 : 0.45} />
      })}
      {list.map((n) => {
        const cx = x(n.month), cy = y(n.snapshot_id)
        const here = n.snapshot_id === cursor
        return (
          <g key={n.snapshot_id} className="mv-node" onClick={() => onJump(n.snapshot_id)}>
            <title>{`${n.label} · month ${n.month}`}</title>
            {here && <rect x={cx - 11} y={cy - 11} width={22} height={22} fill="none" stroke="#4a4063" strokeWidth={2} />}
            <rect x={cx - 6} y={cy - 6} width={12} height={12} fill={n.parent_id ? '#9a7fd8' : '#fff6e9'} stroke="#4a4063" strokeWidth={2} />
            <text x={cx} y={cy - 14} textAnchor="middle" className="mv-label">{n.label}</text>
          </g>
        )
      })}
    </svg>
  )
}
