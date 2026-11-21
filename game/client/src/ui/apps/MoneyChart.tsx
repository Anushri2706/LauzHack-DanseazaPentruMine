import {
  Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceDot, ReferenceLine, ResponsiveContainer, Tooltip,
  XAxis, YAxis,
} from 'recharts'
import type { CheckedGoal, Projection, Roll } from '../../api/types'
import { chf, k } from '../format'

export type Metric = 'net_worth' | 'cash'

/** Adapted from the old LifeChart: lived path (solid), "no surprises" (dotted), P10–P90 band from now on,
 *  stress-test ghosts, one rolled world, and the fog over the future. All values come from the engine. */
export function MoneyChart({ proj, startAge, opening, metric, color, showBand, showGhosts, roll, goals, marks }: {
  proj: Projection
  startAge: number
  opening: { net_worth: number; cash: number }
  metric: Metric
  color: string
  showBand: boolean
  showGhosts: boolean
  roll?: Roll
  goals: CheckedGoal[]
  marks: { month: number; label: string }[]
}) {
  const H = proj.horizon
  const m = proj.month
  const lived = proj.lived[metric]
  const endAge = startAge + H / 12
  const nowAge = startAge + m / 12
  const at = (arr: number[] | undefined, i: number) => (arr ? arr[i - m - 1] : undefined)
  const livedAt = (i: number) => (i === 0 ? opening[metric] : lived[i - 1])
  const data = Array.from({ length: H + 1 }, (_, i) => {
    const d: Record<string, number | number[] | null | undefined> = { age: +(startAge + i / 12).toFixed(3) }
    d.lived = i <= m ? livedAt(i) : null
    if (proj.no_surprises) d.calm = i === m ? livedAt(i) : i > m ? at(proj.no_surprises[metric], i) : null
    if (proj.bands && i >= m) {
      const b = proj.bands[metric]
      d.band = i === m ? [livedAt(i), livedAt(i)] : [b.p10[i - m - 1], b.p90[i - m - 1]]
    }
    proj.ghosts.forEach((g, j) => { d[`g${j}`] = i === m ? livedAt(i) : i > m ? g[metric][i - m - 1] : null })
    if (roll) d.roll = i === m ? livedAt(i) : i > m ? roll[metric][i - m - 1] : null
    return d
  })
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 18, right: 16, bottom: 4, left: 4 }}>
        <defs>
          <linearGradient id="fog" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="#b8a4e3" stopOpacity={0.05} />
            <stop offset="100%" stopColor="#b8a4e3" stopOpacity={0.25} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="#eee3d3" strokeDasharray="2 4" />
        <XAxis dataKey="age" type="number" domain={[startAge, endAge]} tickCount={Math.round(H / 12) + 1}
          tickFormatter={(a) => `${Math.round(a)}`} stroke="#7a7093" fontSize={12} />
        <YAxis tickFormatter={k} stroke="#7a7093" width={48} fontSize={12} />
        <Tooltip
          contentStyle={{ background: '#fff6e9', border: '2px solid #4a4063', borderRadius: 0, fontFamily: 'inherit' }}
          labelFormatter={(a) => `Age ${Number(a).toFixed(1)}`}
          formatter={(v, name) => (Array.isArray(v)
            ? [`${chf(Number(v[0]))} – ${chf(Number(v[1]))}`, 'range under our assumptions (P10–P90)']
            : [chf(Number(v)), String(name)])}
        />
        {m < H && (
          <ReferenceArea x1={nowAge} x2={endAge} fill="url(#fog)" ifOverflow="hidden"
            label={{ value: 'not lived yet: one of many possible futures', position: 'insideTop', fill: '#7a7093', fontSize: 12 }} />
        )}
        <ReferenceLine y={0} stroke="#ff8a8a" strokeDasharray="4 4" />
        {showBand && proj.bands && <Area dataKey="band" name="range" stroke="none" fill={color} fillOpacity={0.22} isAnimationActive={false} />}
        {showGhosts && proj.ghosts.map((g, j) => (
          <Line key={g.kind} dataKey={`g${j}`} name={g.text} stroke="#7a7093" strokeOpacity={0.6} strokeDasharray="2 3" strokeWidth={1.4} dot={false} isAnimationActive={false} />
        ))}
        {roll && <Line dataKey="roll" name={`world #${roll.world}`} stroke="#e8a64a" strokeWidth={1.8} dot={false} isAnimationActive={false} />}
        <Line dataKey="calm" name="if nothing surprising happens" stroke={color} strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} connectNulls={false} />
        <Line dataKey="lived" name="lived so far" stroke={color} strokeWidth={3} dot={false} isAnimationActive={false} connectNulls={false} />
        {m > 0 && m <= H && <ReferenceLine x={nowAge} stroke="#4a4063" strokeOpacity={0.6} label={{ value: 'now', position: 'top', fill: '#4a4063', fontSize: 12 }} />}
        {marks.filter((x) => x.month <= m).map((x, i) => (
          <ReferenceDot key={i} x={startAge + x.month / 12} y={livedAt(x.month)} r={4} fill={color} stroke="#4a4063" strokeWidth={2}
            label={{ value: x.label, position: i % 2 ? 'bottom' : 'top', fill: '#4a4063', fontSize: 11 }} />
        ))}
        {metric === 'net_worth' && goals.filter((g) => g.status !== 'beyond_horizon').map((g) => (
          <ReferenceDot key={g.goal_id} x={startAge + (g.month + 1) / 12} y={g.target} r={5}
            fill={g.status === 'met' ? '#6cc6a0' : '#ff8a8a'} stroke="#4a4063" strokeWidth={1.5}
            label={{ value: `🎯 ${g.name}`, position: 'right', fill: '#4a4063', fontSize: 11 }} />
        ))}
      </ComposedChart>
    </ResponsiveContainer>
  )
}
