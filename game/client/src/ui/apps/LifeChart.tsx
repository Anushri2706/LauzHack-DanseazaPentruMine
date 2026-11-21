import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { HmmFan } from '../../api/types'
import { chf, k } from '../format'

/** Cash fan from the regime engine, with the investments and debt totals on the same axes. */
export function LifeChart({ fan, startAge }: { fan: HmmFan; startAge: number }) {
  const endAge = startAge + fan.horizon_months / 12
  const data = fan.p50.map((p50, i) => ({
    age: +(startAge + i / 12).toFixed(3),
    p50,
    band: [fan.p05[i], fan.p95[i]],
    debt: fan.debt[i],
    investments: fan.investments[i],
  }))
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 18, right: 16, bottom: 4, left: 4 }}>
        <CartesianGrid stroke="#eee3d3" strokeDasharray="2 4" />
        <XAxis dataKey="age" type="number" domain={[startAge, endAge]} tickCount={6}
          tickFormatter={(a) => `${Math.round(Number(a))}`} stroke="#7a7093" fontSize={12} />
        <YAxis tickFormatter={k} stroke="#7a7093" width={48} fontSize={12} />
        <Tooltip
          contentStyle={{ background: '#fff6e9', border: '2px solid #4a4063', borderRadius: 0, fontFamily: 'inherit' }}
          labelFormatter={(a) => `Age ${Number(a).toFixed(1)}`}
          formatter={(v, name) => (Array.isArray(v)
            ? [`${chf(Number(v[0]))} – ${chf(Number(v[1]))}`, 'cash range under our assumptions (P05–P95)']
            : [chf(Number(v)), String(name)])}
        />
        <Area dataKey="band" name="cash range" stroke="none" fill="#9a7fd8" fillOpacity={0.22} isAnimationActive={false} />
        <Line dataKey="p50" name="cash, median" stroke="#4a4063" strokeWidth={2.4} dot={false} isAnimationActive={false} />
        <Line dataKey="investments" name="investments" stroke="#4fb38a" strokeWidth={2} dot={false} isAnimationActive={false} />
        <Line dataKey="debt" name="debt" stroke="#d9785b" strokeWidth={2} dot={false} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
