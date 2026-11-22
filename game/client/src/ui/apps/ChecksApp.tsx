import { useState } from 'react'
import { api } from '../../api/client'
import type { Check } from '../../api/types'

/** The old "Is the maths right?" panel: live consistency checks run by the engine. */
export function ChecksApp() {
  const [checks, setChecks] = useState<Check[] | null>(null)
  const [busy, setBusy] = useState(false)
  const run = async () => { setBusy(true); try { setChecks(await api.validation()) } finally { setBusy(false) } }
  return (
    <div>
      <h3>✅ Is the maths right?</h3>
      <p className="tiny muted">Live checks against the engine (the full suites run with pytest: 131 tests for the game engine, 46 for the regime twin).</p>
      <button className="btn" onClick={() => void run()} disabled={busy}>{busy ? 'Running…' : 'Run live checks'}</button>
      {checks && (
        <>
          <p className="small"><b>{checks.filter((c) => c.passed).length} / {checks.length} passed</b></p>
          <ul className="checks">
            {checks.map((c) => (
              <li key={c.name}><span>{c.passed ? '✅' : '❌'}</span><div><b>{c.name}</b><div className="tiny muted">{c.detail}</div></div></li>
            ))}
          </ul>
        </>
      )}
    </div>
  )
}
