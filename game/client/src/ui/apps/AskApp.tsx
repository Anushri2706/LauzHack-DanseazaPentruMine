// 💬 Ask: the companion. It answers with numbers the engines computed (each one cited), and can propose one
// town action that only happens when you press the button.
import { useRef, useState } from 'react'
import { api, EngineError } from '../../api/client'
import type { AskReply, GameState } from '../../api/types'
import { queueAction } from '../../store/actions'
import { getStore, requestFor } from '../../store/game'

const SUGGESTIONS = ['Why did my cash change last month?', 'How much will I have when I retire?',
  "What's my risk of losing my job?", "What if I do a full-time master's?", 'How accurate is your model?']

interface Msg { who: 'me' | 'twin'; text?: string; reply?: AskReply }

export function AskApp({ st }: { st: GameState }) {
  const [log, setLog] = useState<Msg[]>([])
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const end = useRef<HTMLDivElement>(null)
  const ask = async (q: string) => {
    const message = q.trim()
    if (!message || busy) return
    setLog((l) => [...l, { who: 'me', text: message }])
    setText('')
    setBusy(true)
    try {
      const reply = await api.ask(requestFor(getStore()), message)
      setLog((l) => [...l, { who: 'twin', reply }])
    } catch (e) {
      setLog((l) => [...l, { who: 'twin', text: e instanceof EngineError ? e.message : 'The engine could not answer that.' }])
    } finally {
      setBusy(false)
      setTimeout(() => end.current?.scrollIntoView({ behavior: 'smooth' }), 50)
    }
  }
  const last = [...log].reverse().find((m) => m.reply)?.reply
  return (
    <div className="ask-app">
      <h3>💬 Ask your twin</h3>
      <p className="tiny muted">Answers use only numbers the town engine and the learned twin computed for {st.persona.name}
        {' '}({st.date.label}). Mode: {last ? (last.mode === 'llm' ? `AI (${last.llm})` : 'rules (no AI key set)') : '—'}.</p>
      <div className="chat">
        {log.length === 0 && (
          <div className="chips">{SUGGESTIONS.map((q) => <button key={q} className="chip tiny" onClick={() => void ask(q)}>{q}</button>)}</div>
        )}
        {log.map((m, i) => (
          <div key={i} className={`bubble ${m.who}`}>
            {m.text && <p>{m.text}</p>}
            {m.reply && (
              <>
                {m.reply.answer.length === 0 && <p>I don't have engine numbers for that yet. Try one of the questions below.</p>}
                {m.reply.answer.map((a, j) => <p key={j}>{a.text}</p>)}
                {Object.keys(m.reply.facts_used).length > 0 && (
                  <details className="tiny"><summary>Sources ({Object.keys(m.reply.facts_used).length})</summary>
                    <ul>{Object.entries(m.reply.facts_used).map(([id, f]) => <li key={id}><b>{f.text}</b> · {f.label} · <i>{f.source}</i></li>)}</ul>
                  </details>
                )}
                {m.reply.action && (
                  <div className="ask-action">
                    <div className="small"><b>{m.reply.action.label}</b>: {m.reply.action.preview}</div>
                    {m.reply.action.enabled
                      ? <button className="btn" onClick={() => void queueAction({ type: m.reply!.action!.type, id: m.reply!.action!.id, value: m.reply!.action!.value })}>
                          Plan it for {st.date.label}</button>
                      : <div className="tiny note">🔒 {m.reply.action.reason}</div>}
                  </div>
                )}
                <p className="tiny muted">{m.reply.disclaimer}</p>
              </>
            )}
          </div>
        ))}
        {busy && <div className="bubble twin"><p className="muted">Asking the engines…</p></div>}
        <div ref={end} />
      </div>
      <form className="ask-form" onSubmit={(e) => { e.preventDefault(); void ask(text) }}>
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Ask about your money, your future, a decision…" maxLength={500} />
        <button className="btn" disabled={busy || !text.trim()}>Ask</button>
      </form>
      {log.length > 0 && <div className="chips">{SUGGESTIONS.map((q) => <button key={q} className="chip tiny" onClick={() => void ask(q)}>{q}</button>)}</div>}
    </div>
  )
}
