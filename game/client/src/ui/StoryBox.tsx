import { useState } from 'react'
import { api } from '../api/client'
import type { Story } from '../api/types'
import { getStore, requestFor } from '../store/game'

/** The AI companion's plain-language retelling of the engine's numbers (every number is cited from the engines). */
export function StoryBox({ story, style }: { story?: Story | null; style?: React.CSSProperties }) {
  if (!story) return null
  return (
    <div className="story note" style={style}>
      <b>{story.headline}</b>
      {story.story.map((x, i) => <p key={i} className="small" style={{ margin: '6px 0 0' }}>{x.text}</p>)}
      <p className="tiny muted" style={{ margin: '6px 0 0' }}>
        {story.mode === 'llm' ? '✨ Written by the AI companion from the engine’s numbers' : 'Told from the engine’s numbers'}
        {' · '}{story.disclaimer}
      </p>
    </div>
  )
}

/** On demand (an AI call per press, so not on every refresh): explain the reality you are looking at. */
export function ExplainOutcome() {
  const [story, setStory] = useState<Story | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const go = async () => {
    setBusy(true); setError(null)
    try { setStory(await api.outcome(requestFor(getStore(), undefined, false))) }
    catch (e) { setError((e as Error).message) }
    setBusy(false)
  }
  return (
    <div style={{ margin: '10px 0' }}>
      <button className="btn" onClick={go} disabled={busy}>{busy ? 'Writing…' : story ? '✨ Explain again' : '✨ Explain this in plain words'}</button>
      {error && <p className="error small">{error}</p>}
      <StoryBox story={story} style={{ marginTop: 8 }} />
    </div>
  )
}
