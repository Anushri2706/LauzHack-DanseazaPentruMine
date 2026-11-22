import type { PersonaCard } from './api/types'
import { getStore, resetGame, setStore, useStore } from './store/game'
import { CharacterCreator } from './ui/CharacterCreator'
import { GameScreen } from './ui/GameScreen'
import { PersonaPicker } from './ui/PersonaPicker'

const params = new URLSearchParams(location.search)

/** World #: from ?world=N (reproducible demos), otherwise rolled at game start. */
function rollWorld() {
  const w = Number(params.get('world'))
  return Number.isInteger(w) && w >= 0 && w < 500 && params.has('world') ? w : Math.floor(Math.random() * 500)
}

if (params.has('fresh')) resetGame()

export default function App() {
  const screen = useStore((s) => s.screen)
  const persona = useStore((s) => s.persona)
  const look = useStore((s) => s.look)
  const pick = (p: PersonaCard) => {
    resetGame()
    setStore({ persona: p, screen: 'create', world: rollWorld(), look: getStore().look, studyPlanAsked: false })
  }
  if (screen === 'play' && persona) return <GameScreen />
  if (screen === 'create' && persona) {
    return <CharacterCreator name={persona.name} look={look} setLook={(l) => setStore({ look: l })}
      onBack={() => setStore({ screen: 'pick' })} onDone={() => setStore({ screen: 'play' })} />
  }
  return <PersonaPicker onPick={pick} />
}
