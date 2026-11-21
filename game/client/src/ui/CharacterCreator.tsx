import { HAIR_COLORS, HAIR_STYLES, LEGS, SKINS, TOPS } from '../game/tiles'
import type { Look } from '../store/game'
import { Avatar, SheetSprite } from './Sprite'

const SHEET = 'assets/roguelike-characters/roguelikeChar_transparent.png'
const HAIR_SWATCH = ['#7a4a2a', '#c8611c', '#e8d29a', '#3d3d45', '#e9e9ee']

function Row({ label, n, value, onChange, render }: {
  label: string; n: number; value: number; onChange: (v: number) => void; render: (i: number) => React.ReactNode
}) {
  return (
    <div className="cc-row">
      <span className="cc-label">{label}</span>
      <div className="cc-options">
        {Array.from({ length: n }, (_, i) => (
          <button key={i} className={`cc-opt ${value === i ? 'on' : ''}`} onClick={() => onChange(i)} aria-label={`${label} ${i + 1}`}>
            {render(i)}
          </button>
        ))}
      </div>
    </div>
  )
}

export function CharacterCreator({ name, look, setLook, onDone, onBack }: {
  name: string; look: Look; setLook: (l: Look) => void; onDone: () => void; onBack: () => void
}) {
  const set = (k: keyof Look) => (v: number) => setLook({ ...look, [k]: v })
  return (
    <div className="screen create-screen">
      <div className="px-box create-box">
        <h2>Who is {name} today?</h2>
        <p className="muted small">Looks are just for fun: they never change any number in the simulation.</p>
        <div className="create-grid">
          <div className="preview-stage">
            <div className="walker"><Avatar look={look} scale={9} /></div>
            <div className="stage-ground" />
          </div>
          <div>
            <Row label="Skin" n={SKINS.length} value={look.skin} onChange={set('skin')}
              render={(i) => <SheetSprite src={SHEET} cols={54} spacing={1} frames={[SKINS[i]]} scale={2.5} />} />
            <Row label="Hair" n={HAIR_STYLES.length} value={look.hair} onChange={set('hair')}
              render={(i) => <Avatar look={{ ...look, hair: i }} scale={2.5} />} />
            <Row label="Hair colour" n={HAIR_COLORS.length} value={look.hairColor} onChange={set('hairColor')}
              render={(i) => <span className="swatch" style={{ background: HAIR_SWATCH[i] }} />} />
            <Row label="Top" n={TOPS.length} value={look.top} onChange={set('top')}
              render={(i) => <SheetSprite src={SHEET} cols={54} spacing={1} frames={[TOPS[i]]} scale={2.5} />} />
            <Row label="Trousers" n={LEGS.length} value={look.legs} onChange={set('legs')}
              render={(i) => <SheetSprite src={SHEET} cols={54} spacing={1} frames={[LEGS[i], LEGS[i] + 1]} scale={2.5} />} />
          </div>
        </div>
        <div className="row-end">
          <button className="btn ghost" onClick={onBack}>← Other person</button>
          <button className="btn ghost" onClick={() => setLook({
            skin: rnd(SKINS.length), hair: rnd(HAIR_STYLES.length), hairColor: rnd(HAIR_COLORS.length),
            top: rnd(TOPS.length), legs: rnd(LEGS.length) })}>🎲 Surprise me</button>
          <button className="btn" onClick={onDone}>Walk into town →</button>
        </div>
      </div>
    </div>
  )
}
const rnd = (n: number) => Math.floor(Math.random() * n)
