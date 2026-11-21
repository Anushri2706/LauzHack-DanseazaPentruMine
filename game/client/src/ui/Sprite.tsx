import { useEffect, useRef } from 'react'
import { lookFrames, npcFrame } from '../game/tiles'
import type { Look } from '../store/game'

const imgs = new Map<string, Promise<HTMLImageElement>>()
function loadImg(src: string) {
  let p = imgs.get(src)
  if (!p) {
    p = new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src })
    imgs.set(src, p)
  }
  return p
}

/** Draws one or more 16×16 frames of a sprite sheet on top of each other, scaled crisply. */
export function SheetSprite({ src, frames, cols, spacing = 0, scale = 4, className, flip }: {
  src: string; frames: (number | null)[]; cols: number; spacing?: number; scale?: number; className?: string; flip?: boolean
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const key = frames.join(',')
  useEffect(() => {
    let live = true
    loadImg(src).then((img) => {
      const c = ref.current
      if (!live || !c) return
      const g = c.getContext('2d')!
      g.clearRect(0, 0, 16, 16)
      for (const f of frames) {
        if (f === null) continue
        const r = Math.floor(f / cols), col = f % cols
        g.drawImage(img, col * (16 + spacing), r * (16 + spacing), 16, 16, 0, 0, 16, 16)
      }
    }).catch(() => {})
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, key, cols, spacing])
  return <canvas ref={ref} width={16} height={16} className={`sprite ${className ?? ''}`}
    style={{ width: 16 * scale, height: 16 * scale, transform: flip ? 'scaleX(-1)' : undefined }} />
}

export function Avatar({ look, scale = 4, className }: { look: Look; scale?: number; className?: string }) {
  const f = lookFrames(look)
  return <SheetSprite src="assets/roguelike-characters/roguelikeChar_transparent.png" cols={54} spacing={1}
    frames={[f.body, f.legs, f.shoes, f.top, f.hair]} scale={scale} className={className} />
}

export function Npc({ char, scale = 4 }: { char: number; scale?: number }) {
  return <SheetSprite src="assets/rpg-urban-pack/tilemap_packed.png" cols={27} frames={[npcFrame(char, 1, 0)]} scale={scale} />
}
