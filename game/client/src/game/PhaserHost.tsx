import Phaser from 'phaser'
import { useEffect, useRef } from 'react'
import { bus } from '../bridge'
import { getStore } from '../store/game'
import { TownScene } from './TownScene'

/** React owns this <div>; Phaser renders the town into it. Overlays (HUD, dialogs, phone) sit on top. */
export function PhaserHost() {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    let game: Phaser.Game | null = null
    let dead = false
    const onReady = () => {
      const s = getStore()
      bus.emit('look', s.look)
      bus.emit('season', { month: s.current })
    }
    bus.on('ready', onReady)
    document.fonts.load('27px "Pixelify Sans"').finally(() => {
      if (dead || !ref.current) return
      game = new Phaser.Game({
        type: Phaser.AUTO, parent: ref.current, backgroundColor: '#9fd8a5', pixelArt: true,
        scale: { mode: Phaser.Scale.RESIZE, width: '100%', height: '100%' },
        scene: [TownScene], banner: false, audio: { noAudio: true },
      })
      ;(window as unknown as { __game?: Phaser.Game }).__game = game
    })
    return () => { dead = true; bus.off('ready', onReady); game?.destroy(true) }
  }, [])
  return <div ref={ref} className="phaser-host" />
}
