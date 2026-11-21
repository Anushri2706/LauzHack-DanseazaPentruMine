// Event bus between Phaser (the town) and React (HUD, dialogs, phone).
import mitt from 'mitt'
import type { Look } from './store/game'

export type Events = {
  // Phaser → React
  door: { place: string }
  near: { place: string | null }
  ready: void
  // React → Phaser
  freeze: boolean
  season: { month: number }
  abroad: boolean
  look: Look
  queue: string[] // icons of this month's queued actions, shown above the player
  wake: void // after sleeping: the player steps out of the home door
  walkTo: { place: string }
}
export const bus = mitt<Events>()
