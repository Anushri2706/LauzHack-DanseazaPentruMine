// The town, as an ASCII layout (versionable; no Tiled dependency). 40 × 28 tiles of 16 px.
//
//   M mountain backdrop   T round tree   t pine   b bush   . grass   , flowers   = street/path
//   _ lakeside sand       ~ lake         P pier   S park paving       F fountain  B bench
//   L lamp post           m mailbox
// Buildings are listed separately (footprint + door); their tiles are drawn on top of the layout.
export const LAYOUT = [
  'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM', // 0
  'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM', // 1
  'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM', // 2
  'tTTTTbTtbTbTTTTTTTTbTTbTTbTbbTTTTbTtTTbT', // 3
  't......................................T', // 4
  'T......................................T', // 5
  't......................................T', // 6
  'T..........................,...........T', // 7
  'T...........................,..........t', // 8
  't======================================t', // 9
  't======================================t', // 10
  't.........==...............==..T...,.t.t', // 11
  'T.,.......==......,........==.....b....T', // 12
  'T.,.......==...............==.b.....T..T', // 13
  't.......m.==...............==.,.,..,,..t', // 14
  't======================================t', // 15
  't======================================T', // 16
  'T...,.....==........bBSSSSSSBb.........t', // 17
  'T.........==........SSSSFFSSSS,........t', // 18
  'T.........==........SSSSSSSSSS.........t', // 19
  't.L.....L.==........SBSSSSLSBS.........T', // 20
  '========================================', // 21
  '========================================', // 22
  '___________________PP___________________', // 23
  '~~~~~~~~~~~~~~~~~~~PP~~~~~~~~~~~~~~~~~~~', // 24
  '~~~~~~~~~~~~~~~~~~~PP~~~~~~~~~~~~~~~~~~~', // 25
  '~~~~~~~~~~~~~~~~~~~PP~~~~~~~~~~~~~~~~~~~', // 26
  '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', // 27
]
export const MAP_W = 40
export const MAP_H = 28
export const TILE = 16

LAYOUT.forEach((row, y) => {
  if (row.length !== MAP_W) throw new Error(`map row ${y} has ${row.length} columns, expected ${MAP_W}`)
})
if (LAYOUT.length !== MAP_H) throw new Error('map height')

export type PlaceId = 'home' | 'university' | 'office' | 'cafe' | 'shop' | 'bank' | 'realestate' | 'station'

export interface Building {
  place: PlaceId
  name: string
  x: number // top-left tile
  y: number
  w: number
  h: number // 2 roof rows + 2 wall rows
  doorX: number // door column (bottom wall row); the trigger tile is just below it
  roof: 'red' | 'grey'
  wall: 'wood' | 'stone'
}

export const BUILDINGS: Building[] = [
  { place: 'university', name: 'UNIVERSITY', x: 3, y: 5, w: 7, h: 4, doorX: 6, roof: 'grey', wall: 'stone' },
  { place: 'office', name: 'OFFICE', x: 17, y: 5, w: 6, h: 4, doorX: 19, roof: 'grey', wall: 'stone' },
  { place: 'bank', name: 'BANK', x: 30, y: 5, w: 6, h: 4, doorX: 32, roof: 'red', wall: 'stone' },
  { place: 'home', name: 'HOME', x: 3, y: 11, w: 5, h: 4, doorX: 5, roof: 'red', wall: 'wood' },
  { place: 'cafe', name: 'CAFÉ', x: 13, y: 11, w: 5, h: 4, doorX: 15, roof: 'red', wall: 'wood' },
  { place: 'shop', name: 'SHOP', x: 20, y: 11, w: 6, h: 4, doorX: 22, roof: 'grey', wall: 'wood' },
  { place: 'realestate', name: 'HOMES 4 SALE', x: 13, y: 17, w: 6, h: 4, doorX: 15, roof: 'red', wall: 'wood' },
  { place: 'station', name: 'STATION', x: 31, y: 17, w: 8, h: 4, doorX: 34, roof: 'grey', wall: 'stone' },
]

export const doorTrigger = (b: Building) => ({ x: b.doorX, y: b.y + b.h })

const WALKABLE = new Set(['.', ',', '=', '_', 'P', 'S'])

/** 0 = walkable, 1 = blocked (trees, water, buildings…). Used for movement and click-to-walk paths. */
export function walkGrid(): number[][] {
  const g = LAYOUT.map((row) => [...row].map((c) => (WALKABLE.has(c) ? 0 : 1)))
  for (const b of BUILDINGS) {
    for (let y = b.y; y < b.y + b.h; y++) for (let x = b.x; x < b.x + b.w; x++) g[y][x] = 1
  }
  return g
}

export function buildingAt(tx: number, ty: number): Building | undefined {
  return BUILDINGS.find((b) => tx >= b.x && tx < b.x + b.w && ty >= b.y && ty < b.y + b.h)
}
