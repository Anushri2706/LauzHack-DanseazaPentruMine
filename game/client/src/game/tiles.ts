// Tile and sprite indices, recorded by inspecting the Kenney sheets (scripts rendered each sheet with its
// index numbers; see README "Assets"). Never guess an index: check the sheet first.
//
// tt = Tiny Town  (tilemap_packed.png, 16×16, 12 columns, no spacing)
// ru = RPG Urban  (tilemap_packed.png, 16×16, 27 columns, no spacing)
// ch = Roguelike Characters (roguelikeChar_transparent.png, 16×16, 1px spacing, 54 columns)

export const TT = {
  grass: [0, 1], flowers: 2,
  // dirt path 9-slice with grass border (top-left … bottom-right)
  path: { tl: 12, t: 13, tr: 14, l: 24, c: 25, r: 26, bl: 36, b: 37, br: 38 },
  stone: 43,
  tree: 16, treeAutumn: 15, pine: 4, pineAutumn: 3, bush: 5, mushroom: 29, sprout: 17,
  roofRed: { tl: 52, t: 53, tr: 54, bl: 64, b: 65, br: 66 },
  roofGrey: { tl: 48, t: 49, tr: 50, bl: 60, b: 61, br: 62 },
  wallWood: { l: 72, m: 73, r: 75, window: 84, door: 85 },
  wallStone: { l: 76, m: 77, r: 79, window: 88, door: 89 },
  fence: { l: 44, m: 45, r: 46 },
  sign: 83, well: 104, beehive: 94,
}

export const RU = {
  water: { tl: 170, t: 171, tr: 172, l: 197, c: 198, r: 199, bl: 224, b: 225, br: 226 },
  fountain: 230,
  lampTop: 169, lampPost: 196,
  bench: 381, planks: 383,
  mailbox: 305, trash: 279, hydrant: 278,
  // Walking townsfolk: 4 facing columns (left, down, up, right) starting at `col`, 2 animation rows.
  // Character k uses rows 2k and 2k+1. frame = row * 27 + col + facing.
  npcCol: 23,
}
export const RU_COLS = 27
export const npcFrame = (char: number, facing: 0 | 1 | 2 | 3, step: 0 | 1) => (char * 2 + step) * RU_COLS + RU.npcCol + facing

// ------------------------------------------------------------------ character creator (layered sprite)
export const CH_COLS = 54
export const SKINS = [0, 54, 108]
export const LEGS = [3, 57, 111, 165, 219, 273, 327, 381, 435, 489, 543] // shoes are the next frame (+1)
export const TOPS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] // full shirts only (other rows are straps)
// Hair: colour = base (row, col); style = offset (dRow, dCol) inside that colour block. -1 = no hair.
export const HAIR_COLORS: { name: string; row: number; col: number }[] = [
  { name: 'brown', row: 0, col: 19 }, { name: 'ginger', row: 0, col: 23 }, { name: 'blonde', row: 4, col: 19 },
  { name: 'black', row: 4, col: 23 }, { name: 'white', row: 8, col: 19 },
]
export const HAIR_STYLES: ([number, number] | null)[] = [null, [0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3], [0, 3], [0, 2]]

export interface LookFrames { body: number; legs: number; shoes: number; top: number; hair: number | null }
export function lookFrames(l: { skin: number; hair: number; hairColor: number; top: number; legs: number }): LookFrames {
  const legs = LEGS[l.legs % LEGS.length]
  const style = HAIR_STYLES[l.hair % HAIR_STYLES.length]
  const c = HAIR_COLORS[l.hairColor % HAIR_COLORS.length]
  return {
    body: SKINS[l.skin % SKINS.length], legs, shoes: legs + 1, top: TOPS[l.top % TOPS.length],
    hair: style ? (c.row + style[0]) * CH_COLS + c.col + style[1] : null,
  }
}
