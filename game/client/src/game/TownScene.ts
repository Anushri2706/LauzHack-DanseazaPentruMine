import EasyStar from 'easystarjs'
import Phaser from 'phaser'
import { bus } from '../bridge'
import type { Look } from '../store/game'
import { BUILDINGS, LAYOUT, MAP_H, MAP_W, TILE, buildingAt, doorTrigger, walkGrid, type Building } from './map'
import { RU, TT, lookFrames, npcFrame } from './tiles'

const TT_GID = 1 // tileset gid offsets inside the Phaser tilemap
const RU_GID = 1000
const SPEED = 72 // px per second (world pixels, before the camera zoom)
const FONT = '"PixelDigits", "Pixelify Sans", monospace'

type Tile = { x: number; y: number }

export class TownScene extends Phaser.Scene {
  private grid!: number[][]
  private star!: EasyStar.js
  private ground!: Phaser.Tilemaps.TilemapLayer
  private deco!: Phaser.Tilemaps.TilemapLayer
  private player!: Phaser.GameObjects.Container
  private parts: Phaser.GameObjects.Image[] = []
  private queueText!: Phaser.GameObjects.Text
  private keys!: Record<string, Phaser.Input.Keyboard.Key>
  private path: Tile[] = []
  private pathTarget: string | null = null
  private frozen = false
  private near: string | null = null
  private signs: Record<string, Phaser.GameObjects.Text> = {}
  private trees: { x: number; y: number; kind: 'tree' | 'pine' }[] = []
  private npcs: { sprite: Phaser.GameObjects.Sprite; char: number; path: Tile[]; facing: 0 | 1 | 2 | 3; t: number; wait: number }[] = []
  private walkT = 0
  private unsub: (() => void)[] = []

  constructor() { super('town') }

  preload() {
    this.load.spritesheet('tt', 'assets/tiny-town/tilemap_packed.png', { frameWidth: 16, frameHeight: 16 })
    this.load.spritesheet('ru', 'assets/rpg-urban-pack/tilemap_packed.png', { frameWidth: 16, frameHeight: 16 })
    this.load.spritesheet('ch', 'assets/roguelike-characters/roguelikeChar_transparent.png',
      { frameWidth: 16, frameHeight: 16, spacing: 1 })
  }

  create() {
    this.grid = walkGrid()
    this.star = new EasyStar.js()
    this.star.setGrid(this.grid)
    this.star.setAcceptableTiles([0])

    this.drawBackdrop()
    this.buildMap()
    this.buildBuildings()
    this.spawnPlayer()
    this.spawnNpcs()
    this.drawWaterSparkles()

    const cam = this.cameras.main
    cam.setBounds(0, 0, MAP_W * TILE, MAP_H * TILE)
    cam.startFollow(this.player, true, 0.15, 0.15)
    cam.roundPixels = true
    this.fitZoom()
    this.scale.on('resize', () => this.fitZoom())

    const kb = this.input.keyboard!
    this.keys = kb.addKeys('W,A,S,D,UP,DOWN,LEFT,RIGHT,E,ENTER,SPACE') as Record<string, Phaser.Input.Keyboard.Key>
    kb.disableGlobalCapture() // let React inputs (Settings, Goals) receive keys
    this.input.on('pointerdown', (p: Phaser.Input.Pointer) => this.onClick(p))

    const on = <K extends keyof import('../bridge').Events>(k: K, f: (v: import('../bridge').Events[K]) => void) => {
      bus.on(k, f); this.unsub.push(() => bus.off(k, f))
    }
    on('freeze', (f) => { this.frozen = f; if (f) { this.path = []; this.pathTarget = null } })
    on('look', (l) => this.applyLook(l))
    on('queue', (icons) => this.queueText.setText(icons.join('')))
    on('wake', () => this.wake())
    on('season', ({ month }) => this.setSeason(month))
    on('abroad', (a) => this.setAbroad(a))
    on('walkTo', ({ place }) => this.walkToPlace(place))
    this.events.once('shutdown', () => this.unsub.forEach((u) => u()))
    bus.emit('ready')
  }

  // ------------------------------------------------------------------ world building
  private fitZoom() {
    const w = this.scale.width
    const z = Math.max(2, Math.min(4, Math.floor(w / (26 * TILE))))
    this.cameras.main.setZoom(z)
  }

  private drawBackdrop() {
    const g = this.add.graphics().setDepth(-10)
    const W = MAP_W * TILE
    g.fillStyle(0xd7eefb).fillRect(0, 0, W, 3 * TILE)
    g.fillStyle(0xeaf6fd).fillRect(0, 0, W, 14)
    const peaks = [[-10, 46, 70], [50, 40, 60], [105, 44, 80], [170, 38, 64], [230, 46, 78], [300, 40, 62], [360, 44, 84],
      [430, 39, 66], [495, 46, 76], [560, 40, 70], [620, 44, 60]]
    for (const [x, h, w] of peaks) {
      g.fillStyle(0xb8a4e3).fillTriangle(x, 48, x + w / 2, 48 - h, x + w, 48)
      g.fillStyle(0xfff6e9).fillTriangle(x + w / 2 - 9, 48 - h + 11, x + w / 2, 48 - h, x + w / 2 + 9, 48 - h + 11)
    }
    for (const [x, h, w] of peaks.map(([x, h, w]) => [x + 30, h - 14, w - 10])) {
      g.fillStyle(0xa79ad6).fillTriangle(x, 48, x + w / 2, 48 - h, x + w, 48)
    }
    g.fillStyle(0x9fd8a5).fillRect(0, 44, W, 4)
  }

  private buildMap() {
    const map = this.make.tilemap({ width: MAP_W, height: MAP_H, tileWidth: TILE, tileHeight: TILE })
    const tt = map.addTilesetImage('tt', 'tt', 16, 16, 0, 0, TT_GID)!
    const ru = map.addTilesetImage('ru', 'ru', 16, 16, 0, 0, RU_GID)!
    this.ground = map.createBlankLayer('ground', [tt, ru])!.setDepth(0)
    this.deco = map.createBlankLayer('deco', [tt, ru])!.setDepth(1)
    const at = (x: number, y: number) => (x < 0 || y < 0 || x >= MAP_W || y >= MAP_H ? ' ' : LAYOUT[y][x])
    const isPath = (x: number, y: number) => '=P'.includes(at(x, y)) || (at(x, y) === '_' && false)
    const rnd = new Phaser.Math.RandomDataGenerator(['lakeside'])
    for (let y = 0; y < MAP_H; y++) {
      for (let x = 0; x < MAP_W; x++) {
        const c = at(x, y)
        const grass = TT_GID + (rnd.frac() < 0.25 ? TT.grass[1] : TT.grass[0])
        if (c === 'M') continue
        if (c === '=') {
          const n = isPath(x, y - 1), s = isPath(x, y + 1), w = isPath(x - 1, y), e = isPath(x + 1, y)
          const P = TT.path
          const idx = !n && !w ? P.tl : !n && !e ? P.tr : !n ? P.t : !s && !w ? P.bl : !s && !e ? P.br : !s ? P.b
            : !w ? P.l : !e ? P.r : P.c
          this.ground.putTileAt(TT_GID + idx, x, y)
        } else if (c === '_') {
          this.ground.putTileAt(TT_GID + TT.path.c, x, y)
        } else if (c === '~') {
          this.ground.putTileAt(RU_GID + (at(x, y - 1) !== '~' && at(x, y - 1) !== 'P' ? RU.water.t : RU.water.c), x, y)
        } else if (c === 'P') {
          this.ground.putTileAt(RU_GID + (y === 23 ? RU.water.t : RU.water.c), x, y)
          this.deco.putTileAt(RU_GID + RU.planks, x, y)
        } else if (c === 'S' || c === 'B' || c === 'F') {
          this.ground.putTileAt(TT_GID + TT.stone, x, y)
        } else {
          this.ground.putTileAt(grass, x, y)
        }
        if (c === ',') this.ground.putTileAt(TT_GID + TT.flowers, x, y)
        if (c === 'T' || c === 't') {
          this.deco.putTileAt(TT_GID + (c === 'T' ? TT.tree : TT.pine), x, y)
          this.trees.push({ x, y, kind: c === 'T' ? 'tree' : 'pine' })
        }
        if (c === 'b') this.deco.putTileAt(TT_GID + TT.bush, x, y)
        if (c === 'B') this.deco.putTileAt(RU_GID + RU.bench, x, y)
        if (c === 'F') this.deco.putTileAt(RU_GID + RU.fountain, x, y)
        if (c === 'm') this.deco.putTileAt(RU_GID + RU.mailbox, x, y)
        if (c === 'L') {
          this.deco.putTileAt(RU_GID + RU.lampPost, x, y)
          this.add.image(x * TILE + 8, (y - 1) * TILE + 8, 'ru', RU.lampTop).setDepth(y + 0.5)
        }
      }
    }
  }

  private buildBuildings() {
    for (const b of BUILDINGS) this.drawBuilding(b)
  }

  private drawBuilding(b: Building) {
    const roof = b.roof === 'red' ? TT.roofRed : TT.roofGrey
    const wall = b.wall === 'wood' ? TT.wallWood : TT.wallStone
    for (let i = 0; i < b.w; i++) {
      const x = b.x + i
      const edge = i === 0 ? 'l' : i === b.w - 1 ? 'r' : 'm'
      this.deco.putTileAt(TT_GID + (edge === 'l' ? roof.tl : edge === 'r' ? roof.tr : roof.t), x, b.y)
      this.deco.putTileAt(TT_GID + (edge === 'l' ? roof.bl : edge === 'r' ? roof.br : roof.b), x, b.y + 1)
      const upper = edge === 'l' ? wall.l : edge === 'r' ? wall.r : i % 2 === 1 ? wall.window : wall.m
      this.deco.putTileAt(TT_GID + upper, x, b.y + 2)
      const lower = x === b.doorX ? wall.door : edge === 'l' ? wall.l : edge === 'r' ? wall.r : (i % 2 === 0 ? wall.window : wall.m)
      this.deco.putTileAt(TT_GID + lower, x, b.y + 3)
    }
    const sign = this.add.text((b.x + b.w / 2) * TILE, (b.y + 2) * TILE - 1, b.name, {
      fontFamily: FONT, fontSize: '27px', color: '#4a4063', backgroundColor: '#fff6e9',
      padding: { x: 9, y: 3 },
    }).setOrigin(0.5).setScale(1 / 3).setDepth(30).setResolution(2)
    sign.setStroke('#b8a4e3', 0)
    this.signs[b.place] = sign
    // A clickable invisible zone so the mouse can enter a building.
  }

  private drawWaterSparkles() {
    const rnd = new Phaser.Math.RandomDataGenerator(['water'])
    for (let i = 0; i < 26; i++) {
      const x = rnd.between(4, MAP_W * TILE - 8), y = rnd.between(25 * TILE, MAP_H * TILE - 4)
      const s = this.add.rectangle(x, y, rnd.between(3, 6), 1, 0xffffff, 0.8).setDepth(2)
      this.tweens.add({ targets: s, alpha: { from: 0, to: 0.9 }, x: x + 6, duration: rnd.between(1400, 2600),
        yoyo: true, repeat: -1, delay: rnd.between(0, 2000), ease: 'Sine.inOut' })
    }
    // A little sailing boat drifting on the lake (drawn with shapes).
    const boat = this.add.container(40, 26 * TILE).setDepth(3)
    const g = this.add.graphics()
    g.fillStyle(0xffffff).fillTriangle(0, -2, 0, -12, 7, -2)
    g.fillStyle(0xf7a8b8).fillTriangle(0, -3, 0, -10, -5, -3)
    g.fillStyle(0x8a6a5a).fillRect(-7, -2, 15, 3)
    boat.add(g)
    this.tweens.add({ targets: boat, x: MAP_W * TILE - 40, duration: 60000, yoyo: true, repeat: -1, ease: 'Sine.inOut' })
    this.tweens.add({ targets: boat, y: 26 * TILE + 2, duration: 1600, yoyo: true, repeat: -1, ease: 'Sine.inOut' })
  }

  // ------------------------------------------------------------------ player
  private spawnPlayer() {
    const home = BUILDINGS.find((b) => b.place === 'home')!
    const t = doorTrigger(home)
    this.player = this.add.container(t.x * TILE + 8, t.y * TILE + 8).setDepth(10)
    const shadow = this.add.ellipse(0, 7, 11, 4, 0x4a4063, 0.25)
    this.player.add(shadow)
    for (let i = 0; i < 5; i++) {
      const img = this.add.image(0, 0, 'ch', 0)
      this.parts.push(img)
      this.player.add(img)
    }
    // Thin ink outline so the player stands out on sandy paths (WebGL only; harmless elsewhere).
    if (this.game.renderer.type === Phaser.WEBGL) this.player.postFX?.addGlow(0x4a4063, 1.2, 0, false, 0.2, 6)
    this.queueText = this.add.text(0, -14, '', { fontFamily: FONT, fontSize: '24px' })
      .setOrigin(0.5, 1).setScale(1 / 3).setResolution(2)
    this.player.add(this.queueText)
  }

  private applyLook(l: Look) {
    const f = lookFrames(l)
    const frames = [f.body, f.legs, f.shoes, f.top, f.hair]
    frames.forEach((fr, i) => {
      this.parts[i].setVisible(fr !== null)
      if (fr !== null) this.parts[i].setFrame(fr)
    })
  }

  private wake() {
    const t = doorTrigger(BUILDINGS.find((b) => b.place === 'home')!)
    this.player.setPosition(t.x * TILE + 8, t.y * TILE + 8)
    this.path = []
    this.pathTarget = null
    this.near = null
    this.cameras.main.flash(500, 255, 246, 233)
  }

  private tileOf(x: number, y: number): Tile { return { x: Math.floor(x / TILE), y: Math.floor(y / TILE) } }
  private blocked(px: number, py: number) {
    // Feet box: 10 × 6 px at the bottom of the sprite.
    for (const [dx, dy] of [[-5, 2], [5, 2], [-5, 7], [5, 7]]) {
      const t = this.tileOf(px + dx, py + dy)
      if (t.x < 0 || t.y < 3 || t.x >= MAP_W || t.y >= MAP_H || this.grid[t.y][t.x]) return true
    }
    return false
  }

  private onClick(p: Pointer) {
    if (this.frozen) return
    const wp = this.cameras.main.getWorldPoint(p.x, p.y)
    const t = this.tileOf(wp.x, wp.y)
    const b = buildingAt(t.x, t.y)
    if (b) return this.walkToPlace(b.place)
    if (t.x < 0 || t.y < 0 || t.x >= MAP_W || t.y >= MAP_H || this.grid[t.y][t.x]) return
    this.findPath(t, null)
  }

  private walkToPlace(place: string) {
    const b = BUILDINGS.find((x) => x.place === place)
    if (!b) return
    const t = doorTrigger(b)
    const here = this.tileOf(this.player.x, this.player.y + 4)
    if (here.x === t.x && here.y === t.y) { bus.emit('door', { place }); return }
    this.findPath(t, place)
  }

  private findPath(to: Tile, place: string | null) {
    const from = this.tileOf(this.player.x, this.player.y + 4)
    this.star.findPath(from.x, from.y, to.x, to.y, (path) => {
      if (!path) return
      this.path = path.slice(1)
      this.pathTarget = place
    })
    this.star.calculate()
  }

  update(_t: number, dtMs: number) {
    const dt = Math.min(dtMs, 50) / 1000
    this.updateNpcs(dt)
    if (this.frozen) { this.idle(); return }
    const k = this.keys
    let vx = 0, vy = 0
    if (k.A.isDown || k.LEFT.isDown) vx -= 1
    if (k.D.isDown || k.RIGHT.isDown) vx += 1
    if (k.W.isDown || k.UP.isDown) vy -= 1
    if (k.S.isDown || k.DOWN.isDown) vy += 1
    if (vx || vy) { this.path = []; this.pathTarget = null }
    else if (this.path.length) {
      const n = this.path[0]
      const tx = n.x * TILE + 8, ty = n.y * TILE + 8
      const dx = tx - this.player.x, dy = ty - this.player.y
      if (Math.hypot(dx, dy) < 2) {
        this.player.setPosition(tx, ty)
        this.path.shift()
        if (!this.path.length && this.pathTarget) {
          const place = this.pathTarget
          this.pathTarget = null
          bus.emit('door', { place })
        }
      } else { vx = Math.sign(Math.round(dx)); vy = Math.sign(Math.round(dy)) }
    }
    if (vx || vy) {
      const len = Math.hypot(vx, vy)
      const step = SPEED * dt
      const nx = this.player.x + (vx / len) * step, ny = this.player.y + (vy / len) * step
      if (this.path.length) this.player.setPosition(nx, ny)
      else {
        if (!this.blocked(nx, this.player.y)) this.player.x = nx
        if (!this.blocked(this.player.x, ny)) this.player.y = ny
      }
      if (vx) this.parts.forEach((p) => p.setFlipX(vx < 0))
      this.walkT += dt
      const bob = Math.floor(this.walkT * 8) % 2
      this.parts.forEach((p) => p.setY(-bob))
    } else this.idle()
    this.player.setDepth(10 + this.player.y / 1000)

    const here = this.tileOf(this.player.x, this.player.y + 4)
    const door = BUILDINGS.find((b) => { const t = doorTrigger(b); return t.x === here.x && t.y === here.y })
    const place = door?.place ?? null
    if (place !== this.near) { this.near = place; bus.emit('near', { place }) }
    if (place && (Phaser.Input.Keyboard.JustDown(k.E) || Phaser.Input.Keyboard.JustDown(k.ENTER) ||
      Phaser.Input.Keyboard.JustDown(k.SPACE))) bus.emit('door', { place })
  }

  private idle() { this.parts.forEach((p) => p.setY(0)) }

  // ------------------------------------------------------------------ ambient townsfolk
  private spawnNpcs() {
    const spots: Tile[] = [{ x: 14, y: 10 }, { x: 30, y: 16 }, { x: 25, y: 19 }, { x: 6, y: 22 }, { x: 35, y: 21 }]
    spots.forEach((s, i) => {
      const char = [0, 1, 3, 5, 7][i]
      const sprite = this.add.sprite(s.x * TILE + 8, s.y * TILE + 8, 'ru', npcFrame(char, 1, 0)).setDepth(9)
      this.npcs.push({ sprite, char, path: [], facing: 1, t: 0, wait: 1 + i })
    })
  }

  private updateNpcs(dt: number) {
    for (const n of this.npcs) {
      if (!n.path.length) {
        n.wait -= dt
        n.sprite.setFrame(npcFrame(n.char, n.facing, 0))
        if (n.wait <= 0) {
          n.wait = 1.5 + Math.random() * 4
          const from = this.tileOf(n.sprite.x, n.sprite.y)
          let to: Tile
          do { to = { x: Phaser.Math.Between(1, MAP_W - 2), y: Phaser.Math.Between(9, 23) } }
          while (!'=_S'.includes(LAYOUT[to.y][to.x]) || this.grid[to.y][to.x])
          this.star.findPath(from.x, from.y, to.x, to.y, (p) => { if (p) n.path = p.slice(1, 14) })
          this.star.calculate()
        }
        continue
      }
      const nx = n.path[0].x * TILE + 8, ny = n.path[0].y * TILE + 8
      const dx = nx - n.sprite.x, dy = ny - n.sprite.y
      if (Math.hypot(dx, dy) < 1.5) { n.sprite.setPosition(nx, ny); n.path.shift(); continue }
      n.facing = Math.abs(dx) > Math.abs(dy) ? (dx < 0 ? 0 : 3) : dy < 0 ? 2 : 1
      const s = 28 * dt
      n.sprite.x += Math.sign(dx) * Math.min(s, Math.abs(dx))
      n.sprite.y += Math.sign(dy) * Math.min(s, Math.abs(dy))
      n.t += dt
      n.sprite.setFrame(npcFrame(n.char, n.facing, (Math.floor(n.t * 6) % 2) as 0 | 1))
      n.sprite.setDepth(9 + n.sprite.y / 1000)
    }
  }

  // ------------------------------------------------------------------ seasons & abroad
  private setSeason(month: number) {
    const m = month % 12 // month 0 = January
    const autumn = m >= 8 && m <= 10
    for (const t of this.trees) {
      const idx = t.kind === 'tree' ? (autumn ? TT.treeAutumn : TT.tree) : (autumn ? TT.pineAutumn : TT.pine)
      this.deco.putTileAt(TT_GID + idx, t.x, t.y)
    }
  }

  private setAbroad(abroad: boolean) {
    const st = this.signs.station
    if (st) st.setText(abroad ? 'BACK TO SWITZERLAND' : 'STATION')
    for (const [place, s] of Object.entries(this.signs)) {
      if (place === 'station') continue
      const b = BUILDINGS.find((x) => x.place === place)!
      s.setText(abroad && place === 'home' ? 'HOME (ABROAD)' : b.name)
    }
  }
}

type Pointer = Phaser.Input.Pointer
