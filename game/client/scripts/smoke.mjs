// Headless smoke test of the main flow. Needs backend (:8000) and `npm run dev` (:5173) running.
// Uses the locally installed Google Chrome through playwright-core (no browser download).
//   node scripts/smoke.mjs [--shots dir]
import { chromium } from 'playwright-core'

const BASE = process.env.BASE ?? 'http://localhost:5173'
const CHROME = process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const shotsAt = process.argv.includes('--shots') ? process.argv[process.argv.indexOf('--shots') + 1] : null
const errors = []
const step = (m) => console.log(`✓ ${m}`)

const browser = await chromium.launch({ executablePath: CHROME, headless: true })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`))
page.on('console', (m) => { if (m.type() === 'error') errors.push(`console: ${m.text()}`) })
const shot = async (name) => { if (shotsAt) await page.screenshot({ path: `${shotsAt}/${name}.png` }) }
const door = (place) => page.evaluate((p) => window.__bus.emit('door', { place: p }), place)
const hud = () => page.locator('.hud-date').innerText()

try {
  await page.goto(`${BASE}/?fresh&world=9`)
  await page.locator('.persona-card.featured').click()
  step('picked Alexia')
  await page.getByRole('button', { name: 'Surprise me' }).click()
  await shot('01-creator')
  await page.getByRole('button', { name: /Walk into town/ }).click()
  await page.locator('.phaser-host canvas').waitFor()
  await page.waitForFunction(() => document.querySelector('.hud-date')?.textContent?.includes('Jan 2025'))
  await page.waitForTimeout(1200)
  await shot('02-town')
  step(`town loaded: ${await hud()}`)

  await door('university')
  await page.getByRole('button', { name: /Full-time master's/ }).click()
  await page.locator('.act.queued').first().waitFor()
  await shot('03-university')
  step('queued full-time master\'s (engine accepted)')
  await page.keyboard.press('Escape')

  await door('cafe')
  await page.getByRole('button', { name: /Eating out: 2× per week/ }).click()
  await page.locator('.act.queued').first().waitFor()
  step('queued eating out 2×/week')
  await page.keyboard.press('Escape')

  await door('realestate')
  const studio = page.locator('.act', { hasText: 'Buy: Studio' })
  if (!(await studio.locator('.act-reason').innerText()).includes('down payment')) throw new Error('studio should be locked')
  step('studio greyed out with the engine reason (down payment)')
  await page.keyboard.press('Escape')

  await door('home')
  await page.getByRole('button', { name: /Sleep → next month/ }).click()
  await page.waitForFunction(() => document.querySelector('.hud-date')?.textContent?.includes('Feb 2025'))
  step(`slept one month: ${await hud()} · ${await page.locator('.hud-cash').innerText()}`)
  await page.waitForTimeout(600)
  await shot('04-after-sleep')

  await door('home')
  await page.getByRole('button', { name: /Sleep until next birthday/ }).click()
  await page.waitForFunction(() => document.querySelector('.hud-date')?.textContent?.includes('Jan 2026'))
  step(`slept to the birthday: ${await hud()} · ${await page.locator('.hud-cash').innerText()}`)
  await door('bank')
  await page.locator('.statement').waitFor()
  await shot('05-bank')
  step('bank statement shown')
  await page.keyboard.press('Escape')

  // Phone: every app opens
  await page.keyboard.press('p')
  await page.locator('.phone').waitFor()
  await page.locator('.tile').first().waitFor()
  await page.waitForFunction(() => document.querySelector('.tile .t-main')?.textContent?.includes('–'), null, { timeout: 20000 })
  await shot('06-phone-money')
  step(`phone · Money: ${await page.locator('.tile .t-main').first().innerText()}`)
  await page.getByRole('button', { name: /Roll another world/ }).click()
  await page.locator('.note', { hasText: 'same choices, different luck' }).waitFor()
  step('rolled another world')
  for (const app of ['Goals', 'Settings', 'Mail', 'Checks']) {
    await page.locator('.phone-nav button', { hasText: app }).click()
    await page.waitForTimeout(400)
    await shot(`07-${app.toLowerCase()}`)
  }
  await page.getByRole('button', { name: /Run live checks/ }).click()
  await page.locator('.checks li').first().waitFor()
  const failed = await page.locator('.checks li', { hasText: '❌' }).count()
  if (failed) throw new Error(`${failed} live checks failed`)
  step(`checks: ${await page.locator('.phone-screen p b').first().innerText()}`)

  // Multiverse: go back to the start and choose to work instead
  await page.locator('.phone-nav button', { hasText: 'Multiverse' }).click()
  await page.locator('.mv-node').nth(1).locator('rect').last().click()
  await page.getByRole('button', { name: /Go back/ }).click()
  await page.waitForFunction(() => document.querySelector('.hud-date')?.textContent?.includes('Jan 2025'))
  step('went back to Jan 2025 in the multiverse')
  await door('office')
  await page.getByRole('button', { name: /Start working in Switzerland/ }).click()
  await page.locator('.act.queued').first().waitFor()
  await page.keyboard.press('Escape')
  let bought = false
  for (let i = 0; i < 9 && !bought; i++) {
    await door('home')
    await page.getByRole('button', { name: /Sleep until next birthday/ }).click()
    await page.waitForFunction(() => !document.querySelector('.dialog'))
    await page.waitForTimeout(300)
    await door('realestate')
    const studio = page.locator('.act', { hasText: 'Buy: Studio' }).locator('button')
    if (await studio.isEnabled()) { await studio.click(); await page.locator('.act.queued').waitFor(); bought = true }
    await page.keyboard.press('Escape')
  }
  if (!bought) throw new Error('studio never became affordable')
  step(`bought the studio at ${await hud()}`)
  await door('home')
  await page.getByRole('button', { name: /Sleep → next month/ }).click()
  await page.waitForFunction(() => !document.querySelector('.dialog'))
  await page.waitForTimeout(1500)
  await shot('08-home-owner')

  // Compare the two realities
  await page.keyboard.press('p')
  await page.locator('.phone-nav button', { hasText: 'Multiverse' }).click()
  const pins = page.locator('.reality-list li')
  await pins.nth(0).locator('.pin', { hasText: 'A' }).click()
  await pins.nth(1).locator('.pin', { hasText: 'B' }).click()
  await page.locator('.cmp-table').waitFor({ timeout: 30000 })
  await page.waitForTimeout(500)
  await shot('09-compare')
  step(`compare: ${(await page.locator('.explain li').first().innerText()).slice(0, 90)}…`)
  await page.keyboard.press('Escape')

  // Fast-forward to the end: the postcard
  for (let i = 0; i < 12; i++) {
    if (await page.locator('.postcard').count()) break
    await door('home')
    const b = page.getByRole('button', { name: /Sleep until next birthday/ })
    if (!(await b.count())) break
    await b.click()
    await page.waitForFunction(() => !document.querySelector('.dialog-head'))
    await page.waitForTimeout(400)
  }
  await page.locator('.postcard').waitFor({ timeout: 20000 })
  await page.waitForTimeout(1500)
  await shot('10-postcard')
  step('ten years: postcard shown')
} catch (e) {
  errors.push(`step failed: ${e.message.split('\n')[0]}`)
  await shot('zz-failure')
}
await browser.close()
if (errors.length) { console.log('\n✗ Problems:\n' + errors.map((e) => '  ' + e).join('\n')); process.exit(1) }
console.log('\nSmoke test passed.')
