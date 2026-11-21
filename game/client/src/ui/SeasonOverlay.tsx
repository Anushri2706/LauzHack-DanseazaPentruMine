/** Cheap seasons: a soft colour wash over the town by month, snow in winter, a warmer palette abroad. */
export function SeasonOverlay({ month, abroad }: { month: number; abroad: boolean }) {
  const m = month % 12
  const season = abroad ? 'abroad' : m <= 1 || m === 11 ? 'winter' : m <= 4 ? 'spring' : m <= 7 ? 'summer' : 'autumn'
  const label: Record<string, string> = { abroad: '✈️ Abroad: a new town, new prices', winter: '❄️ Winter', spring: '🌱 Spring', summer: '☀️ Summer', autumn: '🍂 Autumn' }
  return (
    <>
      <div className={`season season-${season}`} aria-hidden />
      {season === 'winter' && <div className="snow" aria-hidden />}
      <span className="season-label px-box tiny">{label[season]}</span>
    </>
  )
}
