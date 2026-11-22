import { useState } from 'react'
import { setStore, type AppId } from '../../store/game'
import { sfx } from '../sound'

/** Short money courses: general education, not financial advice. Worked examples are plain arithmetic. */
interface Course {
  id: string
  icon: string
  title: string
  minutes: number
  intro: string
  sections: { heading: string; text: string[] }[]
  takeaways: string[]
  tryIt?: { text: string; app: AppId }
}

const COURSES: Course[] = [
  {
    id: 'inflation', icon: '🎈', title: 'What is inflation?', minutes: 3,
    intro: 'Inflation means prices in general go up over time, so the same amount of money buys a little less each year.',
    sections: [
      { heading: 'The idea', text: [
        'Imagine a basket with your usual shopping: bread, rent, a bus ticket, a haircut. If that basket costs CHF 100 this year and CHF 102 next year, inflation was 2%.',
        'Your CHF 100 note did not change, but what it can buy did. That is called purchasing power.',
      ] },
      { heading: 'A worked example', text: [
        'At 2% inflation a year, the CHF 100 basket costs about CHF 110 after 5 years and about CHF 122 after 10 years, because each year’s rise is added on top of the last one.',
        'So CHF 1,000 kept as cash for 10 years still says CHF 1,000, but it buys only what about CHF 820 buys today.',
        'A quick rule: divide 72 by the inflation rate to see roughly how many years it takes for prices to double. At 2% that is about 36 years; at 6% only about 12.',
      ] },
      { heading: 'Why it happens', text: [
        'Prices rise when people want to buy more than can be produced, when costs like energy or wages go up, or when there is a lot of money in the economy.',
        'Central banks try to keep inflation low and stable. The Swiss National Bank aims for prices rising by less than 2% a year.',
      ] },
      { heading: 'What it means for you', text: [
        'Salaries often rise with inflation over time, but not always and not right away.',
        'Money in a savings account only keeps its value if the interest is at least as high as inflation. Investments can grow faster than prices, but their value goes up and down.',
        'Debt works the other way: inflation makes a fixed loan a bit lighter over the years.',
      ] },
    ],
    takeaways: ['Inflation = prices rising, so money buys less.', 'Small yearly rates add up over many years.', 'Compare any interest or return with inflation to see if you are really ahead.'],
    tryIt: { text: 'Change “Inflation on costs” in Settings and watch your costs on the timeline.', app: 'settings' },
  },
  {
    id: 'compound', icon: '🌱', title: 'Compound growth', minutes: 2,
    intro: 'When your money earns a return, and that return earns a return too, growth speeds up over time. That is compounding.',
    sections: [
      { heading: 'A worked example', text: [
        'CHF 10,000 growing at 5% a year is about CHF 16,300 after 10 years and about CHF 43,200 after 30 years.',
        'In the last 10 of those 30 years it grows more than in the first 20 together. Time does most of the work.',
      ] },
      { heading: 'The catch', text: [
        'Real investments do not grow by the same amount every year. Some years they fall. The 5% is an average, not a promise.',
        'Compounding also works against you with debt: unpaid interest is added to what you owe and then charges interest itself.',
      ] },
    ],
    takeaways: ['Starting early matters more than starting big.', 'Returns are uncertain: look at ranges, not one number.'],
    tryIt: { text: 'Set a savings rule at the Bank, then see the range of futures in the Money app.', app: 'money' },
  },
  {
    id: 'buffer', icon: '🛟', title: 'Emergency fund', minutes: 2,
    intro: 'An emergency fund is cash you keep for surprises: losing a job, a big bill, a broken laptop.',
    sections: [
      { heading: 'How much?', text: [
        'A common guideline is 3 to 6 months of your usual costs. Someone who spends CHF 2,500 a month would keep CHF 7,500 to 15,000.',
        'It sits in an account you can reach quickly, not in investments that might be down just when you need the money.',
      ] },
      { heading: 'Why it helps', text: [
        'Without a buffer, a surprise often means borrowing, and borrowing costs interest.',
        'With one, a bad month stays a bad month instead of becoming a debt problem.',
      ] },
    ],
    takeaways: ['Build the buffer first, then invest what is above it.', 'Refill it after you use it.'],
    tryIt: { text: 'The Safety score shows whether your cash stays above your buffer.', app: 'money' },
  },
  {
    id: 'debt', icon: '🏦', title: 'Loans and interest', minutes: 2,
    intro: 'A loan gives you money now in exchange for paying back more later. The extra is the interest.',
    sections: [
      { heading: 'What a loan really costs', text: [
        'Borrowing CHF 10,000 at 8% over 3 years means paying about CHF 313 a month, around CHF 1,280 in interest in total.',
        'In Switzerland the law caps interest on consumer loans at 10% a year and on credit-card balances at 12%. Those are still expensive rates.',
      ] },
      { heading: 'Good questions to ask', text: [
        'Can I afford the monthly payment if my income drops?',
        'Is this for something that lasts longer than the loan does?',
        'What is the total I will pay back, not only the monthly amount?',
      ] },
    ],
    takeaways: ['Look at the total cost, not just the monthly payment.', 'Expensive debt is usually the first thing worth paying off.'],
    tryIt: { text: 'The Bank shows the monthly payment and total interest before you borrow.', app: 'money' },
  },
  {
    id: 'pension', icon: '🏛️', title: 'Swiss pensions: the 3 pillars', minutes: 3,
    intro: 'Switzerland’s retirement system rests on three pillars that together should replace part of your salary when you stop working.',
    sections: [
      { heading: 'The pillars', text: [
        '1st pillar (AHV/AVS): the state pension. Almost everyone pays in through their salary. It covers basic needs.',
        '2nd pillar (BVG/LPP): the workplace pension fund. You and your employer pay in once you earn above a minimum salary.',
        '3rd pillar: your own voluntary savings. Pillar 3a has tax advantages and a yearly maximum set by law.',
      ] },
      { heading: 'Why it matters early', text: [
        'Gaps add up: years abroad, part-time work or long breaks can mean a smaller pension later.',
        'Together, the 1st and 2nd pillars are meant to replace roughly 60% of your last salary for a typical income, often less for higher incomes.',
      ] },
    ],
    takeaways: ['Your salary already pays into pillars 1 and 2.', 'Pillar 3a is the part you choose yourself.'],
    tryIt: { text: 'Live to 65 on the timeline and read the retirement letter in Mail.', app: 'mail' },
  },
]

export function LearnApp() {
  const [open, setOpen] = useState<string | null>(null)
  const course = COURSES.find((c) => c.id === open)
  const go = (id: string | null) => { sfx('click'); setOpen(id) }

  if (course) {
    return (
      <article className="course">
        <button className="link" onClick={() => go(null)}>← All courses</button>
        <h3>{course.icon} {course.title}</h3>
        <p className="tiny muted">About {course.minutes} min read</p>
        <p className="course-intro">{course.intro}</p>
        {course.sections.map((s) => (
          <section key={s.heading}>
            <h4>{s.heading}</h4>
            {s.text.map((t, i) => <p key={i} className="small">{t}</p>)}
          </section>
        ))}
        <div className="course-box">
          <h4>✏️ Remember</h4>
          <ul className="small">{course.takeaways.map((t) => <li key={t}>{t}</li>)}</ul>
        </div>
        {course.tryIt && (
          <p className="small">🎮 <b>Try it:</b> {course.tryIt.text}{' '}
            <button className="link" onClick={() => { sfx('click'); setStore({ phone: course.tryIt!.app }) }}>Open →</button></p>
        )}
        <p className="tiny muted">General education, not financial advice.</p>
      </article>
    )
  }

  return (
    <div>
      <h3>📚 Learn</h3>
      <p className="tiny muted">Short reads on the money ideas behind the game. General education, not financial advice.</p>
      <ul className="courses">
        {COURSES.map((c) => (
          <li key={c.id}>
            <button className="course-card" onClick={() => go(c.id)}>
              <span className="course-ico">{c.icon}</span>
              <span><b>{c.title}</b><span className="tiny muted"> · {c.minutes} min</span><br /><span className="small">{c.intro}</span></span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
