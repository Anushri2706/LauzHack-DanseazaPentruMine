export const scenarios = {
  masters: {
    label: "Pursue a master's",
    shortLabel: "Master's degree",
    description: "Study for two years, then return to work with a higher salary.",
    assumption: "For two years, income falls to 35% and tuition costs €12,000 per year. Afterwards, income is 25% higher than the baseline.",
  },
  relocate: {
    label: "Relocate for work",
    shortLabel: "Relocation",
    description: "Move for a better-paying role in a more expensive city.",
    assumption: "Pay €6,000 to move upfront. Income rises by 15% and living expenses rise by 25%.",
  },
  business: {
    label: "Start a business",
    shortLabel: "New business",
    description: "Trade early stability for a potential increase in income.",
    assumption: "Pay €15,000 upfront and earn 50% of baseline income in year one. Income is 40% higher thereafter.",
  },
  home: {
    label: "Buy a home",
    shortLabel: "Home purchase",
    description: "Use savings for an upfront payment and take on higher monthly costs.",
    assumption: "Pay €30,000 upfront and spend €400 more per month. Home equity and property value are not included.",
  },
  break: {
    label: "Take a career break",
    shortLabel: "Career break",
    description: "Take a year away from work, then return at baseline pay.",
    assumption: "Earn no income for the first 12 months, then return to baseline income.",
  },
  shock: {
    label: "Face an economic shock",
    shortLabel: "Economic shock",
    description: "Explore a sudden interruption to income and an unexpected cost.",
    assumption: "Earn no income for six months and pay an unexpected €5,000 cost upfront.",
  },
};

// Annual rates compound monthly; a negative balance represents unfunded debt.
export function simulate(inputs, scenario = null) {
  const { savings, income, expenses, incomeGrowth, inflation, returnRate, years } = inputs;
  let balance = savings;
  let minimum = savings;
  let monthsBelowZero = 0;
  const points = [balance];
  const monthlyReturn = Math.pow(1 + returnRate / 100, 1 / 12) - 1;

  for (let month = 0; month < years * 12; month++) {
    const annualIncome = income * Math.pow(1 + incomeGrowth / 100, month / 12);
    const monthlyExpenses = expenses * Math.pow(1 + inflation / 100, month / 12);
    let incomeMultiplier = 1;
    let extraExpenses = 0;

    if (scenario === "masters") {
      incomeMultiplier = month < 24 ? 0.35 : 1.25;
      if (month < 24) extraExpenses += 1000;
    } else if (scenario === "relocate") {
      incomeMultiplier = 1.15;
      extraExpenses += monthlyExpenses * 0.25;
      if (month === 0) extraExpenses += 6000;
    } else if (scenario === "business") {
      incomeMultiplier = month < 12 ? 0.5 : 1.4;
      if (month === 0) extraExpenses += 15000;
    } else if (scenario === "home") {
      extraExpenses += 400;
      if (month === 0) extraExpenses += 30000;
    } else if (scenario === "break") {
      if (month < 12) incomeMultiplier = 0;
    } else if (scenario === "shock") {
      if (month < 6) incomeMultiplier = 0;
      if (month === 0) extraExpenses += 5000;
    }

    balance = balance * (1 + monthlyReturn) + annualIncome * incomeMultiplier / 12 - monthlyExpenses - extraExpenses;
    minimum = Math.min(minimum, balance);
    if (balance < 0) monthsBelowZero++;
    if ((month + 1) % 12 === 0) points.push(balance);
  }

  return { points, final: balance, minimum, monthsBelowZero };
}
