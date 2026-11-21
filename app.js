import { scenarios, simulate } from "./simulation.js";

const form = document.querySelector("#inputs");
const options = document.querySelector("#scenario-options");
const chart = document.querySelector("#chart");
const money = new Intl.NumberFormat("en-IE", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
let selected = "masters";

for (const [key, scenario] of Object.entries(scenarios)) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "scenario-button";
  button.dataset.scenario = key;
  button.textContent = scenario.label;
  button.addEventListener("click", () => {
    selected = key;
    render();
  });
  options.append(button);
}

function svgElement(name, attributes, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}

function drawChart(baseline, alternative, years) {
  chart.querySelectorAll(".plot").forEach(element => element.remove());
  const all = [...baseline.points, ...alternative.points, 0];
  let min = Math.min(...all);
  let max = Math.max(...all);
  if (max === min) max = min + 1;
  const padding = (max - min) * 0.1;
  min -= padding;
  max += padding;
  const x = index => 72 + index / years * 625;
  const y = value => 18 + (max - value) / (max - min) * 190;
  const add = (name, attrs, text) => chart.append(svgElement(name, { class: "plot", ...attrs }, text));

  for (let i = 0; i <= 4; i++) {
    const value = max - (max - min) * i / 4;
    const yy = y(value);
    add("line", { x1: 72, x2: 697, y1: yy, y2: yy, stroke: "#e8ebe4" });
    add("text", { x: 62, y: yy + 4, "text-anchor": "end", fill: "#858c81", "font-size": 11 }, new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value));
  }
  for (const index of [...new Set([0, Math.round(years / 2), years])]) {
    add("text", { x: x(index), y: 235, "text-anchor": "middle", fill: "#858c81", "font-size": 11 }, index === 0 ? "Today" : `Year ${index}`);
  }
  for (const [result, color, dash] of [[baseline, "#a9b5a4", "5 5"], [alternative, "#48765b", ""]]) {
    const points = result.points.map((value, index) => `${x(index)},${y(value)}`).join(" ");
    add("polyline", { points, fill: "none", stroke: color, "stroke-width": 3, "stroke-linecap": "round", "stroke-linejoin": "round", "stroke-dasharray": dash });
    add("circle", { cx: x(years), cy: y(result.final), r: 4, fill: color });
  }
  document.querySelector("#chart-desc").textContent = `After ${years} years, the current path ends at ${money.format(baseline.final)} and ${scenarios[selected].shortLabel} ends at ${money.format(alternative.final)}.`;
}

function render() {
  for (const button of options.children) {
    const active = button.dataset.scenario === selected;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  const scenario = scenarios[selected];
  document.querySelector("#scenario-assumption").textContent = scenario.assumption;
  document.querySelector("#legend-scenario").textContent = scenario.shortLabel;

  const valid = [...form.elements].every(input => input.checkValidity());
  document.querySelector("#input-error").hidden = valid;
  document.querySelector(".results-panel").hidden = !valid;
  if (!valid) return;
  const inputs = Object.fromEntries([...form.elements].map(input => [input.name, Number(input.value)]));
  const baseline = simulate(inputs);
  const alternative = simulate(inputs, selected);
  const difference = alternative.final - baseline.final;
  document.querySelector("#baseline-final").textContent = money.format(baseline.final);
  document.querySelector("#scenario-final").textContent = money.format(alternative.final);
  document.querySelector("#difference").textContent = `${difference > 0 ? "+" : ""}${money.format(difference)}`;
  document.querySelector("#insight-text").textContent = `${scenario.description} After ${inputs.years} ${inputs.years === 1 ? "year" : "years"}, this path ends ${money.format(Math.abs(difference))} ${difference >= 0 ? "above" : "below"} the current path under these assumptions.`;
  const risk = alternative.monthsBelowZero;
  const baselineRisk = baseline.monthsBelowZero;
  document.querySelector("#risk-text").textContent = (risk
    ? `This path spends ${risk} ${risk === 1 ? "month" : "months"} below zero, reaching a low of ${money.format(alternative.minimum)}. This gap would need funding.`
    : `This path stays at or above zero, with a low of ${money.format(alternative.minimum)}.`)
    + ` The current path spends ${baselineRisk} ${baselineRisk === 1 ? "month" : "months"} below zero, with a low of ${money.format(baseline.minimum)}.`;
  drawChart(baseline, alternative, inputs.years);
}

form.addEventListener("input", render);
render();
