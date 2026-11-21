# Financial Twin

Explore how a life decision could change a fictional person's financial path. Edit the starting snapshot and assumptions, then compare a life event with the current path. The app runs entirely in the browser; it does not save or send the values you enter.

## Run locally

Requires Node.js 18 or newer. No packages need to be installed.

```sh
npm start
```

Open [http://127.0.0.1:3000](http://127.0.0.1:3000). Run the simulation checks with `npm test`.

## How it works

Both paths begin with the same savings, annual take-home income, and monthly living expenses. Each month, the model applies the annual balance growth rate (compounded monthly) to the existing balance, then adds income and subtracts expenses. Income and living expenses grow with their respective annual rates. The selected event modifies these cash flows starting in the first month; its specific assumptions are shown in the app. The chart samples the balance at the start and end of each year, while the funding-gap summary checks every month.

The result is a **liquid balance**, not total net worth. Negative balances represent funding gaps and use the same growth rate as positive balances; the model does not arrange financing. Home equity, property value, debt repayment, taxes beyond take-home income, and other real-world factors are excluded. The assumptions are illustrative, not forecasts or financial advice. Actual outcomes can differ substantially.