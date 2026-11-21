"""Deterministic simulation engine. Imports nothing from FastAPI, HTTP or LLM code.

Modules
  types        Profile, Intervention, RegimePaths, STATES (migrated unchanged)
  transitions  sourced monthly hazards + Dirichlet uncertainty (HMM transition prior)
  regime       HMM emissions, forward filter, regime path sampler (RegimeEngine)
  income       gross pay, unemployment insurance, housing          tax  gross -> net
  accounting   cash paths, one-offs, reconciliation                 wealth  inflation, portfolio, debt
  assumptions  provenance registry                                  decisions  typed decision catalogue
  events       independent life events (not HMM-owned risks)        simulate  orchestration
  montecarlo   P10/P50/P90 result contract                          metrics  metric definitions
  sensitivity  one-at-a-time +/-10% and break-even bisection
"""
