# April 1 AI Team Deliverables

This package turns the current repo into a working April 1 deliverable set for the three inferred focus areas:
- AI activity
- private credit activity
- risk resilience / credit quality

## What Was Generated

- Current bank-level ratings across the three workstreams
- Quarterly ratings for 2024_Q1 through 2025_Q4
- Cluster assignments for each workstream
- A transparent risk workbook with the Risk Resilience Index inputs and FFIEC-based DPD metrics
- A manual Gemini / ChatGPT packet for the external-search task that still sits outside the local pipeline

## Top AI Activity Banks
- COF: Capital One Financial Corp (score 95.0, rating 5)
- SOFI: SoFi Technologies, Inc. (score 93.4, rating 5)
- C: Citigroup Inc (score 89.9, rating 5)
- PNC: Pnc Financial Services Group, Inc. (score 83.1, rating 5)
- AXP: American Express Co (score 83.0, rating 5)
- JPM: Jpmorgan Chase & Co (score 79.5, rating 4)
- CFG: Citizens Financial Group Inc/Ri (score 79.4, rating 4)
- MS: Morgan Stanley (score 79.1, rating 4)
- SSB: SouthState Bank Corp (score 76.6, rating 4)
- KEY: Keycorp (score 71.3, rating 4)

## Top Private Credit Banks
- STT: State Street Corp (score 95.6, rating 5)
- MS: Morgan Stanley (score 92.7, rating 5)
- CFG: Citizens Financial Group Inc/Ri (score 86.3, rating 5)
- WFC: Wells Fargo & Company/Mn (score 84.1, rating 5)
- PNC: Pnc Financial Services Group, Inc. (score 83.5, rating 5)
- JPM: Jpmorgan Chase & Co (score 77.0, rating 4)
- KEY: Keycorp (score 70.4, rating 4)
- RJF: Raymond James Financial Inc (score 70.0, rating 4)
- BKU: BankUnited, Inc. (score 69.9, rating 4)
- FCNCA: First Citizens Bancshares Inc (score 69.2, rating 4)

## Top Risk Resilience Banks
- SOFI: SoFi Technologies, Inc. (score 90.2, rating 5)
- WTFC: Wintrust Financial Corp (score 82.4, rating 5)
- EWBC: East West Bancorp Inc (score 72.5, rating 4)
- SNV: Synovus Financial Corp (score 68.5, rating 4)
- SYF: Synchrony Financial (score 68.0, rating 4)
- SF: Stifel Financial Corp (score 65.9, rating 4)
- FNB: Fnb Corp (score 65.8, rating 4)
- AXP: American Express Co (score 65.0, rating 4)
- RJF: Raymond James Financial Inc (score 64.5, rating 4)
- BK: Bank of New York Mellon Corp (score 63.7, rating 4)

## Cluster Counts
- ai_activity: AI Active (21)
- ai_activity: AI Watchlist (21)
- ai_activity: AI Leaders (8)
- private_credit: Private Credit Active (23)
- private_credit: Private Credit Limited (20)
- private_credit: Private Credit Leaders (7)
- risk_resilience: Risk Resilient (35)
- risk_resilience: Risk Watch (13)
- risk_resilience: Risk Moderate (2)

## Driver Snapshots
## Ai Activity
- operations_efficiency: correlation 0.81
- measurable_outcomes: correlation 0.81
- investment_spend: correlation 0.80
- use_cases: correlation 0.77
- governance: correlation 0.77

## Private Credit
- private_credit: correlation 0.50
- direct_lending: correlation 0.50
- fund_finance: correlation 0.41
- private_capital: correlation 0.30
- asset_based_lending: correlation 0.17

## Risk Resilience
- ROA: correlation 0.64
- ROE: correlation 0.50
- NIMY: correlation 0.33
- equity_assets_ratio: correlation 0.21
- nondepository_requested_rate: correlation -0.11

## Key Caveats

- The three workstreams are inferred from the project context and meeting notes; they are not explicitly codified elsewhere in the repo.
- The commercial-loan 90+ DPD metric follows the requested FFIEC call-report formula with `RCFD/RCON` fallback for banks where one branch is blank.
- The requested nondepository formula was computed exactly as provided (`RCFDPV25 / RCFDJ454`, with `RCON` fallback), but the FFIEC field labels suggest the numerator should be validated before making a final presentation claim that it represents a full nondepository 90+ DPD rate.
- A few banks have latest risk data on a slightly different as-of quarter in the FDIC API; the current risk sheet includes the exact quarter used per bank.