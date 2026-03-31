# Manual Gemini / ChatGPT Search Prompt

Upload the files listed in `manual_ai_upload_manifest.csv`, then use this prompt:

```text
You are helping extend a bank-capstone analysis that already has local ratings and clustering for three workstreams:
1. AI activity
2. private credit activity
3. risk resilience

Use the uploaded files as the current local baseline. Then search beyond the uploaded local corpus and identify only incremental evidence that materially changes or sharpens the conclusions.

Rules:
- Prioritize official bank sources, investor presentations, earnings-call transcripts, SEC filings, and reputable financial reporting.
- Focus first on the banks below because local coverage is thinner or mixed there.
- For each bank, flag whether new external evidence changes the AI, private-credit, or risk-resilience view.
- Give exact source links and publication dates.
- Do not restate what is already in the uploaded files unless it is needed to explain a delta.
- If no incremental value exists for a bank, say so briefly.

Priority banks:
- CFG: Citizens Financial Group Inc/Ri
- RJF: Raymond James Financial Inc
- SOFI: SoFi Technologies, Inc.
- SSB: SouthState Bank Corp
- WAL: Western Alliance Bancorporation
- STT: State Street Corp
- AXP: American Express Co
- ONB: Old National Bancorp
- WBS: Webster Financial Corp
- KEY: Keycorp
- ALLY: Ally Financial Inc.
- PNC: Pnc Financial Services Group, Inc.

Output format:
1. Bank
2. Workstream affected
3. New evidence summary
4. Why it changes or does not change the local rating
5. Source link
```

Note: the local workbook includes the requested commercial-loan 90+ DPD formula. The requested nondepository formula was also computed as provided, but it should be field-validated before any final presentation claims.