# Changes

## 0.2.0

- Verify expected artifacts, nonempty content, Markdown headings, research
  evidence, and citation provenance before reporting completion.
- Execute saved plans, resume version 2 checkpoints, reconcile interrupted file
  writes using hashes, and prevent concurrent execution of the same run.
- Capture usage/latency, preserve uncertain request reservations, and enforce
  token, active-time, tool, turn, step, and estimated-cost budgets.
- Add workspace reads, Tavily page extraction, typed failures, transient retries,
  and bounded repair of unfinished plans.
- Add 16 reproducible offline evaluations and three opt-in live smoke tests.
- Add a token-authenticated local UI with plan preview, history, progress,
  cancellation, resume, verification, and artifact downloads.
- Expand cross-platform CI, documentation, and regression tests.

## 0.1.0

- Initial sequential planner/executor, web search and file tools, CLI,
  deterministic demo, and offline unit tests.
