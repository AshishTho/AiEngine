# Architecture

The application keeps decisions in the LLM adapter and enforcement in Python.
No model response can directly write a file or call an arbitrary URL.

## Execution contract

`Plan` describes ordered steps, whether the goal is supported, expected artifacts
(path, Markdown headings, source count), and minimum research evidence.
`Backend.plan` produces that schema. `Backend.respond` returns Responses-shaped
output items. The runner retains all items, including encrypted reasoning state
when returned, and sends each observation with the matching tool-call ID.

Only one function call is accepted per turn, and only the current step's tool is
allowed. `finish_step` is a control tool. A successful action is required before
a non-reasoning step can finish. Permanent credential failures stop immediately.
A bounded repair call can replace remaining steps, preserving expected outputs.
The original plan and recovery history remain available for inspection.

After all steps finish, deterministic verification checks readable nonempty
files, expected Markdown headings, minimum evidence, and retrieved citation URLs.
A model saying "done" cannot override a failed check. A URL match is provenance,
not semantic entailment. The next research-quality improvement is a separately
evaluated claim/source entailment grader, rather than relaxing these checks.

## Persistence and recovery

Atomic JSON replacement persists checkpoints. An OS file lock serializes access
to a run and is released automatically when the process exits. Conversation
history, pending calls, completed observations, per-step turn counts, and budget
usage survive process restarts.

Before file creation, the intended function call is saved. If no file existed,
the write intent is journaled. On resume, a matching file hash is accepted as the
result of an interrupted write; conflicting content is refused. Previously
completed artifact hashes are also verified. Search/extraction are read-only
and may repeat after an ambiguous interruption. Provider-side billing is not
transactional with local checkpoints.

Requests reserve estimated tokens/cost before sending. Reported usage replaces
that reservation. A crash or missing usage leaves an explicit unconfirmed
reservation, preventing resume from silently forgetting potential spend.
These are conservative local accounting controls, not a provider billing ledger.

## User interface

`RunService` runs jobs in background threads with per-run directories and a local
concurrency quota. The browser polls a redacted state view; raw conversation and
pending arguments stay on disk. The server validates authentication, Host, and
Origin, and files download as attachments. Generated HTML is never evaluated in
the interface. Resume and execute-plan share the same runner used by the CLI.

## Add a tool or provider

Add a strict schema, input validation, bounded handler, and typed failure to
`ToolRegistry`. Add the tool name to `Step.tool`, planner instructions, and tests.
For a new side-effecting tool, design a recovery journal before exposing it to
the model. File-create reconciliation is not a generic transaction mechanism.

An alternative provider implements `plan` and `respond`, optionally `repair`,
`estimate_request`, `configure_runtime`, and per-response `last_usage`. It must
normalize output into the runner's function-call format. New providers should
pass the same offline and opt-in live outcome suites.

## Boundaries

The local service is single-user. Do not expose it publicly by tunneling around
its loopback restriction. Multi-tenant deployment needs actual identity and
authorization, tenant quotas/storage, audited secret handling, and OS isolation.
Likewise, adding shell execution would materially expand the capability boundary
and needs a separate sandbox design and evaluation suite.
