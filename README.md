# AiEngine

Turn a plain-English goal into a plan, execute research and file tools, and check
the resulting artifacts. Version 0.2 adds verified outputs, resumable runs,
resource budgets, evaluations, and an authenticated local web interface.

## Why this is useful

- **Research to deliverable:** ask for a report and receive a real local file
  with links to sources retrieved during the run.
- **Inspectable results:** the plan names expected files, headings, and source
  requirements; failed checks prevent a run from reporting completion.
- **Recoverable work:** preview a plan, execute that exact plan, and resume an
  interrupted run without repeating completed steps or rewriting saved files.
- **Measurable changes:** compare task results, citation provenance, token usage,
  and latency using a repeatable evaluation suite.
- **Controlled execution:** bound steps, tools, model turns, tokens, active
  elapsed time, and estimated model cost. Inspect the reason execution stopped.

OpenAI handles planning and tool decisions. Python validates, executes, persists,
and verifies. Tavily provides search and page extraction. There is no agent
framework or frontend build system to configure.

## Install

Requires Python 3.10+ and pip. Windows PowerShell:

```powershell
git clone https://github.com/AshishTho/AiEngine.git
cd AiEngine
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
Copy-Item .env.example .env
```

If you already have a checkout, run `git pull` and repeat the install command.
Do not replace an existing `.env` when upgrading.

On macOS/Linux, use `python3 -m venv .venv`, `.venv/bin/python -m pip install -e .`,
and `cp .env.example .env`. The commands below use `python` to mean the Python
inside your activated virtual environment; alternatively use its full path.

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell activation is unavailable, replace `python` below with
`.\.venv\Scripts\python.exe`; changing system execution policy is unnecessary.

## Try it without API keys

```text
python -m goal_agent --demo
python -m goal_agent --ui
```

The demo uses a fixed plan and source fixture, executes real file operations,
and verifies the generated `pathlib-note.md`. It makes no API calls.

For the interface, open the full launch URL printed in the terminal. Its
fragment contains a per-launch access token. Select **Offline demo**, preview the
plan, and execute it. The interface shows steps, usage, verification checks,
history, cancellation, resume, and artifact downloads. Keep the server running
while using it; stop it with Ctrl+C. The installed `goal-agent` command is also
equivalent to `python -m goal_agent`.

## Configure live runs

Edit `.env` locally:

```dotenv
OPENAI_API_KEY=your-local-openai-key
TAVILY_API_KEY=your-local-tavily-key
OPENAI_MODEL=gpt-4.1-mini
INPUT_COST_PER_MILLION=
OUTPUT_COST_PER_MILLION=
```

Get keys from [OpenAI](https://platform.openai.com/api-keys) and
[Tavily](https://app.tavily.com/). File-only goals need OpenAI; search and page
extraction also need Tavily. Live runs incur provider charges. Never include keys
in goals or commit `.env`; it is ignored by Git. The application loads `.env`
from the current directory, and exported environment values take priority.
Restart the UI server after changing its environment configuration.

Select an available model supporting Responses, structured outputs, and function
calling using `OPENAI_MODEL` or `--model`. Optional prices are **USD per million
input/output tokens** for your chosen model; set both or leave both blank.
Unconfigured costs display as unknown. Estimates exclude Tavily fees and cached
input discounts and are not billing statements.

```text
python -m goal_agent "Create a travel packing checklist in packing.md with Clothing and Documents sections."
python -m goal_agent "Research Python pathlib and summarize three useful features with source links."
python -m goal_agent "Research Python pathlib. Save report.md with Overview, Examples, and Sources headings and cite two retrieved sources."
```

The final command succeeds only if the saved plan's expected report exists,
contains its required headings, and cites enough distinct retrieved source URLs.
Plans needing unsupported actions or more clarification return `blocked`.

## Preview, execute, and resume

```text
python -m goal_agent "Research SQLite backups and save a cited report to backup.md." --plan-only
python -m goal_agent --execute-plan runs/<run-id>/plan.json
python -m goal_agent --resume runs/<run-id>
```

Replace `<run-id>` with the directory printed by the preceding command.
`--plan-only` uses one live planning request but runs no action tools.
`--execute-plan` uses the saved goal and plan in a **new** run without replanning.
`--resume` continues the original run directory, including a plan-only run.
An editable example is in [examples/packing-plan.json](examples/packing-plan.json).

Resume preserves completed steps, conversation items, tool-call IDs, usage, and
budgets. A write-ahead record and SHA-256 hashes allow recovery when a file was
created just before interruption. Changed, missing, or partially written files
are refused instead of overwritten. A stopped search may be requested again;
exactly-once network billing is not guaranteed. Version 0.1 records cannot resume.

The same run cannot execute in two processes at once. CLI resumes preserve the
original model and prices. To extend a depleted budget explicitly:

```text
python -m goal_agent --resume runs/<run-id> --max-tokens 200000 --max-seconds 1200 --max-tool-calls 40
```

## Budgets and recovery

```text
python -m goal_agent "Research pathlib and save notes.md with citations." --max-steps 8 --max-turns-per-step 6 --max-tool-calls 24 --max-tokens 100000 --max-seconds 600 --max-replans 1
python -m goal_agent "Create a checklist in list.md." --max-cost-usd 0.50
```

The cost-cap command requires both configured prices. Defaults match the first
command. Automatic plan repair can revise only unfinished steps, at most once;
it preserves output requirements and completed work. Set `--max-replans 0` to
disable it. Authentication/configuration failures stop promptly. Tavily retries
transient errors at most twice total; OpenAI uses its SDK's two retries.

Token and cost admission uses a conservative byte-based input estimate plus the
configured output allowance, then records provider-reported usage. This can
stop conservatively. If an in-flight request fails or the process crashes before
usage arrives, its reservation remains recorded as unconfirmed usage. Counters
survive resume. Provider accounting or retries may differ from estimates; use
provider account limits as an additional billing control.

Cancellation and elapsed-time limits are cooperative: no new work starts after
they are observed, but an in-flight synchronous HTTP request must return or time
out. SDK retries and per-I/O timeouts can delay that return. Idle time between
resumes does not consume the active-time budget.

## Available tools

- `web_search(query, max_results)`: Tavily search, up to five bounded snippets
  with public source URLs. Empty usable evidence is a failed tool result.
- `fetch_page(url)`: Tavily extraction through its fixed HTTPS endpoint; returns
  bounded Markdown, with truncation indicated. Rejects private/local URLs.
- `create_file(path, content)`: new UTF-8 text files under the current run's
  artifact directory; no overwrite, traversal, or symlink paths.
- `read_file(path)`: reads only the current artifact workspace and returns content
  and a SHA-256 digest. Files have a 100,000-byte limit by default.

Only the current step's tool is exposed. Source content is untrusted evidence,
not an instruction channel. The agent cannot execute generated code, send
messages, or operate arbitrary desktop applications.

## Evaluate and test

```text
python -m unittest discover -s tests -v
python -m goal_agent --eval --eval-output evaluations
python -m goal_agent --eval --live --eval-output evaluations
```

The **16 offline fixtures** cover report/file success, missing keys, empty
research, unsupported and ambiguous goals, malformed calls, missing artifacts
and sections, invented citations, file protection, reading, tool boundaries,
and budgets. They use scripted model responses with real local file operations.
The injection fixture checks tool containment, not live-model injection resistance.

`--eval --live` explicitly enables **three paid smoke tests**: file-only,
research-only, and research-to-report. It requires both keys. Reports preserve
per-case checks, observed metrics, source URLs, and artifact paths. Correctly
rejecting a negative fixture counts as an evaluation pass; it does not count as
a successfully completed user task. Old reports are preserved in separate folders.

Citation checks establish URL provenance and required structure. They do not
prove that a source supports every claim, that the plan captured every nuance
of the goal, or that research is complete. Review important reports and expand
the evaluation dataset for your domain.

CI runs the tests, demo, and offline suite on Windows/Linux with Python 3.10 and
3.12. Live tests are opt-in and API secrets are never required in CI.

## Run records and deployment scope

```text
runs/<run-id>/
  plan.json        # saved goal and expected outputs
  run.json         # checkpoint, history, observations, metrics, verification
  artifacts/       # generated text files
```

Records include goals, retrieved text, and generated content. Keep the run
directory private. Do not edit checkpoint internals while a run is active.

The UI is a **local, single-user application** bound to loopback. It uses a
per-launch token, same-origin mutation checks, safe text rendering, and forced
file downloads. Workspaces are isolated per run; local quotas allow one active
run and 1,000 saved runs by default. The Python `RunService` constructor exposes
these quotas. It is deliberately not a public multi-tenant hosting service.
Shared deployment requires individual user identities, tenant-specific storage
and billing quotas, TLS, and process/container isolation before exposure.

Filesystem guards assume another local process is not concurrently replacing
directories. They do not provide an OS sandbox. The web server refuses public
bindings and does not enable cross-origin access.

## Development

See [architecture and extension notes](docs/ARCHITECTURE.md) and
[release changes](CHANGELOG.md). Main components:

```text
src/goal_agent/
  models.py / llm.py          # validated plans and OpenAI adapter
  runner.py / storage.py      # sequential execution, journals, resume, locks
  tools.py / verification.py  # tool boundaries and artifact checks
  service.py / cli.py         # shared run service and commands
  web.py / static/index.html  # authenticated local interface
  evals.py                    # independent outcome evaluation
  evaluation_cases.json      # reproducible fixture dataset
```

API references: [function calling](https://developers.openai.com/api/docs/guides/function-calling),
[structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[token accounting](https://developers.openai.com/api/docs/guides/token-counting),
[agent evaluations](https://developers.openai.com/api/docs/guides/agent-evals),
[Tavily search](https://docs.tavily.com/documentation/api-reference/endpoint/search),
[Tavily extraction](https://docs.tavily.com/documentation/api-reference/endpoint/extract).
