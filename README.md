# AiEngine

A small Python application that turns a plain-English goal into a validated plan,
executes its steps sequentially, searches the web, and creates files.

Uses OpenAI's Responses API for planning and tool decisions, plus a custom Tavily
search adapter. Python controls execution and checks tool results. No agent
framework is needed.

## Quick start

Requires Python 3.10+ and pip. In this project directory, on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
Copy-Item .env.example .env
# Edit .env and add OPENAI_API_KEY and TAVILY_API_KEY.
.\.venv\Scripts\python.exe -m goal_agent "Research three practical uses of Python pathlib and save a source-linked report to pathlib.md."
```

On macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
cp .env.example .env
# Edit .env and add your keys.
.venv/bin/python -m goal_agent "Research three practical uses of Python pathlib and save a source-linked report to pathlib.md."
```

Create keys in your [OpenAI API account](https://platform.openai.com/api-keys) and
[Tavily account](https://app.tavily.com/). Live runs use paid API services according
to your account plans. `OPENAI_API_KEY` is required for live planning/execution.
`TAVILY_API_KEY` is required only when the goal needs web search. Missing search
credentials produce a tool error. Never paste keys into the goal or commit `.env`.

The default model is `gpt-4.1-mini`. Set `OPENAI_MODEL` in `.env`, or use `--model`,
to select a model available to your account that supports Responses, structured
outputs, and function calling. Exported environment values take priority over
`.env`; the CLI loads `.env` from the current working directory only.

## Try it without keys

After installation, run:

```powershell
.\.venv\Scripts\python.exe -m goal_agent --demo
```

The demo uses a fixed two-step plan and canned search data. It runs the same
execution loop and real file creation tool, producing `pathlib-note.md` without
any network requests. It demonstrates the wiring, not live model behavior.

If the virtual environment is activated, the installed `goal-agent` command is
equivalent to `python -m goal_agent` in these examples:

```text
goal-agent "Research SQLite backup strategies and save a short report with links." --plan-only
goal-agent "Create a packing checklist in checklist.md." --max-steps 4 --max-tool-calls 8
goal-agent --demo --runs-dir ./my-runs
```

`--plan-only` still makes one live planning request unless paired with `--demo`.
It saves the plan and runs no action tools. To execute a live goal, run again
without that flag; the new run generates a fresh plan.

## How execution works

1. The planner returns a structured list of steps, each labeled `web_search`,
   `create_file`, or `reason` for synthesis.
2. The executor receives the goal, plan, current step, and all earlier results.
3. The model can request the current step's tool. Python validates and executes
   the call, then sends the observation back with its original call ID.
4. The model calls `finish_step` with a result or blocker. A search/file step
   cannot complete unless its required tool actually succeeded.
5. The next step starts only after completion. An explicit failure or exhausted
   budget stops the run and leaves later steps pending.

The registry exposes custom `web_search(query, max_results)` and
`create_file(path, content)` functions with strict JSON schemas. Search results
include source URLs and bounded excerpts. The executor is instructed to treat
those excerpts as untrusted data and include source links in research artifacts.

Every run gets its own directory:

```text
runs/<timestamp>-<id>/
  run.json             # goal, plan, status, step summaries, tool results/errors
  artifacts/           # files created by the agent
```

Records are saved after each action and step transition. They contain the goal
and retrieved content; handle them according to the sensitivity of your task.
The full conversation remains in memory during the run and is not a resumable
checkpoint. Failed runs retain files already created.

## Limits and scope

- Default budgets: 8 steps, 6 model turns per step, and 24 custom tool attempts.
  Completion calls count toward turns, but not the custom-tool budget.
- OpenAI requests time out after 45 seconds per attempt and retry transient
  failures twice. A live run makes at most `1 + steps * turns` logical model
  requests; HTTP retries can increase the number of network attempts. These are
  operation limits, not dollar or total elapsed-time limits.
- Files are limited to 100,000 UTF-8 bytes each, use relative paths, and cannot
  overwrite existing files. Paths outside the artifact directory, Windows device
  paths, alternate data streams, and symlink/reparse-point paths are rejected.
- This is a local scaffold, not an operating-system security sandbox. Do not let
  another process concurrently replace directories in the artifact workspace.
  For untrusted multi-user deployments, isolate the process and filesystem.
- The agent can research and write text files. It cannot browse full pages,
  execute generated code, send messages, install software, or perform arbitrary
  desktop actions. Plans are fixed; the model can retry tools within a step but
  does not rewrite the plan automatically.
- Completion is a model assessment backed by tool success checks, not an
  independent fact-check of the generated document. Review source accuracy and
  completeness before relying on a report.

## Project layout and extension

```text
src/goal_agent/
  cli.py        # command-line options and environment configuration
  models.py     # validated plan, step completion, budgets
  llm.py        # OpenAI adapter and provider-independent Backend protocol
  runner.py     # sequential orchestration and run records
  tools.py      # custom search and file tools
  demo.py       # deterministic offline demonstration
tests/          # offline tool, execution-loop, and API-adapter tests
```

To add a capability, add its validated schema and handler to `ToolRegistry`, add
the tool name to `Step.tool`, and update the planner's capability instructions.
The runner automatically exposes only the tool declared for the current step.
To use another LLM service, implement `Backend.plan()` and `Backend.respond()`;
the latter returns Responses-shaped output items consumed by the runner.

Run the tests without credentials or external network calls:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## API references

- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [GPT-4.1 mini capabilities](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
- [Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search)
