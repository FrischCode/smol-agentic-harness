#  S   ╱─╲   H  •  Smol Agentic Harness

**Small on purpose** - minimal but complete and with some polish! Understandable, extendable,
composable, local friendly.

A **synchronous loop**, a **stdlib-only core**, one **append-only log**, and a **neat command line
interface**. Driven by you, your agent or your pipeline.

> [!NOTE]
> It is a *harness*, not a framework: it runs a model against tools under permissions you
> grant per run, records every step to a log that is the ground-truth report of what happened.

> [!CAUTION]
> *Agentic execution is inherently unsafe. Sandboxing and other mitigations may reduce
> risk, but do not guarantee containment. Use a VM and discretion.*

## Screenshot

<details><summary>Click here to show</summary>
A `sah` session on Windows — the banner, one objective, the tool call framed
with its arguments and result, and the end state:

<img src="screenshot-win.png" alt="A `sah` session: the banner names the backend (demo-model @
127.0.0.1:8081) and the capabilities in force (think=on, vision=on, exec granted, spawn, web
public only); the objective asks the run to commit a fact to its memory tree and list its tools;
the assistant's reasoning shows dimmed above each answer; a `mem_note` call appends the fact to
`glossary.md`; the answer lists the run's tools; `complete (exit 0)`" />

The banner is the run's own account of what it may do — the backend and its reasoning posture,
vision, the granted shell, spawn, what it may read and write (`rw: c:\work\sah`), its memory tree
and recall root, and the web's reach. When the model reasons, its thought shows dimmed under an
assistant box; each tool's box carries its arguments as JSON and its result. The line under a box is
the turn's own numbers — context occupancy against the window, tokens, rate, and the seconds it
took. Everything on screen is a projection of the run's one append-only log: `sah export` on the
folder renders the same conversation as Markdown.
</details>

## Quickstart

The tree is self-contained: a copy of it is the same product on another machine, the venv being a
copyable directory. Using the repo's own uv venv:

```bash
# point it at any OpenAI-compatible server (tabbyAPI, llama.cpp, vLLM, ...) and talk to it:
.venv/Scripts/python.exe -m sah run \
    --host 127.0.0.1 --port 8080 --model demo-model \
    --grant ./notes --write ./out --exec

# or hand it one objective and exit (this is how a spawned child runs — no keyboard):
.venv/Scripts/python.exe -m sah run --task "summarize notes/ and write notes/summary.md" --write ./notes
```

`--grant`/`--write` decide what the run may touch; if you pass none, it can read the working
directory and write nothing. What a run is allowed comes **only from the flags you typed this time**
— never from a saved config or a previous run.

## The front door (`sah` / `sah-call`)

`python -m sah` is the whole engine and every flag. The launchers (`sah.bat`/`sah.sh` and their
one-shot twins `sah-call.bat`/`sah-call.sh`) are a *thin* shell over it that sets one machine's good
defaults, so the everyday shape is just a word from whatever folder you are in:

```bat
C:\mydir>  sah "rename these files by content"           :: a session: runs it, then stays open
C:\mydir>  sah                                           :: an empty session here
C:\mydir>  sah-call "what is in this folder"             :: one shot: the answer as text, then exit
C:\mydir>  sah-call "list the .py files" | grep model    :: plain text out: it pipes like any other tool
C:\mydir>  sah -r                                        :: resume the last session here
C:\mydir>  sah -r 14                                     :: cut back to step 14 of the last session and steer again
```

Both verbs share every default — a new run dir, **read+write+exec on the current folder**, the
server's own reasoning default, and the full loadout (memory, recall, web, vision, exec, spawn).
They differ by **one** thing, set by `SAH_ONCE` in the wrapper:

| | stays open | shows |
|---|---|---|
| `sah "task"` | yes — "anything else?" | boxes, streaming, the dimmed thought |
| `sah-call "task"` | no | just the answer as text, so it pipes cleanly |

The defaults are the *shim's opinion*, not core's, and each is overridable: `SAH_HOST` `SAH_PORT`
`SAH_MODEL` `SAH_THINK` (`off`/`on`/`xhigh`/…) `SAH_RUNS` — or per run, by naming the flag again:
the CLI is one flat parser of options, so `sah run --think none --no-vision` narrows what the shim
asserted, the last word winning. `SAH_RAW=1` drops every flag the shim adds, which is the way out
for the one thing a second flag cannot undo — a `--grant` that appends rather than replaces. Naming
your own endpoint or adding `--no-confine` is still a shim run and still gets the loadout; the
launcher never hides the engine it wraps, it just stops withholding its own opinion from a run that
spoke the engine's dialect. The engine stays deny-by-default on purpose — a reasoning-on, web-on,
this-host's-IP policy living in a launcher is *your* box's choice, so a spawned child or another
machine never silently inherits it.

## Four commands

`python -m sah run …` — talk to a model in this terminal.
`python -m sah send …` — send a command to *another* run's queue (steer it, answer its question).
`python -m sah export …` — render a finished run's log to Markdown (read-only, no model).
`python -m sah result …` — a run's final answer and how it ended (read-only, no model).

```bash
# a run you intend to steer from elsewhere: name its folder once, on the run that creates it
.venv/Scripts/python.exe -m sah run --run-dir runs/demo --grant ./notes --exec "check the tests"

# while it is going, from a second shell:
.venv/Scripts/python.exe -m sah send --run-dir runs/demo --from alice steer "check the tests first"
.venv/Scripts/python.exe -m sah send --run-dir runs/demo --from alice --reply 000042-request answer yes

# after a run, turn its log into a readable report:
.venv/Scripts/python.exe -m sah export --run-dir runs/demo --out report.md

# or read its answer — what it last said, how it ended, and its exit code back to your shell
# (0 = answered, 1 = ended badly, 2 = usage, 3 = still running) — `--json` for the machine form:
.venv/Scripts/python.exe -m sah result --run-dir runs/demo
```

### `run` flags

| flag | meaning |
|---|---|
| `--run-dir DIR` | the run's folder (`log.jsonl`, `control.md`, `out/`, `children/`). **Unset by default**: a fresh folder is minted under the runs root, because two runs sharing a folder merge into one damaged history. Naming a folder that already holds a run is refused unless `--resume` is given; naming an empty one is how you give a run a stable handle (see the steer example above). |
| `--grant DIR` | a folder the run may **read** (repeatable). Defaults to the working directory. |
| `--write DIR` | a folder it may **write** (repeatable); implies reading it. |
| `--exec` | give it a shell (`bash` tool). This makes path confinement *advisory* — a command can reach anything your account can. Separate from read/write. |
| `--no-confine` | **lift path confinement for this run**: any path your account can reach, for the mounted drive nobody thought to grant. Its own axis — it grants no shell, still refuses a call that names no path, prints `rw: anywhere this user can reach (--no-confine)` instead of a root list, is recorded as `unrestricted` in the run's `started` step (so a resume reports the change), and is **never** passed to a child run. |
| `--memory DIR` | a memory **tree** (browse/read/grep/append notes), separate from `--grant`. Notes cannot leave it. |
| `--runs-root DIR` | let the run read its **history**: a folder of past run directories (browse via `runs`, read one via `run_log`), **and** where a fresh run folder is minted when `--run-dir` is unnamed — a run writes where recall reads. Read-only, distinct from `--memory` (curated notes); a run id is one folder name inside it, never a path. Also read from `$SAH_RUNS`. |
| `--web` | let the run reach the network (`web_search`, `web_fetch`). Search uses an endpoint **you** choose (`SAH_SEARCH_URL`, default a local SearXNG at `127.0.0.1:8888`); `web_fetch` reaches the **public internet only** — loopback/private/LAN/metadata addresses are refused, and so is a redirect that lands on one. |
| `--web-private` | with `--web`, also let `web_fetch` reach private/LAN addresses (turns off the SSRF guard). A widening: printed in the banner, disclosed to the model, never inherited by a child it wasn't given to. |
| `--search-url URL` | override the `web_search` endpoint; `%s` is the query. A SearXNG (`…/search?q=%s&format=json`) or a DuckDuckGo HTML URL both work — the backend is chosen by the response's shape. |
| `-i`, `--interactive` | run a given `--task` (or positional objective) **and then stay open** at the prompt, instead of exiting after one objective. Without it a one-shot runs once and returns — what a spawned or scripted run needs (no keyboard, must not block). |
| `--view MODE` | how much the terminal shows: `normal` (framed turn boxes with live output), `quiet` (one line per turn), `text` (prose only), `off` (silent). Default `normal` when stdout is a terminal, `off` when it is redirected — a spawned child has no witness, so its log is the report. |
| `--no-color` | never emit ANSI colour (also honoured via `NO_COLOR`; auto-off when stdout is not a terminal). |
| `--model / --host / --port / --api-key` | the backend. Env defaults: `SAH_MODEL`, `SAH_HOST` (127.0.0.1), `SAH_PORT` (8080) — or a `.env` at the tree root (see `.env.example`), which a flag or an exported variable overrides. |
| `--context N` | context window advertised to the harness (default 32768). |
| `--think off\|on\|LEVEL` | the reasoning level. `none` and `on` **omit the field entirely** — the server's own default, and the one spelling a backend that predates reasoning levels (or validates strictly) cannot reject (a named level is a 400 on a template that doesn't spell it — `high` is refused by the Qwen3.8 family, which takes `xhigh`/`medium`/`low`); `off` asks the server to stop reasoning; any other word goes out verbatim in the backend's own vocabulary. `SAH_THINK` (shell or `.env`) is the default. |
| `--no-streaming` | take the single-body round trip instead of the SSE line stream — for a server that cannot stream at all, or a run that wants one JSON body per turn. Usage then comes from the body, and `stream_options` is not sent (the dialect refuses it on a non-streaming request). On by default. |
| `--no-usage` | omit the usage request (`stream_options`) a server's dialect may not declare. The run falls back to its local token estimate. On by default — the compactor calibrates on the counts. |
| `--budget N` | `max_tokens` for the run. |
| `--max-turns N` | the objective's work-turn budget (default 12). When it runs out the model is **told to wrap up** and gets `--grace-turns` more to deliver; a run that answers in them ends `complete` (exit 0), one that spends the grace ends `max_turns` (exit 1). Judged per objective: a resumed run gets a fresh budget. |
| `--max-runtime SEC` | seconds this objective may take (default 0: no limit). Same wrap-up and grace as the turn budget; the clock starts when the objective starts, so wall time spent inside a tool call counts. A run that spends the grace ends `max_runtime`. |
| `--grace-turns N` | turns of leeway after either budget runs out (default 3). `0` makes a budget a hard stop at the turn boundary, with no word to the model. |
| `--task "…"` | run one objective then exit; reads no stdin. |
| `--resume [STEP]` | continue the run in `--run-dir` from its own log. Rebuilds the *conversation*; **re-grant** with `--grant/--write` — permissions are never resumed. With a step number (the digits of a step's id, as logged: `000000` is the first step), the conversation is rebuilt from the prefix ending at that step and the later steps are **detached**: kept in `log.jsonl` for the audit, never in any conversation again — cut back to a step and steer again. A cut is refused where it would end the conversation mid-exchange (a call owed its result) or at the run's `ended` step, and the line says how many steps came off. If this invocation's reach differs from the run you are continuing, it prints what changed in both directions (`reach changed: this invocation does not grant --exec …`) instead of letting a missing tool be the message. |
| `--system "…" / --system-file F` | replace the preamble. The live-bounds block is always appended; a verbatim prompt cannot hide the run's real permissions from the model. |
| `--ask-timeout SEC` | how long a question waits before the answer counts as *no*. Default: **no deadline** when the run is attended (a person can answer at any hour), 0 when it is not. |
| `--reasoning` | show reasoning as it streams. |

### `send` flags

Positional: the action, then the text. `--reply STEP_ID` for an `answer`; `--from NAME` (or
`SAH_USER`) records who claims to have sent it; `--prompt` adds a line of context; `--cancel` marks
the block cancelled. The sender line is a *claim*, visible in the log — never an authority.

## What a run leaves behind

Everything for a run lives in its `--run-dir`:

- `log.jsonl` — the step log: the append-only, ground-truth record. Read it to know what happened.
- `report.md` — the Markdown projection of the log, written by the harness when an unattended run
  ends — for every ending, since the bad run's report is the one a caller most wants. Attended runs
  leave none: the operator has the live view, and `sah export` renders the file on demand. The log
  stays the record.
- `control.md` — the inbound command queue (`send` appends here; the run drains it at turn
  boundaries).
- `out/` — full tool output that was too big for the context, saved here so the run can read it
  back.
- `children/` — any runs this one spawned (a child owns its own log; the parent can always read
  them,  
  and `sah result --run-dir children/<id>` reads the child's answer).

## How it works

- **One log, one way in.** Only the harness core appends authoritative steps; it stamps who issued
  each one, and every path that ends a run writes its `ended` line — a crash included, with its
  traceback beside it — so no reader of the log can mistake a dead run for a running one.
  A tool returns text; a plugin sees a finished step and cannot change it or inject a
  fact. You cannot forge an accepted permission by writing to the log.
- **One truth, three projections.** The log is written once; everything else is a projection of it:
  JSONL for persistence and debugging, Markdown (`sah export`, and what a child reports to its
  parent) that both a human and a model can parse, and a rich console (`sah run`, human-only and
  ephemeral). Producers yield a chunk stream — a tool that doesn't stream is a stream of one — and
  one event bus fans the live stream out to whoever is watching. No projection is a second writer.
- **Commands in, events out.** The loop reads a queue at each turn boundary and accepts or refuses
  each command by rule, recording the decision before acting. The same surface serves a keypress, a
  second terminal, or a spawned caller.
- **A veto point, not permission-shaped code.** One policy callable decides every tool call: it
  resolves the real path first (symlinks can't escape a grant), and a refusal tells the model the
  exact act that would have allowed it. Execution is its own axis and always states its cost.
- **Capabilities are widened, never assumed.** Shell, memory, history and the web are each off until
  asked for; what is in force is printed in the banner *and* handed to the model in its own
  `<situation>`, so no run can act on a permission its operator never granted. The web is the
  sharpest example: `web_fetch` reaches the public internet only and refuses loopback/LAN/metadata
  addresses by resolving each name and re-checking every redirect — because a fetch tool on your
  workstation is a port scanner unless it isn't. Reaching your own network is a deliberate widening
  (`--web-private`), and a child never inherits a widening its parent didn't have.
- **A refusal can become a question.** Instead of silently denying, a run can *ask* you one specific
  thing (this path? this command?). Only a yes-word grants it; with nobody to answer, the answer is
  no. Recording a question grants nothing.
- **Big results can't blow the context.** A tool result enters as a bounded preview with a "continue
  from offset N" hint; a huge file is streamed, never loaded whole; command output that overflows is
  saved to `out/` with the path handed back. The log keeps the preview + pointer.
- **A session survives its own size.** When the window fills, the oldest turns are summarised.
  Resume rebuilds the conversation from the log — but never the permissions.
- **Extensions are not back doors.** A capability plugs in as a plain tool and/or a step watcher. It
  buys no permission, because the gate is on every *call*, not on who registered it. (Commands and
  "answering a core question" are deliberately not exposed to plugins.)
- **A run that spins in place is caught.** A repetition guard, reasoning only about the shape of the
  act stream (never the meaning), nudges the model to change approach and then stops the run rather
  than letting it repeat forever.
- **Completion is a contract, and it depends on attendance.** An attended run ends when the model
  says done in words — a no-tool turn is read as finished, and the human is the backstop. A run no
  operator is watching ends on an explicit act: the model calls `verify_objective` (offered only to
  that shape, takes no argument, touches nothing), the log records the act and when it happened,
  and the caller that set the task verifies the result against the world itself. A no-call turn in
  an unattended run is asked for the call or the declaration, bounded; a run that cannot prove done
  ends `blocked` (exit 1) rather than `complete` — its caller gets the exit code and nothing else.

## Extensions

A capability plugs in through one surface and only one: `sah.ext` gives a contributor exactly two
contracts — **tools** (plain type-hinted functions added to the run's tool set) and **on(step)**
(see each step as it is written; notification only, no way back into the loop). Nothing else: an
extension cannot answer a core question, cannot register a command, and buys no permission, because
every call — built-in or contributed — still passes the veto point. The gate is on the call, not on
who registered it.

The door is `--extension FILE`, repeatable. The file must define `EXTENSION` — one extension
instance (or a tuple of them), each shaped the way `sah.ext.assemble` reads it. The file is an
explicit act of the operator, loaded at boot, and there is no discovery: a file reaches a run only
if the operator named it — on the command line, or in the machine's own `.env` (`SAH_EXTENSION`,
comma-separated; the flag wins when it is present). What the operator types shows up where what's in
force shows up — the boot box names the file and its tools, the model is told the same in its
`<situation>`, and a `notice` step beside the log records it. A child never inherits its parent's
extensions, and a resume does not restore them: grants come from the invocation, and this is one. A
file that cannot be loaded — missing, mis-named, a class where an instance belongs, a tool that
never declared how it touches the world, a name the built-ins already hold — is a named refusal at
boot (exit 2), not a traceback and not a half-run.

`extensions/speak.py` is the demo: a `speak` tool (words to an OpenAI-compatible TTS endpoint named
in the environment — `SAH_SPEAK_HOST`/`SAH_SPEAK_PORT`/`SAH_SPEAK_VOICE`/`SAH_SPEAK_MODEL`, see
`.env.example` — the audio to a wav file the answer names, played through the player the operator
named — `SAH_SPEAK_PLAYER`, default `ffplay`; `SAH_SPEAK_PLAY=0` mutes a box whose runs would talk
over each other — and a step watcher that announces the ask on the console's transient line — a line
the record does not hold, because the fact is already in the record (the call step carries the
words), and the line self-names its speaker, the way a finding carries its own naming
(`SAH_SPEAK_ANNOUNCE=0` silences it for a box whose run flow already carries the ask in its own tool
box). It travels the whole path a contributor's file would, and it passes the same boot gate as the
built-ins; the live suite (kept out of the shipped tree on purpose, like the rest of it — see
Testing) carries its check, which proves the endpoint for real and says a few words out loud.

Where to look: the docstrings in `sah/ext.py` (the two contracts, the findings and status surfaces,
and the doors that are *not* open) and `sah/contract.py` (the shape a tool you write must take: the
signature is the schema, and `access`/`address`/`description` are its declaration — a tool that never
declared is refused at boot, and that refusal is the point).

### MCP

`extensions/mcp_client.py` is `--extension FILE`'s second occupant, and it shows the door carrying
something whose contents the operator cannot see ahead of time: MCP servers. The operator names one
or more of them in the environment:

- `SAH_MCP_STDIO` — the servers to launch; semicolon-separated, each a comma-separated argv.
- `SAH_MCP_HTTP` — the Streamable HTTP endpoints to post to; semicolon-separated full URLs.
- `SAH_MCP_HTTP_HEADERS` — extra request headers, `Name: Value` pairs semicolon-separated, sent on
  every request to every named endpoint: the door a Bearer token or an API key comes through.
- `SAH_MCP_TIMEOUT` — the seconds a call may take before the wrapper answers with the timeout
  (default 120), on either wire.

At boot, each named server is reached over its own wire and its handshake is run. Every tool it
advertises then joins the run's table through the same boot gate as every other tool — and nothing
about a tool is written or pre-declared by anyone: the wrapper is generated at boot from what the
server advertised, name, description and signature all derived, and the declaration the gate reads
is derived with them, `self`, because the wrapper is self-confined by construction — it closes over
the server the operator named, the model supplies only the arguments, and no value it sends chooses
or changes the destination. The model sees the server's own names and descriptions and never sees
the transport. A server that changes what it advertises changes what the next run gets; nothing on
this side is rewritten to keep up. In both cases the server is bought by naming it, and every call
still passes the veto point.

What a mis-named or dead server becomes is the same on either wire. A server that will not answer the
handshake is a named refusal at boot (exit 2, the server named), with nothing left running. A call
that fails comes back as the tool result naming the failure, so the run reads it and changes course
instead of dying. What the handshake found — the connection, the tools it refused and why — is not
said to a screen but handed up as findings the core records beside the load notice: a boot fact no
one watched arrive is still in the record where an audit looks.

The wire's own detail — how the HTTP half tells a JSON answer from an SSE stream, what it does with
a session id, why it refuses a redirect — lives in the module's docstring, next to the refusal words
it speaks.

## Skills

Where an extension adds a tool, a skill adds a procedure: a named how-to the user has dictated,
which the run finds and follows when the user references it ("use your X skill"). A skill is a note
in the memory tree the run is pointed at with `--memory`, under a `skills/` folder — one `.md` per
skill: a name, a description, and the instructions. The folder is the machine's memory, not part of
the shipped tree; the operator adds it, and the user may add, edit or remove skills at any time. A
child that shares the memory inherits them.

The run is told nothing about skills unless the user names one. SAH injects no generated index into
the prompt — a generated index can be wrong about the note it indexes, and the reader cannot tell
which of the two to trust — so discovery is by reference: the user names the skill, the run finds
the note and follows it, and asked what it has, it lists the folder. The contract the model works
from is the standing guidance in `sah/prompts/agents.md`, and it carries the load-bearing line: a
skill is an *instruction*, not a grant — it says how to do what the run's grants already allow, and
a step a skill names that the confine refuses stays refused.

Where to look: `sah/prompts/agents.md` (the contract the run follows) and `sah/tools/memory.py` (the
address space a skill lives in, and the no-generated-index invariant the operation rests on).

## Testing

`python -m pytest` — in-process, no real spawns/servers/network (a structural test forbids spawning
calls in the battery), no skips or xfails, and the step taxonomy is a closed set enforced by tests.
The battery is hermetic by design, so it cannot prove the wire to a real model server or the
behaviour of real processes. That is what the opt-in **live suite** is for: checks that run only
when pointed at a real endpoint or a real machine — the wire, a detached child and its resume, ask →
answer → widen, whether an abandoned command dies. It is kept out of the shipped tree on purpose: an
out-of-band check that never sits skipped is cleaner than a test that pretends to pass, and every
check prints what it found, so a green run is a measurement, not a claim. Contributor rules and the
invariants are in `AGENTS.md`.

## License

MIT — the full text is in `LICENSE`. Permissive on purpose: the harness's claim is that its owner
reads all of it and changes it without asking anyone, and a copyleft would force that owner to
re-license their own modifications when they share them. The one condition is a notice, and the one
warranty is the absence of one.

## Credits

Created, tested and inspired by using [pi](https://github.com/earendil-works/pi), [Qwen 3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) & [Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B)

<sup>Jan - C. Frischmuth • 2026</sup>