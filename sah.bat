@echo off
rem ============================================================================
rem  sah.bat -- the SAH front door: the shape you reach for from any folder.
rem
rem      C:\mydir>  sah "rename these files by content"
rem
rem  It is ONE machine's opinion of good defaults, assembled into flags the real
rem  CLI already understands. It adds no logic of its own -- read it, edit it.
rem  Everything here is overridable by your own flag or environment variable.
rem
rem  Two verbs share these defaults, split by what the output is FOR:
rem      sah "task"        run it, then KEEP TALKING -- a session you watch (boxes)
rem      sah               an empty interactive session in this folder
rem      sah-call "task"   run it, print the answer as text, exit  (see sah-call.bat)
rem
rem  What a plain `sah "task"` gives you (each is the shim's choice, not core's):
rem    - a NEW session          (a fresh run dir under the runs root)
rem    - a grant on THIS folder (read + write + exec on the current directory)
rem    - the server's own reasoning default (SAH_THINK overrides: off | on | a level the backend speaks, e.g. xhigh)
rem    - the full loadout       (memory + recall + web + vision + exec + spawn)
rem    - interactive            (runs the task, then stays open -- "anything else?")
rem
rem  Also:
rem      sah -r [STEP]      resume the last session; with a step, cut back to that step and steer again
rem      sah --help          the real CLI help (the door's flags ride along and change nothing)
rem      sah run --task "x"  the CLI's own verbs, still with this file's loadout
rem
rem  Overrides (env):  SAH_THINK  SAH_RUNS -- and the machine's host/port/model from the
rem  .env at %ROOT%.env (the core reads it; a shell value or a flag wins over it).
rem  SAH_RAW=1 drops every flag this file adds (see :raw)
rem  One-shot is set by sah-call.bat via SAH_ONCE=1; that is the only difference
rem  between the two verbs, and it lives in ONE place so they cannot drift.
rem ============================================================================
setlocal enabledelayedexpansion
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo sah.bat: no venv at "%PY%"
    echo            create it first:  uv venv ^&^& uv pip install -e .
    exit /b 1
)

rem ---- policy defaults: an already-set env var wins -----------------------
rem The machine's endpoint + model come from the .env at %ROOT%.env (the core reads it at start),
rem so this shim carries no machine-specific address -- a fresh clone reads its own .env.
rem SAH_THINK and SAH_RUNS are the shim's generic defaults; it builds the --think / --memory /
rem --runs-root flags from them.
if not defined SAH_THINK  set "SAH_THINK=on"
if not defined SAH_RUNS   set "SAH_RUNS=%ROOT%runs"

rem Let `python -m sah` import the package no matter which folder we work in.
set "PYTHONPATH=%ROOT%"

rem The current directory is the working folder AND the grant.
set "CWD=%CD%"
if not exist "%SAH_RUNS%" mkdir "%SAH_RUNS%"

rem The run dir is NOT named here. The CLI mints a fresh one under --runs-root (fixed-format UTC stamp plus
rem a random tail) and refuses to write into a folder that already holds a run. Minting it here is what put a
rem locale's date format into a directory name: on this box %DATE% is 26/10/2026, so slices written for
rem MM/DD/YYYY named a run r20261020 on the 26th -- a date that never happened. The name has to be trustworthy
rem rather than merely pretty, because `-r` below reads it off a directory listing.

rem ---- the shared defaults (every flag both verbs use) ---------------------
rem cwd read+write+exec, reasoning on, and the full loadout: memory + recall +
rem web + vision (+ exec + spawn, which is always offered). The view is the ONE line that
rem differs: a session shows boxes, a one-shot call shows clean text.
set "ARGS=--grant "%CWD%" --write "%CWD%" --exec"
set "ARGS=%ARGS% --think %SAH_THINK%"
set "ARGS=%ARGS% --memory "%SAH_RUNS%\.memory""
set "ARGS=%ARGS% --runs-root "%SAH_RUNS%""
set "ARGS=%ARGS% --web --vision"
rem Vision is a default because the operator asked for it and I kept removing it. The caveat stays beside
rem the line rather than in a commit message: the endpoint (from the .env) answers a picture inside the TOOL
rem RESULT with HTTP 400 (measured: the vision check of the opt-in live suite exits 1, and the server
rem itself says "image_url requires a user message and a string URL"), while the same picture on a
rem user turn goes through clean.
rem So an image `read` in a door run against that endpoint fails at the NEXT request until core moves the
rem bytes onto a user turn. `--no-vision` declines the default per run; `--vision` is what it overrides.
rem --think (server-side reasoning quality) is shared; --reasoning (SHOWING the thought) is the
rem session's, not the call's. A one-shot reasons just as hard but prints only the answer as text,
rem so `sah-call "x" | grep` gets a clean pipe. The view line is the whole verb difference.
if defined SAH_ONCE (set "VIEW=--view text") else (set "VIEW=--view normal --reasoning")

rem The endpoint and bounds are NOT echoed here -- the CLI's boot box (sah.render.header) draws them,
rem so an interactive run's first line is that box. A one-shot (text view) prints no box and, with no
rem echo here, its stdout is the answer alone -- a clean pipe.

rem ---- routing (never SHIFTs, so %* stays whole for the objective) ---------
rem SAH_RAW=1 is the only way out of this file's flags. The CLI is one flat parser whose first positional
rem is `run|send|export` and every entry in ARGS is an option, so prefixing them cannot swallow a
rem positional or reorder an objective. A repeated scalar takes the LAST word, which is how `--think none`
rem and `--no-vision` narrow what this file asserts; --grant and --write append and cannot be narrowed,
rem and that is the one thing SAH_RAW is genuinely for.
if defined SAH_RAW goto raw
if "%~1"==""     goto dispatch_empty
if /i "%~1"=="-r"       goto resume
if /i "%~1"=="--resume" goto resume
if /i "%~1"=="--help"   goto with_args
if /i "%~1"=="-h"       goto with_args
if /i "%~1"=="run"      goto with_args
if /i "%~1"=="send"     goto with_args
if /i "%~1"=="export"   goto with_args
set "A1=%~1"
if "%A1:~0,1%"=="-"    goto with_args

rem Anything else is the objective. cmd splits it on spaces and hands each quoted word over as its own
rem argument, so `sah "rename" "these"` reaches the CLI as `--task rename these` — several arguments, not
rem one. The CLI joins the words of an objective, which is why this line can stay `%*`: the door that
rem quoted its whole prompt (`sah.sh`, which passes "$*") and this one now mean the same sentence. What cmd
rem cannot carry is an UNquoted &, which splits the line before this batch starts — the shell's concern,
rem not the harness's.
if defined SAH_ONCE (
    "%PY%" -m sah run %ARGS% %VIEW% --task %*
) else (
    "%PY%" -m sah run %ARGS% %VIEW% -i --task %*
)
goto end

:dispatch_empty
if defined SAH_ONCE (
    echo sah-call: a single call needs an objective.  e.g.  sah-call "what is in this folder"
    goto end
)
rem An empty interactive session.
"%PY%" -m sah run %ARGS% %VIEW%
goto end

:resume
rem The newest run folder under the runs root, by name. The CLI mints those names in fixed-format UTC, so
rem name order is clock order, and it refuses to share a folder between runs -- so a listing is the whole
rem truth. The filter for log.jsonl skips the sibling folders the runs root holds (.memory), and a pointer
rem file is not consulted: it went stale the first time a folder was moved and then resumed the wrong run.
set "LAST="
for /f "delims=" %%i in ('dir /b /o:n /a:d "%SAH_RUNS%" 2^>nul') do if exist "%SAH_RUNS%\%%i\log.jsonl" set "LAST=%%i"
if not defined LAST (
    echo sah.bat: nothing to resume yet -- no run folder under "%SAH_RUNS%".
    goto end
)
echo [sah] resuming %SAH_RUNS%\%LAST%
rem The one word after -r is the jump-back: the CLI's --resume <step>. A word the CLI cannot read as
rem a step number is refused by the CLI's own named line; a third word is not a step, and a dropped
rem word is a silent gap, so it is refused here rather than let it vanish.
if not "%~3"=="" (
    echo sah.bat: -r takes at most a step number (the step's digits, as logged)
    goto end
)
rem Grants are never resumed -- re-issue this run's cwd grant the same way, and let the CLI say so out
rem loud: it prints `reach changed:` when this folder's last run could touch something this one cannot.
if "%~2"=="" (
    "%PY%" -m sah run --run-dir "%SAH_RUNS%\%LAST%" %ARGS% %VIEW% --resume
) else (
    "%PY%" -m sah run --run-dir "%SAH_RUNS%\%LAST%" %ARGS% %VIEW% --resume %~2
)
goto end

:with_args
rem The door's flags, then whatever was typed. This is the whole point of the file: an invocation that
rem names its own endpoint, or adds --no-confine, is still a door run and still gets the loadout.
rem Measured before this existed: `sah.bat --no-confine run --host ... --port ...` fell through to the raw
rem CLI and booted with exec off, web off, vision off, think=none and no memory tree -- a run that looked
rem like a door session and was none of it.
"%PY%" -m sah %ARGS% %*
rem Note: no %VIEW% here on purpose, so `sah run --task "x"` takes the CLI's own default, which is `off`
rem for a one-shot -- the log is the report. `sah "x"` above passes a view, so an objective run draws the
rem box and a verb run says nothing unless asked. Ask with --view normal; --view off keeps it silent.
goto end

:raw
rem SAH_RAW=1: none of this file's flags. The env block above still applies, because it ran before
rem routing. The runs value is printed (not described) so the line cannot drift from what this file
rem holds. The machine's endpoint + model come from the .env at %ROOT%.env (the core reads it), and
rem the log records the model that answered -- so this echo names the .env rather than the values it
rem holds.
echo [sah] SAH_RAW: no flags from this file - grant, write, exec, memory, runs-root, web, think, vision 1>&2
echo [sah]   machine config (host/port/model) from %ROOT%.env; runs=%SAH_RUNS% think=%SAH_THINK% 1>&2
"%PY%" -m sah %*
goto end

:end
rem The code the caller sees must be the RUN's, not the shim's: `sah-call "x" && next-step` only
rem means something if a failed run exits non-zero. A bare `endlocal` swallowed it (measured: a
rem refused endpoint drew `failed (exit 1)` and left ERRORLEVEL=0), so the code is taken on the SAME
rem line — %ERRORLEVEL% expands when the line is parsed, which is before endlocal runs. sah-call.bat
rem calls this file and forwards %ERRORLEVEL%, so this one line is both verbs.
endlocal & exit /b %ERRORLEVEL%
