@echo off
rem ============================================================================
rem  sah-call.bat -- the single-call front door: fire one objective, get the
rem  answer as clean text, exit. The one-shot twin of sah.bat.
rem
rem      C:\mydir>  sah-call "summarize what is in this folder"
rem      C:\mydir>  sah-call "list the .py files" | grep model
rem
rem  Identical defaults to sah.bat -- new session, cwd read+write+exec, reasoning
rem  on, full loadout -- with two differences, both from SAH_ONCE: it does NOT
rem  stay open after the task, and it shows text (the answer) instead of boxes.
rem
rem  Every default lives in sah.bat. This file is a wrapper on purpose: the
rem  two verbs share one body, so their defaults can never drift apart.
rem ============================================================================
setlocal
set "SAH_ONCE=1"
call "%~dp0sah.bat" %*
exit /b %ERRORLEVEL%
