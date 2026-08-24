@echo off
setlocal

rem Canonical submission run: final two-stage Hyb-STEX, four datasets, seeds 1-3.
rem Extra command-line arguments are forwarded to scripts\run_hybstex.py.
python scripts\run_hybstex.py %*
if errorlevel 1 exit /b %errorlevel%

endlocal
