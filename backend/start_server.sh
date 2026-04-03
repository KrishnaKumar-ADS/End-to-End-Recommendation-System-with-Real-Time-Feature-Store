@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM DS19 — Start FastAPI Server (Windows)
REM Run from PROJECT ROOT: E:\DS19\

echo Starting DS19 Recommendation API...
echo   Logs: logs\api_requests.jsonl
echo   Latency: logs\latency_log.csv
echo.

if "%API_HOST%"=="" set "API_HOST=0.0.0.0"
if "%API_PORT%"=="" set "API_PORT=8000"
if "%API_WORKERS%"=="" set "API_WORKERS=1"

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found at venv\
)

REM Create __init__.py files if missing
python -c "from pathlib import Path; [Path(d).mkdir(parents=True, exist_ok=True) or (Path(d)/'__init__.py').touch() for d in ['backend','backend/app','backend/app/api','backend/app/services','backend/app/schemas','backend/app/middleware','backend/app/core']]"

REM If port is already bound, stop only an old DS19 uvicorn process.
set "PORT_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%API_PORT% .*LISTENING"') do (
    set "PORT_PID=%%P"
    goto :port_check_done
)

:port_check_done
if defined PORT_PID (
    set "EXISTING_CMD="
    for /f "usebackq delims=" %%C in (`powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"ProcessId = %PORT_PID%\" | Select-Object -ExpandProperty CommandLine) 2>$null"`) do (
        set "EXISTING_CMD=%%C"
    )

    echo Port %API_PORT% is already in use by PID %PORT_PID%.
    if defined EXISTING_CMD echo Existing command: !EXISTING_CMD!

    echo !EXISTING_CMD! | findstr /I "uvicorn backend.app.main:app" >nul
    if errorlevel 1 (
        echo ERROR: Port %API_PORT% is occupied by another process.
        echo        Stop PID %PORT_PID% manually or set API_PORT to a free port.
        goto :end
    )

    echo Stopping existing DS19 uvicorn process on port %API_PORT%...
    taskkill /PID %PORT_PID% /F >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Failed to stop PID %PORT_PID%. Close it manually and retry.
        goto :end
    )
    timeout /t 1 >nul
)

REM Start server from project root
REM IMPORTANT: Run from E:\DS19\ so imports resolve correctly
python -m uvicorn backend.app.main:app --host %API_HOST% --port %API_PORT% --workers %API_WORKERS% --log-level info

:end
pause
endlocal