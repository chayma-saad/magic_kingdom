@echo off
echo ================================================
echo  AETHER SCRIPT — Install and Run
echo ================================================
echo.

REM Try to find a Python 3.10+ installation
set PYTHON=
for %%P in (python python3 py) do (
    %%P --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON=%%P
        goto :found
    )
)

echo ERROR: Python not found in PATH.
echo Please install Python 3.10+ from https://python.org
echo and make sure to check "Add Python to PATH" during install.
pause
exit /b 1

:found
echo Using: %PYTHON%
%PYTHON% --version
echo.

echo Installing dependencies...
%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 (
    echo pip install failed. Try: %PYTHON% -m pip install --upgrade pip
    pause
    exit /b 1
)

echo.
echo ================================================
echo  Launching AETHER SCRIPT
echo  ESC=quit  r=reset  g=toggle-HUD  d=debug
echo ================================================
echo.
%PYTHON% main.py %*
pause
