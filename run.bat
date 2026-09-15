@echo off
title Smart Desk Agent
echo.
echo  ================================
echo   Smart Desk Agent - Starting...
echo  ================================
echo.

:: Try to find python in common locations
where python >nul 2>&1
if %ERRORLEVEL% == 0 (
    set PYTHON=python
    goto :run
)

:: Fallback to known Python 3.13 install path
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set PYTHON=%LOCALAPPDATA%\Programs\Python\Python313\python.exe
    goto :run
)

echo [ERROR] Python not found. Please install Python 3.10+ from https://python.org
pause
exit /b 1

:run
echo [1/2] Checking dependencies...
%PYTHON% -m pip install -r requirements.txt -q
echo [2/2] Launching app...
echo.
echo  Open your browser at: http://localhost:8501
echo  Press Ctrl+C to stop.
echo.
%PYTHON% -m streamlit run app.py
pause
