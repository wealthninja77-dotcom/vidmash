@echo off
title VidMash Server
echo ========================================================
echo            Launching VidMash Intelligence App
echo ========================================================
echo.
echo Starting backend server on http://localhost:8000 ...
echo Opening your web browser...
start http://localhost:8000
echo.
.venv\Scripts\python.exe run.py
pause
