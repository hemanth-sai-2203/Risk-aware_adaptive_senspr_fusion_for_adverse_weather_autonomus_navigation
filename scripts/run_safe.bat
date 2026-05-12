@echo off
color 0A
echo ====================================================
echo   RA-ASF SAFE HEADLESS LAUNCHER
echo ====================================================
echo.

echo [1/4] Killing any stuck CARLA servers...
taskkill /F /IM CarlaUE4.exe /T >nul 2>&1
taskkill /F /IM CarlaUE4-Win64-Shipping.exe /T >nul 2>&1
timeout /t 3 /nobreak >nul

echo [2/4] Launching CARLA in HEADLESS (-nullrhi) mode...
cd C:\Users\heman\Downloads\CARLA_0.9.16
start CarlaUE4.exe -nullrhi -log

echo [3/4] Waiting 15 seconds for CARLA to load completely...
timeout /t 15 /nobreak >nul

echo [4/4] Starting the Python AI Brain (SAFE MODE)...
cd C:\Users\heman\Music\ra_asf
call .\carla16_env\Scripts\activate.bat
cd final_implementation
python run_demo.py

echo.
echo Demo finished or crashed. Press any key to exit.
pause >nul
