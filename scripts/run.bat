@echo off
color 0A
echo ====================================================
echo   RA-ASF FINAL DEMO LAUNCHER
echo ====================================================
echo.

echo [1/4] Killing any stuck CARLA servers...
taskkill /F /IM CarlaUE4.exe /T >nul 2>&1
taskkill /F /IM CarlaUE4-Win64-Shipping.exe /T >nul 2>&1
timeout /t 3 /nobreak >nul

echo [2/4] Launching CARLA directly into Town02...
cd C:\Users\heman\Downloads\CARLA_0.9.16
start CarlaUE4.exe /Game/Maps/Town02 -carla-rpc-port=2000 -windowed -ResX=640 -ResY=480 -quality-level=Low -nosound -dx11

echo [3/4] Waiting 15 seconds for CARLA to load completely...
timeout /t 15 /nobreak >nul

echo [4/4] Starting the Python AI Brain...
cd C:\Users\heman\Music\ra_asf
call .\carla16_env\Scripts\activate.bat
cd final_implementation
python run_demo.py

echo.
echo Demo finished or crashed. Press any key to exit.
pause >nul
