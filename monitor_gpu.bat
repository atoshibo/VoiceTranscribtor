@echo off
echo Monitoring GPU usage during transcription...
echo.
echo Upload a file at: https://localhost:8443/?token=Y3faMWCXbO1R0cqxznK7kQtmyjeZSpsh
echo.
echo GPU Status (refreshes every 2 seconds):
echo ============================================
:loop
docker compose exec -T worker nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo.
timeout /t 2 /nobreak >nul
goto loop
