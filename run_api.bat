@echo off
REM Laptop 1: start the PathSnap API on all network interfaces, port 8000.
REM Evaluation trips go in data\eval\<trip_id>\ (falls back to data\student_test if empty).
cd /d "%~dp0"
echo Your IPv4 address(es) - give one of these to Laptop 2:
ipconfig | findstr /i "IPv4"
echo.
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
pause
