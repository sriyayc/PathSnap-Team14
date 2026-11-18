@echo off
REM Laptop 2: serve the evaluator UI on port 8501 and open it in the browser.
cd /d "%~dp0ui"
start "" http://localhost:8501/index.html
python -m http.server 8501
pause
