@echo off
echo Starting Surgical Monitor App...
echo Please wait for the browser to open automatically.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py

pause
