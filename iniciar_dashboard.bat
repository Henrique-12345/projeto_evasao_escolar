@echo off
cd /d "%~dp0"
echo Iniciando dashboard de Evasao Escolar...
python -m streamlit run dashboard/app.py --server.port 8501 --server.headless true
pause
