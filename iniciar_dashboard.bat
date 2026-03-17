@echo off
cd /d "%~dp0"
echo Iniciando dashboard de Evasao Escolar...
echo.

:: Usa o Python 3.13 onde os pacotes estao instalados
python -m streamlit run dashboard/app.py --server.port 8501 --server.headless true

pause
