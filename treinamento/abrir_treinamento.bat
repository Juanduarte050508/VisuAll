@echo off
setlocal
cd /d "%~dp0"
python interface_treinamento.py
if errorlevel 1 (
  echo.
  echo Nao foi possivel abrir a interface. Verifique se o Python esta instalado.
  pause
)
endlocal
