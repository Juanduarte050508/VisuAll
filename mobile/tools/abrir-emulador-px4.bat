@echo off
setlocal EnableExtensions EnableDelayedExpansion
title VisuAll - emulador Pixel_4

REM ============================================================
REM  VisuAll - abre (e cria, se preciso) o emulador Pixel_4.
REM
REM  Objetivo: o time so precisa dar "git pull" e rodar este
REM  arquivo. Se a AVD nao existir, ele baixa a imagem x86
REM  (32-bit, roda o MediaPipe NATIVO = rapido) e cria a AVD
REM  ja com a webcam ligada.
REM
REM  Unico requisito: ter o Android Studio / SDK instalado.
REM ============================================================

set "IMG=system-images;android-30;google_apis_playstore;x86"
set "AVD=Pixel_4"

REM ---------- localizar o SDK ----------
set "SDK=%LOCALAPPDATA%\Android\Sdk"
if not exist "%SDK%\emulator\emulator.exe" set "SDK=%ANDROID_HOME%"
if not exist "%SDK%\emulator\emulator.exe" set "SDK=%ANDROID_SDK_ROOT%"
if not exist "%SDK%\emulator\emulator.exe" (
    echo [ERRO] Nao encontrei o Android SDK.
    echo Instale o Android Studio ou defina ANDROID_HOME.
    goto erro
)
set "EMU=%SDK%\emulator\emulator.exe"

REM ---------- a AVD ja existe? entao so abre ----------
"%EMU%" -list-avds | findstr /x "%AVD%" >nul
if not errorlevel 1 goto abrir

echo A AVD "%AVD%" nao existe neste PC. Vou criar (uma vez so)...
echo.

REM ---------- localizar sdkmanager / avdmanager ----------
set "SDKMAN="
for %%P in (
    "%SDK%\cmdline-tools\latest\bin\sdkmanager.bat"
    "%SDK%\cmdline-tools\bin\sdkmanager.bat"
    "%SDK%\tools\bin\sdkmanager.bat"
) do if exist %%~P set "SDKMAN=%%~P"

set "AVDMAN="
for %%P in (
    "%SDK%\cmdline-tools\latest\bin\avdmanager.bat"
    "%SDK%\cmdline-tools\bin\avdmanager.bat"
    "%SDK%\tools\bin\avdmanager.bat"
) do if exist %%~P set "AVDMAN=%%~P"

if not defined SDKMAN goto semTools
if not defined AVDMAN goto semTools

REM ---------- baixar a imagem x86 (aceita licenca) ----------
echo [1/3] Baixando a imagem x86 (pode demorar na 1a vez)...
(for /l %%i in (1,1,30) do @echo y) | "%SDKMAN%" "%IMG%"
if errorlevel 1 goto falhaImg

REM ---------- criar a AVD ----------
echo [2/3] Criando a AVD "%AVD%"...
echo no | "%AVDMAN%" create avd -n "%AVD%" -k "%IMG%" -d pixel_4 --force
if errorlevel 1 goto falhaAvd

REM ---------- ligar a webcam real na camera frontal ----------
echo [3/3] Configurando a webcam...
set "CFG=%USERPROFILE%\.android\avd\%AVD%.avd\config.ini"
if exist "%CFG%" >>"%CFG%" echo hw.camera.front=webcam0

echo.
echo AVD criada com sucesso!
echo.

:abrir
echo Abrindo a Pixel_4...
start "" "%EMU%" -avd %AVD% -no-boot-anim
exit /b 0

REM ---------- erros ----------
:semTools
echo [ERRO] Nao encontrei sdkmanager/avdmanager.
echo No Android Studio: Settings ^> Languages ^& Frameworks ^> Android SDK
echo   ^> aba "SDK Tools" ^> marque "Android SDK Command-line Tools".
goto erro
:falhaImg
echo [ERRO] Falha ao baixar a imagem do sistema.
goto erro
:falhaAvd
echo [ERRO] Falha ao criar a AVD.
goto erro
:erro
echo.
pause
exit /b 1
