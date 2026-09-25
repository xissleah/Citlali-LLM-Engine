@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Citlali Remote Inference

set "ROOT=%~dp0"
set "CLI=%ROOT%build\Release\citlali_cli.exe"
if not exist "%CLI%" set "CLI=%ROOT%build\Debug\citlali_cli.exe"
if not exist "%CLI%" (
    echo citlali_cli.exe was not found. Build the project first.
    pause
    exit /b 1
)

set "ADB="
if exist "%ROOT%platform-tools\adb.exe" set "ADB=%ROOT%platform-tools\adb.exe"
if not defined ADB if defined ANDROID_HOME if exist "%ANDROID_HOME%\platform-tools\adb.exe" set "ADB=%ANDROID_HOME%\platform-tools\adb.exe"
if not defined ADB if exist "C:\Android\platform-tools\adb.exe" set "ADB=C:\Android\platform-tools\adb.exe"
if not defined ADB for %%A in (adb.exe) do set "ADB=%%~$PATH:A"
if not defined ADB (
    echo adb.exe was not found. Install Android platform-tools first.
    pause
    exit /b 1
)

set "ADB_PORT=5037"
"%ADB%" -P 5038 devices 2>nul | findstr /r /c:"[ ]device$" >nul
if not errorlevel 1 set "ADB_PORT=5038"

set "USB_SERIAL="
set "WIFI_SERIAL="
for /f "skip=1 tokens=1,2" %%A in ('"%ADB%" -P %ADB_PORT% devices') do (
    if "%%B"=="device" (
        echo %%A | %SystemRoot%\System32\findstr.exe /c:":" >nul
        if errorlevel 1 if not defined USB_SERIAL set "USB_SERIAL=%%A"
        if not errorlevel 1 if not defined WIFI_SERIAL set "WIFI_SERIAL=%%A"
    )
)

echo.
echo Select remote transport:
echo   [1] USB ^(ADB TCP forwarding^)
echo   [2] Wi-Fi 6 ^(wireless ADB TCP tunnel^)
set "TRANSPORT="
set /p "TRANSPORT=Enter 1 or 2 [default: 1]: "
if not defined TRANSPORT set "TRANSPORT=1"
if not "%TRANSPORT%"=="1" if not "%TRANSPORT%"=="2" (
    echo Invalid selection: %TRANSPORT%
    pause
    exit /b 1
)

set "DEFAULT_MODEL=C:\work\citlali\Qwen3-0.6B-Q4_K_M.gguf"
set "MODEL="
set /p "MODEL=GGUF model path [default: %DEFAULT_MODEL%]: "
if not defined MODEL set "MODEL=%DEFAULT_MODEL%"
if not exist "%MODEL%" (
    echo Model not found: %MODEL%
    pause
    exit /b 1
)

set "OFFLOAD=layer:0,1,2,3,4,5,6,7,8,9,10,11,12,13"
set /p "OFFLOAD=Offload range [default: %OFFLOAD%]: "
if not defined OFFLOAD set "OFFLOAD=layer:0,1,2,3,4,5,6,7,8,9,10,11,12,13"

set "PROMPT="
set /p "PROMPT=Prompt: "
if not defined PROMPT (
    echo Prompt cannot be empty.
    pause
    exit /b 1
)

set "MAX_NEW=64"
set /p "MAX_NEW=Maximum new tokens [default: 64]: "
if not defined MAX_NEW set "MAX_NEW=64"

set "CTX=2048"
set /p "CTX=Context length [default: 2048]: "
if not defined CTX set "CTX=2048"

if "%TRANSPORT%"=="1" goto USB_MODE
goto WIFI_MODE

:USB_MODE
if not defined USB_SERIAL (
    echo No authorized USB ADB device was found.
    pause
    exit /b 1
)
"%ADB%" -P %ADB_PORT% -s "%USB_SERIAL%" usb >nul 2>nul
ping 127.0.0.1 -n 2 >nul
"%ADB%" -P %ADB_PORT% -s "%USB_SERIAL%" shell "pkill -f citlali_remote_server" >nul 2>nul
"%ADB%" -P %ADB_PORT% -s "%USB_SERIAL%" shell "nohup /data/local/tmp/citlali_remote_server 27183 /data/local/tmp/q4k_matvec.spv 127.0.0.1 >/data/local/tmp/citlali_remote_server.log 2>&1 </dev/null &"
if errorlevel 1 goto SERVER_FAILED
"%ADB%" -P %ADB_PORT% -s "%USB_SERIAL%" forward tcp:27183 tcp:27183 >nul
if errorlevel 1 goto SERVER_FAILED
set "ENDPOINT=127.0.0.1:27183"
set "TRANSPORT_NAME=usb"
goto RUN_ENGINE

:WIFI_MODE
echo.
echo The tablet and PC must be connected to the same Wi-Fi 6 network.
if defined USB_SERIAL (
    "%ADB%" -P %ADB_PORT% -s "%USB_SERIAL%" shell "pkill -f citlali_remote_server" >nul 2>nul
    "%ADB%" -P %ADB_PORT% -s "%USB_SERIAL%" shell "ip -f inet addr show wlan0"
) else if defined WIFI_SERIAL (
    "%ADB%" -P %ADB_PORT% -s "%WIFI_SERIAL%" shell "pkill -f citlali_remote_server" >nul 2>nul
    "%ADB%" -P %ADB_PORT% -s "%WIFI_SERIAL%" shell "ip -f inet addr show wlan0"
) else (
    echo No authorized ADB device was found.
    pause
    exit /b 1
)
set "TABLET_IP="
set /p "TABLET_IP=Enter the tablet Wi-Fi IPv4 address shown above: "
if not defined TABLET_IP (
    echo Tablet IP cannot be empty.
    pause
    exit /b 1
)
if defined USB_SERIAL (
    "%ADB%" -P %ADB_PORT% -s "%USB_SERIAL%" tcpip 5555
    if errorlevel 1 goto SERVER_FAILED
)
"%ADB%" -P %ADB_PORT% connect %TABLET_IP%:5555
if errorlevel 1 goto SERVER_FAILED
"%ADB%" -P %ADB_PORT% -s %TABLET_IP%:5555 shell "nohup /data/local/tmp/citlali_remote_server 27183 /data/local/tmp/q4k_matvec.spv 127.0.0.1 >/data/local/tmp/citlali_remote_server.log 2>&1 </dev/null &"
if errorlevel 1 goto SERVER_FAILED
"%ADB%" -P %ADB_PORT% -s %TABLET_IP%:5555 forward tcp:27184 tcp:27183 >nul
if errorlevel 1 goto SERVER_FAILED
set "ENDPOINT=127.0.0.1:27184"
set "TRANSPORT_NAME=wifi6"

:RUN_ENGINE
echo.
echo Running through %TRANSPORT_NAME% at %ENDPOINT%
"%CLI%" --model "%MODEL%" --prompt "%PROMPT%" --max-new-tokens "%MAX_NEW%" --ctx "%CTX%" --quantized --remote-transport "%TRANSPORT_NAME%" --remote "%ENDPOINT%" --remote-backend gpu --offload "%OFFLOAD%"
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" echo Inference failed with exit code %RESULT%.
if "%RESULT%"=="0" echo Inference completed.
pause
exit /b %RESULT%

:SERVER_FAILED
echo Could not start the tablet server or create ADB forwarding.
echo Check device authorization and ADB server port %ADB_PORT%.
pause
exit /b 1
