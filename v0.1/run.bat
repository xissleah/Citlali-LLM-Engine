@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title Citlali CUDA CLI

rem 使用脚本所在目录，避免依赖当前工作目录。
set "ROOT=%~dp0"
set "EXE="

if exist "%ROOT%build\Release\citlali_cli.exe" set "EXE=%ROOT%build\Release\citlali_cli.exe"
if not defined EXE if exist "%ROOT%build\Release\main.exe" set "EXE=%ROOT%build\Release\main.exe"
if not defined EXE if exist "%ROOT%build\Debug\citlali_cli.exe" set "EXE=%ROOT%build\Debug\citlali_cli.exe"
if not defined EXE if exist "%ROOT%build\Debug\main.exe" set "EXE=%ROOT%build\Debug\main.exe"
if defined EXE goto HAVE_EXE

echo 未找到 CLI 可执行文件。
echo 请先编译，并确认 build\Release 下存在 citlali_cli.exe 或 main.exe。
pause
exit /b 1

:HAVE_EXE
set "DEFAULT_MODEL=C:\work\citlali\Qwen3-0.6B-Q4_K_M.gguf"
set "MODEL="
set /p "MODEL=GGUF 模型路径 [默认: %DEFAULT_MODEL%]: "
if defined MODEL goto MODEL_INPUT_DONE
set "MODEL=%DEFAULT_MODEL%"

:MODEL_INPUT_DONE
if exist "%MODEL%" goto MODEL_EXISTS
echo 找不到模型文件：%MODEL%
pause
exit /b 1

:MODEL_EXISTS
echo.
echo 请选择运行模式：
echo   [1] Single generation
echo   [2] Interactive chat
echo   [3] Model info
echo   [4] Prompt tokens
set "MODE="
set /p "MODE=模式 [默认: 1]: "
if defined MODE goto MODE_INPUT_DONE
set "MODE=1"

:MODE_INPUT_DONE
if "%MODE%"=="1" goto SINGLE
if "%MODE%"=="2" goto INTERACTIVE
if "%MODE%"=="3" goto INFO
if "%MODE%"=="4" goto TOKENS
echo 无效的模式：%MODE%
pause
exit /b 1

:SINGLE
set "PROMPT="
set /p "PROMPT=请输入 prompt: "
if defined PROMPT goto SINGLE_PROMPT_DONE
echo prompt 不能为空。
pause
exit /b 1

:SINGLE_PROMPT_DONE
set "MAX_NEW=64"
set /p "MAX_NEW=最大生成 token 数 [默认: 64]: "
if defined MAX_NEW goto SINGLE_MAX_DONE
set "MAX_NEW=64"

:SINGLE_MAX_DONE
set "CTX=2048"
set /p "CTX=最大上下文长度 [默认: 2048]: "
if defined CTX goto SINGLE_CTX_DONE
set "CTX=2048"

:SINGLE_CTX_DONE
set "EXTRA="
set "RAW="
set /p "RAW=不使用 chat template？输入 y，否则直接回车: "
if /i not "%RAW%"=="y" goto SINGLE_RUN
set "EXTRA=--no-chat-template"

:SINGLE_RUN
echo.
echo 正在启动：%EXE%
"%EXE%" --model "%MODEL%" --prompt "%PROMPT%" --max-new-tokens "%MAX_NEW%" --ctx "%CTX%" %EXTRA%
goto RESULT

:INTERACTIVE
set "MAX_NEW=128"
set /p "MAX_NEW=最大生成 token 数 [默认: 128]: "
if defined MAX_NEW goto INTERACTIVE_MAX_DONE
set "MAX_NEW=128"

:INTERACTIVE_MAX_DONE
set "CTX=2048"
set /p "CTX=最大上下文长度 [默认: 2048]: "
if defined CTX goto INTERACTIVE_CTX_DONE
set "CTX=2048"

:INTERACTIVE_CTX_DONE
echo.
echo 输入 /exit 退出，输入 /reset 清空对话历史。
"%EXE%" --model "%MODEL%" --interactive --max-new-tokens "%MAX_NEW%" --ctx "%CTX%"
goto RESULT

:INFO
echo.
"%EXE%" --model "%MODEL%" --info
goto RESULT

:TOKENS
set "PROMPT="
set /p "PROMPT=请输入要分词的 prompt: "
if defined PROMPT goto TOKENS_RUN
echo prompt 不能为空。
pause
exit /b 1

:TOKENS_RUN
"%EXE%" --model "%MODEL%" --prompt "%PROMPT%" --tokens
goto RESULT

:RESULT
echo.
if errorlevel 1 goto FAILED
echo CLI 执行完成。
goto END

:FAILED
echo CLI 执行失败，退出码：%errorlevel%

:END
pause
endlocal
