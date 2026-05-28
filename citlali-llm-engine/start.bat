@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Citlali LLM Engine - Launcher
echo ============================================
echo.

:: ========== Check Rust environment ==========
where cargo >nul 2>&1
if !errorlevel! equ 0 goto :rust_ok

echo [!] Rust not found. Installing...
echo.
echo [*] Downloading rustup-init.exe ...
curl -o "%TEMP%\rustup-init.exe" https://win.rustup.rs/x86_64 --silent --show-error
if !errorlevel! neq 0 goto :err_download

echo [*] Installing Rust (minimal profile)...
"%TEMP%\rustup-init.exe" -y --profile minimal --default-toolchain stable
if !errorlevel! neq 0 goto :err_install

set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
del "%TEMP%\rustup-init.exe" >nul 2>&1
echo [OK] Rust installed successfully.
echo.
goto :find_model

:rust_ok
for /f "tokens=*" %%v in ('rustc --version') do echo [OK] Found: %%v
echo.

:find_model
:: ========== Find .gguf model file ==========
set "MODEL_DIR=%~dp0model"

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
if not exist "%MODEL_DIR%\*.gguf" goto :err_no_model

set "GGUF_FILE="
for %%f in ("%MODEL_DIR%\*.gguf") do (
    if not defined GGUF_FILE set "GGUF_FILE=%%f"
)

echo [OK] Model: !GGUF_FILE!
echo.

:: ========== Build project (release) ==========
echo [*] Building project (release mode, may take a few minutes)...
cargo build --release -p citlali-cli
if !errorlevel! neq 0 goto :err_build
echo [OK] Build complete.
echo.

:: ========== Launch inference ==========
echo [*] Starting Citlali CLI...
echo ============================================
echo.
"%~dp0target\release\citlali.exe" "!GGUF_FILE!"
goto :done

:err_download
echo [ERROR] Download failed. Check your network.
echo You can install manually: https://rustup.rs
goto :done

:err_install
echo [ERROR] Rust installation failed.
goto :done

:err_no_model
echo [ERROR] No .gguf file found in: %MODEL_DIR%
echo Please put a GGUF model file in the model folder.
goto :done

:err_build
echo [ERROR] Build failed.
goto :done

:done
echo.
pause
