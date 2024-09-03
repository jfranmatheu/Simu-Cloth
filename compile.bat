@echo off
setlocal enabledelayedexpansion

:: Get the directory of the batch script
set "SCRIPT_DIR=%~dp0"

:: Change to the script directory
cd /d "%SCRIPT_DIR%"

:: Check if Python is installed and in PATH
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not found in your PATH. Please install Python and add it to your PATH.
    pause
    exit /b 1
)

:: Run the setup.py script to compile the cloth_simulation module
python setup.py build_ext

if %errorlevel% neq 0 (
    echo Compilation failed. Please check your mypyc installation and the cloth_simulation.py file.
    pause
    exit /b 1
)

:: Find the compiled .pyd file
for /r "build" %%F in (cloth_simulation.cp*-win_amd64.pyd) do (
    set "COMPILED_FILE=%%F"
)

if not defined COMPILED_FILE (
    echo Compiled file not found. Compilation may have failed.
    pause
    exit /b 1
)

:: Create the simucloth directory if it doesn't exist
if not exist "simucloth" mkdir "simucloth"

:: Copy and rename the file
copy "!COMPILED_FILE!" "simucloth\cloth_simulation.pyd"

if %errorlevel% neq 0 (
    echo Failed to copy the compiled file.
    pause
    exit /b 1
)

echo Compilation successful! File copied to simucloth\cloth_simulation.pyd
