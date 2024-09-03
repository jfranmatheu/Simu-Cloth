@echo off
setlocal enabledelayedexpansion

:: Get the directory of the batch script
set "SCRIPT_DIR=%~dp0"

:: Change to the script directory
cd /d "%SCRIPT_DIR%"

:: Use Python 3.11 specifically (adjust path as necessary)
set PYTHON_PATH="C:\\Users\\JF\\AppData\\Local\\Programs\\Python\\Python311\\python.exe"

:: Check if Python 3.11 is available
if not exist "%PYTHON_PATH%" (
    echo Python 3.11 not found at %PYTHON_PATH%. Please install it or adjust the path.
    pause
    exit /b 1
)

:: Ensure NumPy is installed
"%PYTHON_PATH%" -m pip install numpy

:: Run the setup.py script to compile the cloth_simulation module
"%PYTHON_PATH%" setup.py build_ext

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
