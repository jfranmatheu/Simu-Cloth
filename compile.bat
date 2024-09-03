@echo off
setlocal enabledelayedexpansion

:: Use Python 3.11 specifically (adjust path as necessary)
set PYTHON_PATH="C:\Users\JF\AppData\Local\Programs\Python\Python311\python.exe"

:: Get the directory of the batch script
d "%~dp0"

"%PYTHON_PATH%" -m pip install --upgrade cython
"%PYTHON_PATH%" setup.py build_ext --inplace

if %errorlevel% neq 0 (
    echo Compilation failed. Please check your Cython installation and the cloth_simulation.pyx file.
    pause
    exit /b 1
)
