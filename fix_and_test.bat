@echo off
REM Clean Python cache and run tests on Windows
REM Execute: fix_and_test.bat

echo.
echo ======================================================================
echo Cleaning Python cache...
echo ======================================================================
echo.

REM Remove pycache directories
for /d /r . %%d in (__pycache__) do @if exist "%%d" (
    echo Removing %%d
    rmdir /s /q "%%d"
)

REM Remove pytest cache
for /d /r . %%d in (.pytest_cache) do @if exist "%%d" (
    echo Removing %%d
    rmdir /s /q "%%d"
)

echo.
echo ======================================================================
echo Running pytest with fresh cache...
echo ======================================================================
echo.

python -m pytest tests/ -v --tb=short

echo.
echo ======================================================================
echo Test run complete!
echo ======================================================================
echo.

pause
