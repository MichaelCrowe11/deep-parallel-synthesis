@echo off
echo === Verifying Docker Installation ===
echo.

where docker >nul 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Docker is installed!
    echo.
    docker --version
    echo.
    
    docker info >nul 2>&1
    if %errorlevel% equ 0 (
        echo [SUCCESS] Docker is running!
        echo.
        echo Docker is ready for DPS deployment.
        echo.
        echo Next step: Run start_deployment.bat
    ) else (
        echo [WARNING] Docker is installed but not running.
        echo Please start Docker Desktop from the Start Menu.
    )
) else (
    echo [ERROR] Docker is not installed or not in PATH.
    echo Please complete the Docker Desktop installation.
)

echo.
pause