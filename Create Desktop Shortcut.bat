@echo off
echo Creating desktop shortcut...

:: Get desktop path
for /f "tokens=2*" %%a in ('reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders" /v Desktop 2^>nul') do set "DESKTOP=%%b"

:: Create shortcut using PowerShell
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%DESKTOP%\RAG Document Analysis.lnk'); $s.TargetPath = '%~dp0RAG System.bat'; $s.WorkingDirectory = '%~dp0'; $s.IconLocation = '%SystemRoot%\System32\SHELL32.dll,41'; $s.Description = 'RAG Document Analysis System'; $s.Save()"

if exist "%DESKTOP%\RAG Document Analysis.lnk" (
    echo.
    echo ✓ Desktop shortcut created successfully!
    echo.
    echo You can now launch the application from your desktop.
) else (
    echo.
    echo ✗ Failed to create shortcut.
    echo   Please run this script as administrator.
)

echo.
pause
