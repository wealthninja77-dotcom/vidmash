@echo off
echo ========================================================
echo        Pushing VidMash to your GitHub Repository
echo ========================================================
echo.
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo ========================================================
    echo  [SUCCESS] Code successfully pushed to GitHub!
    echo ========================================================
) else (
    echo ========================================================
    echo  [NOTICE] If GitHub asked for a password, you need to
    echo  use a GitHub Personal Access Token or browser sign-in.
    echo ========================================================
)
echo.
pause
