@echo off
title VidMash - Share to Live Internet ($0 Cloudflare Tunnel)
echo =====================================================================
echo  VidMash - Instant $0 Global Deployment via Cloudflare Tunnel
echo =====================================================================
echo.

if not exist cloudflared.exe (
    echo [1/2] Downloading official Cloudflare Tunnel client (free, secure)...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile 'cloudflared.exe'"
    if not exist cloudflared.exe (
        echo [ERROR] Failed to download cloudflared.exe. Please verify internet connection.
        pause
        exit /b 1
    )
    echo [1/2] Cloudflare client downloaded successfully!
) else (
    echo [1/2] Cloudflare client verified.
)

echo.
echo [2/2] Launching secure public tunnel for port 8000...
echo.
echo =====================================================================
echo  Look for the line below containing:
echo  https://*.trycloudflare.com
echo.
echo  Share that HTTPS link with ANYONE in the world - on mobile or web!
echo  Press Ctrl+C to close the public tunnel at any time.
echo =====================================================================
echo.

cloudflared.exe tunnel --url http://localhost:8000
pause
