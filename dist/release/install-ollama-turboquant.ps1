# Ollama TurboQuant Installer for Windows
# This script installs Ollama with TurboQuant KV cache compression support

$ErrorActionPreference = "Stop"

Write-Host "Ollama TurboQuant Installer" -ForegroundColor Cyan
Write-Host "Version: 0.6.0-turboquant" -ForegroundColor Gray
Write-Host ""

# Default install directory
$InstallDir = "$env:LOCALAPPDATA\Programs\Ollama"

Write-Host "Installing to: $InstallDir"

# Create directory
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

# Extract ollama.exe to install directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OllamaExe = Join-Path $ScriptDir "ollama.exe"

if (Test-Path $OllamaExe) {
    Copy-Item $OllamaExe "$InstallDir\ollama.exe" -Force
} else {
    Write-Host "Error: ollama.exe not found in script directory" -ForegroundColor Red
    Write-Host "Please ensure ollama.exe is in the same directory as this script" -ForegroundColor Yellow
    exit 1
}

# Add to PATH if not already there
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
    Write-Host "Added $InstallDir to user PATH" -ForegroundColor Green
}

# Set TurboQuant environment variable
$CurrentTQ = [Environment]::GetEnvironmentVariable("OLLAMA_TURBOQUANT", "User")
if (-not $CurrentTQ) {
    $EnableTQ = Read-Host "Enable TurboQuant KV cache compression? (Y/n)"
    if ($EnableTQ -ne "n" -and $EnableTQ -ne "N") {
        [Environment]::SetEnvironmentVariable("OLLAMA_TURBOQUANT", "true", "User")
        Write-Host "TurboQuant enabled (OLLAMA_TURBOQUANT=true)" -ForegroundColor Green
        Write-Host "  - 3-bit compression (~5.3x theoretical, 4x effective)" -ForegroundColor Gray
        Write-Host "  - Requires NVIDIA CUDA GPU" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start Ollama:" -ForegroundColor Cyan
Write-Host "  ollama serve" -ForegroundColor White
Write-Host ""
Write-Host "To run a model:" -ForegroundColor Cyan
Write-Host "  ollama run gemma3" -ForegroundColor White
Write-Host ""
Write-Host "TurboQuant can be configured via:" -ForegroundColor Cyan
Write-Host '  $env:OLLAMA_TURBOQUANT="true"   # Enable 3-bit (default)' -ForegroundColor White
Write-Host '  $env:OLLAMA_TURBOQUANT="4"      # 4-bit (higher quality)' -ForegroundColor White
Write-Host '  $env:OLLAMA_TURBOQUANT="2"      # 2-bit (max compression)' -ForegroundColor White
