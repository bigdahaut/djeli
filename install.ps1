$ErrorActionPreference = "Stop"

$Base = "https://github.com/bigdahaut/djeli/releases/latest/download"
$Bin = "djeli.exe"
$InstallDir = "$env:LOCALAPPDATA\djeli\bin"
$ModelsDir = "$env:USERPROFILE\.djeli\models"

$arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
switch ($arch) {
    "X64"   { $asset = "djeli-windows-x86_64.exe" }
    default { throw "Unsupported architecture: $arch" }
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Invoke-WebRequest -Uri "$Base/$asset" -OutFile "$InstallDir\$Bin"

# Automatically create the models directory
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

Write-Host "Installed $Bin to $InstallDir"
Write-Host "Created models directory at $ModelsDir"

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$InstallDir", "User")
    Write-Host "Added $InstallDir to your user PATH. Restart your terminal."
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Download a .gguf model and place it in: $ModelsDir"
Write-Host "  2. Restart your terminal."
Write-Host "  3. Start the engine: djeli serve"
