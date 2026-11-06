$ErrorActionPreference = "Stop"
$Base = "https://github.com/bigdahaut/djeli/releases/latest/download"
$Bin = "djeli.exe"
$InstallDir = "$env:LOCALAPPDATA\djeli\bin"

$arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
switch ($arch) {
    "X64"   { $asset = "djeli-windows-x86_64.exe" }
    default { throw "Unsupported architecture: $arch" }
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Invoke-WebRequest -Uri "$Base/$asset" -OutFile "$InstallDir\$Bin"
Write-Host "Installed $Bin to $InstallDir"
