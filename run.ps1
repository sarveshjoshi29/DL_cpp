

$BuildDir = "build"
$ExeTargetName = "test"

if (!(Test-Path -Path $BuildDir)) {
    Write-Host "--- Creating Build Directory ---" -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}
cd $BuildDir

# Run CMake
Write-Host "--- Running CMake Configuration ---" -ForegroundColor Cyan
cmake ..

# Build 
Write-Host "--- Compiling Project ---" -ForegroundColor Cyan
cmake --build . 


# Run generated executable
cd ../exe

$ExeFile = "$ExeTargetName.exe"
Write-Host "--- Successfully Built! Executing $ExeFile ---" -ForegroundColor Green
Write-Host "------------------------------------------------"

& ".\$ExeFile"
cd ..