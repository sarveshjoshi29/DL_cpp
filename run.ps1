
# 1. Define folder names and target
$BuildDir = "build"
$ExeTargetName = "test" # Ensure this matches the project name in CMakeLists.txt

# 2. Create the build directory if it doesn't exist
if (!(Test-Path -Path $BuildDir)) {
    Write-Host "--- Creating Build Directory ---" -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# 3. Enter build directory
cd $BuildDir

# 4. Run CMake to generate build files
Write-Host "--- Running CMake Configuration ---" -ForegroundColor Cyan
cmake ..

# 5. Build the project using CMake's build tool abstraction
Write-Host "--- Compiling Project ---" -ForegroundColor Cyan
cmake --build . 

# 6. Locate and run the executable
# Note: Windows CMake often places binaries in Debug/ or Release/ subfolders

cd ../exe

$ExeFile = "$ExeTargetName.exe"
Write-Host "--- Successfully Built! Executing $ExeFile ---" -ForegroundColor Green
Write-Host "------------------------------------------------"

& ".\$ExeFile"
cd ..

