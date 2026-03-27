param(
    [string]$EnvName = "dvc",
    [string]$PythonVersion = "3.10",
    [string]$ProjectPath = ".",
    [string]$RemoteName = "gdrive",
    [Parameter(Mandatory = $true)]
    [string]$GDriveFolderId,
    [string]$TrackPath = "",
    [switch]$RunFirstPush,
    [string]$GDriveClientId = "",
    [string]$GDriveClientSecret = ""
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) {
    Write-Host "`n==== $msg ====" -ForegroundColor Cyan
}

function Require-Command($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Command not found: $name. Please install it first and make sure it is available in PATH."
    }
}

Write-Step "Check required commands"
Require-Command "git"
Require-Command "conda"

$ProjectPath = (Resolve-Path $ProjectPath).Path

Write-Step "Check whether conda is available"
$condaInfo = conda info --base 2>$null
if (-not $condaInfo) {
    throw "Conda is not working correctly. Please install Miniconda or Anaconda, and run this script in an initialized PowerShell or Anaconda Prompt."
}

Write-Step "Create conda environment if it does not exist"
$envsText = conda env list
if ($envsText -notmatch "(?m)^\s*$([regex]::Escape($EnvName))\s") {
    conda create -y -n $EnvName "python=$PythonVersion"
} else {
    Write-Host "Environment $EnvName already exists. Skipping creation."
}

Write-Step "Upgrade pip and install DVC with Google Drive support"
conda run -n $EnvName python -m pip install --upgrade pip
conda run -n $EnvName python -m pip install "dvc[gdrive]"

Write-Step "Show DVC version"
conda run -n $EnvName dvc version

Push-Location $ProjectPath
try {
    Write-Step "Check whether the current folder is a Git repository"
    git rev-parse --is-inside-work-tree *> $null

    Write-Step "Initialize DVC if it has not been initialized"
    if (-not (Test-Path ".dvc")) {
        conda run -n $EnvName dvc init
        git add .dvc .dvcignore 2>$null
        git commit -m "init dvc" 2>$null
    } else {
        Write-Host ".dvc already exists. Skipping dvc init."
    }

    Write-Step "Configure Google Drive remote"
    $existingRemotes = conda run -n $EnvName dvc remote list
    if ($existingRemotes -notmatch "(?m)^$([regex]::Escape($RemoteName))\s") {
        conda run -n $EnvName dvc remote add --default $RemoteName "gdrive://$GDriveFolderId"
    } else {
        Write-Host "Remote $RemoteName already exists. Updating URL and setting it as default."
        conda run -n $EnvName dvc remote modify $RemoteName url "gdrive://$GDriveFolderId"
        conda run -n $EnvName dvc remote default $RemoteName
    }

    conda run -n $EnvName dvc remote modify $RemoteName gdrive_acknowledge_abuse true

    if ($GDriveClientId -and $GDriveClientSecret) {
        Write-Step "Write custom Google Cloud OAuth credentials to local private config"
        conda run -n $EnvName dvc remote modify --local $RemoteName gdrive_client_id $GDriveClientId
        conda run -n $EnvName dvc remote modify --local $RemoteName gdrive_client_secret $GDriveClientSecret
    } else {
        Write-Host "No custom client_id or client_secret provided. The default authorization flow will be used."
    }

    if ($TrackPath) {
        Write-Step "Track data path: $TrackPath"
        if (-not (Test-Path $TrackPath)) {
            throw "TrackPath does not exist: $TrackPath"
        }

        conda run -n $EnvName dvc add $TrackPath
        git add . 2>$null
        Write-Host "dvc add has been executed. Please review the changes before committing."
    }

    if ($RunFirstPush) {
        Write-Step "Run the first dvc push"
        Write-Host "The first push or pull may open a browser window for Google authorization."
        conda run -n $EnvName dvc push
    }

    Write-Step "Done"
    Write-Host "Project path: $ProjectPath"
    Write-Host "Conda environment: $EnvName"
    Write-Host "Remote name: $RemoteName"
    Write-Host "Google Drive Folder ID: $GDriveFolderId"

    Write-Host "`nCommon commands to use later:"
    Write-Host "  conda activate $EnvName"
    Write-Host "  dvc status"
    Write-Host "  dvc push"
    Write-Host "  dvc pull"
    Write-Host "  git add ."
    Write-Host '  git commit -m "update dvc config/data"'
    Write-Host "  git push"
}
finally {
    Pop-Location
}