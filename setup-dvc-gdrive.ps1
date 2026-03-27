param(
    [string]$EnvName = "dvc",
    [string]$PythonVersion = "3.10",
    [string]$ProjectPath = ".",
    [string]$RemoteName = "gdrive",
    [string]$GDriveFolderId = "",
    [string]$TrackPath = "",
    [switch]$RunFirstPush,
    [string]$GDriveClientId = "",
    [string]$GDriveClientSecret = ""
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) {
    Write-Host "`n==== $msg ====" -ForegroundColor Cyan
}

function Write-WarnMsg($msg) {
    Write-Host "[WARN] $msg" -ForegroundColor Yellow
}

function Write-InfoMsg($msg) {
    Write-Host "[INFO] $msg" -ForegroundColor Gray
}

function Require-Command($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Command not found: $name. Please install it first and make sure it is available in PATH."
    }
}

function Invoke-Safe($scriptBlock, $errorMessage) {
    try {
        & $scriptBlock
        return $true
    }
    catch {
        Write-WarnMsg "$errorMessage"
        Write-WarnMsg $_.Exception.Message
        return $false
    }
}

function Test-RemoteExists($envName, $remoteName) {
    try {
        $remoteList = conda run -n $envName dvc remote list 2>$null
        if (-not $remoteList) { return $false }
        return ($remoteList -match "(?m)^$([regex]::Escape($remoteName))\s")
    }
    catch {
        return $false
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
    Write-InfoMsg "Environment $EnvName already exists. Skipping creation."
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
        try {
            git commit -m "init dvc" 2>$null
        }
        catch {
            Write-WarnMsg "Git commit for DVC initialization was skipped."
        }
    } else {
        Write-InfoMsg ".dvc already exists. Skipping dvc init."
    }

    Write-Step "Optionally configure Google Drive remote"
    if ([string]::IsNullOrWhiteSpace($GDriveFolderId)) {
        Write-InfoMsg "No Google Drive folder ID was provided. Remote configuration is skipped."
        Write-InfoMsg "You can configure it later with:"
        Write-Host "  dvc remote add -d $RemoteName `"gdrive://YOUR_FOLDER_ID`""
    }
    else {
        Write-InfoMsg "A Google Drive folder ID was provided. The script will try to configure the remote."

        if (Test-RemoteExists $EnvName $RemoteName) {
            Write-InfoMsg "Remote $RemoteName already exists. Skipping removal."
        }
        else {
            Write-InfoMsg "Remote $RemoteName does not exist yet."
        }

        $addedRemote = Invoke-Safe {
            conda run -n $EnvName dvc remote remove $RemoteName 2>$null
            conda run -n $EnvName dvc remote add -d $RemoteName "gdrive://$GDriveFolderId"
        } "Failed to add or reset Google Drive remote."

        if ($addedRemote) {
            Invoke-Safe {
                conda run -n $EnvName dvc remote modify $RemoteName gdrive_acknowledge_abuse true
            } "Failed to set gdrive_acknowledge_abuse. This can be configured later."

            if (-not [string]::IsNullOrWhiteSpace($GDriveClientId) -and -not [string]::IsNullOrWhiteSpace($GDriveClientSecret)) {
                Invoke-Safe {
                    conda run -n $EnvName dvc remote modify --local $RemoteName gdrive_client_id $GDriveClientId
                    conda run -n $EnvName dvc remote modify --local $RemoteName gdrive_client_secret $GDriveClientSecret
                } "Failed to write custom Google OAuth credentials. You can configure them later."
            }
            else {
                Write-InfoMsg "No custom Google OAuth credentials were provided. Default authorization flow will be used later."
            }
        }
        else {
            Write-WarnMsg "Google Drive remote configuration was skipped due to an error."
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($TrackPath)) {
        Write-Step "Track data path: $TrackPath"
        if (-not (Test-Path $TrackPath)) {
            Write-WarnMsg "TrackPath does not exist: $TrackPath"
            Write-WarnMsg "Skipping dvc add for the track path."
        }
        else {
            conda run -n $EnvName dvc add $TrackPath
            git add . 2>$null
            Write-InfoMsg "dvc add has been executed. Please review the changes before committing."
        }
    }
    else {
        Write-InfoMsg "No TrackPath was provided. Skipping dvc add."
    }

    if ($RunFirstPush) {
        Write-Step "Optionally run the first dvc push"
        if ([string]::IsNullOrWhiteSpace($GDriveFolderId)) {
            Write-WarnMsg "RunFirstPush was requested, but no Google Drive folder ID was provided."
            Write-WarnMsg "Skipping dvc push."
        }
        elseif (-not (Test-RemoteExists $EnvName $RemoteName)) {
            Write-WarnMsg "RunFirstPush was requested, but the remote is not configured correctly."
            Write-WarnMsg "Skipping dvc push."
        }
        else {
            Write-InfoMsg "The first push or pull may open a browser window for Google authorization."
            Invoke-Safe {
                conda run -n $EnvName dvc push
            } "The first dvc push failed. You can retry it later manually."
        }
    }

    Write-Step "Done"
    Write-Host "Project path: $ProjectPath"
    Write-Host "Conda environment: $EnvName"
    Write-Host "Remote name: $RemoteName"

    if ([string]::IsNullOrWhiteSpace($GDriveFolderId)) {
        Write-Host "Google Drive folder ID: <not configured>"
    }
    else {
        Write-Host "Google Drive folder ID: <provided>"
    }

    Write-Host "`nCommon commands to use later:"
    Write-Host "  conda activate $EnvName"
    Write-Host "  dvc status"
    Write-Host "  dvc push"
    Write-Host "  dvc pull"
    Write-Host "  git add ."
    Write-Host '  git commit -m "update dvc config/data"'
    Write-Host "  git push"

    Write-Host "`nManual Google Drive setup later:"
    Write-Host "  dvc remote add -d $RemoteName `"gdrive://YOUR_FOLDER_ID`""
    Write-Host "  dvc remote modify $RemoteName gdrive_acknowledge_abuse true"
    Write-Host "  dvc remote modify --local $RemoteName gdrive_client_id `"YOUR_CLIENT_ID`""
    Write-Host "  dvc remote modify --local $RemoteName gdrive_client_secret `"YOUR_CLIENT_SECRET`""
}
finally {
    Pop-Location
}