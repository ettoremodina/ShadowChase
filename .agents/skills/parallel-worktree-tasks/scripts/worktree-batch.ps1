[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Create", "List", "Remove")]
    [string]$Action,

    [string]$RepositoryPath = ".",
    [string]$BatchId = "",
    [string]$TaskId = "",
    [ValidateSet("task", "integration")]
    [string]$Kind = "task",
    [string]$BaseRef = "HEAD",
    [string]$WorktreeRoot = "",
    [string]$WorktreePath = "",
    [switch]$DeleteBranch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Git {
    param(
        [string]$WorkingDirectory,
        [string[]]$Arguments
    )

    $output = @(& git -C $WorkingDirectory @Arguments 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed:`n$($output -join [Environment]::NewLine)"
    }
    return @($output)
}

function ConvertTo-SafeSlug {
    param(
        [string]$Value,
        [string]$Label
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "$Label is required."
    }

    $slug = $Value.Trim().ToLowerInvariant() -replace "[^a-z0-9._-]+", "-"
    $slug = $slug.Trim([char[]]"-._")
    if ([string]::IsNullOrWhiteSpace($slug)) {
        throw "$Label does not contain a usable identifier."
    }
    if ($slug.Length -gt 64) {
        $slug = $slug.Substring(0, 64).TrimEnd([char[]]"-._")
    }
    return $slug
}

function Get-RepositoryRoot {
    param([string]$Path)

    $candidate = (Resolve-Path -LiteralPath $Path).Path
    $root = (Invoke-Git $candidate @("rev-parse", "--show-toplevel") | Select-Object -First 1).ToString().Trim()
    return [IO.Path]::GetFullPath($root)
}

function Get-DefaultWorktreeRoot {
    param([string]$RepositoryRoot)

    $parent = Split-Path -Parent $RepositoryRoot
    $name = Split-Path -Leaf $RepositoryRoot
    return [IO.Path]::GetFullPath((Join-Path $parent "$name.worktrees"))
}

function Assert-PathWithin {
    param(
        [string]$Candidate,
        [string]$Root,
        [string]$Label
    )

    $candidateFull = [IO.Path]::GetFullPath($Candidate).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $rootFull = [IO.Path]::GetFullPath($Root).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $prefix = $rootFull + [IO.Path]::DirectorySeparatorChar
    if (-not $candidateFull.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label is outside the managed worktree root: $candidateFull"
    }
}

$repositoryRoot = Get-RepositoryRoot $RepositoryPath
if ([string]::IsNullOrWhiteSpace($WorktreeRoot)) {
    $WorktreeRoot = Get-DefaultWorktreeRoot $repositoryRoot
} else {
    $WorktreeRoot = [IO.Path]::GetFullPath($WorktreeRoot)
}

if ($Action -eq "List") {
    Invoke-Git $repositoryRoot @("worktree", "list", "--porcelain")
    exit 0
}

if ($Action -eq "Create") {
    $batchSlug = ConvertTo-SafeSlug $BatchId "BatchId"
    $taskSlug = ConvertTo-SafeSlug $TaskId "TaskId"
    if ($Kind -eq "integration") {
        $taskSlug = "integration"
    }

    $status = @(Invoke-Git $repositoryRoot @("status", "--porcelain", "--untracked-files=all"))
    if ($status.Count -gt 0) {
        throw "The primary worktree is not clean. Commit or otherwise handle existing changes before creating write worktrees."
    }

    $baseCommit = (Invoke-Git $repositoryRoot @("rev-parse", "$BaseRef^{commit}") | Select-Object -First 1).ToString().Trim()
    $branch = "codex/$batchSlug/$taskSlug"
    $existingBranch = @(Invoke-Git $repositoryRoot @("branch", "--list", $branch))
    if ($existingBranch.Count -gt 0) {
        throw "Branch already exists: $branch"
    }

    $batchRoot = Join-Path $WorktreeRoot $batchSlug
    $targetPath = [IO.Path]::GetFullPath((Join-Path $batchRoot $taskSlug))
    Assert-PathWithin $targetPath $WorktreeRoot "Worktree path"
    if (Test-Path -LiteralPath $targetPath) {
        throw "Worktree path already exists: $targetPath"
    }

    New-Item -ItemType Directory -Force -Path $batchRoot | Out-Null
    Invoke-Git $repositoryRoot @("worktree", "add", "-b", $branch, $targetPath, $baseCommit) | Out-Null

    [PSCustomObject][ordered]@{
        action = "created"
        kind = $Kind
        repository_root = $repositoryRoot
        batch = $batchSlug
        task = $taskSlug
        branch = $branch
        base_commit = $baseCommit
        worktree_path = $targetPath
    } | ConvertTo-Json
    exit 0
}

if ([string]::IsNullOrWhiteSpace($WorktreePath)) {
    throw "WorktreePath is required for Remove."
}

$resolvedWorktreePath = [IO.Path]::GetFullPath($WorktreePath).TrimEnd([IO.Path]::DirectorySeparatorChar)
Assert-PathWithin $resolvedWorktreePath $WorktreeRoot "Worktree path"

$registeredPaths = @(
    Invoke-Git $repositoryRoot @("worktree", "list", "--porcelain") |
        Where-Object { $_ -like "worktree *" } |
        ForEach-Object { [IO.Path]::GetFullPath($_.Substring(9)).TrimEnd([IO.Path]::DirectorySeparatorChar) }
)
if ($resolvedWorktreePath -notin $registeredPaths) {
    throw "Path is not a registered worktree for this repository: $resolvedWorktreePath"
}

$worktreeStatus = @(Invoke-Git $resolvedWorktreePath @("status", "--porcelain", "--untracked-files=all"))
if ($worktreeStatus.Count -gt 0) {
    throw "Worktree is not clean and will not be removed: $resolvedWorktreePath"
}

$branchName = (Invoke-Git $resolvedWorktreePath @("symbolic-ref", "--short", "HEAD") | Select-Object -First 1).ToString().Trim()
Invoke-Git $repositoryRoot @("worktree", "remove", $resolvedWorktreePath) | Out-Null

$branchDeleted = $false
if ($DeleteBranch) {
    Invoke-Git $repositoryRoot @("branch", "-d", $branchName) | Out-Null
    $branchDeleted = $true
}

[PSCustomObject][ordered]@{
    action = "removed"
    repository_root = $repositoryRoot
    branch = $branchName
    branch_deleted = $branchDeleted
    worktree_path = $resolvedWorktreePath
} | ConvertTo-Json
