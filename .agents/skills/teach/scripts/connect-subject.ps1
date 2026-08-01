[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ProjectPath,
    [Parameter(Mandatory)][ValidatePattern('^[a-z0-9]+(?:-[a-z0-9]+)*$')][string]$Subject,
    [string]$SourceRepository,
    [switch]$Scaffold,
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'

function Resolve-AbsoluteDirectory {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Label)

    if (-not [IO.Path]::IsPathFullyQualified($Path)) {
        throw "$Label must be an absolute path: $Path"
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label was not found: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Resolve-SourceRepository {
    param([Parameter(Mandatory)][string]$ResolvedProject, [string]$ExplicitSource)

    if ($ExplicitSource) {
        return Resolve-AbsoluteDirectory $ExplicitSource 'SourceRepository'
    }

    $manifestPath = Join-Path $ResolvedProject '.agents\control-center.json'
    if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
        $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
        if ($manifest.sourceRepository) {
            return Resolve-AbsoluteDirectory ([string]$manifest.sourceRepository) 'Manifest sourceRepository'
        }
    }

    $candidate = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
    if (Test-Path -LiteralPath (Join-Path $candidate 'workflow.json') -PathType Leaf) {
        return (Resolve-Path -LiteralPath $candidate).Path
    }

    throw 'Cannot resolve the AgentWorkflow source repository. Bootstrap the project first or pass -SourceRepository.'
}

function Get-LinkTargetPath {
    param([Parameter(Mandatory)][System.IO.FileSystemInfo]$Item)

    $target = @($Item.Target) | Select-Object -First 1
    if (-not $target) { return $null }
    if (-not [IO.Path]::IsPathFullyQualified([string]$target)) {
        $target = Join-Path $Item.Parent.FullName ([string]$target)
    }
    return [IO.Path]::GetFullPath([string]$target).TrimEnd('\')
}

function Test-ManagedLink {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Target)

    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $item = Get-Item -LiteralPath $Path -Force
    $actualTarget = Get-LinkTargetPath $item
    $expectedTarget = [IO.Path]::GetFullPath($Target).TrimEnd('\')
    if ($item.LinkType -eq 'Junction' -and $actualTarget -eq $expectedTarget) { return $true }
    throw "Project knowledge path already exists and will not be replaced: $Path"
}

$resolvedProject = Resolve-AbsoluteDirectory $ProjectPath 'ProjectPath'
$resolvedSource = Resolve-SourceRepository $resolvedProject $SourceRepository
$knowledgeRoot = Join-Path $resolvedSource 'knowledge'
$subjectSource = Join-Path $knowledgeRoot $Subject
$projectKnowledgeRoot = Join-Path $resolvedProject '.agents\knowledge'

$subjectDirectories = @(
    'good-examples',
    'learning-records',
    'lessons',
    'raw-sources\papers',
    'raw-sources\websites',
    'reference',
    'resources',
    'scripts'
)

$links = [ordered]@{
    $Subject = $subjectSource
    'assets' = Join-Path $knowledgeRoot 'assets'
    'templates' = Join-Path $knowledgeRoot 'templates'
}

Write-Host "Subject: $Subject"
foreach ($name in $links.Keys) {
    Write-Host "Link $name`: $($links[$name])"
}

if (-not (Test-Path -LiteralPath $subjectSource -PathType Container) -and -not $Scaffold) {
    throw "Canonical subject does not exist. Re-run with -Scaffold after reviewing the path: $subjectSource"
}

$missingLinks = @()
foreach ($name in $links.Keys) {
    $projectLink = Join-Path $projectKnowledgeRoot $name
    if (-not (Test-ManagedLink $projectLink $links[$name])) {
        $missingLinks += $name
    }
}

if (-not $Apply) {
    if (-not (Test-Path -LiteralPath $subjectSource)) {
        Write-Host 'Preview: scaffold the canonical subject.'
    }
    foreach ($name in $missingLinks) {
        Write-Host "Preview: create project knowledge junction '$name'."
    }
    if ($missingLinks.Count -eq 0) { Write-Host 'Status: already linked' }
    return
}

if (-not (Test-Path -LiteralPath $subjectSource)) {
    New-Item -ItemType Directory -Path $subjectSource -Force | Out-Null
}
foreach ($relativeDirectory in $subjectDirectories) {
    New-Item -ItemType Directory -Path (Join-Path $subjectSource $relativeDirectory) -Force | Out-Null
}

foreach ($sharedName in @('assets', 'templates')) {
    $sharedPath = Join-Path $knowledgeRoot $sharedName
    if (-not (Test-Path -LiteralPath $sharedPath -PathType Container)) {
        throw "Shared knowledge directory was not found: $sharedPath"
    }
}

New-Item -ItemType Directory -Path $projectKnowledgeRoot -Force | Out-Null
foreach ($name in $missingLinks) {
    New-Item -ItemType Junction -Path (Join-Path $projectKnowledgeRoot $name) -Target $links[$name] | Out-Null
}
Write-Host 'Status: linked'
