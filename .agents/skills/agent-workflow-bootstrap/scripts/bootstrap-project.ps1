[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$ProjectPath,

    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string[]]$Skills,

    [ValidateSet("codex", "claude", "gemini")]
    [string[]]$Providers = @("codex"),

    [string]$ProjectName = "",
    [string]$Purpose = "Not established yet",
    [string]$Technology = "Not established yet",
    [string]$Commands = "Not established yet",
    [string]$Constraints = "Not established yet",
    [switch]$SkipProjectRegistry,
    [switch]$Apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-NormalizedPath {
    param([Parameter(Mandatory)][string]$Path)

    return [IO.Path]::GetFullPath($Path).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
}

function Normalize-Field {
    param(
        [string]$Value,
        [string]$Fallback = "Not established yet"
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $Fallback
    }
    return (($Value -replace "[\r\n]+", " ").Trim() -replace "\s{2,}", " ")
}

function Write-FileIfMissing {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Content
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        Set-Content -LiteralPath $Path -Value $Content -Encoding utf8
    }
}

function New-AgentsContentFromTemplate {
    param(
        [Parameter(Mandatory)][string]$TemplatePath,
        [Parameter(Mandatory)][string]$ManagedBlock
    )

    if (-not (Test-Path -LiteralPath $TemplatePath -PathType Leaf)) {
        throw "Base AGENTS.md template was not found: $TemplatePath"
    }

    $templateContent = Get-Content -Raw -LiteralPath $TemplatePath
    $templateContent = [regex]::Replace(
        $templateContent,
        '(?ms)\A# Project Agent Guidelines\s*\r?\n\s*(?:>[^\r\n]*(?:\r?\n|$))+\s*',
        ''
    )
    $templateContent = [regex]::Replace(
        $templateContent,
        '(?ms)^## Project context\s*\r?\n.*?(?=^## |\z)',
        ''
    )
    $templateContent = $templateContent.Replace(
        '[Add repository layout rules, coding conventions, test expectations, safety boundaries, release processes, and other project-specific guidance here.]',
        'Add repository layout rules, coding conventions, test expectations, safety boundaries, release processes, and other project-specific guidance here when they become established.'
    ).Trim()

    return "# Project Agent Guidelines`r`n`r`n$ManagedBlock`r`n`r`n$templateContent`r`n"
}

if (-not [IO.Path]::IsPathFullyQualified($ProjectPath)) {
    throw "ProjectPath must be absolute: $ProjectPath"
}

$resolvedProject = Get-NormalizedPath $ProjectPath
$sourceRepository = Get-NormalizedPath (Join-Path $PSScriptRoot "..\..\..\..")
$controlScript = Join-Path $sourceRepository "scripts\project-control-center.ps1"
if (-not (Test-Path -LiteralPath $controlScript -PathType Leaf)) {
    $existingManifest = Join-Path $resolvedProject ".agents\control-center.json"
    if (Test-Path -LiteralPath $existingManifest -PathType Leaf) {
        $manifest = Get-Content -Raw -LiteralPath $existingManifest | ConvertFrom-Json
        if (-not [string]::IsNullOrWhiteSpace([string]$manifest.sourceRepository)) {
            $sourceRepository = Get-NormalizedPath ([string]$manifest.sourceRepository)
            $controlScript = Join-Path $sourceRepository "scripts\project-control-center.ps1"
        }
    }
}
if (-not (Test-Path -LiteralPath $controlScript -PathType Leaf)) {
    throw "Project Control Center was not found: $controlScript"
}

$selectedSkills = @($Skills | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
if ($selectedSkills.Count -eq 0) {
    throw "Select at least one skill."
}
$requiredSkills = @("agent-workflow-bootstrap", "memory-manager")
$selectedSkills = @($requiredSkills + $selectedSkills | Select-Object -Unique)

$applyArguments = @{
    Action = "Apply"
    ProjectPath = $resolvedProject
    SourceRepository = $sourceRepository
    Skills = $selectedSkills
    Providers = $Providers
    SkipProjectRegistry = $SkipProjectRegistry
}

Write-Host "Previewing AgentWorkflow bootstrap for $resolvedProject"
& $controlScript @applyArguments -WhatIf

if (-not $Apply) {
    Write-Host "Preview complete. Run again with -Apply to configure the project."
    return
}

& $controlScript @applyArguments

$memoryScript = Join-Path $sourceRepository "scripts\memory-control-center.ps1"
if (-not (Test-Path -LiteralPath $memoryScript -PathType Leaf)) {
    throw "Memory Control Center was not found: $memoryScript"
}
$localMemory = Join-Path $resolvedProject ".agents\memory\local"
$globalMemory = Join-Path $resolvedProject ".agents\memory\global"
& $memoryScript -Action Reindex -MemoryRoot $localMemory
& $memoryScript -Action Reindex -MemoryRoot $globalMemory
& $memoryScript -Action Check -MemoryRoot $localMemory
& $memoryScript -Action Check -MemoryRoot $globalMemory

if ([string]::IsNullOrWhiteSpace($ProjectName)) {
    $ProjectName = Split-Path -Leaf $resolvedProject
}

$safeName = Normalize-Field $ProjectName "Unnamed project"
$safePurpose = Normalize-Field $Purpose
$safeTechnology = Normalize-Field $Technology
$safeCommands = Normalize-Field $Commands
$safeConstraints = Normalize-Field $Constraints
$skillList = ($selectedSkills | Sort-Object) -join ", "

$startMarker = "<!-- agent-workflow:start -->"
$endMarker = "<!-- agent-workflow:end -->"
$managedBlock = @"
$startMarker
## Agent workflow

### Project context

- **Project:** $safeName
- **Purpose:** $safePurpose
- **Technology:** $safeTechnology
- **Important commands:** $safeCommands
- **Important constraints:** $safeConstraints

### Connected capabilities

- **Managed skills:** $skillList
- **Global memory index:** `.agents/memory/global/INDEX.md`
- **Local memory index:** `.agents/memory/local/INDEX.md`

### Memory rules

- Start from the global and local `INDEX.md`; open only entries relevant to the task.
- Use `memory-manager` to classify durable knowledge, create categories, and validate indexes.
- Keep always-on rules here; move conditional detail to memory, `docs/agent-guides/`, or a skill as appropriate.
- Do not store secrets, transient task state, or information that is easy to derive from the repository.
$endMarker
"@

$agentsPath = Join-Path $resolvedProject "AGENTS.md"
$agentsTemplatePath = Join-Path $sourceRepository "templates\base-project\AGENTS.md"
if (Test-Path -LiteralPath $agentsPath -PathType Container) {
    throw "AGENTS.md must be a file: $agentsPath"
}

if (Test-Path -LiteralPath $agentsPath -PathType Leaf) {
    $agentsContent = Get-Content -Raw -LiteralPath $agentsPath
    $startIndex = $agentsContent.IndexOf($startMarker, [StringComparison]::Ordinal)
    $endIndex = $agentsContent.IndexOf($endMarker, [StringComparison]::Ordinal)
    if (($startIndex -ge 0) -xor ($endIndex -ge 0)) {
        throw "AGENTS.md contains only one AgentWorkflow marker; repair it before bootstrapping."
    }
    if ($startIndex -ge 0) {
        if ($endIndex -lt $startIndex) {
            throw "AGENTS.md AgentWorkflow markers are out of order."
        }
        $afterMarker = $endIndex + $endMarker.Length
        $unmanagedContent = ($agentsContent.Substring(0, $startIndex) + $agentsContent.Substring($afterMarker)).Trim()
        if ($unmanagedContent -eq "# Project Agent Guidelines") {
            $agentsContent = New-AgentsContentFromTemplate `
                -TemplatePath $agentsTemplatePath `
                -ManagedBlock $managedBlock
        } else {
            $agentsContent = $agentsContent.Substring(0, $startIndex) + $managedBlock + $agentsContent.Substring($afterMarker)
        }
    } else {
        $agentsContent = $agentsContent.TrimEnd() + "`r`n`r`n" + $managedBlock + "`r`n"
    }
} else {
    $agentsContent = New-AgentsContentFromTemplate `
        -TemplatePath $agentsTemplatePath `
        -ManagedBlock $managedBlock
}
Set-Content -LiteralPath $agentsPath -Value $agentsContent -Encoding utf8 -NoNewline

& $controlScript `
    -Action Status `
    -ProjectPath $resolvedProject `
    -SourceRepository $sourceRepository

Write-Host "AgentWorkflow bootstrap complete."
