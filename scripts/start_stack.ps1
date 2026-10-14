# Bring the whole ELVARN data stack up: Docker Desktop, then Neo4j, Chroma + Redis, SearXNG (web search), and Kafka + Spark + Airflow.
# Safe to run any time (docker compose up -d only starts what is not already running). Registered to run at every Windows logon by
# scripts/register_autostart.ps1, so after a restart the daily collection (Airflow, 02:00) is back without any manual step.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = Join-Path $root "backend\data\start_stack.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
function Log($m) { "$(Get-Date -Format 's') $m" | Add-Content -Path $log -Encoding utf8 }

function Docker-Ready { try { docker info *> $null; return ($LASTEXITCODE -eq 0) } catch { return $false } }

if (-not (Docker-Ready)) {
    $exe = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (Test-Path $exe) { Log "starting Docker Desktop"; Start-Process $exe }
    $deadline = (Get-Date).AddMinutes(6)
    while (-not (Docker-Ready) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 5 }
    if (-not (Docker-Ready)) { Log "Docker did not become ready in 6 minutes; giving up"; exit 1 }
}
Log "docker is ready"
foreach ($f in @("docker-compose.neo4j.yml", "docker-compose.stores.yml", "docker-compose.searxng.yml", "docker-compose.pipeline.yml")) {
    Log "up: $f"
    docker compose --env-file .env -f $f up -d 2>&1 | Add-Content -Path $log -Encoding utf8
    if ($LASTEXITCODE -ne 0) { Log "FAILED: $f (exit $LASTEXITCODE)" }
}
Log "done"
