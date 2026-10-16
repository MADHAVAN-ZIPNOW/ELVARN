# Runs the collector agent, so the Start button on the DEPLOYED site reaches this computer.
# It makes outbound HTTPS calls only (nothing here is opened to the internet) and sends OWNER_KEY from .env to the site you name in PUBLIC_BASE_URL.
# Leave the window open (or register it to run at logon in Task Scheduler). It restarts itself if it stops.
#   powershell -ExecutionPolicy Bypass -File scripts\start_agent.ps1
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "backend\data\agent.log"
while ($true) {
    & $py -m backend.pipeline.agent *>> $log
    Start-Sleep -Seconds 15
}
