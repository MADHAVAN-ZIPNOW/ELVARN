# Register (or remove) the Windows scheduled task that starts the ELVARN stack at every logon.
#   .\scripts\register_autostart.ps1            create / update the task
#   .\scripts\register_autostart.ps1 -Remove    delete it
# Runs as the current user, no admin rights, hidden window. The PC must be on and signed in for the nightly collection to run.
param([switch]$Remove)
$name = "ELVARN-Stack-Start"
if ($Remove) { schtasks /Delete /TN $name /F; exit $LASTEXITCODE }
$script = Join-Path $PSScriptRoot "start_stack.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Description "Starts Docker Desktop and the ELVARN data stack (Neo4j, Chroma, Redis, Kafka, Spark, Airflow)" -Force | Out-Null
Get-ScheduledTask -TaskName $name | Select-Object TaskName, State
