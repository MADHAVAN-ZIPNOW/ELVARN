# The GPU embedding service for Chroma (about ten times faster than the CPU). Leave it running while the judgments collector works.
#   powershell -ExecutionPolicy Bypass -File scripts\start_embed_gpu.ps1
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& (Join-Path $root ".venv-gpu\Scripts\python.exe") -m backend.pipeline.embed_server
