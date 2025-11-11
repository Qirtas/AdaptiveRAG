#!/usr/bin/env pwsh
# Setting Configs
$env:LLM_PROVIDER = "claude"
# $env:ANTHROPIC_API_KEY = ""
if (-not $env:PERSIST_DIR) { $env:PERSIST_DIR = "RAG/ProcessedDocuments/chroma_db" }
$PORT = if ($env:PORT) { $env:PORT } else { 8000 }
$QUESTION = if ($args.Count -ge 1) { $args[0] } else { "What is CAPEX?" }

# Checking API key
if (-not $env:ANTHROPIC_API_KEY) {
  Write-Host "ANTHROPIC_API_KEY is not set. Export it and then rerun"
  exit 1
}

Write-Host "Using LLM_PROVIDER=$($env:LLM_PROVIDER)"
Write-Host "PERSIST_DIR=$($env:PERSIST_DIR)"

# build vector DB
python main.py --mode index --persist_dir "$env:PERSIST_DIR"

# Starting API server in background
Write-Host "Starting Uvicorn on port $PORT ..."
$log = ".server_claude.log"
$proc = Start-Process -FilePath "python" `
  -ArgumentList @("-m","uvicorn","RAG.api:app","--host","0.0.0.0","--port",$PORT,"--reload") `
  -PassThru -RedirectStandardOutput $log -RedirectStandardError $log
Start-Sleep -Seconds 3

# Sending request
Write-Host "Sending test request to http://127.0.0.1:$PORT/ask"
$body = @{ question = $QUESTION; k = 3; min_similarity = 0.28 } | ConvertTo-Json
try {
  $resp = Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:$PORT/ask" -ContentType "application/json" -Body $body
  $resp | ConvertTo-Json -Depth 10
} catch {
  Write-Host "Request failed. See log $log"
}

# Ending
Write-Host "Stopping server (PID $($proc.Id))"
Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue
Wait-Process -Id $proc.Id -ErrorAction SilentlyContinue
