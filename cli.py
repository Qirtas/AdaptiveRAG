# cli.py
"""
Cross-platform CLI for RAGForDatamite dev tasks.

"""

import json
import os
import signal
import subprocess
import sys
import time
from typing import Optional

import requests
import typer

# Optional: load keys from .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

app = typer.Typer(help="Dev commands for RAGForDatamite (cross-platform).")

# ---- helpers -----------------------------------------------------------------

def _echo(msg: str):
    typer.echo(msg)

def _exit_with_error(msg: str, code: int = 1):
    _echo(typer.style(msg, fg=typer.colors.RED, bold=True))
    return code

def _check_env(llm_provider: str):
    p = (llm_provider or "").lower()
    if p == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise typer.Exit(
                _exit_with_error("ANTHROPIC_API_KEY is not set. Export it (or put it in a .env) and re-run.")
            )
    elif p == "deepseek":
        # Using OpenRouter for DeepSeek
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise typer.Exit(
                _exit_with_error("OPENROUTER_API_KEY is not set. Export it (or put it in a .env) and re-run.")
            )
    else:
        raise typer.Exit(_exit_with_error(f"Unknown --llm-provider '{llm_provider}'. Use 'claude' or 'deepseek'."))

def _env(**extra):
    env = os.environ.copy()
    for k, v in extra.items():
        if v is not None:
            env[k] = str(v)
    return env

def _wait_for_api(host: str, port: int, timeout: float = 20.0):
    """Poll the API until it responds or timeout is reached."""
    url = f"http://{host}:{port}/docs"
    start = time.time()
    last_err = None
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code < 500:
                return True
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    if last_err:
        _echo(f"Last connection error: {last_err}")
    return False

def _start_uvicorn(app_path: str, host: str, port: int, reload: bool, log_file: Optional[str] = None):
    args = [sys.executable, "-m", "uvicorn", app_path, "--host", host, "--port", str(port)]
    if reload:
        args.append("--reload")

    stdout = subprocess.PIPE
    stderr = subprocess.STDOUT
    if log_file:
        f = open(log_file, "a", buffering=1, encoding="utf-8")
        proc = subprocess.Popen(args, stdout=f, stderr=f, env=os.environ.copy())
        return proc, f
    else:
        proc = subprocess.Popen(args, stdout=stdout, stderr=stderr, env=os.environ.copy())
        return proc, None

def _stop_process(proc: subprocess.Popen, log_handle=None):
    try:
        if os.name == "nt":
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        if log_handle:
            try:
                log_handle.flush()
                log_handle.close()
            except Exception:
                pass

# ---- commands ----------------------------------------------------------------

@app.command()
def index(
    persist_dir: str = typer.Option("RAG/ProcessedDocuments/chroma_db", "--persist-dir", help="Chroma/VectorDB path"),
):
    """Build vector DB (calls: python main.py --mode index --persist_dir ...)."""
    _echo(f"Building vector DB at: {persist_dir}")
    cmd = [sys.executable, "main.py", "--mode", "index", "--persist_dir", persist_dir]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise typer.Exit(_exit_with_error(f"Indexing failed (exit {e.returncode})."))

@app.command("run-api")
def run_api(
    port: int = typer.Option(8000, "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host"),
    app_path: str = typer.Option("RAG.api:app", "--app"),
    reload: bool = typer.Option(True, "--reload/--no-reload"),
):
    """Start FastAPI via uvicorn. Press Ctrl+C to stop."""
    _echo(f"Starting API: {app_path} at http://{host}:{port} (reload={reload})")
    proc, log_handle = _start_uvicorn(app_path, host, port, reload, log_file=None)
    try:
        # foreground attach: print uvicorn output
        while True:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                continue
            try:
                sys.stdout.write(line.decode("utf-8", errors="ignore"))
            except Exception:
                pass
    except KeyboardInterrupt:
        _echo("\nStopping server...")
    finally:
        _stop_process(proc, log_handle)

@app.command()
def ask(
    question: str = typer.Option("What is CAPEX?", "--question", "-q"),
    k: int = typer.Option(3, "--k"),
    min_similarity: float = typer.Option(0.28, "--min-similarity"),
    port: int = typer.Option(8000, "--port", "-p"),
    host: str = typer.Option("127.0.0.1", "--host"),
    persist_dir: str = typer.Option("RAG/ProcessedDocuments/chroma_db", "--persist-dir"),
    llm_provider: str = typer.Option("claude", "--llm-provider", help="claude | deepseek"),
    app_path: str = typer.Option("RAG.api:app", "--app"),
    reload: bool = typer.Option(False, "--reload/--no-reload"),
    log_file: str = typer.Option(".server.log", "--log-file"),
    build_index: bool = typer.Option(False, "--index/--no-index", help="(Default: no) Build index before starting API"),
    reset_index: bool = typer.Option(False, "--reset-index/--no-reset-index", help="Delete persist_dir before indexing"),
):
    """
    One-shot test: (optional) index -> start server -> POST /ask -> stop.
    """
    import shutil

    effective_provider = (os.environ.get("LLM_PROVIDER") or llm_provider or "claude").lower()

    # Pre-flight checks for the chosen provider
    _check_env(effective_provider)

    # Make sure the API subprocess sees the effective value
    os.environ["LLM_PROVIDER"] = effective_provider
    os.environ.setdefault("PERSIST_DIR", persist_dir)

    _echo(f"LLM_PROVIDER={os.environ['LLM_PROVIDER']}")
    _echo(f"PERSIST_DIR={os.environ['PERSIST_DIR']}")

    # 1) (optional) index once
    if build_index:
        if reset_index and os.path.exists(persist_dir):
            _echo(f"--reset-index: removing {persist_dir}")
            shutil.rmtree(persist_dir, ignore_errors=True)

        _echo("Building vector DB...")
        cmd = [sys.executable, "main.py", "--mode", "index", "--persist_dir", persist_dir]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise typer.Exit(_exit_with_error(f"Indexing failed (exit {e.returncode})."))

    # 2) start API
    _echo(f"Starting Uvicorn on port {port} ...")
    proc, handle = _start_uvicorn(app_path, host="0.0.0.0", port=port, reload=reload, log_file=log_file)

    # 3) wait until ready
    ok = _wait_for_api("127.0.0.1", port, timeout=25.0)
    if not ok:
        _stop_process(proc, handle)
        raise typer.Exit(_exit_with_error("Server did not come up in time. Check the log file for details."))

    # 4) send request
    url = f"http://127.0.0.1:{port}/ask"
    payload = {"question": question, "k": k, "min_similarity": min_similarity}
    _echo(f"POST {url}")
    try:
        r = requests.post(url, json=payload, timeout=120)
        try:
            print(json.dumps(r.json(), indent=2)[:20000])
        except Exception:
            print(f"HTTP {r.status_code}\n{r.text[:20000]}")
    except Exception as e:
        _echo(f"Request failed: {e}")

    # 5) stop
    _echo(f"Stopping server (PID {proc.pid})")
    _stop_process(proc, handle)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app()
