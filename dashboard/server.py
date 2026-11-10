"""Training dashboard for open-fret.

    .venv/bin/python dashboard/server.py [--port 8765]

Then open http://127.0.0.1:8765. Reads models/*/metrics.jsonl (written by
scripts/metrics_callback.py) and samples Apple GPU use from ioreg every 2 s.
Standard library only; listens on localhost only.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
INDEX_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
GPU_HISTORY = deque(maxlen=450)  # 15 min at one sample every 2 s


def sample_gpu():
    """Whole-Mac GPU use from the IOAccelerator stats (no sudo needed)."""
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    util = re.search(r'"Device Utilization %"=(\d+)', out)
    mem = re.search(r'"In use system memory"=(\d+)', out)
    if not util:
        return None
    return {"time": time.time(), "util": int(util.group(1)), "mem": int(mem.group(1)) if mem else None}


def gpu_sampler():
    while True:
        sample = sample_gpu()
        if sample:
            GPU_HISTORY.append(sample)
        time.sleep(2)


def list_runs():
    runs = []
    if os.path.isdir(MODELS_DIR):
        for name in os.listdir(MODELS_DIR):
            path = os.path.join(MODELS_DIR, name, "metrics.jsonl")
            if os.path.isfile(path):
                runs.append({"name": name, "updated": os.path.getmtime(path)})
    return sorted(runs, key=lambda r: -r["updated"])


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_run(name):
    records = []
    with open(os.path.join(MODELS_DIR, name, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a half-written last line while training is writing

    # One session per training process. A fresh start (resumed_from_step 0) replaces
    # earlier history; a resume keeps the earlier logs up to the resume step. Until the
    # current process starts training, show only its own logs, so a new run never
    # displays the previous run's numbers.
    sessions = []
    for r in records:
        if r.get("event") == "session" or not sessions:
            sessions.append([])
        sessions[-1].append(r)

    logs = []
    for i, session in enumerate(sessions):
        session_start = next((r for r in session if r.get("event") == "start"), None)
        if session_start:
            resume = session_start.get("resumed_from_step") or 0
            logs = [r for r in logs if r.get("step", 0) <= resume] if resume else []
        elif i == len(sessions) - 1:
            logs = []
        logs += [r for r in session if r.get("event") == "log"]

    current = sessions[-1] if sessions else []
    start = next((r for r in current if r.get("event") == "start"), None)
    end = next((r for r in current if r.get("event") == "end"), None)
    pid = current[0].get("pid") if current else None
    if end:
        status = "finished"
    elif pid and pid_alive(pid):
        status = "running" if start else "preparing"
    else:
        status = "stopped"

    return {
        "name": name,
        "status": status,
        "start": start,
        "session_time": current[0].get("time") if current else None,
        "end_time": end["time"] if end else None,
        "end_step": end.get("step") if end else None,
        "logs": logs,
    }


def state(run_name):
    runs = list_runs()
    names = {r["name"] for r in runs}
    if run_name not in names:  # also guards against path tricks in ?run=
        run_name = runs[0]["name"] if runs else None
    disk = shutil.disk_usage(ROOT)
    return {
        "now": time.time(),
        "runs": runs,
        "run": read_run(run_name) if run_name else None,
        "gpu": list(GPU_HISTORY),
        "disk": {"free": disk.free, "total": disk.total},
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            with open(INDEX_HTML, "rb") as f:
                self._send(200, "text/html; charset=utf-8", f.read())
        elif url.path == "/api/state":
            run = parse_qs(url.query).get("run", [None])[0]
            self._send(200, "application/json", json.dumps(state(run)).encode())
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep the console quiet; it polls every few seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    port = parser.parse_args().port
    threading.Thread(target=gpu_sampler, daemon=True).start()
    print(f"open-fret dashboard on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
