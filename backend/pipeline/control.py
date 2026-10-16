"""Start and stop the politician collector from the app (the button), and tell the app whether it is running.

The collector runs as its own process (`python -m backend.pipeline.person_collector run`), started detached so that restarting the
chat server does not kill it. The two sides talk through a small JSON file:

    backend/data/kb_state/collector.control.json
        {"state": "idle|running|stopping|error", "pid": 123, "started_at": "...", "stop_requested": false, "message": "..."}

Stop is graceful: the collector finishes the document it is on, saves where it was (the person stays "in progress"), and exits.
Start again and it resumes from that person. Start also makes sure Kafka and Spark are up (they are normally stopped to save CPU).
"""
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATE_DIR = REPO / "backend" / "data" / "kb_state"
CONTROL_PATH = STATE_DIR / "collector.control.json"
LOG_PATH = STATE_DIR / "collector.log"
STACK_CONTAINERS = ("elvarn-kafka", "elvarn-spark")
STARTING_GRACE = 240      # seconds the Start button may spend bringing Kafka and Spark up before a process exists
IDLE = {"state": "idle", "pid": None, "started_at": None, "stop_requested": False, "message": ""}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _age(iso):
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()
    except (TypeError, ValueError):
        return None


def read(path=CONTROL_PATH):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(IDLE)
    return {**IDLE, **data}


def write(update, path=CONTROL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {**read(path), **update}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)
    return data


def pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))        # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259                                       # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def state(path=CONTROL_PATH):
    """The control file, corrected for a collector that died without saying so (crash, machine restart)."""
    data = read(path)
    starting = data["pid"] is None and (_age(data["started_at"]) or 0) < STARTING_GRACE          # Kafka / Spark are being started, no process yet
    if data["state"] in ("running", "stopping") and not starting and not pid_alive(data["pid"]):
        data = write({"state": "idle", "pid": None, "stop_requested": False,
                      "message": "The collector is not running (it ended or was interrupted). Start it to resume where it stopped."}, path)
    return data


def is_running(path=CONTROL_PATH):
    return state(path)["state"] in ("running", "stopping")


def stop_requested(path=CONTROL_PATH):
    return bool(read(path).get("stop_requested"))


def request_stop(path=CONTROL_PATH):
    data = state(path)
    if data["state"] == "running":
        data = write({"state": "stopping", "stop_requested": True, "message": "Stopping: finishing the current document, then the collector exits."}, path)
    return data


def _docker(*args, timeout=60):
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def kafka_up(bootstrap=None, timeout=4):
    try:
        from confluent_kafka import Producer
        from backend.pipeline.kafka_io import BOOTSTRAP, CLIENT_OPTIONS, TOPIC
        meta = Producer({"bootstrap.servers": bootstrap or BOOTSTRAP, **CLIENT_OPTIONS}).list_topics(TOPIC, timeout=timeout)
        return TOPIC in meta.topics and not meta.topics[TOPIC].error
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        return False


def ensure_stack(wait_seconds=120, say=lambda m: None):
    """Make sure Kafka (and Spark) are running: start the two containers if they are stopped. True when Kafka answers."""
    if kafka_up():
        return True
    say("Starting Kafka and Spark ...")
    try:
        for name in STACK_CONTAINERS:
            _docker("start", name)
    except (OSError, subprocess.TimeoutExpired):
        return False
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if kafka_up():
            return True
        time.sleep(3)
    return False


def hidden_process_options():
    """How to start the collector so that it outlives this process and shows NO window. On Windows a DETACHED process has no console, and the
    virtual environment's python.exe starts the real interpreter as a child: that child then gets a console window of its own (a black cmd
    window pops up). CREATE_NO_WINDOW gives the process a hidden console that its children share, so nothing is shown."""
    if os.name != "nt":
        return {"start_new_session": True}
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = 0                                   # SW_HIDE
    return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP, "startupinfo": info}


PERSON_MODULE = "backend.pipeline.person_collector"
JUDGMENTS_MODULE = "backend.pipeline.judgments"


def _launch(path, python, check_stack, extra=(), module=PERSON_MODULE):
    if check_stack and not ensure_stack(say=lambda m: write({"message": m}, path)):
        write({"state": "error", "pid": None, "message": "Kafka is not reachable and could not be started. Start Docker Desktop, then press Start again "
                                                         "(scripts/start_stack.ps1 starts everything)."}, path)
        return
    if stop_requested(path):                                        # Stop was pressed while the stack was starting
        write({"state": "idle", "pid": None, "stop_requested": False, "message": "Stopped before it began."}, path)
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "ab") as log:
        proc = subprocess.Popen([python or sys.executable, "-m", module, "run", *extra], cwd=str(REPO), stdout=log, stderr=log,
                                stdin=subprocess.DEVNULL, close_fds=True, **hidden_process_options())
    write({"pid": proc.pid, "message": "Collecting politician by politician."}, path)


def start(path=CONTROL_PATH, python=None, check_stack=True, background=True, extra=(), message=None, module=PERSON_MODULE):
    """Start the collector process unless one is running. Returns the control state at once; with `background` the work of bringing
    Kafka and Spark up and launching the process happens in a thread (the dashboard shows the message meanwhile)."""
    data = state(path)
    if data["state"] in ("running", "stopping"):
        return data
    data = write({"state": "running", "pid": None, "started_at": _now(), "stop_requested": False, "mode": "judgments" if module == JUDGMENTS_MODULE else "people",
                  "message": message or ("Starting ..." if check_stack else "Collecting politician by politician.")}, path)
    if background:
        threading.Thread(target=_launch, args=(path, python, check_stack, extra, module), daemon=True).start()
    else:
        _launch(path, python, check_stack, extra, module)
    return data
