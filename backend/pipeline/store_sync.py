"""The store sync as its own program: embed new passages into Chroma and write what Neo4j still lacks, so that a big collection does not have to do it.

Before, the collector ran this pass in a thread of its own process. During a big judgments run that process already uses four CPU cores for
downloads and PDF reading, so the pass waited for CPU and Chroma gained only a few passages a second although the GPU can embed about 200.
Now the collectors start `python -m backend.pipeline.store_sync run` (hidden, its own process) and their in-process pass steps aside while it lives.

    heartbeat   backend/data/kb_state/store_sync.heartbeat   touched on every loop: "the sync program is alive" (fresh = under 90 s old)
    lock        backend/data/kb_state/store_sync.lock        a pass rewrites Parquet files, so two passes (even from two programs) never overlap
The program ends by itself after `idle_exit` seconds without new passages while no collection is running.
"""
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

from backend.pipeline import control
from backend.pipeline.registry import STATE_DIR, Registry

HEARTBEAT = STATE_DIR / "store_sync.heartbeat"
LOCK = STATE_DIR / "store_sync.lock"
LOG = STATE_DIR / "store_sync.log"
FRESH = 90


def alive(path=HEARTBEAT, max_age=FRESH):
    try:
        return time.time() - Path(path).stat().st_mtime < max_age
    except OSError:
        return False


def beat(path=HEARTBEAT):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(int(time.time())), encoding="utf-8")


@contextlib.contextmanager
def pass_lock(path=LOCK):
    """An exclusive lock between PROGRAMS (a threading lock only works inside one). Yields True when we hold it, False when another pass is running."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, got = open(path, "a+b"), False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            got = True
        except OSError:
            got = False
        yield got
    finally:
        handle.close()                                                    # closing the file releases the lock


def do_pass(registry=None, say=print, lock_path=LOCK):
    """One pass: Chroma, Neo4j, flags and the report (check_stores) and the person-document links. -> the report, or None when another pass is running
    or a store is not configured."""
    with pass_lock(lock_path) as got:
        if not got:
            return None
        try:
            from backend.knowledge import check_stores
            from backend.knowledge.kb import KnowledgeBase
            from backend.knowledge.vectorstore import RagStore
            kb = KnowledgeBase.from_env()
            if kb is None:
                raise RuntimeError("Neo4j is not configured (NEO4J_URI / NEO4J_PASSWORD)")
            try:
                report = check_stores.run(kb=kb, rag=RagStore.from_env())
                pairs = registry.unlinked_in_graph() if registry is not None else []
                if pairs:
                    registry.mark_graph_linked(kb.link_person_documents(pairs))
            finally:
                kb.close()
            say(f"stores synced: {report['passages']} passages, {report['embedded_passages']} in Chroma, {report['neo4j_indexed_passages']} in Neo4j")
            return report
        except Exception as exc:  # noqa: BLE001 - a store being down must not stop anything; the next pass tries again
            say(f"store sync skipped: {exc.__class__.__name__}: {str(exc)[:120]}")
            return None


def ensure_running():
    """Start the sync program (hidden, detached) unless it is already alive. Returns True when one is running afterwards."""
    if alive():
        return True
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        beat()                                                            # claim it now: two collectors starting together launch only one
        with open(LOG, "ab") as log:
            subprocess.Popen([sys.executable, "-m", "backend.pipeline.store_sync", "run"], cwd=str(control.REPO), stdout=log, stderr=log,
                             stdin=subprocess.DEVNULL, close_fds=True, **control.hidden_process_options())
        return True
    except OSError:
        return False


def run_forever(interval=10, idle_exit=900, say=lambda m: print(m, flush=True)):
    try:
        from dotenv import load_dotenv
        load_dotenv(control.REPO / ".env", override=False)
    except ImportError:
        pass
    registry = Registry()
    last_embedded, last_work = -1, time.time()
    while True:
        beat()
        report = do_pass(registry, say)
        if report is not None and report["embedded_passages"] != last_embedded:
            last_embedded, last_work = report["embedded_passages"], time.time()
        if control.is_running():
            last_work = time.time()
        if time.time() - last_work > idle_exit:
            say("nothing new for a while and no collection is running: the sync program ends (the next Start launches it again)")
            try:
                Path(HEARTBEAT).unlink()
            except OSError:
                pass
            return
        time.sleep(interval)


def main(argv=None):
    cmd = (argv or sys.argv[1:] or ["run"])[0]
    if cmd == "run":
        run_forever()
    else:
        print(f"alive: {alive()}")


if __name__ == "__main__":
    main()
