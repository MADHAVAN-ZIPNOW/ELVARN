"""The link between the deployed site and the computer that does the collecting.

The collector needs Kafka, Spark, Neo4j, Chroma and the downloaded files, so it can only run on the owner's computer, never on the hosted site.
To let the owner press Start on the deployed site anyway, the computer runs a small agent (agent.py) that makes OUTBOUND calls only (nothing on the
computer is exposed to the internet):

    owner's browser --> hosted site (this Bridge) <--poll-- agent on the owner's computer --> collector, stores

* The agent posts its latest status every couple of seconds and receives the commands that are waiting (start, stop, a typed request, a list
  of people ...); it answers them on its next post.
* The hosted site keeps the last status and the waiting commands in memory (one process). A restart of the site loses nothing that matters: the
  agent reports the true state again within seconds.
* A command that nobody picked up for two minutes is dropped. If no agent reported for 20 seconds the dashboard says the computer is offline.
"""
import threading
import time
from datetime import datetime, timezone

STALE_AFTER = 20.0          # seconds without a report: the agent is offline
COMMAND_TTL = 120.0         # seconds a command may wait for the agent
RESULT_TTL = 300.0

EMPTY_PEOPLE = {"total": None, "done": None, "in_progress": None, "pending": None, "failed": None, "current": None, "recent": [], "queue": []}
STAGES = ("collector", "raw", "kafka", "spark", "corpus", "neo4j", "chroma")


class CollectBridge:
    def __init__(self, stale_after=STALE_AFTER, clock=time.time):
        self.stale_after, self.clock = stale_after, clock
        self.cv = threading.Condition()
        self.report, self.report_at = None, None
        self.queue, self.results, self.seq = [], {}, 0

    # ---- what the dashboard sees -------------------------------------------------------------------------
    def online(self):
        return self.report_at is not None and self.clock() - self.report_at < self.stale_after

    def _last_seen(self):
        return datetime.fromtimestamp(self.report_at, timezone.utc).isoformat(timespec="seconds") if self.report_at else None

    def status(self):
        """The status document of the contract, from the agent's last report; a 'state: offline' document when the agent is not reporting."""
        with self.cv:
            agent = {"online": self.online(), "last_seen": self._last_seen()}
            if self.report is None:
                return {"running": False, "state": "offline", "started_at": None, "elapsed_s": 0, "scope": {"from_year": 1947, "to_year": 2026},
                        "message": "Your collection computer is offline. Start the agent there (python -m backend.pipeline.agent), then try again.",
                        "people": dict(EMPTY_PEOPLE), "stages": {k: None for k in STAGES}, "movement": [], "agent": agent}
            doc = dict(self.report)
            if not agent["online"]:
                doc.update(state="offline", running=False,
                           message="Your collection computer stopped reporting (last seen " + (agent["last_seen"] or "never") + "). Showing its last numbers.")
            doc["agent"] = agent
            return doc

    # ---- commands from the dashboard ---------------------------------------------------------------------
    def enqueue(self, action, args=None):
        with self.cv:
            self.seq += 1
            self.queue.append({"seq": self.seq, "action": action, "args": args or {}, "at": self.clock()})
            return self.seq

    def call(self, action, args=None, wait=12.0):
        """Queue a command and wait for the agent's answer. None when the agent did not answer in time."""
        seq = self.enqueue(action, args)
        deadline = self.clock() + wait
        with self.cv:
            while seq not in self.results and self.clock() < deadline:
                self.cv.wait(0.2)
            hit = self.results.pop(seq, None)
        return None if hit is None else hit["result"]

    # ---- the agent ---------------------------------------------------------------------------------------
    def poll(self, report=None, results=()):
        """The agent's call: store its report and answers, hand over the commands that are waiting."""
        now = self.clock()
        with self.cv:
            if isinstance(report, dict):
                self.report, self.report_at = report, now
            for r in results or ():
                if isinstance(r, dict) and "seq" in r:
                    self.results[r["seq"]] = {"result": r.get("result"), "at": now}
            self.results = {k: v for k, v in self.results.items() if now - v["at"] < RESULT_TTL}
            commands = [c for c in self.queue if now - c["at"] < COMMAND_TTL]
            self.queue = []
            self.cv.notify_all()
        return [{k: c[k] for k in ("seq", "action", "args")} for c in commands]
