"""The agent that runs on the computer which does the collecting. It lets the owner press Start on the DEPLOYED site.

    python -m backend.pipeline.agent [--site https://elvarn.onrender.com]

It only makes outbound HTTPS calls to the site (nothing on this computer is opened to the internet), every call carries the owner key
(OWNER_KEY in .env), and it understands exactly five commands: start, stop, targeted, people, person (see remote_ops.py). Leave it running;
if the computer sleeps or the agent stops, the dashboard on the site says "offline" and nothing else is affected.
"""
import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request

from backend.pipeline import remote_ops

POLL_SECONDS = 1.0          # how often the site is told the state and asked for commands
REFRESH_SECONDS = 1.5       # how often the (slower) status document is rebuilt in the background
MAX_BACKOFF = 60.0


class Agent:
    def __init__(self, site, owner_key, monitor, post=None, say=print):
        if len(owner_key or "") < 16:
            raise ValueError("OWNER_KEY is missing or shorter than 16 characters")
        self.url = site.rstrip("/") + "/api/collect/agent/poll"
        self.key, self.monitor, self.say = owner_key, monitor, say
        self.post = post or self._post
        self.answers = []
        self.latest, self._lock = None, threading.Lock()          # the last status document; rebuilt by a background thread so a slow probe never delays a report

    def _post(self, payload):
        req = urllib.request.Request(self.url, data=json.dumps(payload, default=str).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "X-Owner-Key": self.key, "User-Agent": "ELVARN-agent/1.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            return json.load(resp)

    def refresh(self):
        """Rebuild the status document now (it is cached, so the next report carries it)."""
        try:
            report = self.monitor.status()
        except Exception as exc:  # noqa: BLE001 - a broken stage must not silence the agent
            self.say(f"status unavailable: {exc.__class__.__name__}: {str(exc)[:80]}")
            return None
        with self._lock:
            self.latest = report
        return report

    def _keep_fresh(self):
        while True:
            self.refresh()
            time.sleep(REFRESH_SECONDS)

    def step(self):
        """One round: report the state, receive the waiting commands, run them. Returns how many commands ran.
        After commands, a FRESH status goes out at once together with their answers, so the dashboard shows the effect without waiting."""
        with self._lock:
            report = self.latest
        if report is None:
            report = self.refresh()
        reply = self.post({"report": report, "results": self.answers})
        self.answers = []
        commands = reply.get("commands") or []
        for c in commands:
            self.say(f"command {c.get('action')}")
            self.answers.append({"seq": c["seq"], "result": remote_ops.run(self.monitor, c.get("action"), c.get("args"))})
        if commands:
            if any(c.get("action") in ("start", "stop", "targeted") for c in commands):
                report = self.refresh()                                        # a control command: the answer carries the NEW state
            else:
                with self._lock:                                               # a roster search or person lookup: answer at once, the status is not rebuilt for it
                    report = self.latest
            self.post({"report": report, "results": self.answers})
            self.answers = []
        return len(commands)

    def run_forever(self, sleep=time.sleep):
        threading.Thread(target=self._keep_fresh, daemon=True).start()
        wait = POLL_SECONDS
        while True:
            try:
                ran = self.step()
                wait = POLL_SECONDS
            except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:
                wait = min(MAX_BACKOFF, wait * 2 if wait >= POLL_SECONDS else POLL_SECONDS)
                self.say(f"the site did not answer ({exc.__class__.__name__}); trying again in {wait:.0f}s")
            sleep(wait)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=None, help="the deployed site, default PUBLIC_BASE_URL from .env")
    a = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        from backend.pipeline import control
        load_dotenv(control.REPO / ".env", override=False)
    except ImportError:
        pass
    site = a.site or os.environ.get("PUBLIC_BASE_URL", "")
    local = site.startswith(("http://127.0.0.1", "http://localhost"))              # a copy of the site on this computer (for testing)
    if not (site.startswith("https://") or local):
        raise SystemExit("Give the https address of the deployed site: --site https://... (or PUBLIC_BASE_URL in .env). The owner key is only sent over https.")
    from backend.pipeline.monitor import Monitor
    agent = Agent(site, os.environ.get("OWNER_KEY", ""), Monitor())
    print(f"agent running for {site}; press Ctrl+C to stop", flush=True)
    try:
        agent.run_forever()
    except KeyboardInterrupt:
        print("agent stopped")


if __name__ == "__main__":
    main()
