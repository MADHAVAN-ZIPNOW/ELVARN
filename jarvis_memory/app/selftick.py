"""The service runs its own schedule (phone calls, reminders, e-mails) every few minutes, so it no longer depends on a GitHub job or the
owner's PC pinging /tick. It can only run while the process is awake: on Render's free plan something outside (any free uptime monitor
opening the public /api/jarvis/health link, which needs no key) keeps it from sleeping."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from . import jobs

log = logging.getLogger("jarvis_memory.selftick")
SELF_TICK_SECONDS = 5 * 60


async def run(app: Any, every: float = SELF_TICK_SECONDS, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
              rounds: Optional[int] = None) -> None:
    """Tick every ``every`` seconds, for ever (``rounds`` is for tests). A failed tick is only logged: the next one tries again."""
    await sleep(30)       # let the service finish starting first
    done = 0
    while rounds is None or done < rounds:
        try:
            await asyncio.to_thread(jobs.run_tick, app.state.service, app.state.mailbox, pusher=app.state.pusher)
        except Exception as exc:
            log.info("Self tick failed (%s)", type(exc).__name__)
        done += 1
        await sleep(every)
