"""Process memory watchdog for Render's free 512 MB instance.

The service idles around 50 MB. We track its resident memory against a LIMIT (default 400 MB, safely below the
512 MB the host kills at) and report "close" at 80% and "over" at 100%, so there is a warning long before a crash.
"""
from __future__ import annotations

import gc
import logging
import os
from typing import Optional

log = logging.getLogger("jarvis_memory.resources")
CLOSE_FRACTION = 0.8


def rss_mb() -> Optional[float]:
    """Resident memory of this process in MB (Linux /proc, else psutil if present, else unknown)."""
    try:
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024), 1)
    except (OSError, ValueError, AttributeError, IndexError):
        pass
    try:
        import psutil  # type: ignore

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


def status(limit_mb: int, current: Optional[float] = None) -> dict:
    """``{"process_mb", "ram_limit_mb", "ram_status": ok|close|over|unknown}``; frees memory when over the limit."""
    current = rss_mb() if current is None else current
    if current is None:
        return {"process_mb": None, "ram_limit_mb": limit_mb, "ram_status": "unknown"}
    if current >= limit_mb:
        log.warning("Memory %.0f MB is over the %d MB limit; collecting garbage", current, limit_mb)
        gc.collect()
        state = "over"
    elif current >= limit_mb * CLOSE_FRACTION:
        log.warning("Memory %.0f MB is close to the %d MB limit", current, limit_mb)
        state = "close"
    else:
        state = "ok"
    return {"process_mb": current, "ram_limit_mb": limit_mb, "ram_status": state}
