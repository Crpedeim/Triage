"""
ratelimit.py — simple per-session and per-IP rate limits for the public UI.

Every chat turn costs LLM calls, so the public Cloud Run URL limits how fast a
single browser session and a single client IP can send messages. Limits are
sliding windows held in process memory:

    RATE_LIMIT_SESSION="20/600"   # 20 messages per 600 s per session (default)
    RATE_LIMIT_IP="60/3600"       # 60 messages per 3600 s per client IP (default)
    MAX_MESSAGE_CHARS=2000        # longer messages are refused
    RATE_LIMIT_XFF_INDEX=-1       # which X-Forwarded-For entry is the client IP

Limits are per app instance. With Cloud Run max-instances 2, one client whose
requests land on both instances can get up to twice the limit; session
affinity keeps a browser on one instance in the normal case.

The client IP comes from X-Forwarded-For (Cloud Run sits behind Google's front
end, so the socket peer is not the client). Earlier entries in that header can
be set by the client itself; the index used is configurable and was checked on
the deployed service (see PHASES.md, R2 Phase 4).
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque


def parse_spec(spec: str | None, default: tuple[int, int]) -> tuple[int, int]:
    """'20/600' -> (20, 600). Invalid or empty -> default. '0/…' disables."""
    if not spec:
        return default
    try:
        n, w = spec.split("/")
        return int(n), int(w)
    except ValueError:
        return default


class SlidingWindowLimiter:
    """At most `limit` events per `window_s` seconds per key."""

    MAX_KEYS = 50_000  # bound memory under abuse

    def __init__(self, limit: int, window_s: int, clock=time.monotonic):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._events: dict[str, deque] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> float:
        """
        Record an event for `key` if allowed. Returns 0.0 when allowed, else the
        number of seconds until the next event would be allowed (nothing recorded).
        """
        if self.limit <= 0:
            return 0.0
        now = self.clock()
        with self._lock:
            q = self._events.get(key)
            if q is None:
                if len(self._events) >= self.MAX_KEYS:
                    self._prune(now)
                q = self._events[key] = deque()
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return max(0.0, self.window_s - (now - q[0]))
            q.append(now)
            return 0.0

    def _prune(self, now: float) -> None:
        for k in [k for k, q in self._events.items() if not q or now - q[-1] >= self.window_s]:
            del self._events[k]


class RateLimiter:
    """Session + IP limits and a message-length cap, with user-facing messages."""

    def __init__(self, session_spec: str | None = None, ip_spec: str | None = None,
                 max_chars: int | None = None, clock=time.monotonic):
        s_n, s_w = parse_spec(session_spec if session_spec is not None else os.getenv("RATE_LIMIT_SESSION"), (20, 600))
        i_n, i_w = parse_spec(ip_spec if ip_spec is not None else os.getenv("RATE_LIMIT_IP"), (60, 3600))
        self.session = SlidingWindowLimiter(s_n, s_w, clock)
        self.ip = SlidingWindowLimiter(i_n, i_w, clock)
        self.max_chars = max_chars if max_chars is not None else int(os.getenv("MAX_MESSAGE_CHARS", "2000"))

    def check(self, message: str, session_id: str, ip: str | None) -> str | None:
        """None if the message may be processed, else the message to show the user."""
        if self.max_chars and len(message) > self.max_chars:
            return (f"⚠️ **Message too long** ({len(message)} characters). This demo accepts up to "
                    f"{self.max_chars} characters per message. Please shorten it and send again.")
        wait = self.session.hit(f"s:{session_id}")
        if wait:
            return self._limited("this conversation", self.session, wait)
        if ip:
            wait = self.ip.hit(f"ip:{ip}")
            if wait:
                return self._limited("your network address", self.ip, wait)
        return None

    @staticmethod
    def _limited(scope: str, lim: SlidingWindowLimiter, wait: float) -> str:
        return (f"⏳ **Rate limit reached.** This public demo allows {lim.limit} messages per "
                f"{_human(lim.window_s)} for {scope}. Your message was **not** processed. "
                f"Please try again in about {max(1, round(wait))} seconds.")


def _human(seconds: int) -> str:
    if seconds % 3600 == 0:
        h = seconds // 3600
        return "hour" if h == 1 else f"{h} hours"
    if seconds % 60 == 0:
        m = seconds // 60
        return "minute" if m == 1 else f"{m} minutes"
    return f"{seconds} seconds"


def client_ip(request) -> str | None:
    """Client IP from a Gradio/Starlette request (X-Forwarded-For first, then the socket peer)."""
    if request is None:
        return None
    headers = getattr(request, "headers", None) or {}
    xff = headers.get("x-forwarded-for") if hasattr(headers, "get") else None
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            idx = int(os.getenv("RATE_LIMIT_XFF_INDEX", "-1"))
            try:
                return parts[idx]
            except IndexError:
                return parts[-1]
    client = getattr(request, "client", None)
    return getattr(client, "host", None)
