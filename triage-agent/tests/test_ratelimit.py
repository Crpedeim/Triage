"""Tests for ratelimit.py (fake clock, no sleeping)."""

from types import SimpleNamespace

from ratelimit import RateLimiter, SlidingWindowLimiter, client_ip, parse_spec


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_parse_spec():
    assert parse_spec("20/600", (1, 1)) == (20, 600)
    assert parse_spec("", (1, 1)) == (1, 1)
    assert parse_spec("garbage", (5, 60)) == (5, 60)


def test_sliding_window_allows_limit_then_blocks_then_recovers():
    clock = FakeClock()
    lim = SlidingWindowLimiter(3, 60, clock)
    assert [lim.hit("k") for _ in range(3)] == [0.0, 0.0, 0.0]
    wait = lim.hit("k")
    assert 59 <= wait <= 60
    clock.t += 30
    assert lim.hit("k") > 0          # still inside the window
    clock.t += 31
    assert lim.hit("k") == 0.0       # oldest event expired
    assert lim.hit("other") == 0.0   # keys are independent


def test_zero_limit_disables():
    lim = SlidingWindowLimiter(0, 60, FakeClock())
    assert all(lim.hit("k") == 0.0 for _ in range(100))


def test_session_limit_message_and_ip_limit_across_sessions():
    clock = FakeClock()
    rl = RateLimiter("2/600", "3/3600", max_chars=100, clock=clock)
    assert rl.check("hi", "s1", "1.2.3.4") is None
    assert rl.check("hi", "s1", "1.2.3.4") is None
    msg = rl.check("hi", "s1", "1.2.3.4")
    assert msg and "Rate limit reached" in msg and "2 messages per 10 minutes" in msg
    assert "not** processed" in msg
    # a new session from the same IP still counts against the IP limit
    assert rl.check("hi", "s2", "1.2.3.4") is None   # IP event 3
    msg = rl.check("hi", "s3", "1.2.3.4")
    assert msg and "network address" in msg and "per hour" in msg
    # another IP is unaffected
    assert rl.check("hi", "s4", "5.6.7.8") is None


def test_session_block_does_not_consume_ip_quota():
    rl = RateLimiter("1/600", "2/3600", max_chars=100, clock=FakeClock())
    assert rl.check("a", "s1", "ip") is None
    assert rl.check("a", "s1", "ip")  # session-limited, IP not charged
    assert rl.check("a", "s2", "ip") is None


def test_message_too_long():
    rl = RateLimiter("5/60", "5/60", max_chars=10, clock=FakeClock())
    assert "too long" in rl.check("x" * 11, "s", "ip")


def test_client_ip_from_forwarded_header(monkeypatch):
    req = SimpleNamespace(headers={"x-forwarded-for": "203.0.113.9, 198.51.100.7"},
                          client=SimpleNamespace(host="10.0.0.1"))
    monkeypatch.setenv("RATE_LIMIT_XFF_INDEX", "-1")
    assert client_ip(req) == "198.51.100.7"
    monkeypatch.setenv("RATE_LIMIT_XFF_INDEX", "0")
    assert client_ip(req) == "203.0.113.9"
    assert client_ip(SimpleNamespace(headers={}, client=SimpleNamespace(host="10.0.0.1"))) == "10.0.0.1"
    assert client_ip(None) is None
