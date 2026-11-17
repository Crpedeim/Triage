"""tests/test_eval_metrics.py — the eval runner's metric code (no LLM calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.run_eval import SEVERITY, _cost, _p95, summarise  # noqa: E402


def _r(cid, exp, pred, wall=10.0, status="ok", per_model=None, calls=5):
    under = pred is None or SEVERITY[pred] < SEVERITY[exp]
    over = pred is not None and SEVERITY[pred] > SEVERITY[exp]
    return dict(id=cid, expected=exp, predicted=pred, correct=pred == exp, under_triage=under,
                over_triage=over, status=status, system_wall_s=wall, llm_calls=calls, llm_errors=[],
                per_model=per_model or {})


def test_under_and_over_triage_counting():
    rs = [_r("a", "emergency", "urgent"), _r("b", "urgent", "emergency"), _r("c", "standard", "standard"),
          _r("d", "self_care", None)]
    s = summarise(rs, {"label": "t"})
    assert s["accuracy"] == {"correct": 1, "n": 4}
    assert s["under_triage"]["count"] == 2 and s["under_triage"]["ids"] == ["a", "d"]  # no result counts as under
    assert s["over_triage"]["ids"] == ["b"]
    assert s["no_result"] == 1


def test_p95_nearest_rank():
    assert _p95([]) is None
    assert _p95([5.0]) == 5.0
    assert _p95(list(range(1, 7))) == 6            # n=6 → max
    assert _p95(list(range(1, 21))) == 19          # n=20 → 19th value
    assert _p95(list(range(1, 101))) == 95


def test_cost_known_only_when_every_model_priced():
    pm = {"gemini-3.1-pro-preview": dict(input=1_000_000, output=100_000, reasoning=100_000, total=1_200_000)}
    c = _cost(pm)
    assert c["priced_low"] == 2.0 + 1.2 and c["priced_high"] == 2.0 + 2.4 and c["total_known"] is True
    pm["gemini-3.8-flash"] = dict(input=10, output=10, reasoning=0, total=20)
    c2 = _cost(pm)
    assert c2["total_known"] is False and c2["unpriced_models"] == ["gemini-3.8-flash"]


def test_error_cases_excluded_from_latency():
    rs = [_r("a", "urgent", "urgent", wall=10.0), _r("b", "urgent", None, wall=0.0, status="error: x")]
    s = summarise(rs, {"label": "t"})
    assert s["latency_s"]["n"] == 1 and s["latency_s"]["median"] == 10.0


def test_cases_file_is_well_formed():
    cases = json.load(open(Path(__file__).parent.parent / "eval" / "cases.json"))
    assert len(cases) >= 30
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert c["expected"] in SEVERITY
        for k in ("basis", "opening", "vignette"):
            assert c[k].strip(), (c["id"], k)


def test_priced_cost_is_unknown_not_zero_when_no_model_priced():
    c = _cost({"gemini-3.8-flash": dict(input=1000, output=100, reasoning=0, total=1100)})
    assert c["priced_low"] is None and c["priced_high"] is None and c["total_known"] is False
    s = summarise([_r("a", "urgent", "urgent", per_model={"gemini-3.8-flash": dict(input=1, output=1, reasoning=0, total=2)})],
                  {"label": "t"})
    assert s["priced_cost_per_triage_usd"]["low_median"] is None
