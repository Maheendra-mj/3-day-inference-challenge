import math

import numpy as np
import pytest

from infer_lab.bench.metrics import RequestRecord, summarize
from infer_lab.bench.workload import poisson_arrivals


def test_request_record_derived_times():
    r = RequestRecord(0, prompt_tokens=10, output_tokens=5, t_submit=1.0, t_first_token=1.2, t_end=2.0)
    assert r.ttft == pytest.approx(0.2)
    assert r.e2e == pytest.approx(1.0)
    assert r.tpot == pytest.approx(0.8 / 4)


def test_single_token_has_no_tpot():
    assert RequestRecord(0, 10, 1, 0.0, 0.1, 0.1).tpot is None


def test_summarize_throughput_and_percentiles():
    recs = [RequestRecord(i, 100, 50, 0.0, 0.1 * (i + 1), 1.0) for i in range(10)]
    m = summarize(recs, wall_time_s=2.0)
    assert m["num_requests"] == 10
    assert m["output_tokens_per_s"] == pytest.approx(500 / 2.0)
    assert m["total_tokens_per_s"] == pytest.approx(1500 / 2.0)
    assert m["ttft_p50_ms"] == pytest.approx(550.0)
    assert m["ttft_p99_ms"] <= 1000.0


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([], 1.0)


def test_poisson_arrivals_rate():
    t = poisson_arrivals(20_000, rate_rps=10.0, seed=1)
    assert t[0] == 0.0 and np.all(np.diff(t) >= 0)
    assert len(t) / t[-1] == pytest.approx(10.0, rel=0.05)
    assert np.all(poisson_arrivals(5, math.inf) == 0)
