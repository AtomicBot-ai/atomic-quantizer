"""kld_diff: per chunk values out of running means, paired intervals over chunks, bench checks."""
import random

import pytest

import kld_diff
from synth import kld_log, write_kld_log


def chunks(n, mu, sd, seed):
    rnd = random.Random(seed)
    return [max(0.0, rnd.gauss(mu, sd)) for _ in range(n)]


def test_per_chunk_values_come_back_from_running_means():
    kld = [0.0137, 0.0126, 0.0118, 0.0102, 0.0099]
    top = [93.0, 95.0, 94.0, 96.0, 95.5]
    r = kld_diff.parse(kld_log(kld, top))
    assert r["chunks"] == 5 and r["n_ctx"] == 4096 and r["ppl_base"] == 5.421271
    # the running KLD is printed with 5 decimals: chunk i carries up to ~i * 1e-5 of rounding
    for i, (got, want) in enumerate(zip(r["kld"], kld), 1):
        assert got == pytest.approx(want, abs=i * 1e-5)
    assert r["top1"] == pytest.approx(top, abs=0.01)
    assert sum(r["kld"]) / 5 == pytest.approx(r["mean_kld"], abs=1e-5)   # the mean telescopes


def test_paired_difference_and_interval():
    n = 30
    a = chunks(n, 0.011, 0.002, 1)
    b = [x + 0.0005 + random.Random(i).gauss(0, 0.0002) for i, x in enumerate(a)]
    ra, rb = kld_diff.parse(kld_log(a, [95] * n)), kld_diff.parse(kld_log(b, [95] * n))
    r = kld_diff.paired(ra, rb, reps=2000)
    assert r["d_kld"] == pytest.approx(0.0005, abs=1.5e-4)
    # the spread of the per chunk difference, not of either side: 0.0002 / sqrt(30) plus rounding
    assert 2e-5 < r["d_kld_se"] < 8e-5
    lo, hi = r["d_kld_ci95"]
    assert lo < r["d_kld"] < hi and hi - lo < 4.5 * r["d_kld_se"]
    assert kld_diff.verdict(r) == "worse"
    assert kld_diff.verdict(kld_diff.paired(rb, ra, reps=500)) == "better"


def test_a_tie_is_a_tie():
    a = chunks(30, 0.011, 0.002, 2)
    b = [x + random.Random(100 + i).gauss(0, 0.0003) for i, x in enumerate(a)]
    r = kld_diff.paired(kld_diff.parse(kld_log(a, [95] * 30)), kld_diff.parse(kld_log(b, [95] * 30)), reps=500)
    assert kld_diff.verdict(r) == "tie"


def test_per_gb_scales_the_interval():
    a = chunks(10, 0.01, 0.001, 3)
    b = [x + 0.001 for x in a]
    r = kld_diff.paired(kld_diff.parse(kld_log(a, [95] * 10)), kld_diff.parse(kld_log(b, [95] * 10)), reps=500)
    g = kld_diff.per_gb(r, 0.25)
    assert g["per_gb"] == pytest.approx(r["d_kld"] * 4)
    assert g["per_gb_se"] == pytest.approx(r["d_kld_se"] * 4)
    assert kld_diff.per_gb(r, 0) is None


@pytest.mark.parametrize("change, message", [
    ({"n": 29}, "chunk counts differ"),
    ({"n_ctx": 2048}, "context differs"),
    ({"ppl_base": 5.5}, "different references"),
])
def test_refuses_a_different_bench(change, message):
    a = kld_diff.parse(kld_log([0.01] * 30, [95] * 30))
    n = change.pop("n", 30)
    b = kld_diff.parse(kld_log([0.01] * n, [95] * n, **change))
    with pytest.raises(kld_diff.KldLogError, match=message):
        kld_diff.paired(a, b)


def test_an_unfinished_log_is_not_a_log(tmp_path):
    p = tmp_path / "kld.log"
    p.write_text(kld_log([0.01] * 3, [95] * 3).split("======")[0])
    assert not kld_diff.finished(str(p))
    assert kld_diff.finished(write_kld_log(tmp_path / "ok.log", [0.01] * 3))


def test_real_logs_of_the_4b_bench(local_fixture):
    """Stock Q4_K_M and the hand mask B2 on the 24.09 bench: the numbers in the masks README."""
    d = kld_diff.parse_file(local_fixture("qwen3.5-4b-bands", "kld-D-Q4_K_M.log"))
    b2 = kld_diff.parse_file(local_fixture("qwen3.5-4b-bands", "kld-mask-B2.log"))
    assert (d["chunks"], d["n_ctx"], d["mean_kld"], d["top1_pct"]) == (30, 4096, 0.024006, 92.778)
    assert b2["mean_kld"] == 0.011308
    r = kld_diff.paired(d, b2, reps=2000)
    assert r["d_kld"] == pytest.approx(-0.012698, abs=1e-6)
    assert r["b_lower_in"] == 30 and kld_diff.verdict(r) == "better"
