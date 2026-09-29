"""band_select on the August Flash-Next imatrix against the band Boris chose by hand.

The August "after" build (tech brief: KLD 0.2277 -> 0.1074 at 85 GB) put blocks
0-3 and 40-47 one step up. He read the top of --show-statistics (all attn_gate:
40, 46, 42, 41, 45, 44, then blk.0) and rounded it to two runs of blocks. If
band_select, fed the same matrix, lands on the same blocks, the automation can
be trusted for MoE; these tests pin that it does not, and how it misses.

The matrix is the August one (dense-share corpus, llama.cpp before the GDN
normalisation fix), so on the day the check is repeated on the new matrix;
the conclusion here is what that check is compared with.
"""
import os

import pytest

import ladder_gen

band_select = pytest.importorskip("band_select", reason="band_select.py comes with the bands PR")

AUGUST = set(range(0, 4)) | set(range(40, 48))


@pytest.fixture(scope="module")
def stats(local_fixture):
    with open(local_fixture("qwen3.8-flash-next", "imatrix-4000.stats.txt")) as f:
        return f.read()


@pytest.fixture(scope="module")
def inv(local_fixture):
    import json
    with open(local_fixture("qwen3.8-flash-next", "inventory.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def prof(profiles):
    return ladder_gen.load_profile(os.path.join(profiles, "moe-qwen4exp.yaml"))


def picked(b):
    return set(b["edge"]) | set(b["mid"])


def test_the_profile_band_is_the_august_size(stats, prof, inv):
    b = band_select.build(stats, prof, inv)
    assert len(picked(b)) == len(AUGUST) == 12


def test_expert_energy_takes_the_tail_and_misses_the_head(stats, prof, inv):
    # the default group (ffn_gate_exps): Sum(Act^2) grows with depth, so the
    # ranking is nearly the block order reversed and the head never gets in
    b = band_select.build(stats, prof, inv)
    assert picked(b) == set(range(36, 48))
    order = [blk for blk, _, _ in b["ranking"]]
    assert order[-3:] == [0, 1, 2]
    assert picked(b) & AUGUST == set(range(40, 48))


def test_attn_gate_is_what_boris_read(stats, prof, inv):
    b = band_select.build(stats, prof, inv, select="attn_gate")
    order = [blk for blk, _, _ in b["ranking"]]
    assert order[:7] == [40, 46, 0, 42, 41, 45, 44]
    # past the seventh block the ranking scatters instead of filling 1-3 and 43/47
    assert picked(b) == {0, 17, 18, 32, 33, 34, 40, 41, 42, 44, 45, 46}
    assert len(picked(b) & AUGUST) == 7


def test_neither_ranking_reproduces_the_august_band(stats, prof, inv):
    for select in (None, "attn_gate", "ffn_down_exps", "."):
        assert picked(band_select.build(stats, prof, inv, select=select)) != AUGUST
