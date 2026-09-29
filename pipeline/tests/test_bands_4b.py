"""The band methods on Qwen3.5-4B (2026-09-29), replayed from their logs.

AD-Q5_K_M-Q4_K_M of dense-hybrid, bands at three sizes, from three sources:
depth fractions (the profile, August 1:3 head:tail), band_select (imatrix
Sum(Act^2) of ffn_down) and scan (quarters ranked by measured dKLD per GB).
All on the bench of the hand masks (fixtures/README.md). The tests pin the
decisions the numbers carry, so a change to a tool that would have led to
another decision shows up here.
"""
import json
import os

import pytest

import band_select
import gguf_inventory
import kld_diff
import ladder_gen
import scan

EDGE_MID = {8: 4, 12: 4, 16: 4}
# file sizes as built; the hand masks B2 (3 105 597 504) and G (3 159 009 344) sit at the two larger ones
SIZE = {8: 3041536064, 12: 3102730304, 16: 3163924544}


@pytest.fixture(scope="module")
def fx(local_fixture):
    return lambda name: local_fixture("qwen3.5-4b-bands", name)


@pytest.fixture(scope="module")
def log(fx):
    return lambda name: kld_diff.parse_file(fx(name))


@pytest.fixture(scope="module")
def inv4b(local_fixture):
    return gguf_inventory.from_log(local_fixture("qwen3.5-4b-masks", "quantize-E-Q5_K_S.log"))


@pytest.fixture(scope="module")
def dense(profiles):
    return ladder_gen.load_profile(os.path.join(profiles, "dense-hybrid.yaml"))


@pytest.fixture(scope="module")
def scan_doc(fx):
    with open(fx("scan.json")) as f:
        return json.load(f)


def fractions(n_edge, n_mid, eligible):
    head = n_edge // 4
    return {"edge": eligible[:head] + eligible[-(n_edge - head):], "mid": eligible[head:head + n_mid]}


def test_the_bands_that_were_built(fx, dense, inv4b, scan_doc):
    eligible = ladder_gen.compute_bands(dense, inv4b)["eligible"]
    with open(fx("imatrix-stats.txt")) as f:
        stats = f.read()
    assert ladder_gen.compute_bands(dense, inv4b)["edge"] == fractions(8, 4, eligible)["edge"]
    built = {
        ("bs", 8): ([22, 23, 26, 27, 28, 29, 30, 31], [19, 21, 24, 25]),
        ("bs", 12): ([19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], [16, 17, 18, 20]),
        ("bs", 16): (list(range(16, 32)), [6, 7, 14, 15]),
        ("scan", 8): (list(range(24, 32)), [0, 1, 2, 3]),
        ("scan", 12): ([0, 1, 2, 3] + list(range(24, 32)), [4, 5, 6, 7]),
        ("scan", 16): (list(range(0, 8)) + list(range(24, 32)), [20, 21, 22, 23]),
        ("frac", 12): ([0, 1, 2] + list(range(23, 32)), [3, 4, 5, 6]),
        ("frac", 16): ([0, 1, 2, 3] + list(range(20, 32)), [4, 5, 6, 7]),
    }
    for (method, e), (edge, mid) in built.items():
        if method == "bs":
            b = band_select.build(stats, dense, inv4b, n_edge=e, n_mid=EDGE_MID[e])
        elif method == "scan":
            b = scan.scan_bands(scan_doc, eligible, e, EDGE_MID[e])
        else:
            b = fractions(e, EDGE_MID[e], eligible)
        assert (b["edge"], b["mid"]) == (edge, mid), (method, e)
        ladder, problems = ladder_gen.build(dense, inv4b, only=["AD-Q5_K_M-Q4_K_M"], bands_override=b)
        assert not problems
        # 10 969 152 bytes of GGUF header and tokenizer on top of the tensors
        assert ladder["rungs"][0]["predicted_bytes"] + 10969152 == SIZE[e], (method, e)


def test_scan_ranks_the_ends_first(scan_doc):
    assert scan_doc["protocol"]["base"] == "q8_0" and scan_doc["protocol"]["chunks"] == 30
    assert len(scan_doc["groups"]) == 18 and not scan_doc["missing"]
    assert sum(scan_doc["wall_s"]) < 2 * 3600                  # the whole scan in under two hours
    q = scan.quarter_ranking(scan_doc, {"ffn_down", "ffn_gate_up"})
    assert [x["quarter"] for x in q] == [4, 1, 3, 2]
    for hi, lo in zip(q, q[1:]):                               # every step of the ranking beyond 2 SE
        assert hi["per_gb"] - lo["per_gb"] > 2 * (hi["per_gb_se"] ** 2 + lo["per_gb_se"] ** 2) ** 0.5
    per_gb = {g["group"]: g["per_gb"] for g in scan_doc["groups"]}
    # attn_k/attn_v at q8_0 on every rung: the most damage per byte of the whole model
    assert max(per_gb, key=per_gb.get) == "attn_kv"
    assert per_gb["ssm_out"] > per_gb["attn_gate"]


def test_band_select_loses_to_depth(log):
    """Sum(Act^2) of ffn_down grows with depth, so it bands only the tail and drops blocks 0-7."""
    for e in (8, 12, 16):
        r = kld_diff.paired(log(f"kld-frac-{e}-4.log"), log(f"kld-bs-{e}-4.log"), reps=2000)
        assert kld_diff.verdict(r) == "worse" and r["rel"] > 0.09, e


def test_scan_and_fractions_are_the_same_bands_in_effect(log):
    for e in (8, 12, 16):
        r = kld_diff.paired(log(f"kld-frac-{e}-4.log"), log(f"kld-scan-{e}-4.log"), reps=2000)
        assert abs(r["rel"]) < 0.01 and abs(r["z"]) < 2.5, e


def test_against_the_hand_masks(log):
    b2 = log("kld-mask-B2.log")
    for m in ("frac", "scan"):   # at the size of B2 both beat it by about a sixth
        r = kld_diff.paired(b2, log(f"kld-{m}-12-4.log"), reps=2000)
        assert kld_diff.verdict(r) == "better" and r["rel"] < -0.15
    assert kld_diff.verdict(kld_diff.paired(b2, log("kld-bs-12-4.log"), reps=2000)) == "tie"
    # at the size of G no band choice on this rung is enough: G has all of the ffn at q5_K
    g = log("kld-mask-G.log")
    for m in ("frac", "scan", "bs"):
        assert kld_diff.verdict(kld_diff.paired(g, log(f"kld-{m}-16-4.log"), reps=2000)) == "worse"
    # the rung with the ffn at q5_K and no bands at all, 0.3 % larger, beats it
    r = kld_diff.paired(g, log("kld-q5-noband.log"), reps=2000)
    assert kld_diff.verdict(r) == "better" and r["rel"] < -0.08


def test_q5_rung_without_bands_is_what_was_built(dense, inv4b):
    ladder, problems = ladder_gen.build(dense, inv4b, only=["AD-Q5_K_M"], bands_override={"edge": [], "mid": []})
    assert not problems and ladder["rungs"][0]["predicted_bytes"] + 10969152 == 3168184384
    eff = ladder["rungs"][0]["_effective"]
    assert {eff[f"blk.{b}.ffn_{k}.weight"] for b in range(32) for k in ("down", "gate", "up")} == {"q5_k"}
