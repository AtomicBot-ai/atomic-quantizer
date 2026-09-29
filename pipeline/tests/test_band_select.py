"""band_select: imatrix statistics -> explicit bands, and ladder_gen taking bands as block lists."""
import copy
import json
import os
import subprocess
import sys

import pytest

import band_select
import gguf_inventory
import ladder_gen

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def inv4b(local_fixture):
    return gguf_inventory.from_log(local_fixture("qwen3.5-4b-masks", "quantize-E-Q5_K_S.log"))


@pytest.fixture(scope="module")
def stats4b(local_fixture):
    with open(local_fixture("qwen3.5-4b-bands", "imatrix-stats.txt")) as f:
        return f.read()


@pytest.fixture(scope="module")
def dense(profiles):
    return ladder_gen.load_profile(os.path.join(profiles, "dense-hybrid.yaml"))


def test_parse_the_4b_statistics(stats4b):
    rows = band_select.parse_stats(stats4b)
    # 248 tensors in the imatrix; the per layer summary (32 rows) is not taken for tensors
    assert len(rows) == 248
    top = max(rows, key=lambda r: r["sum_act2"])
    assert (top["block"], top["tensor"], top["sum_act2"]) == (31, "ffn_down", 6582.43)
    assert {r["tensor"] for r in rows} >= {"ffn_down", "ffn_gate", "attn_gate", "ssm_out", "attn_k"}


@pytest.mark.parametrize("line", [
    "    28\tattn_gate           \t   4712.58\t  0.8382\t   356.9317",                  # plain
    "0.00.054.285 I    28\tattn_gate           \t   4712.58\t  0.8382\t   356.9317",   # with log timestamps
    "  28 | attn_gate | 4712.58 | 0.8382 | 356.9317 |",                                # pipes
])
def test_parse_accepts_the_layouts(line):
    assert band_select.parse_stats(line) == [{"block": 28, "tensor": "attn_gate", "sum_act2": 4712.58}]


def test_parse_skips_the_layer_summary_and_refuses_noise():
    assert band_select.parse_stats("    3\t       1390.34\t    2.7037%\t0.2452\n  1 ffn_down 5.9") == \
        [{"block": 1, "tensor": "ffn_down", "sum_act2": 5.9}]
    with pytest.raises(band_select.BandError):
        band_select.parse_stats("nothing to see\n")


def test_4b_bands_are_the_tail(stats4b, dense, inv4b):
    b = band_select.build(stats4b, dense, inv4b)
    # same counts as the profile's fractions give on 32 blocks: 2 + 6 edge, 4 mid
    assert b["select"] == dense["bands"]["group"]
    assert b["edge"] == [22, 23, 26, 27, 28, 29, 30, 31] and b["mid"] == [19, 21, 24, 25]
    assert [x[0] for x in b["ranking"]][:4] == [31, 30, 29, 28]
    assert 32 not in [x[0] for x in b["ranking"]]   # the MTP block never carries a band
    wide = band_select.build(stats4b, dense, inv4b, select=r"^blk\.", n_edge=12, n_mid=4)
    assert len(wide["edge"]) == 12 and len(wide["mid"]) == 4 and 3 in wide["edge"] + wide["mid"]


def test_too_many_blocks(stats4b, dense, inv4b):
    with pytest.raises(band_select.BandError, match="only 32"):
        band_select.build(stats4b, dense, inv4b, n_edge=30, n_mid=4)


def test_bands_from_changes_only_the_blocks(stats4b, dense, inv4b):
    b = band_select.build(stats4b, dense, inv4b)
    frac, _ = ladder_gen.build(dense, inv4b, only=["AD-Q5_K_M-Q4_K_M"])
    sel, problems = ladder_gen.build(dense, inv4b, only=["AD-Q5_K_M-Q4_K_M"], bands_override=b)
    assert not problems
    assert sel["bands"] == {"edge": b["edge"], "mid": b["mid"]} and sel["bands_from"]["method"] == "band_select"
    rf, rs = frac["rungs"][0], sel["rungs"][0]
    assert rs["predicted_bytes"] == rf["predicted_bytes"]    # same counts, same bytes
    moved = {n for n in rf["_effective"] if rf["_effective"][n] != rs["_effective"][n]}
    assert moved and all(ladder_gen.block_of(n) is not None and ".ffn_" in n for n in moved)
    assert rs["_effective"]["blk.0.ffn_gate.weight"] == "q4_k" and rs["_effective"]["blk.22.ffn_gate.weight"] == "q6_k"


def test_profile_block_lists(dense, inv4b):
    p = copy.deepcopy(dense)
    p["bands"] = {"group": dense["bands"]["group"], "edge_blocks": [0, 31], "mid_blocks": [15]}
    ladder, _ = ladder_gen.build(p, inv4b, only=["AD-Q4_K_M"])
    assert ladder["bands"] == {"edge": [0, 31], "mid": [15]}
    p["bands"]["tail"] = 3
    with pytest.raises(ladder_gen.LadderError, match="both block lists and counts"):
        ladder_gen.build(p, inv4b)


@pytest.mark.parametrize("bands, message", [
    ({"edge": [0, 32], "mid": []}, r"blocks \[32\] that do not carry"),      # the MTP block
    ({"edge": [0, 40], "mid": []}, r"blocks \[40\]"),
    ({"edge": [0, 1], "mid": [1]}, "in both"),
    ({"edge": [3, 3], "mid": []}, "twice"),
])
def test_bad_block_lists_are_refused(bands, message, dense, inv4b):
    with pytest.raises(ladder_gen.LadderError, match=message):
        ladder_gen.build(dense, inv4b, bands_override=bands)


def test_cli_round_trip(tmp_path, local_fixture, profiles, inv4b):
    inv = tmp_path / "inv.json"
    inv.write_text(json.dumps(inv4b))
    lib = os.path.join(HERE, "..", "lib")
    prof = os.path.join(profiles, "dense-hybrid.yaml")
    subprocess.run([sys.executable, os.path.join(lib, "band_select.py"), local_fixture("qwen3.5-4b-bands", "imatrix-stats.txt"),
                    "--inventory", str(inv), "--profile", prof, "--edge", "12", "--mid", "4", "-o", str(tmp_path / "b.json")],
                   check=True, capture_output=True)
    out = subprocess.run([sys.executable, os.path.join(lib, "ladder_gen.py"), "--inventory", str(inv), "--profile", prof,
                          "--only", "AD-Q5_K_M-Q4_K_M", "--bands-from", str(tmp_path / "b.json"), "--out", str(tmp_path / "l")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    ladder = json.loads((tmp_path / "l" / "ladder.json").read_text())
    assert len(ladder["bands"]["edge"]) == 12 and ladder["bands_from"]["source"] == "imatrix-stats.txt"
