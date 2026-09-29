"""The qwen4exp (Qwen3.8-Flash-Next) profile against the model and its August release.

The inventory is Flash-Next as convert_hf_to_gguf.py would write it (a --remote
--dry-run, no download); the release types are read from the shard headers of
AtomicChat/Qwen3.8-Flash-Next-GGUF. See fixtures/README.md.
"""
import copy
import json
import os

import pytest

import ladder_gen

GB = 1e9


@pytest.fixture(scope="module")
def inv(local_fixture):
    with open(local_fixture("qwen3.8-flash-next", "inventory.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def release(local_fixture):
    with open(local_fixture("qwen3.8-flash-next", "release-AD-3.84bpw.types.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def prof(profiles):
    return ladder_gen.load_profile(os.path.join(profiles, "moe-qwen4exp.yaml"))


def test_inventory_is_the_release_tensor_set(inv, release):
    # the dry-run inventory names exactly the tensors of the published file
    assert {t["name"] for t in inv["tensors"]} == set(release["types"])
    assert inv["arch"] == "qwen4exp" and inv["block_count"] == 48 and inv["mtp_blocks"] == []


def test_every_rung_passes(prof, inv):
    ladder, problems = ladder_gen.build(prof, inv)
    assert problems == {}
    assert len(ladder["rungs"]) == 12
    # the default band is the August one: blocks 0-3 and 40-47
    assert ladder["bands"]["edge"] == [0, 1, 2, 3, 40, 41, 42, 43, 44, 45, 46, 47]


def test_short_rows_only_take_block32_types(prof, inv):
    ladder, _ = ladder_gen.build(prof, inv)
    short = {t["name"] for t in inv["tensors"] if t["shape"][0] % 256 and ladder_gen.quantizable(t)}
    assert "per_layer_token_embd.weight" in short and "blk.0.ffn_down_exps.weight" in short
    for r in ladder["rungs"]:
        for name in short:
            typ = r["_effective"][name]
            assert ladder_gen.GGML[typ][0] <= 32, (r["label"], name, typ)


def test_iq2_xs_is_the_august_after_build(prof, inv):
    # tech brief: table q4_1, gate/up iq2_xs, blocks 0-3 and 40-47 iq3_xxs, ffn_down iq4_nl: 85.3 GB
    ladder, _ = ladder_gen.build(prof, inv, ["AD-IQ2_XS"])
    assert ladder["rungs"][0]["predicted_bytes"] / GB == pytest.approx(85.3, abs=0.05)


def august_before(prof):
    """The August "before" build as a rung of this profile: what is on the hub today."""
    p = copy.deepcopy(prof)
    p["bands"] = {"group": prof["bands"]["group"], "head": 6, "tail": 6}   # blocks 0-5 and 42-47
    p["defaults"]["ple_conv"] = "f16"
    p["rungs"] = [{"label": "BEFORE", "ftype": "IQ1_M", "types": {
        "ple_table": "q5_1", "edge_down_exps": "mxfp4", "edge_gate_up_exps": "iq2_s",
        "down_exps": "mxfp4", "gate_up_exps": "iq1_m"}}]
    return p


def test_published_release_is_replayed_tensor_for_tensor(prof, inv, release):
    p = august_before(prof)
    p.pop("refuse_types")
    ladder, problems = ladder_gen.build(p, inv)
    assert problems == {}
    eff = ladder["rungs"][0]["_effective"]
    assert {n: t for n, t in eff.items()} == release["types"]
    # 84.9 GB in the brief, 79.1 GiB on the Mac
    assert ladder["rungs"][0]["predicted_bytes"] / GB == pytest.approx(84.92, abs=0.01)
    assert release["kv"]["general.file_type"] == 31   # labelled IQ1_M, 3.84 bpw on disk


def test_mxfp4_on_the_experts_is_refused(prof, inv):
    problems = ladder_gen.build(august_before(prof), inv)[1]["BEFORE"]
    assert len(problems) == 48 and all("ffn_down_exps" in e and "mxfp4" in e for e in problems)


def test_k_quant_on_down_exps_is_refused(prof, inv):
    p = copy.deepcopy(prof)
    p["rungs"] = [dict(prof["rungs"][4], label="R")]
    p["rungs"][0]["types"] = dict(p["rungs"][0]["types"], down_exps="q4_k")
    problems = ladder_gen.build(p, inv)[1]["R"]
    assert problems and all("ffn_down_exps" in e and "not a multiple of 256" in e for e in problems)


def test_token_embedding_flag_would_hit_the_table(prof, inv):
    p = copy.deepcopy(prof)
    p["flags"] = {"token_embedding_type": "q8_0"}
    problems = ladder_gen.build(p, inv, ["AD-IQ2_XS"])[1]["AD-IQ2_XS"]
    assert any(e.startswith("per_layer_token_embd.weight") and "flag" in e for e in problems)


def test_low_bit_on_uncalibrated_output_hc_is_refused(prof, inv):
    p = copy.deepcopy(prof)
    p["defaults"]["output_hc"] = "iq2_xs"
    problems = ladder_gen.build(p, inv, ["AD-IQ2_XS"])[1]["AD-IQ2_XS"]
    assert any(e.startswith("output_hc_down.weight") and "needs imatrix" in e for e in problems)


def test_band_override_from_band_select(prof, inv):
    if not hasattr(ladder_gen, "explicit_bands"):
        pytest.skip("bands from band_select.py land in their own commit")
    ladder, problems = ladder_gen.build(prof, inv, ["AD-IQ2_XS"], {"edge": [0] + list(range(36, 47)), "mid": []})
    assert problems == {}
    eff = ladder["rungs"][0]["_effective"]
    assert eff["blk.36.ffn_up_exps.weight"] == "iq3_xxs" and eff["blk.47.ffn_up_exps.weight"] == "iq2_xs"


def test_profile_rungs_are_ordered_by_size(prof, inv):
    ladder, _ = ladder_gen.build(prof, inv)
    sizes = [r["predicted_bytes"] for r in ladder["rungs"] if not r["control"]]
    assert sizes == sorted(sizes, reverse=True)
