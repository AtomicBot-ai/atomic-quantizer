"""The generated ladders must reproduce the published releases tensor for tensor.

For every rung the simulated llama-quantize outcome (which type each tensor
ends up with, which tensors print "applying manual override", file size) is
compared with the real log of the build that was published.
"""
import os

import pytest

import gguf_inventory
import ladder_gen
import quantlog

# published label -> log of the build behind it. The August logs kept the
# names from before rename_quant (AD-Q4_K is the file AD-Q4_K_M).
DENSE_RUNGS = {
    "Q8_0": "quantize-Q8_0.log",
    "AD-Q6_K": "quantize-AD-Q6_K.log",
    "AD-Q6_K-Q5_K": "quantize-AD-Q6_K-Q5_K.log",
    "AD-Q5_K_M": "quantize-AD-Q5_K.log",
    "AD-Q5_K_M-Q4_K_M": "quantize-AD-Q5_K-Q4_K.log",
    "AD-Q4_K_M": "quantize-AD-Q4_K.log",
    "AD-IQ4_XS": "quantize-AD-IQ4_XS.log",
    "AD-IQ4_XS-IQ3_S": "quantize-AD-IQ4_XS-IQ3_S.log",
    "AD-IQ3_S": "quantize-AD-IQ3_S.log",
    "AD-IQ3_S-IQ3_XXS": "quantize-AD-IQ3_S-IQ3_XXS.log",
    "AD-IQ3_XXS": "quantize-AD-IQ3_XXS.log",
    "AD-IQ2_S": "quantize-AD-IQ2_S.log",
    "AD-IQ2_S-IQ2_XS": "quantize-AD-IQ2_S-IQ2_XS.log",
    "AD-IQ2_XS": "quantize-AD-IQ2_XS.log",
    "AD-IQ2_XXS": "quantize-AD-IQ2_XXS.log",
    "AD-IQ1_M": "quantize-AD-IQ1_M.log",
}

# every Ling rung with a complete log (IQ2_M, IQ2_XXS and NVFP4 logs are truncated or absent)
MOE_RUNGS = {f"AD-{r}": f"quant-{r}.log" for r in (
    "Q6_K", "Q5_K_L", "Q5_K_M", "Q5_K_S", "Q4_K_L", "Q4_K_M", "IQ4_NL", "IQ4_XS", "Q4_K_S",
    "IQ4_XXS", "IQ3_M", "IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS", "IQ1_M", "IQ1_S")}
MOE_RUNGS.update({"Q4_K_FLAT": "quant-Q4_K_FLAT.log", "IQ4_XS_FLAT": "quant-IQ4_XS_FLAT.log"})


def replay(profile_path, inventory, label, log):
    profile = ladder_gen.load_profile(profile_path)
    ladder, problems = ladder_gen.build(profile, inventory, only=[label])
    assert not problems, problems
    rung = ladder["rungs"][0]
    # the August high rungs were built before the MTP pin fix: leave that block out there only
    mtp = inventory["mtp_blocks"]
    pin = profile["mtp"]["type"]
    in_mtp = [t for n, t in log["tensors"].items() if ladder_gen.block_of(n) in mtp and t["src"] == "bf16"]
    pinned = bool(in_mtp) and all(t["dst"] == pin for t in in_mtp)
    skip = None if (not mtp or pinned) else r"^blk\.(%s)\." % "|".join(map(str, mtp))
    return rung, skip


def check(rung, log, skip):
    import re
    rx = re.compile(skip) if skip else None
    diffs = ladder_gen.compare(rung, log, skip)
    assert not diffs, f"{len(diffs)} tensors differ, first: {diffs[:5]}"
    want = {n for n in rung["_overrides"] if not (rx and rx.search(n))}
    got = {n for n in log["overrides"] if not (rx and rx.search(n))}
    assert want == got, f"override sets differ: extra {sorted(want - got)[:5]} missing {sorted(got - want)[:5]}"
    assert log["fallbacks"] == 0
    if skip is None:
        # same tensors, same types: the predicted size must be the logged size
        assert rung["predicted_bytes"] / 2**20 == pytest.approx(log["quant_mib"], rel=0.002)
        assert rung["predicted_bpw"] == pytest.approx(log["quant_bpw"], abs=0.011)


@pytest.fixture(scope="module")
def dense_inventory(dense_log):
    return gguf_inventory.from_log(dense_log("quantize-Q8_0.log"))


@pytest.fixture(scope="module")
def moe_inventory(moe_log):
    return gguf_inventory.from_log(moe_log("quant-Q4_K_M.log"))


def test_dense_inventory(dense_inventory):
    assert dense_inventory["arch"] == "qwen35"
    assert dense_inventory["block_count"] == 65
    assert dense_inventory["mtp_blocks"] == [64]
    assert len(dense_inventory["tensors"]) == 866


@pytest.mark.parametrize("label", DENSE_RUNGS)
def test_dense_rung_matches_august(label, dense_log, dense_inventory, profiles):
    log = quantlog.parse_file(dense_log(DENSE_RUNGS[label]))
    rung, skip = replay(os.path.join(profiles, "dense-hybrid.yaml"), dense_inventory, label, log)
    check(rung, log, skip)


def test_dense_bands(dense_inventory, profiles):
    profile = ladder_gen.load_profile(os.path.join(profiles, "dense-hybrid.yaml"))
    bands = ladder_gen.compute_bands(profile, dense_inventory)
    assert bands["edge"] == [0, 1, 2, 3] + list(range(52, 64))
    assert bands["mid"] == list(range(4, 12))


@pytest.mark.parametrize("label", MOE_RUNGS)
def test_moe_rung_matches_ling(label, moe_log, moe_inventory, profiles):
    log = quantlog.parse_file(moe_log(MOE_RUNGS[label]))
    profile = os.path.join(profiles, "moe-hybrid.yaml")
    ladder, problems = ladder_gen.build(ladder_gen.load_profile(profile), moe_inventory, only=[label])
    assert not problems, problems
    rung = ladder["rungs"][0]
    assert rung["ftype"] == log["ftype"]
    check(rung, log, None)


def test_moe_bands(moe_inventory, profiles):
    profile = ladder_gen.load_profile(os.path.join(profiles, "moe-hybrid.yaml"))
    assert ladder_gen.compute_bands(profile, moe_inventory)["edge"] == [2, 3, 39, 40, 41]
