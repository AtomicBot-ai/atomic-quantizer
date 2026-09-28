"""Small tied-embedding Qwen3.5: the 4B masks and the 2B smoke run, kept as tests.

The masks were hand-written --tensor-type-files on Qwen3.5-4B (2026-09-24);
profiles/qwen35-masks.yaml spells them out as rungs, and here the generator
must rebuild each one tensor for tensor from its original quantize log.

The 2B smoke run (2026-09-25) was the first run of the pipeline on a rented
box. Its ladder was built before tied_embeddings existed, which gives two
checks: the old behaviour reproduces that ladder byte for byte, and the new
one changes nothing but the head.
"""
import copy
import json
import os
import re

import pytest

import gguf_inventory
import ladder_gen
import quantlog

MASKS = {  # rung label -> quantize log of the hand-made build
    "Q5_K_S": "quantize-E-Q5_K_S.log",
    "F1": "quantize-F1-q5embd.log",
    "AD-Q5_K_S": "quantize-G-q5edges-embd6.log",
}


@pytest.fixture(scope="module")
def inv4b(local_fixture):
    return gguf_inventory.from_log(local_fixture("qwen3.5-4b-masks", MASKS["Q5_K_S"]))


@pytest.fixture(scope="module")
def inv2b(local_fixture):
    return gguf_inventory.from_log(local_fixture("qwen3.5-2b-smoke", "quantize-Q8_0.log"))


@pytest.fixture(scope="module")
def smoke_ladder(local_fixture):
    with open(local_fixture("qwen3.5-2b-smoke", "ladder.json")) as f:
        return json.load(f)


def test_small_inventories_are_tied(inv4b, inv2b):
    for inv, blocks in ((inv4b, 33), (inv2b, 25)):
        assert inv["arch"] == "qwen35" and inv["block_count"] == blocks and inv["mtp_blocks"] == [blocks - 1]
        names = {t["name"] for t in inv["tensors"]}
        assert "token_embd.weight" in names and "output.weight" not in names


@pytest.mark.parametrize("label", MASKS)
def test_mask_rebuilds_the_4b_build(label, inv4b, local_fixture, profiles):
    profile = ladder_gen.load_profile(os.path.join(profiles, "qwen35-masks.yaml"))
    ladder, problems = ladder_gen.build(profile, inv4b, only=[label])
    assert not problems, problems
    assert ladder["tied_embeddings"] == "output"
    assert ladder["bands"]["edge"] == [0, 1, 2, 3, 28, 29, 30, 31]
    rung = ladder["rungs"][0]
    log = quantlog.parse_file(local_fixture("qwen3.5-4b-masks", MASKS[label]))
    diffs = ladder_gen.compare(rung, log)
    assert not diffs, f"{len(diffs)} tensors differ, first: {diffs[:5]}"
    assert log["fallbacks"] == 0
    # the hand builds ran on a Q5_K_S base, these rungs on Q8_0 with every tensor named:
    # same types, so the same bytes (the override sets differ by construction)
    assert rung["predicted_bytes"] / 2**20 == pytest.approx(log["quant_mib"], rel=0.002)


def test_mask_bands_scale_to_the_2b(inv2b, profiles):
    profile = ladder_gen.load_profile(os.path.join(profiles, "qwen35-masks.yaml"))
    ladder, problems = ladder_gen.build(profile, inv2b)
    assert not problems, problems
    assert ladder["bands"]["edge"] == [0, 1, 2, 21, 22, 23]
    g = next(r for r in ladder["rungs"] if r["label"] == "AD-Q5_K_S")["_effective"]
    assert g["token_embd.weight"] == "q6_k" and g["blk.24.ffn_down.weight"] == "q8_0"   # head, MTP pin


def untied(profile):
    p = copy.deepcopy(profile)
    p.pop("tied_embeddings", None)
    return p


def test_smoke_ladder_is_the_generator_before_tied_embeddings(inv2b, smoke_ladder, profiles):
    profile = untied(ladder_gen.load_profile(os.path.join(profiles, "dense-hybrid.yaml")))
    ladder, problems = ladder_gen.build(profile, inv2b)
    assert not problems, problems
    assert [r["label"] for r in ladder["rungs"]] == [r["label"] for r in smoke_ladder["rungs"]]
    for new, old in zip(ladder["rungs"], smoke_ladder["rungs"]):
        assert new["rules_sha256"] == old["rules_sha256"], new["label"]
        assert new["predicted_bytes"] == old["predicted_bytes"], new["label"]


def test_tied_head_takes_the_output_type(inv2b, smoke_ladder, profiles):
    profile = ladder_gen.load_profile(os.path.join(profiles, "dense-hybrid.yaml"))
    assert profile["tied_embeddings"] == "output"
    ladder, problems = ladder_gen.build(profile, inv2b)
    assert not problems, problems
    assert ladder["tied_embeddings"] == "output"
    rungs = {r["label"]: r for r in profile["rungs"]}
    for new, old in zip(ladder["rungs"], smoke_ladder["rungs"]):
        want = (rungs[new["label"]].get("types") or {}).get("output", "q8_0")
        assert new["_effective"]["token_embd.weight"] == want, new["label"]
        # nothing but the head moved
        changed = [(p, t) for p, t in new["rules"] if [p, t] not in old["rules"]]
        assert all(p == r"^token_embd\.weight$" for p, _ in changed), (new["label"], changed)
    q4 = next(r for r in ladder["rungs"] if r["label"] == "AD-Q4_K_M")
    assert q4["_effective"]["token_embd.weight"] == "q6_k"   # was iq4_xs in the smoke run


def test_tied_key_is_inert_with_an_output_tensor(inv2b, profiles):
    profile = ladder_gen.load_profile(os.path.join(profiles, "dense-hybrid.yaml"))
    inv = copy.deepcopy(inv2b)
    head = next(t for t in inv["tensors"] if t["name"] == "token_embd.weight")
    inv["tensors"].append(dict(head, name="output.weight"))
    a, _ = ladder_gen.build(profile, inv)
    b, _ = ladder_gen.build(untied(profile), inv)
    assert a["tied_embeddings"] is None
    assert [r["rules_sha256"] for r in a["rungs"]] == [r["rules_sha256"] for r in b["rungs"]]


def test_tied_key_must_name_a_role(inv2b, profiles):
    profile = ladder_gen.load_profile(os.path.join(profiles, "dense-hybrid.yaml"))
    profile["tied_embeddings"] = "lm_head"
    with pytest.raises(ladder_gen.LadderError, match="unknown role lm_head"):
        ladder_gen.build(profile, inv2b)


def test_smoke_results(local_fixture):
    with open(local_fixture("qwen3.5-2b-smoke", "results.json")) as f:
        rows = json.load(f)
    assert len(rows) == 16
    assert all(re.fullmatch(r"Qwen3\.5-2B-(Q8_0|AD-.+)", r["name"]) for r in rows)
    q = [r["quality"]["neutral"] for r in rows]
    assert {(x["chunks"], x["ctx"]) for x in q} == {(48, 4096)}
    assert all(r["llama_commit"].startswith("1692f9e50") for r in rows)
    # one ladder, one reference: KLD falls and top-1 rises as the file grows
    by_size = sorted(zip((r["size_bytes"] for r in rows), q))
    kld = [x["mean_kld"] for _, x in by_size]
    top1 = [x["top1_pct"] for _, x in by_size]
    assert kld == sorted(kld, reverse=True) and top1 == sorted(top1)
