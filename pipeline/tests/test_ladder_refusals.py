"""The generator must refuse every ladder llama-quantize would quietly change."""
import os

import pytest

import ladder_gen


def tensor(name, ne0, ne1=1, ne2=1, typ="bf16"):
    return {"name": name, "type": typ, "shape": [ne0, ne1, ne2, 1]}


def inventory(tensors, blocks):
    return {"source": "synthetic", "arch": "test", "block_count": blocks, "tensors": tensors,
            "mtp_blocks": sorted({ladder_gen.block_of(t["name"]) for t in tensors if ".nextn." in t["name"]})}


def dense_block(b, ff=1024):
    return [tensor(f"blk.{b}.ffn_down.weight", ff, 512), tensor(f"blk.{b}.ffn_gate.weight", 512, ff),
            tensor(f"blk.{b}.ffn_up.weight", 512, ff), tensor(f"blk.{b}.attn_qkv.weight", 512, 1024),
            tensor(f"blk.{b}.attn_norm.weight", 512, typ="f32")]


def moe_block(b, ff=512):
    return [tensor(f"blk.{b}.ffn_down_exps.weight", ff, 512, 8), tensor(f"blk.{b}.ffn_gate_exps.weight", 512, ff, 8),
            tensor(f"blk.{b}.ffn_up_exps.weight", 512, ff, 8), tensor(f"blk.{b}.ffn_down_shexp.weight", ff, 512),
            tensor(f"blk.{b}.ffn_gate_inp.weight", 512, 8, typ="f32"), tensor(f"blk.{b}.attn_q.weight", 512, 512)]


HEAD = [tensor("token_embd.weight", 512, 1000), tensor("output.weight", 512, 1000)]


def profile(**over):
    p = {"name": "t", "ftype": "Q8_0", "roles": [
        {"name": "ffn", "pattern": r"^blk\.\d+\.ffn_(down|gate|up)\.weight$"},
        {"name": "exps", "pattern": r"^blk\.\d+\.ffn_(down|gate|up)_exps\.weight$"},
        {"name": "rest", "pattern": r"^blk\.\d+\.(attn_\w+|ffn_\w+_shexp)\.weight$"},
        {"name": "head", "pattern": r"^(token_embd|output)\.weight$"}],
        "rungs": [{"label": "R", "types": {"ffn": "q4_k", "exps": "q4_k", "rest": "q8_0", "head": "q6_k"}}]}
    p.update(over)
    return p


def problems(p, inv):
    return ladder_gen.build(p, inv)[1].get("R", [])


def test_clean_ladder_passes():
    inv = inventory(HEAD + [t for b in range(4) for t in moe_block(b)], 4)
    assert problems(profile(), inv) == []


def test_dense_ladder_on_moe_model_is_refused():
    # the trap: dense rules miss every expert, the file would be a stock Q8_0 labelled AD
    p = profile(rungs=[{"label": "R", "types": {"ffn": "q4_k", "rest": "q8_0", "head": "q6_k"}}])
    inv = inventory(HEAD + [t for b in range(4) for t in moe_block(b)], 4)
    errs = problems(p, inv)
    assert errs and all("_exps" in e and "silently" in e for e in errs)


def test_row_not_multiple_of_256_is_refused():
    inv = inventory(HEAD + [t for b in range(2) for t in moe_block(b, ff=640)], 2)
    errs = problems(profile(), inv)
    assert any("ffn_down_exps" in e and "not a multiple of 256" in e for e in errs)


def test_block32_type_is_accepted_on_short_rows():
    p = profile(roles=profile()["roles"][:1] + [
        {"name": "down_exps", "pattern": r"^blk\.\d+\.ffn_down_exps\.weight$"},
        {"name": "exps", "pattern": r"^blk\.\d+\.ffn_(gate|up)_exps\.weight$"}] + profile()["roles"][2:],
        rungs=[{"label": "R", "types": {"down_exps": "iq4_nl", "exps": "q4_k", "rest": "q8_0", "head": "q6_k"}}])
    inv = inventory(HEAD + [t for b in range(2) for t in moe_block(b, ff=640)], 2)
    assert problems(p, inv) == []


def test_low_bit_on_mtp_block_is_refused_without_pin():
    blocks = [t for b in range(3) for t in dense_block(b)]
    mtp = dense_block(3) + [tensor("blk.3.nextn.eh_proj.weight", 1024, 512)]
    p = profile(rungs=[{"label": "R", "types": {"ffn": "iq2_xxs", "rest": "q8_0", "head": "q6_k"}}],
                allow_fallback=[r"nextn"])
    errs = problems(p, inventory(HEAD + blocks + mtp, 4))
    assert any(e.startswith("blk.3.ffn_down.weight") and "imatrix" in e for e in errs)
    # with the pin the same ladder is fine
    assert problems(dict(p, mtp={"type": "q5_k"}), inventory(HEAD + blocks + mtp, 4)) == []


def test_uncovered_tensor_under_mixture_ftype_is_refused():
    p = profile(rungs=[{"label": "R", "ftype": "Q4_K_M", "types": {"ffn": "q4_k", "exps": "q4_k", "head": "q6_k"}}])
    inv = inventory(HEAD + [t for b in range(2) for t in moe_block(b)], 2)
    errs = problems(p, inv)
    assert any("attn_q" in e and "per tensor logic" in e for e in errs)


def test_flag_that_overrides_a_rule_is_refused():
    p = profile(flags={"token_embedding_type": "q8_0"})
    errs = problems(p, inventory(HEAD + [t for b in range(2) for t in moe_block(b)], 2))
    assert any(e.startswith("token_embd.weight") and "flag forces" in e for e in errs)


def test_token_embedding_flag_also_hits_ngram_table():
    ngram = tensor("per_layer_token_embd.weight", 256, 5000)
    p = profile(flags={"token_embedding_type": "q8_0"},
                rungs=[{"label": "R", "types": {"ffn": "q4_k", "exps": "q4_k", "rest": "q8_0"}}],
                allow_fallback=[r"^output\.weight$"])
    ladder, errs = ladder_gen.build(p, inventory(HEAD + [ngram] + [t for b in range(2) for t in moe_block(b)], 2))
    assert ladder["rungs"][0]["_effective"]["per_layer_token_embd.weight"] == "q8_0"


def test_ftype_mix_in_rule_is_rejected():
    with pytest.raises(ValueError, match="not a ggml type"):
        ladder_gen.norm_type("iq2_m")


def test_anchored_roles_do_not_collide(profiles):
    # dense-hybrid rules must never reach MoE or router tensors
    p = ladder_gen.load_profile(os.path.join(profiles, "dense-hybrid.yaml"))
    names = ["blk.1.ffn_down_exps.weight", "blk.1.ffn_down_shexp.weight", "blk.1.ffn_gate_inp.weight",
             "blk.1.attn_q_norm.weight", "blk.1.attn_qkv.weight", "blk.1.attn_output.weight"]
    rules = [(r["pattern"].replace("{edge}", r"(1)").replace("{mid}", r"(2)"), r["name"]) for r in p["roles"]]
    import re
    hits = {n: [role for pat, role in rules if re.search(pat, n)] for n in names}
    assert hits == {"blk.1.ffn_down_exps.weight": [], "blk.1.ffn_down_shexp.weight": [], "blk.1.ffn_gate_inp.weight": [],
                    "blk.1.attn_q_norm.weight": [], "blk.1.attn_qkv.weight": ["attn_q"],
                    "blk.1.attn_output.weight": ["attn_output"]}
