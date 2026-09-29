"""scan: groups from the inventory, rule files ladder_gen accepts, the run loop, report and bands."""
import json
import os
import threading

import pytest

import gguf_inventory
import ladder_gen
import scan
from synth import write_kld_log


@pytest.fixture(scope="module")
def inv4b(local_fixture):
    return gguf_inventory.from_log(local_fixture("qwen3.5-4b-masks", "quantize-E-Q5_K_S.log"))


def tensor(name, ne0, ne1=1, ne2=1, typ="bf16"):
    return {"name": name, "type": typ, "shape": [ne0, ne1, ne2, 1]}


def moe_inventory(blocks=8, ff=640):
    """A Flash-Next like MoE: expert rows of 640 (not a multiple of 256), shared expert, a PLE table."""
    ts = [tensor("token_embd.weight", 1024, 4000), tensor("output.weight", 1024, 4000),
          tensor("per_layer_token_embd.weight", 2048, 4000)]
    for b in range(blocks):
        ts += [tensor(f"blk.{b}.ffn_down_exps.weight", ff, 1024, 16), tensor(f"blk.{b}.ffn_gate_exps.weight", 1024, ff, 16),
               tensor(f"blk.{b}.ffn_up_exps.weight", 1024, ff, 16), tensor(f"blk.{b}.ffn_down_shexp.weight", ff, 1024),
               tensor(f"blk.{b}.ffn_gate_shexp.weight", 1024, ff), tensor(f"blk.{b}.ffn_up_shexp.weight", 1024, ff),
               tensor(f"blk.{b}.attn_q.weight", 1024, 1024), tensor(f"blk.{b}.ffn_gate_inp.weight", 1024, 16, typ="f32"),
               tensor(f"blk.{b}.attn_norm.weight", 1024, typ="f32")]
    return {"source": "synthetic", "arch": "test", "block_count": blocks, "mtp_blocks": [], "tensors": ts}


def test_4b_groups(inv4b):
    groups, skipped, parts = scan.make_groups(inv4b)
    names = [g["name"] for g in groups]
    assert parts == [list(range(0, 8)), list(range(8, 16)), list(range(16, 24)), list(range(24, 32))]
    # the big roles by quarter, the rest whole, ssm_alpha/beta too small to measure
    assert sorted(names) == sorted([f"{k}-Q{q}" for k in ("ffn_gate_up", "ffn_down", "attn_qkv") for q in (1, 2, 3, 4)]
                                   + ["token_embd", "attn_gate", "ssm_out", "attn_q", "attn_output", "attn_kv"])
    assert [s["kind"] for s in skipped] == ["ssm_ab"]
    assert all(32 not in g["blocks"] for g in groups)          # MTP stays out
    qkv = next(g for g in groups if g["name"] == "attn_qkv-Q1")
    assert qkv["blocks"] == [0, 1, 2, 4, 5, 6]                  # DeltaNet blocks only; 3 and 7 are full attention
    assert sum(g["share"] for g in groups) + sum(s["share"] for s in skipped) == pytest.approx(1.0, abs=1e-4)


def test_4b_plan(inv4b, tmp_path):
    doc = scan.plan(inv4b, str(tmp_path))
    assert len(doc["groups"]) == 19 and doc["groups"][0]["name"] == "base"
    assert (tmp_path / "types" / "base.types").read_text() == "^blk\\.32\\.=q8_0\n.=q8_0\n"
    fd = (tmp_path / "types" / "ffn_down-Q4.types").read_text().splitlines()
    assert fd == ["^blk\\.32\\.=q8_0", r"^blk\.(24|25|26|27|28|29|30|31)\.(ffn_down)\.weight$=q5_k", ".=q8_0"]
    for g in doc["groups"][1:]:
        assert g["target"] == ["q5_k"] and g["predicted_saved"] > 0
    emb = next(g for g in doc["groups"] if g["name"] == "token_embd")
    assert emb["predicted_saved"] == 248320 * 2560 // 32 * 34 - 248320 * 2560 // 256 * 176


@pytest.mark.parametrize("base, down, row, want", [
    ("q8_0", 2, 2560, "q5_k"), ("q8_0", 6, 2560, "iq3_xxs"), ("q6_k", 2, 2560, "q4_k"),
    ("q8_0", 2, 640, "q4_0"), ("q8_0", 1, 640, "q5_0"), ("q6_k", 2, 640, "q4_0"), ("iq1_m", 3, 2560, "iq1_m"),
])
def test_step_down(base, down, row, want):
    assert scan.step_down(base, down, row) == want


def test_moe_plan_walks_legacy_steps_on_640_rows(tmp_path):
    inv = moe_inventory()
    groups, _, _ = scan.make_groups(inv)
    names = {g["name"] for g in groups}
    assert {"exps_gate_up-Q1", "exps_down-Q4", "shexp", "per_layer_token_embd", "attn_q"} <= names
    doc = scan.plan(inv, str(tmp_path), base="q6_k")
    exps = next(g for g in doc["groups"] if g["name"] == "exps_down-Q1")
    assert exps["target"] == ["q4_0"]
    q = next(g for g in doc["groups"] if g["name"] == "attn_q")
    assert q["target"] == ["q4_k"]
    # at a k base the 640 rows sit at the legacy neighbour, never at a type that would fall back
    base = ladder_gen.build(scan.group_profile(inv, "q6_k", 2), inv)[0]["rungs"][0]["_effective"]
    assert base["blk.0.ffn_down_exps.weight"] == "q5_0" and base["blk.0.attn_q.weight"] == "q6_k"


def fake_bench(monkeypatch, d, kld_of):
    """quantize writes a file, measure writes a synthetic log; records the order and the files on disk."""
    events, lock = [], threading.Lock()

    def quantize(d_, g, inv, p, cfg):
        path = os.path.join(cfg["work"], f"scan-{g['name']}.gguf")
        with open(path, "wb") as f:
            f.write(b"x")
        with lock:
            events.append(("q", g["name"], len(os.listdir(cfg["work"]))))
        return path, 3_000_000_000 - g.get("predicted_saved", 0), 0.0

    def measure(d_, g, path, cfg):
        with lock:
            events.append(("k", g["name"], len(os.listdir(cfg["work"]))))
        write_kld_log(os.path.join(d_, "logs", f"kld-{g['name']}.log"), kld_of(g["name"]))
        return 0.0

    monkeypatch.setattr(scan, "quantize", quantize)
    monkeypatch.setattr(scan, "measure", measure)
    return events


def cfg(tmp_path):
    return {"bin": "/x", "bf16": "m.gguf", "imatrix": "i.gguf", "eval": "e.txt", "ref": "r.kld", "chunks": 6, "ctx": 4096,
            "ngl": 99, "threads": 8, "work": str(tmp_path / "gguf"), "keep": False}


def kld_of(name):
    """Base 0.010; the tail quarters of ffn hurt most, per byte too."""
    bump = {"ffn_down-Q4": 0.004, "ffn_gate_up-Q4": 0.006, "ffn_down-Q1": 0.002, "ffn_gate_up-Q1": 0.003,
            "ffn_down-Q2": 0.0005, "ffn_gate_up-Q2": 0.0008, "ffn_down-Q3": 0.001, "ffn_gate_up-Q3": 0.0015}
    d = bump.get(name, 0.0001) if name != "base" else 0.0
    return [0.010 + d + 0.0003 * ((i * 7) % 5) for i in range(6)]


def test_run_report_bands(inv4b, tmp_path, monkeypatch, profiles):
    d = str(tmp_path / "scan")
    scan.plan(inv4b, d)
    events = fake_bench(monkeypatch, d, kld_of)
    scan.run(d, cfg(tmp_path), say=lambda s: None)
    qs = [e for e in events if e[0] == "q"]
    assert len(qs) == 19 and qs[0][1] == "base"
    assert max(n for _, _, n in events) <= 2            # never more than two GGUFs on disk
    assert os.listdir(tmp_path / "gguf") == []          # all deleted
    # resumable: nothing left to do
    events.clear()
    scan.run(d, cfg(tmp_path), say=lambda s: None)
    assert events == []

    doc = scan.report(d)
    assert json.load(open(os.path.join(d, "scan.json")))["base"]["kld"] == doc["base"]["kld"]
    assert len(doc["groups"]) == 18 and not doc["missing"]
    top = doc["groups"][0]
    assert top["group"] in ("ffn_down-Q4", "ffn_gate_up-Q4") and top["per_gb"] > 0
    assert top["d_kld"] == pytest.approx(kld_of(top["group"])[0] - 0.010, abs=2e-5)

    dense = ladder_gen.load_profile(os.path.join(profiles, "dense-hybrid.yaml"))
    eligible = ladder_gen.compute_bands(dense, inv4b)["eligible"]
    b = scan.scan_bands(doc, eligible, 8, 4)
    assert [q["quarter"] for q in b["quarters"]] == [4, 1, 3, 2]
    assert b["edge"] == list(range(24, 32)) and b["mid"] == [0, 1, 2, 3]   # Q1, nearest the end first
    b12 = scan.scan_bands(doc, eligible, 12, 4)
    assert b12["edge"] == [0, 1, 2, 3] + list(range(24, 32)) and b12["mid"] == [4, 5, 6, 7]
    ladder, problems = ladder_gen.build(dense, inv4b, bands_override=b12, only=["AD-Q4_K_M"])
    assert not problems and ladder["bands"]["edge"] == b12["edge"]


def test_parts_split_the_groups(inv4b, tmp_path, monkeypatch):
    d = str(tmp_path / "scan")
    scan.plan(inv4b, d)
    events = fake_bench(monkeypatch, d, kld_of)
    scan.run(d, cfg(tmp_path), part=(1, 3), say=lambda s: None)
    got = [n for k, n, _ in events if k == "q"]
    assert "base" not in got and len(got) == 6
    scan.run(d, cfg(tmp_path), part=(0, 3), say=lambda s: None)
    scan.run(d, cfg(tmp_path), part=(2, 3), say=lambda s: None)
    assert sorted(n for k, n, _ in events if k == "q") == sorted(g["name"] for g in scan.load_plan(d)[0]["groups"])


def test_a_changed_rule_file_is_measured_again(inv4b, tmp_path, monkeypatch):
    d = str(tmp_path / "scan")
    scan.plan(inv4b, d)
    fake_bench(monkeypatch, d, kld_of)
    scan.run(d, cfg(tmp_path), say=lambda s: None)
    scan.plan(inv4b, d, down=3)                      # q4_k now: every group but the base changes
    events = fake_bench(monkeypatch, d, kld_of)
    scan.run(d, cfg(tmp_path), say=lambda s: None)
    assert [n for k, n, _ in events if k == "q"][0] != "base" and len([e for e in events if e[0] == "q"]) == 18


def test_report_needs_the_base(inv4b, tmp_path):
    d = str(tmp_path / "scan")
    scan.plan(inv4b, d)
    with pytest.raises(scan.ScanError, match="base is not measured"):
        scan.report(d)
