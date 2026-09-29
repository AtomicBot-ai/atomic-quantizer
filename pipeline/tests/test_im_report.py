"""im_report.py on imatrix files written the way llama-imatrix writes them."""
import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

import im_report  # noqa: E402

ROW = 64


def write_imatrix(path, entries, chunks):
    """entries: {name: (values[n_mat, row], counts[n_mat])}, stored as imatrix.cpp save_imatrix does."""
    w = gguf.GGUFWriter(str(path), arch="imatrix")
    w.add_type("imatrix")
    w.add_uint32("imatrix.chunk_count", chunks)
    w.add_uint32("imatrix.chunk_size", 512)
    for name, (vals, counts) in entries.items():
        w.add_tensor(f"{name}.in_sum2", np.asarray(vals, dtype=np.float32))
        w.add_tensor(f"{name}.counts", np.asarray(counts, dtype=np.float32).reshape(-1, 1))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def model(rng, counts_by_block, scale=1.0, noise=0.0):
    """Two blocks: a dense attn_q and 8 experts of gate/up/down each."""
    base = rng.random((8, ROW)) + 0.1
    out = {}
    for b, counts in counts_by_block.items():
        c = np.asarray(counts, dtype=np.float64)
        m = base * (1 + noise * rng.standard_normal(base.shape)).clip(0.5)
        for t in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            out[f"blk.{b}.{t}.weight"] = (m * c[:, None] * scale, c)
        out[f"blk.{b}.attn_q.weight"] = ((base[0] * 1000 * (b + 1))[None, :], np.array([1000.0]))
    return out


def test_dead_and_weak_experts_are_named(tmp_path):
    rng = np.random.default_rng(0)
    counts = {0: [500, 0, 480, 510, 2, 505, 490, 500], 1: [500] * 8}
    path = write_imatrix(tmp_path / "im.gguf", model(rng, counts), 100)
    im, chunks = im_report.load(path)
    rep = im_report.coverage(im)
    assert chunks == 100
    assert rep["dead_by_block"] == {"0": [1]}
    assert rep["dead_experts"] == 3                 # gate, up and down of the same expert
    assert rep["weak_experts"] == 3 and all(t["weak"] in ([], [4]) for t in rep["tensors"])


def test_coverage_exit_code(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    path = write_imatrix(tmp_path / "im.gguf", model(rng, {0: [500, 0] + [500] * 6}), 100)
    monkeypatch.setattr("sys.argv", ["im_report.py", "coverage", path])
    assert im_report.main() == im_report.DEAD_EXIT
    path = write_imatrix(tmp_path / "ok.gguf", model(rng, {0: [500] * 8}), 100)
    monkeypatch.setattr("sys.argv", ["im_report.py", "coverage", path])
    assert im_report.main() == 0


def test_uncovered_weights_with_reasons():
    inv = {"tensors": [
        {"name": "per_layer_token_embd.weight", "type": "bf16", "shape": [160, 1000, 1, 1]},
        {"name": "output.weight", "type": "bf16", "shape": [64, 100, 1, 1]},
        {"name": "output_hc_up.weight", "type": "bf16", "shape": [32, 64, 1, 1]},
        {"name": "blk.0.attn_q.weight", "type": "bf16", "shape": [64, 64, 1, 1]},
        {"name": "blk.0.attn_norm.weight", "type": "f32", "shape": [64, 1, 1, 1]}]}
    im = {"blk.0.attn_q.weight": (np.ones((1, 64)), np.array([10.0]))}
    u = im_report.uncovered(im, inv)
    assert list(u["groups"]) == ["per_layer_token_embd.weight", "output.weight", "output_hc_up.weight"]
    assert "GET_ROWS" in u["groups"]["per_layer_token_embd.weight"]["why"]
    assert "--process-output" in u["groups"]["output.weight"]["why"]


def test_twice_the_tokens_is_not_movement(tmp_path):
    rng = np.random.default_rng(2)
    small = model(rng, {0: [500] * 8, 1: [300] * 8})
    big = {n: (v * 2, c * 2) for n, (v, c) in small.items()}
    rep = im_report.converge(small, big)
    assert rep["converged"] and rep["min"] == pytest.approx(1.0)


def test_routing_shift_alone_is_not_movement():
    # same per-expert statistics, the router sent the tokens elsewhere: the raw sums move, the means do not
    rng = np.random.default_rng(3)
    a = model(rng, {0: [100, 900, 500, 500, 500, 500, 500, 500]})
    b = {n: (v / c[:, None] * c2[:, None], c2) for n, (v, c) in a.items()
         for c2 in [np.array([900, 100, 500, 500, 500, 500, 500, 500], dtype=float) if c.size > 1 else c]}
    rep = im_report.converge(a, b)
    assert rep["converged"]
    assert min(r["cos_raw"] for r in rep["worst"]) < 0.995   # converge.py would have called this movement


def test_noisy_experts_fail_and_name_the_role(tmp_path):
    rng = np.random.default_rng(4)
    a = model(rng, {0: [500] * 8})
    b = model(np.random.default_rng(4), {0: [500] * 8}, noise=0.5)
    b["blk.0.attn_q.weight"] = a["blk.0.attn_q.weight"]
    rep = im_report.converge(a, b, 0.995)
    assert not rep["converged"]
    assert set(k for k, v in rep["role_min"].items() if v < 0.995) == {
        "blk.N.ffn_gate_exps.weight", "blk.N.ffn_up_exps.weight", "blk.N.ffn_down_exps.weight"}


def test_first_seen_experts_are_counted():
    rng = np.random.default_rng(5)
    a = model(rng, {0: [500, 0] + [500] * 6})
    b = model(np.random.default_rng(5), {0: [500] * 8})
    assert im_report.converge(a, b)["experts_new"] == 3


def test_stats_follow_show_statistics(tmp_path):
    rng = np.random.default_rng(6)
    im = model(rng, {0: [500, 0] + [500] * 6, 1: [400] * 8})
    rows = im_report.stats(im)
    down0 = next(r for r in rows if r["tensor"] == "blk.0.ffn_down_exps.weight")
    vals, c = im["blk.0.ffn_down_exps.weight"]
    assert down0["n"] == 7 * ROW                         # the dead expert is skipped, as in imatrix.cpp
    assert down0["sum"] == pytest.approx(float((vals[c > 0] / c[c > 0, None]).sum()))
    # llama-imatrix order: by short tensor name, then falling sum
    names = [im_report.short_names(r["tensor"])[1] for r in rows]
    assert names == sorted(names)
    assert [r["tensor"] for r in rows if "attn_q" in r["tensor"]] == ["blk.1.attn_q.weight", "blk.0.attn_q.weight"]


def test_band_select_reads_the_table(tmp_path):
    band_select = pytest.importorskip("band_select")   # the band selection lands in its own commit
    rng = np.random.default_rng(7)
    im = model(rng, {0: [500] * 8, 1: [500] * 8})
    text = im_report.stats_text(im_report.stats(im), "im.gguf")
    rows = band_select.parse_stats(text)
    assert {(r["block"], r["tensor"]) for r in rows} == {(b, t) for b in (0, 1) for t in (
        "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "attn_q")}
    ranking = band_select.rank_blocks(rows, r"attn_q", [0, 1])
    assert [b for b, _, _ in ranking] == [1, 0]


def test_shards_add_up_to_the_whole(tmp_path):
    rng = np.random.default_rng(8)
    a = model(rng, {0: [100, 0, 300, 50, 200, 100, 100, 100]})
    b = model(np.random.default_rng(9), {0: [50, 20, 100, 50, 200, 100, 0, 100]})
    pa = write_imatrix(tmp_path / "s0.gguf", a, 10)
    pb = write_imatrix(tmp_path / "s1.gguf", b, 12)
    total, chunks = im_report.load_sum(f"{pa},{pb}")
    assert chunks == 22
    v, c = total["blk.0.ffn_down_exps.weight"]
    assert np.allclose(c, a["blk.0.ffn_down_exps.weight"][1] + b["blk.0.ffn_down_exps.weight"][1])
    assert np.allclose(v, a["blk.0.ffn_down_exps.weight"][0] + b["blk.0.ffn_down_exps.weight"][0], rtol=1e-6)
    assert im_report.coverage(total)["dead_experts"] == 0   # each shard has a hole, the sum has none
