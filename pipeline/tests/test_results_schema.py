"""results.json rows: the pinned schema, parsed from real August logs."""
import argparse
import json

import pytest

import results

KEYS = {"name", "label", "publisher", "size_bytes", "size_gb", "bpw", "file_type", "llama_commit", "measured_on",
        "quality"}
QUALITY = {"mean_kld", "median_kld", "q99_kld", "q999_kld", "max_kld", "top1_pct", "mean_dp_pct", "rms_dp_pct",
           "ppl", "ppl_base", "ppl_ratio", "chunks", "ctx"}


def make_row(dense_log, label="AD-Q4_K", public="AD-Q4_K_M", size=17_123_000_000):
    a = argparse.Namespace(name=f"Qwen3.8-27B-{public}", label=public, kld=dense_log(f"kld-neutral--Qwen3.8-27B-{label}.log"),
                           quant_log=dense_log(f"quantize-{label}.log"), size_bytes=size, file_type=15,
                           commit="1692f9e50bb2", evalset="neutral")
    return results.row(a)


def test_row_has_the_pinned_schema(dense_log):
    r = make_row(dense_log)
    assert set(r) == KEYS
    q = r["quality"]["neutral"]
    assert QUALITY <= set(q)
    assert q["mean_kld"] == pytest.approx(0.011262, abs=1e-6)
    assert q["chunks"] == 87 and q["ctx"] == 4096
    assert r["size_gb"] == 17.12 and r["publisher"] == "atomicchat"
    assert 4.5 < r["bpw"] < 5.5
    json.dumps(r)


def test_merge_sorts_by_size_and_joins_evalsets(dense_log, tmp_path):
    a = make_row(dense_log)
    b = make_row(dense_log, "AD-IQ1_M", "AD-IQ1_M", 8_500_000_000)
    agentic = dict(a, quality={"agentic": {"mean_kld": 0.02}})
    paths = []
    for i, r in enumerate([a, b, agentic]):
        p = tmp_path / f"{i}.json"
        p.write_text(json.dumps(r))
        paths.append(str(p))
    merged = results.merge(paths)
    assert [r["label"] for r in merged] == ["AD-IQ1_M", "AD-Q4_K_M"]
    assert set(merged[1]["quality"]) == {"neutral", "agentic"}


def test_merge_refuses_a_row_without_size(dense_log, tmp_path):
    r = make_row(dense_log)
    r["size_bytes"] = None
    p = tmp_path / "r.json"
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="no size_bytes"):
        results.merge([str(p)])


def test_unfinished_measurement_is_an_error():
    with pytest.raises(ValueError, match="did not finish"):
        results.parse_kld("kl_divergence: computing over 87 chunks, n_ctx=4096\n[1]0.01,")
