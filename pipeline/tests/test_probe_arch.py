"""probe_arch.py rows from a saved inventory, no hub."""
import json

import pytest

import probe_arch

CFG = {"architectures": ["Qwen4ExpForConditionalGeneration"], "vision_config": {}, "tie_word_embeddings": False,
       "text_config": {"num_hidden_layers": 48, "layer_types": ["linear_attention"] * 36 + ["full_attention"] * 12,
                       "mtp_num_hidden_layers": 1, "num_experts": 512, "num_experts_per_tok": 10,
                       "hidden_size": 2560, "vocab_size": 248320}}
FP = {"revision": "de4b8e4d", "tokenizer_sha256": "8f6ca429", "template_sha256": "924d3fcc",
      "tokens": {"vocab_entries": 248044, "added": 33},
      "files": {"tokenizer.json": "0997f410c57a1f4e", "chat_template.jinja": "c3cf9e34abf4f9e3"}}


@pytest.fixture(scope="module")
def inv(local_fixture):
    with open(local_fixture("qwen3.8-flash-next", "inventory.json")) as f:
        return json.load(f)


def test_flash_next_row(inv):
    r = probe_arch.row("Qwen/Qwen3.8-Flash-Next", CFG, FP, inv)
    assert (r["arch"], r["layers"], r["full_attention"], r["mtp_layers"], r["mtp_in_gguf"]) == ("qwen4exp", 48, 12, 1, False)
    short = {x["ne0"]: x for x in r["rows"] if not x["div256"]}
    assert set(short) == {160, 640, 320, 4}
    assert short[640]["share"] == pytest.approx(0.2276, abs=1e-3)
    assert short[640]["tensors"] == ["ffn_down_exps", "ffn_down_shexp"]
    assert r["get_rows_share"]["per_layer_token_embd.weight"] == pytest.approx(0.2894, abs=1e-3)
    assert r["bf16_gb"] == pytest.approx(354.0, abs=0.1)
    assert r["tokenizer_entries"] == 248077


def test_markdown_row(inv):
    md = probe_arch.markdown([probe_arch.row("Qwen/Qwen3.8-Flash-Next", CFG, FP, inv),
                              probe_arch.row("Qwen/Qwen4-X", CFG, FP, None, "converter failed: not supported")])
    lines = md.splitlines()
    assert lines[2].startswith("| Qwen3.8-Flash-Next | qwen4exp | 48 (12) | 1 (dropped) | 512/10 | 160 (28.9%")
    assert "| 0997f410c57a | c3cf9e34abf4 | yes | 0.36% + 28.9% | 354.0 GB |" in lines[2]
    assert "converter failed" in lines[4]
    assert all(line.count("|") == lines[0].count("|") for line in lines)
