"""The card draft renders from what the pipeline publishes, and says what the August card says.

The rows are the published Qwen3.8-27B table (sizes, KLD, top-1); inventory,
ladder, KLD manifest, reference log and corpus manifest are the real files.
"""
import json
import os
import shutil
import sys

from conftest import DENSE, fetch

import gguf_inventory
import ladder_gen

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import make_card  # noqa: E402

CALIB = ("AtomicChat/calib-corpora", "814d662f6c94d207fc8f38545a1b4abea11484b5")
PUBLISHED = [  # label, GB, mean KLD, top-1, from the August card
    ("Q8_0", 28.9, 0.00064, 98.92), ("AD-Q6_K", 25.0, 0.00107, 98.67), ("AD-Q6_K-Q5_K", 23.1, 0.00252, 97.94),
    ("AD-Q5_K_M", 20.2, 0.00419, 97.34), ("AD-Q5_K_M-Q4_K_M", 18.6, 0.00730, 96.43),
    ("AD-Q4_K_M", 17.1, 0.01126, 95.59), ("AD-IQ4_XS", 16.5, 0.01248, 95.39),
    ("AD-IQ4_XS-IQ3_S", 14.4, 0.02660, 93.15), ("AD-IQ3_S", 13.8, 0.03247, 92.41),
    ("AD-IQ3_S-IQ3_XXS", 13.0, 0.04337, 91.33), ("AD-IQ3_XXS", 12.1, 0.06972, 89.13),
    ("AD-IQ2_S", 11.1, 0.09832, 87.18), ("AD-IQ2_S-IQ2_XS", 10.2, 0.13807, 84.77),
    ("AD-IQ2_XS", 9.9, 0.16170, 83.48), ("AD-IQ2_XXS", 9.0, 0.25663, 79.44), ("AD-IQ1_M", 8.5, 0.34212, 76.34),
]


class Args:
    model = "Qwen/Qwen3.8-27B"
    stem = "Qwen3.8-27B"
    main = "AtomicChat/Qwen3.8-27B-GGUF"
    metrics = "AtomicChat/Qwen3.8-27B-GGUF-metrics"
    recommended = None
    evalset = "neutral"
    main_files = ["Qwen3.8-27B-AD-Q4_K_M.gguf", "mmproj-Qwen3.8-27B-F16.gguf", "mmproj-Qwen3.8-27B-BF16.gguf"]


def fetched(tmp_path, dense_log, profiles, rows=PUBLISHED):
    d = tmp_path / "metrics"
    (d / "logs").mkdir(parents=True)
    (d / "kld").mkdir()
    (d / "imatrix").mkdir()
    inv = gguf_inventory.from_log(dense_log("quantize-Q8_0.log"))
    (d / "inventory.json").write_text(json.dumps(inv))
    ladder, problems = ladder_gen.build(ladder_gen.load_profile(os.path.join(profiles, "dense-hybrid.yaml")), inv)
    assert not problems
    ladder_gen.write(ladder, str(d / "ladder"))
    res = [{"name": f"Qwen3.8-27B-{lab}", "label": lab, "size_bytes": int(g * 1e9), "size_gb": g, "bpw": None,
            "quality": {"neutral": {"mean_kld": k, "top1_pct": t, "chunks": 87, "ctx": 4096}}} for lab, g, k, t in rows]
    (d / "results.json").write_text(json.dumps(res))
    shutil.copy(fetch(*DENSE, "kld/base-neutral.manifest.txt"), d / "kld" / "base-neutral.manifest.txt")
    shutil.copy(fetch(*DENSE, "logs/base-neutral.log"), d / "logs" / "base-neutral.log")
    shutil.copy(fetch(*DENSE, "logs/env.txt"), d / "logs" / "env-node_base.txt")
    shutil.copy(fetch(*CALIB, "builds/qwen3.8-27b/manifest.json"), d / "imatrix" / "corpus-manifest.json")
    (d / "imatrix" / "params.txt").write_text("recipe qwen3.8-27b\nctx 512\nbatch 4096\nshards 7\nper_shard 1400\n")
    return str(d)


def render(tmp_path, dense_log, profiles, **kw):
    a = Args()
    a.dir = fetched(tmp_path, dense_log, profiles)
    for k, v in kw.items():
        setattr(a, k, v)
    card = make_card.Card(a)
    return card, card.render()


def test_card_matches_the_published_facts(tmp_path, dense_log, profiles):
    card, text = render(tmp_path, dense_log, profiles)
    assert text.startswith("---\nbase_model: Qwen/Qwen3.8-27B\n")
    assert "pipeline_tag: image-text-to-text" in text and "license: apache-2.0" in text
    for lab, g, k, t in PUBLISHED:
        assert f"| `{lab}` | {g:.1f} GB | {k:.5f} | {t:.2f}% |" in text
    assert text.index("`Q8_0`") < text.index("`AD-IQ1_M`")   # largest first
    assert ("64 layers, hidden size 5120, feed-forward size 17408, vocabulary 248,320, context 262,144. "
            "One multi token prediction head.") in text
    assert "Perplexity on the held-out set is 4.5234 plus or minus 0.02381." in text
    assert "Hardware: 4x RTX 5090." in text
    assert "87 chunks at 4096 context." in text
    assert "--kl-divergence-base base-neutral.kld --kl-divergence -c 4096 -ngl 99\n" in text
    assert "recipe `qwen3.8-27b`. It is 4,967,044 tokens across 3,004 documents" in text
    assert "| agentic and tool use | 24.7% |" in text
    assert "--spec-type draft-mtp" in text
    assert "mmproj-Qwen3.8-27B-F16.gguf` for most people" in text


def test_card_reproduces_the_recommended_rung(tmp_path, dense_log, profiles):
    card, text = render(tmp_path, dense_log, profiles)
    assert card.rec == "AD-Q4_K_M"
    assert "--tensor-type '^blk\\.64\\.=q5_k'" in text
    assert "--tensor-type '^token_embd\\.weight$=iq4_xs'" in text
    assert "--override-kv general.file_type=int:15" in text
    assert "Qwen3.8-27B-BF16.gguf Qwen3.8-27B-AD-Q4_K_M.gguf Q8_0" in text
    assert "split across 7 workers" in text and "-b 4096 -ub 4096" in text


def test_card_kv_cache_counts_only_attention_layers(tmp_path, dense_log, profiles):
    """16 of the 64 blocks keep a KV cache (full_attention_interval 4), 4 KV heads of 256, f16."""
    card, text = render(tmp_path, dense_log, profiles)
    assert card.kv_bytes_per_token() == 16 * 2 * 4 * 256 * 2
    assert "keeps 64 KB of" in text
    fits = text[text.index("## Which one fits"):text.index("## Running it")]
    rows = [ln for ln in fits.splitlines() if ln.startswith("| ") and "GB" in ln]
    assert rows and rows[-1].endswith("| full context |") and "`Q8_0`" in rows[-1]
    sizes = [next(g for lab, g, _, _ in PUBLISHED if f"`{lab}`" in ln) for ln in rows]
    assert sizes == sorted(sizes)


def test_card_lists_what_needs_a_person(tmp_path, dense_log, profiles):
    card, text = render(tmp_path, dense_log, profiles)
    assert "<!-- TODO(editorial): " in text
    assert any("other publishers" in t for t in card.todo)
    assert all(f"TODO(editorial): {t}" in text for t in card.todo)
