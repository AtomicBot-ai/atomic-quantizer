"""The fingerprint tells a reusable build from one that needs re-rendering or rebuilding.

Local directories stand in for hub repositories, so this runs offline.
"""
import json

import tok_fingerprint as tf

TOK = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": ["a b"]},
       "added_tokens": [{"id": 2, "content": "]~b]", "special": True}],
       "normalizer": None, "pre_tokenizer": None, "post_processor": None, "decoder": None}


def model_dir(tmp_path, name, tok=TOK, template="{{ messages }}", indent=None):
    d = tmp_path / name
    d.mkdir()
    (d / "tokenizer.json").write_text(json.dumps(tok, indent=indent))
    (d / "tokenizer_config.json").write_text(json.dumps({"bos_token": "]~b]"}))
    (d / "chat_template.jinja").write_text(template)
    return str(d)


def fp(path):
    return tf.fingerprint(tf.read_files(path))


def test_reformatted_file_is_the_same_tokenizer(tmp_path):
    old = fp(model_dir(tmp_path, "old"))
    new = fp(model_dir(tmp_path, "new", indent=2))
    assert old["files"]["tokenizer.json"] != new["files"]["tokenizer.json"]
    verdict, diffs = tf.compare(old, new)
    assert verdict.startswith("SAME TOKENIZER AND TEMPLATE") and not diffs


def test_new_template_keeps_the_sweep(tmp_path):
    old = fp(model_dir(tmp_path, "old"))
    new = fp(model_dir(tmp_path, "new", template="{{ messages | tojson }}"))
    assert tf.compare(old, new)[0].startswith("SAME TOKENIZER, NEW TEMPLATE")


def test_new_special_token_is_a_new_tokenizer(tmp_path):
    tok = json.loads(json.dumps(TOK))
    tok["added_tokens"].append({"id": 3, "content": "<mm:think>", "special": True})
    old = fp(model_dir(tmp_path, "old"))
    verdict, diffs = tf.compare(old, fp(model_dir(tmp_path, "new", tok=tok)))
    assert verdict.startswith("NEW TOKENIZER (added tokens differ)")
    assert "added: 1 -> 2" in diffs
