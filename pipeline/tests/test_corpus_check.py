"""corpus_check.py on a toy tokenizer whose template writes Qwen-shaped markup."""
import json

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

import corpus_check  # noqa: E402

SPECIALS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
            "<tool_response>", "</tool_response>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>",
            "<|fim_prefix|>"]

# the shape of the Qwen3.8 template, cut down: tools, parallel calls, results, reasoning, thinking off, images
TEMPLATE = """{%- for m in messages -%}
{%- if m.role == 'tool' -%}<|im_start|>user
<tool_response>
{{ m.content }}
</tool_response><|im_end|>
{% else -%}<|im_start|>{{ m.role }}
{%- if m.role == 'assistant' %}
<think>
{{ m.reasoning_content or '' }}
</think>

{% else %}
{% endif -%}
{%- if m.content is string %}{{ m.content }}{% else %}{% for c in m.content %}{% if c.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% elif c.type == 'text' %}{{ c.text }}{% endif %}{% endfor %}{% endif -%}
{%- for tc in m.tool_calls or [] %}<tool_call>
<function={{ tc.function.name }}>
</function>
</tool_call>{% endfor -%}
<|im_end|>
{% endif -%}
{%- endfor -%}
{%- if add_generation_prompt %}<|im_start|>assistant
{% if enable_thinking is defined and enable_thinking is false %}<think>

</think>

{% else %}<think>
{% endif %}{% endif %}"""


def toy_tokenizer(path, template=TEMPLATE, specials=SPECIALS):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["[UNK]"] + [chr(c) for c in range(32, 127)] + ["\n"]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok.add_special_tokens(specials)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
    fast.chat_template = template
    fast.save_pretrained(str(path))
    (path / "chat_template.jinja").write_text(template)
    return str(path)


@pytest.fixture
def toy(tmp_path):
    return toy_tokenizer(tmp_path / "tok")


def test_template_reports_every_construct(toy):
    rep = corpus_check.check_template(toy)
    assert rep["fails"] == []
    assert "</think>" in rep["markers"] and "<tool_call>" in rep["markers"]
    assert "</function>" in rep["plain_markers"]
    notes = " ".join(rep["notes"])
    assert "KEPT" in notes
    assert "vision markers: <|image_pad|>, <|vision_end|>, <|vision_start|>" in notes
    assert "<think>\\n\\n</think>\\n\\n'" in notes


def test_template_that_drops_a_parallel_call_fails(tmp_path):
    first_only = TEMPLATE.replace("{%- for tc in m.tool_calls or [] %}", "{%- for tc in (m.tool_calls or [])[:1] %}")
    rep = corpus_check.check_template(toy_tokenizer(tmp_path / "tok", first_only))
    assert any("parallel tool calls" in f for f in rep["fails"])


def test_pipe_marker_outside_the_vocabulary_fails(tmp_path):
    rep = corpus_check.check_template(toy_tokenizer(tmp_path / "tok", TEMPLATE.replace("<|im_end|>", "<|eot|>")))
    assert any("<|eot|>" in f and "not a token" in f for f in rep["fails"])


def test_specials_count_only_template_markers(toy, tmp_path):
    doc = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nok<|im_end|>\n"
    corpus = tmp_path / "calib.txt"
    corpus.write_text("\n\n".join([doc] * 120) + "\n\n<|fim_prefix|>")
    rep = corpus_check.check_specials(str(corpus), toy)
    counts = {r["marker"]: r["count"] for r in rep["markers"]}
    assert "<|fim_prefix|>" not in counts                  # an added token the chat template never writes
    assert counts["<|im_start|>"] == 240 and counts["<think>"] == 120
    assert {f.split()[0] for f in rep["fails"]} == {
        "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
        "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"}


def test_coverage_over_reachable_ids(toy, tmp_path):
    corpus = tmp_path / "calib.txt"
    corpus.write_text("".join(chr(c) for c in range(32, 127)) + "\n" + "".join(SPECIALS) + "[UNK]")  # unk is an added token here
    rep = corpus_check.check_coverage(str(corpus), toy, 0.999)
    assert rep["fails"] == [] and rep["share"] == 1.0
    corpus.write_text("abc")
    rep = corpus_check.check_coverage(str(corpus), toy, 0.999)
    assert rep["covered"] == 3 and rep["fails"]


def test_cli_exit_code(toy, tmp_path, monkeypatch):
    corpus = tmp_path / "calib.txt"
    corpus.write_text("hello")
    out = tmp_path / "rep.json"
    monkeypatch.setattr("sys.argv", ["corpus_check.py", "specials", str(corpus), "--tokenizer", toy,
                                     "--json", str(out)])
    assert corpus_check.main() == 1
    assert json.loads(out.read_text())["fails"]
