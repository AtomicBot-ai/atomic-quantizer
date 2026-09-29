"""Chat rendering for any model whose template transformers can render.

Same contract as ``nemotron_fmt``: ``render()`` turns a pool dialogue record into
the exact byte string the model sees at inference.

This is ``nemotron_fmt`` with the model path unpinned. That module was already
not a hand port: it drives the model's own ``chat_template.jinja`` through
``transformers.apply_chat_template``, which is the reference implementation, so
byte equality is true by construction rather than by assertion. Nothing in it
was specific to Nemotron except the name of one environment variable, so this
file generalises that and removes the need for a new renderer per model.

Point it at the original weights::

    export FOUNDRY_MODEL_DIR=/src

Reproducibility. If the template calls ``strftime_now``, an unpinned build
depends on the day it ran, and two builds a week apart differ. Set::

    export FOUNDRY_PIN_DATE=2026-08-14

and that call is frozen. If the template needs a date and none is pinned, this
module refuses to render rather than quietly emit a corpus nobody can rebuild.

Register it once in ``build.py``::

    FORMATTERS = {
        "glimmer":  "glimmer_fmt",
        "nemotron": "nemotron_fmt",
        "auto":     "auto_fmt",
    }

and then every new model is ``chat.format: auto`` in its recipe, with no new
code at all.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

# Kept for interface parity with the other renderers.
DEFAULT_REASONING = "high"
DEFAULT_KNOWLEDGE_CUTOFF = None

MODEL_DIR = os.environ.get("FOUNDRY_MODEL_DIR", "/src")
PIN_DATE = os.environ.get("FOUNDRY_PIN_DATE")

# Reasoning strengths in the pool that mean "do not emit a thinking channel".
_NO_THINK = {"none", "off", "no", "disabled", "low0"}

# Keys a chat template may read. Anything else in a pool record is routing
# metadata for a different model family and would confuse the template.
_KEEP = {"role", "content", "reasoning_content", "tool_calls",
         "tool_call_id", "name"}

_tok = None
_date_checked = False


def _template_text(tok) -> str:
    """The template as a string, whichever shape transformers hands back."""
    tpl = tok.chat_template
    if isinstance(tpl, list):
        tpl = tpl[0].get("template", "") if tpl else ""
    elif isinstance(tpl, dict):
        tpl = tpl.get("default") or next(iter(tpl.values()), "")
    return tpl or ""


def _pin_the_date(date_str: str) -> bool:
    """Freeze strftime_now so the build does not depend on today's date."""
    import datetime as _dt
    target = _dt.datetime.strptime(date_str, "%Y-%m-%d")
    try:
        from transformers.utils import chat_template_utils as ctu
        ctu.strftime_now = lambda fmt: target.strftime(fmt)
        return True
    except Exception:
        return False


def _tokenizer():
    global _tok, _date_checked
    if _tok is None:
        from transformers import AutoTokenizer  # heavy; imported lazily
        _tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
        if not _tok.chat_template:
            raise SystemExit(
                f"no chat_template found in {MODEL_DIR}.\n"
                "Without one the corpus cannot be rendered the way the model "
                "actually reads text. Either the weights directory is wrong, or "
                "this model ships no template and you must decide explicitly "
                "whether to build from plain text."
            )

    if not _date_checked:
        _date_checked = True
        if "strftime_now" in _template_text(_tok):
            if not PIN_DATE:
                raise SystemExit(
                    "This model's chat template calls strftime_now, so the "
                    "rendered corpus would carry today's date and could never "
                    "be rebuilt byte for byte.\n"
                    "Set FOUNDRY_PIN_DATE=YYYY-MM-DD and run again, and record "
                    "the same date under chat.pin_date in the recipe."
                )
            if not _pin_the_date(PIN_DATE):
                raise SystemExit(
                    "The template calls strftime_now but the pin could not be "
                    "installed: transformers.utils.chat_template_utils is not "
                    "where this version keeps it. Refusing to build an "
                    "unreproducible corpus. Check the transformers version."
                )
    return _tok


def _clean(messages: Sequence[dict]) -> list[dict]:
    out = []
    for m in messages:
        d = {k: v for k, v in m.items() if k in _KEEP and v is not None}
        d.setdefault("content", "")
        out.append(d)
    return out


def render(messages: Sequence[dict],
           tools: Sequence[dict] | None = None,
           reasoning_strength: str | None = None,
           knowledge_cutoff: Any = None,        # unused, interface parity
           current_date: Any = None,            # unused, pinned via env instead
           namespace_descriptions: Any = None,  # unused, Glimmer only
           add_generation_prompt: bool = False,
           add_bos: bool = False) -> str:
    tok = _tokenizer()

    # Templates that have no thinking channel simply never read these variables,
    # so passing them unconditionally is safe and keeps one code path. Qwen reads
    # enable_thinking, MiniMax-M3 reads thinking_mode and ignores the other.
    enable_thinking = str(reasoning_strength or DEFAULT_REASONING).lower() not in _NO_THINK

    text = tok.apply_chat_template(
        _clean(messages),
        tools=list(tools) if tools else None,
        enable_thinking=enable_thinking,
        thinking_mode="enabled" if enable_thinking else "disabled",
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )

    if add_bos and tok.bos_token and not text.startswith(tok.bos_token):
        text = tok.bos_token + text
    return text
