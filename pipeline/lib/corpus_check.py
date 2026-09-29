#!/usr/bin/env python3
"""Does the calibration corpus say what the model sees at inference?

    corpus_check.py template  Qwen/Qwen3.8-Flash-Next          (or a local dir with the tokenizer files)
    corpus_check.py specials  calib_train.txt --tokenizer Qwen/Qwen3.8-Flash-Next [--min 100]
    corpus_check.py coverage  calib_train.txt --tokenizer Qwen/Qwen3.8-Flash-Next [--min 0.999]

template  Renders, through the model's own apply_chat_template, every construct
          the corpus calibrates on and reports how the template writes it:
          a tool call, parallel tool calls in one turn, tool results matched by
          tool_call_id, a multi-turn dialogue with reasoning in an earlier turn,
          thinking off (the empty think block), reasoning effort if the template
          has one, an image and a video. Then every marker the renders contain
          that is an added token must come out as that one id, and a <|...|>
          marker that is not in the vocabulary means template and tokenizer do
          not belong together. Plain tags (<tools>, </function> on Qwen) are
          text the model was trained on as text. This is foundry.sh
          corpus_check, extended; it FAILs where the old one only warned.
specials  How often each marker of the template occurs in a corpus, tokenised
          with special tokens parsed (what llama-imatrix --parse-special does).
          A marker under --min (100) has no statistics worth the name; zero on
          every marker means the file was tokenised as plain text.
coverage  The share of the vocabulary the corpus reaches, over the ids that can
          appear at all (an id that is a fragment of a multi-byte character
          never comes out of the tokenizer on its own). The vocab sweep is
          there to push this to --min (99.9%).

Exit 1 on a FAIL. Needs `transformers` for template, `tokenizers` for the rest.
"""
import argparse
import collections
import json
import os
import re
import sys

MARKER = re.compile(r"<\|[^|<>\s]{1,40}\|>|</?[a-z_]{2,24}>")
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                     "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_time", "description": "Local time in a city",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                     "required": ["city"]}}},
]
SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
PAST = "SENTINEL_PAST_REASONING"
NOW = "SENTINEL_LAST_REASONING"
ASK = "SENTINEL_QUESTION"


def call(i, name, city):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": {"city": city}}}


CASES = {
    "tool call": dict(messages=[
        SYSTEM, {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [call(0, "get_weather", "Paris")]},
        {"role": "tool", "tool_call_id": "call_0", "content": '{"temp_c": 19}'},
        {"role": "assistant", "content": "It is 19 degrees."}], tools=TOOLS),
    "parallel tool calls": dict(messages=[
        SYSTEM, {"role": "user", "content": "Weather and time in Oslo?"},
        {"role": "assistant", "content": "", "tool_calls": [call(0, "get_weather", "Oslo"), call(1, "get_time", "Oslo")]},
        {"role": "tool", "tool_call_id": "call_0", "content": '{"temp_c": 4}'},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"time": "09:12"}'},
        {"role": "assistant", "content": "4 degrees, 09:12."}], tools=TOOLS),
    "reasoning in an earlier turn": dict(messages=[
        SYSTEM, {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": PAST},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6", "reasoning_content": NOW}]),
    "thinking off": dict(messages=[SYSTEM, {"role": "user", "content": ASK}],
                         add_generation_prompt=True, enable_thinking=False),
    "thinking on": dict(messages=[SYSTEM, {"role": "user", "content": ASK}], add_generation_prompt=True),
    "image and video": dict(messages=[
        {"role": "user", "content": [{"type": "image"}, {"type": "video"}, {"type": "text", "text": "Describe."}]},
        {"role": "assistant", "content": "A cat."}]),
}


def load_tokenizer(target):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(target)


def render(tok, case):
    kw = dict(case)
    msgs = kw.pop("messages")
    kw.setdefault("add_generation_prompt", False)
    return tok.apply_chat_template(msgs, tokenize=False, **kw)


def check_template(target):
    tok = load_tokenizer(target)
    tpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
    out, fails, notes = {}, [], []
    for name, case in CASES.items():
        try:
            out[name] = render(tok, case)
        except Exception as e:  # the template raising is itself the finding
            out[name] = None
            fails.append(f"{name}: the template refuses it ({str(e)[:90]})")
    r = out
    if r.get("tool call") is not None and "get_weather" not in r["tool call"]:
        fails.append("tool call: no tool markup in the render, the agentic slice would calibrate nothing of it")
    if r.get("parallel tool calls") is not None:
        p = r["parallel tool calls"]
        if not ("get_weather" in p and "get_time" in p):
            fails.append("parallel tool calls: one of the two calls is missing from the render")
        if p.count('{"temp_c": 4}') != 1 or p.count('{"time": "09:12"}') != 1:
            fails.append("parallel tool calls: a tool result is missing or doubled")
    if r.get("reasoning in an earlier turn") is not None:
        m = r["reasoning in an earlier turn"]
        notes.append("reasoning of earlier turns: " + ("KEPT (render corpus dialogues with it)" if PAST in m
                                                      else "dropped by the template (only the last turn keeps it)"))
        if NOW not in m:
            fails.append("reasoning of the last turn is not rendered")
    off, on = r.get("thinking off"), r.get("thinking on")
    if off is not None and on is not None:
        tail_off, tail_on = off[off.rfind(ASK) + len(ASK):], on[on.rfind(ASK) + len(ASK):]
        if off == on:
            notes.append("thinking off: the template ignores enable_thinking")
        else:
            notes.append(f"thinking off: the prompt ends {tail_off!r}")
            notes.append(f"thinking on: the prompt ends {tail_on!r}")
            sys_off, sys_on = off.split("<|im_end|>")[0], on.split("<|im_end|>")[0]
            if sys_off != sys_on:
                notes.append("thinking on also changes the system block: " + repr(sys_on[len(os.path.commonprefix([sys_on, sys_off])):][:90]))
    if "reasoning_effort" in tpl:
        for effort in ("low", "medium", "high", "xhigh"):
            try:
                render(tok, dict(CASES["thinking on"], reasoning_effort=effort))
                notes.append(f"reasoning_effort={effort}: accepted")
            except Exception as e:
                notes.append(f"reasoning_effort={effort}: refused ({str(e)[:60]})")
    if r.get("image and video") is not None:
        v = r["image and video"]
        rest = {m for k, s in out.items() if s and k != "image and video" for m in MARKER.findall(s)}
        vm = sorted(set(MARKER.findall(v)) - rest)
        notes.append("vision markers: " + (", ".join(vm) if vm else "none"))
        if not vm:
            notes.append("  the template emits nothing for an image: no vision markers can be calibrated")

    # A marker the tokenizer defines as an added token must come out as that one id
    # (llama-imatrix --parse-special); a <|...|> marker that is not one is a template
    # and tokenizer that do not belong together. Other tags (<tools>, </function> on
    # Qwen) are plain text the model was trained on as text, and calibrate as such.
    markers = sorted({m for s in out.values() if s for m in MARKER.findall(s)})
    added = {t.content for t in tok.added_tokens_decoder.values()}
    special, plain = [], []
    for m in markers:
        ids = tok.encode(m, add_special_tokens=False)
        if m in added:
            special.append(m)
            if len(ids) != 1:
                fails.append(f"marker {m} is an added token but encodes to {len(ids)} ids")
        else:
            plain.append(m)
            if m.startswith("<|"):
                fails.append(f"marker {m} looks special but is not a token of this vocabulary")
    if not special:
        fails.append("not one marker is a real token: not this model's template, or not its tokenizer")
    return {"target": target, "renders": out, "markers": special, "plain_markers": plain,
            "notes": notes, "fails": fails}


def print_template(rep):
    for name, text in rep["renders"].items():
        print(f"--- {name}")
        print(text if text is not None else "(refused)")
    print("=" * 60)
    for n in rep["notes"]:
        print("note  " + n)
    print(f"ok    {len(rep['markers'])} special markers, each one token: {' '.join(rep['markers'])}")
    if rep["plain_markers"]:
        print(f"note  plain text tags, calibrated as text: {' '.join(rep['plain_markers'])}")
    for f in rep["fails"]:
        print("FAIL  " + f)
    print("RESULT " + ("FAIL" if rep["fails"] else "ok"))


# ------------------------------------------------------------------ corpus side

def fast_tokenizer(target):
    from tokenizers import Tokenizer
    if os.path.isdir(target):
        return Tokenizer.from_file(os.path.join(target, "tokenizer.json")), target
    if target.endswith(".json") and os.path.isfile(target):
        return Tokenizer.from_file(target), os.path.dirname(target)
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(target, "tokenizer.json")
    try:
        hf_hub_download(target, "chat_template.jinja")
    except Exception:
        pass
    return Tokenizer.from_file(path), os.path.dirname(path)


def template_markers(tok, where):
    """Added tokens that the chat template writes literally (not every added token: the
    sweep puts each of those in once, and <|fim_prefix|> is not part of a chat)."""
    src = ""
    p = os.path.join(where, "chat_template.jinja")
    if os.path.exists(p):
        src = open(p, encoding="utf-8").read()
    elif os.path.exists(os.path.join(where, "tokenizer_config.json")):
        tpl = json.load(open(os.path.join(where, "tokenizer_config.json"), encoding="utf-8")).get("chat_template")
        src = json.dumps(tpl) if not isinstance(tpl, str) else tpl or ""
    if not src:
        raise SystemExit(f"no chat template next to the tokenizer in {where}")
    added = [(t.content, i) for i, t in tok.get_added_tokens_decoder().items()]
    return sorted((c, i) for c, i in added if c in src)


def token_ids(tok, path, piece=1 << 20):
    """Ids of the whole file, special tokens parsed, read in pieces cut at blank lines."""
    counts = collections.Counter()
    with open(path, encoding="utf-8") as f:
        text = f.read()
    start = 0
    while start < len(text):
        end = min(len(text), start + piece)
        if end < len(text):
            cut = text.rfind("\n\n", start, end)
            end = cut + 2 if cut > start else end
        counts.update(tok.encode(text[start:end], add_special_tokens=False).ids)
        start = end
    return counts


def check_specials(path, target, minimum=100, counts=None):
    tok, where = fast_tokenizer(target)
    counts = counts or token_ids(tok, path)
    rows = [{"marker": c, "id": i, "count": counts.get(i, 0)} for c, i in template_markers(tok, where)]
    low = [r for r in rows if r["count"] < minimum]
    return {"file": path, "tokens": sum(counts.values()), "min": minimum, "markers": rows,
            "fails": [f"{r['marker']} occurs {r['count']} times, under {minimum}" for r in low]}


def reachable_ids(tok):
    """Ids that the tokenizer can produce on their own: decode, re-encode, get the id back."""
    vocab = tok.get_vocab(with_added_tokens=True)
    ids = sorted(vocab.values())
    texts = tok.decode_batch([[i] for i in ids], skip_special_tokens=False)
    enc = tok.encode_batch(texts, add_special_tokens=False)
    return {i for i, e in zip(ids, enc) if e.ids == [i]}, len(ids)


def check_coverage(path, target, minimum=0.999, counts=None):
    tok, _ = fast_tokenizer(target)
    counts = counts or token_ids(tok, path)
    reach, total = reachable_ids(tok)
    seen = set(counts)
    covered = len(seen & reach)
    missing = sorted(reach - seen)
    share = covered / len(reach)
    return {"file": path, "vocab": total, "reachable": len(reach), "covered": covered, "share": share,
            "seen_outside_reachable": len(seen - reach), "missing_sample": missing[:40], "min": minimum,
            "fails": [] if share >= minimum else [f"covers {100 * share:.3f}% of reachable ids, under {100 * minimum:.1f}%"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("template", "specials", "coverage"))
    ap.add_argument("target", help="template: model repo or dir; specials/coverage: the corpus file")
    ap.add_argument("--tokenizer", help="specials/coverage: model repo, dir or tokenizer.json")
    ap.add_argument("--min", type=float)
    ap.add_argument("--json")
    a = ap.parse_args()

    if a.what == "template":
        rep = check_template(a.target)
        print_template(rep)
    else:
        if not a.tokenizer:
            ap.error(f"{a.what} needs --tokenizer")
        if a.what == "specials":
            rep = check_specials(a.target, a.tokenizer, int(a.min or 100))
            print(f"{rep['file']}: {rep['tokens']:,} tokens")
            for r in rep["markers"]:
                print(f"  {r['count']:9,d}  {r['marker']}  (id {r['id']})")
        else:
            rep = check_coverage(a.target, a.tokenizer, a.min or 0.999)
            print(f"{rep['file']}: {rep['covered']:,} of {rep['reachable']:,} reachable ids "
                  f"({100 * rep['share']:.3f}%), vocab {rep['vocab']:,}")
            if rep["missing_sample"]:
                print(f"  first missing ids: {rep['missing_sample'][:20]}")
        for f in rep["fails"]:
            print("FAIL  " + f)
        print("RESULT " + ("FAIL" if rep["fails"] else "ok"))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)
            f.write("\n")
    return 1 if rep["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
