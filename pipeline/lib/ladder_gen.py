#!/usr/bin/env python3
"""Generate an Atomic Dynamic (AD) ladder from a tensor inventory and a profile.

    ladder_gen.py --inventory inventory.json --profile profiles/dense-hybrid.yaml --out ladder/
    ladder_gen.py --inventory inventory.json --profile ... --explain AD-Q4_K_M
    ladder_gen.py --inventory inventory.json --profile ... --compare-log quantize-AD-Q4_K.log --label AD-Q4_K_M

The profile says which tensor roles exist and which type each rung gives them.
This script turns that into ordered llama-quantize rules, then simulates
llama-quantize on the inventory (first rule that matches wins, the rest take
the positional type) and refuses to write anything that llama-quantize would
silently change: an uncovered tensor under a mixture ftype, a k/i type on a
row that is not a multiple of 256, a very low bit type on a tensor that the
imatrix never sees.

Output per rung: <LABEL>.types (one regex=type per line, the format of
--tensor-type-file) and ladder.json with the rules, flags, expected per
tensor types, expected override count and predicted size.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys

import yaml

# ggml type -> (elements per block, bytes per block), from ggml-common.h
GGML = {
    "f32": (1, 4), "f16": (1, 2), "bf16": (1, 2),
    "q4_0": (32, 18), "q4_1": (32, 20), "q5_0": (32, 22), "q5_1": (32, 24), "q8_0": (32, 34),
    "q2_k": (256, 84), "q3_k": (256, 110), "q4_k": (256, 144), "q5_k": (256, 176), "q6_k": (256, 210),
    "iq1_s": (256, 50), "iq1_m": (256, 56), "iq2_xxs": (256, 66), "iq2_xs": (256, 74), "iq2_s": (256, 82),
    "iq3_xxs": (256, 98), "iq3_s": (256, 110), "iq4_xs": (256, 136), "iq4_nl": (32, 18),
    "tq1_0": (256, 54), "tq2_0": (256, 66), "mxfp4": (32, 17), "nvfp4": (64, 36),
}
# types that llama-quantize refuses on a tensor the imatrix has no data for
NEEDS_IMATRIX = {"iq3_xxs", "iq2_xxs", "iq2_xs", "iq2_s", "iq1_m", "iq1_s"}
# types that do not use the imatrix at all
IGNORES_IMATRIX = {"q8_0", "f16", "bf16", "f32", "mxfp4", "nvfp4"}

# positional ftype -> (default type before per tensor logic, LLAMA_FTYPE_MOSTLY_* value)
FTYPE = {
    "F16": ("f16", 1), "BF16": ("bf16", 32), "Q4_0": ("q4_0", 2), "Q4_1": ("q4_1", 3), "Q8_0": ("q8_0", 7),
    "Q5_0": ("q5_0", 8), "Q5_1": ("q5_1", 9), "Q2_K": ("q2_k", 10), "Q3_K_S": ("q3_k", 11),
    "Q3_K_M": ("q3_k", 12), "Q3_K_L": ("q3_k", 13), "Q4_K_S": ("q4_k", 14), "Q4_K_M": ("q4_k", 15),
    "Q5_K_S": ("q5_k", 16), "Q5_K_M": ("q5_k", 17), "Q6_K": ("q6_k", 18), "IQ2_XXS": ("iq2_xxs", 19),
    "IQ2_XS": ("iq2_xs", 20), "IQ3_XXS": ("iq3_xxs", 23), "IQ1_S": ("iq1_s", 24), "IQ4_NL": ("iq4_nl", 25),
    "IQ3_S": ("iq3_s", 26), "IQ3_M": ("iq3_s", 27), "IQ2_S": ("iq2_xs", 28), "IQ2_M": ("iq2_s", 29),
    "IQ4_XS": ("iq4_xs", 30), "IQ1_M": ("iq1_m", 31), "NVFP4": ("nvfp4", 39),
}
# llama-quant.cpp tensor_get_category: --token-embedding-type hits both, before any rule
TOKEN_EMBD_NAMES = ("token_embd.weight", "per_layer_token_embd.weight")
# ftypes whose per tensor logic is the identity: an uncovered tensor gets exactly the default type
PLAIN_FTYPES = {"F16", "BF16", "Q8_0"}
# file_type value to declare for a rung named after this ggml type
FILE_TYPE_OF = {
    "q8_0": 7, "q6_k": 18, "q5_k": 17, "q4_k": 15, "iq4_xs": 30, "iq4_nl": 25, "q3_k": 12, "iq3_s": 26,
    "iq3_xxs": 23, "q2_k": 10, "iq2_s": 28, "iq2_xs": 20, "iq2_xxs": 19, "iq1_m": 31, "iq1_s": 24,
}

# subset of llama-quant.cpp tensor_allows_quantization that matters for text models
NO_QUANT_SUBSTR = ("_norm.weight", "ffn_gate_inp.weight", "ffn_gate_tid2eid.weight", "altup", "laurel",
                   "per_layer_model_proj", "ssm_conv1d", "shortconv", "attn_rel_b.weight", ".position_embd",
                   "indexer.k_proj.weight", "indexer.q_proj.weight", ".rel_pos", ".patch_embd")


def n_dims(shape):
    n = 1
    for i, d in enumerate(shape):
        if d > 1:
            n = i + 1
    return n


def quantizable(t):
    name = t["name"]
    if n_dims(t["shape"]) < 2 or not name.endswith("weight"):
        return False
    return not any(s in name for s in NO_QUANT_SUBSTR)


def norm_type(t):
    t = t.lower()
    if t not in GGML:
        raise ValueError(f"'{t}' is not a ggml type (ftype mixes such as iq2_m or q4_k_m cannot go in a rule)")
    return t


def nbytes(shape, typ):
    n = 1
    for d in shape:
        n *= d
    be, bb = GGML[typ]
    return n // be * bb


class LadderError(Exception):
    pass


def load_profile(path):
    with open(path) as f:
        p = yaml.safe_load(f)
    for key in ("name", "roles", "rungs"):
        if key not in p:
            raise LadderError(f"{path}: profile has no '{key}'")
    return p


def block_of(name):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def compute_bands(profile, inv):
    spec = profile.get("bands") or {}
    if not spec:
        return {}
    group = re.compile(spec["group"])
    mtp = set(inv.get("mtp_blocks") or [])
    blocks = sorted({block_of(t["name"]) for t in inv["tensors"] if group.search(t["name"])} - mtp - {None})
    if not blocks:
        raise LadderError(f"band group {spec['group']!r} matches no tensor in the inventory")
    def count(key):
        # a band is either a block count or a fraction of the depth (half rounds up, so 64 blocks
        # at 0.0625/0.1875/0.125 give exactly the August 4/12/8)
        if f"{key}_frac" in spec:
            return int(math.floor(float(spec[f"{key}_frac"]) * len(blocks) + 0.5))
        return int(spec.get(key, 0))

    head, tail, mid = count("head"), count("tail"), count("mid")
    if head + tail + mid > len(blocks):
        raise LadderError(f"bands need {head}+{mid}+{tail} blocks but only {len(blocks)} carry the band group")
    edge = blocks[:head] + (blocks[-tail:] if tail else [])
    return {"edge": edge, "mid": blocks[head:head + mid], "eligible": blocks}


def render(pattern, bands):
    def alt(xs):
        return "(" + "|".join(str(x) for x in xs) + ")"
    out = pattern
    for k in ("edge", "mid"):
        if "{" + k + "}" in out:
            if not bands.get(k):
                return None  # band is empty for this model, the role does not apply
            out = out.replace("{" + k + "}", alt(bands[k]))
    return out


def rung_rules(profile, rung, bands, inv):
    """Ordered (pattern, type, role) list for one rung."""
    rules = []
    mtp = profile.get("mtp")
    if mtp and inv.get("mtp_blocks"):
        for b in inv["mtp_blocks"]:
            rules.append((rf"^blk\.{b}\.", norm_type(mtp["type"]), "mtp"))
    types = {} if rung.get("uniform") else dict(profile.get("defaults") or {})
    types.update(rung.get("types") or {})
    known = {r["name"] for r in profile["roles"]}
    unknown = set(types) - known
    if unknown:
        raise LadderError(f"{rung['label']}: types for unknown roles {sorted(unknown)}")
    names = [t["name"] for t in inv["tensors"]]
    for role in profile["roles"]:
        src = role.get("same_as", role["name"])
        if src not in types:
            continue
        pat = render(role["pattern"], bands)
        if pat is None:
            continue
        # a role this model does not have would only be noise in the rule file
        if not any(re.search(pat, n) for n in names):
            continue
        rules.append((pat, norm_type(types[src]), role["name"]))
    return rules


def rung_flags(profile, rung):
    f = dict(profile.get("flags") or {})
    f.update(rung.get("flags") or {})
    args = []
    if f.get("output_tensor_type"):
        args += ["--output-tensor-type", norm_type(f["output_tensor_type"])]
    if f.get("token_embedding_type"):
        args += ["--token-embedding-type", norm_type(f["token_embedding_type"])]
    return f, args


def simulate(profile, rung, rules, inv):
    """What llama-quantize will give every tensor, and the problems it would hide."""
    ftype = rung.get("ftype", profile.get("ftype", "Q8_0"))
    if ftype not in FTYPE:
        raise LadderError(f"{rung['label']}: unknown ftype {ftype}")
    default = FTYPE[ftype][0]
    flags, _ = rung_flags(profile, rung)
    compiled = [(re.compile(p), t, role) for p, t, role in rules]
    allow = [re.compile(a) for a in profile.get("allow_fallback") or []]
    no_im = [re.compile(a) for a in profile.get("no_imatrix") or []]
    mtp = set(inv.get("mtp_blocks") or [])

    eff, why, errors, overrides = {}, {}, [], set()
    for t in inv["tensors"]:
        name = t["name"]
        if not quantizable(t) or t["type"] not in ("bf16", "f16", "f32"):
            eff[name], why[name] = t["type"], "kept"
            continue
        hit = next(((typ, role) for rx, typ, role in compiled if rx.search(name)), None)
        flag = None
        if name in TOKEN_EMBD_NAMES and flags.get("token_embedding_type"):
            flag = norm_type(flags["token_embedding_type"])
        elif name == "output.weight" and flags.get("output_tensor_type"):
            flag = norm_type(flags["output_tensor_type"])
        if flag:
            if hit and hit[0] != flag:
                errors.append(f"{name}: rule '{hit[1]}' says {hit[0]} but a flag forces {flag} first")
            eff[name], why[name] = flag, "flag"
            continue
        if hit:
            typ, role = hit
            if typ != default:
                overrides.add(name)  # llama-quantize logs "applying manual override" for exactly these
        else:
            typ, role = default, "fallback"
            if ftype not in PLAIN_FTYPES:
                errors.append(f"{name}: no rule matches and {ftype} applies its own per tensor logic")
            elif not rung.get("uniform") and not any(a.search(name) for a in allow):
                errors.append(f"{name}: no rule matches, it would silently take {default}")
        be = GGML[typ][0]
        if be > 1 and t["shape"][0] % be:
            errors.append(f"{name}: row length {t['shape'][0]} is not a multiple of {be}, {typ} would fall back")
        exempt = name in TOKEN_EMBD_NAMES or name == "output.weight"
        if typ in NEEDS_IMATRIX and not exempt and (block_of(name) in mtp or any(a.search(name) for a in no_im)):
            errors.append(f"{name}: {typ} needs imatrix data this tensor never gets")
        eff[name], why[name] = typ, role
    return eff, why, errors, overrides


def predict(inv, eff):
    total = sum(nbytes(t["shape"], eff[t["name"]]) for t in inv["tensors"])
    elems = sum(math.prod(t["shape"]) for t in inv["tensors"])
    return total, 8.0 * total / elems


def file_type(profile, eff, inv):
    """Declared file type: the type most tensors of the file_type_from role ended up with."""
    role = next((r for r in profile["roles"] if r["name"] == profile["file_type_from"]), None)
    if role is None:
        raise LadderError(f"file_type_from names unknown role {profile['file_type_from']}")
    rx = re.compile(role["pattern"].replace("{edge}", r"\d+").replace("{mid}", r"\d+"))
    counts = {}
    for t in inv["tensors"]:
        if rx.search(t["name"]):
            counts[eff[t["name"]]] = counts.get(eff[t["name"]], 0) + 1
    return FILE_TYPE_OF.get(max(counts, key=counts.get)) if counts else None


def build(profile, inv, only=None):
    bands = compute_bands(profile, inv)
    out = {"profile": profile["name"], "inventory": inv.get("source"), "arch": inv.get("arch"),
           "block_count": inv.get("block_count"), "mtp_blocks": inv.get("mtp_blocks"),
           "bands": {k: v for k, v in bands.items() if k != "eligible"}, "rungs": []}
    problems = {}
    for rung in profile["rungs"]:
        if only and rung["label"] not in only:
            continue
        rules = rung_rules(profile, rung, bands, inv)
        eff, why, errors, overrides = simulate(profile, rung, rules, inv)
        size, bpw = predict(inv, eff)
        _, flag_args = rung_flags(profile, rung)
        ftype = rung.get("ftype", profile.get("ftype", "Q8_0"))
        text = "".join(f"{p}={t}\n" for p, t, _ in rules)
        ft = file_type(profile, eff, inv) if profile.get("file_type_from") else None
        counts = {}
        for t in inv["tensors"]:
            counts[eff[t["name"]]] = counts.get(eff[t["name"]], 0) + 1
        out["rungs"].append({
            "label": rung["label"], "ftype": ftype, "flags": flag_args, "control": bool(rung.get("control")),
            "file_type": ft if ft is not None and ft != FTYPE[ftype][1] else None,
            "rules": [[p, t] for p, t, _ in rules], "rules_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "expected_overrides": len(overrides), "type_counts": dict(sorted(counts.items())),
            "predicted_bytes": size, "predicted_gib": round(size / 2**30, 3), "predicted_bpw": round(bpw, 3),
            "imatrix": any(t not in IGNORES_IMATRIX for t in eff.values() if t not in ("f32",)),
            "no_imatrix_tensors": sorted(n for n in eff if block_of(n) in set(inv.get("mtp_blocks") or [])),
        })
        out["rungs"][-1]["_text"] = text
        out["rungs"][-1]["_effective"] = eff
        out["rungs"][-1]["_why"] = why
        out["rungs"][-1]["_overrides"] = overrides
        if errors:
            problems[rung["label"]] = errors
    return out, problems


def write(ladder, outdir):
    os.makedirs(outdir, exist_ok=True)
    public = {k: v for k, v in ladder.items() if k != "rungs"}
    public["rungs"] = []
    for r in ladder["rungs"]:
        with open(os.path.join(outdir, f"{r['label']}.types"), "w") as f:
            f.write(r["_text"])
        public["rungs"].append({k: v for k, v in r.items() if not k.startswith("_")})
    with open(os.path.join(outdir, "ladder.json"), "w") as f:
        json.dump(public, f, indent=1)
        f.write("\n")


def compare(ladder_rung, log, skip=None):
    """Differences between the simulated types and the types a real log shows."""
    rx = re.compile(skip) if skip else None
    diffs = []
    for name, want in ladder_rung["_effective"].items():
        if rx and rx.search(name):
            continue
        got = log["tensors"].get(name)
        if got is None:
            diffs.append((name, want, "missing in log"))
        elif got["dst"] != want:
            diffs.append((name, want, got["dst"]))
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--explain")
    ap.add_argument("--compare-log")
    ap.add_argument("--label")
    ap.add_argument("--skip", help="regex of tensor names to leave out of --compare-log")
    a = ap.parse_args()

    with open(a.inventory) as f:
        inv = json.load(f)
    profile = load_profile(a.profile)
    ladder, problems = build(profile, inv, a.only)

    print(f"{profile['name']}: {inv.get('arch')}, {inv.get('block_count')} blocks, mtp {inv.get('mtp_blocks')}, "
          f"bands {ladder['bands']}")
    print(f"{'label':22s} {'ftype':8s} {'rules':>5s} {'overr':>5s} {'GiB':>7s} {'BPW':>6s}  file_type")
    for r in ladder["rungs"]:
        print(f"{r['label']:22s} {r['ftype']:8s} {len(r['rules']):5d} {r['expected_overrides']:5d} "
              f"{r['predicted_gib']:7.2f} {r['predicted_bpw']:6.2f}  {r['file_type']}")

    if a.explain:
        r = next(x for x in ladder["rungs"] if x["label"] == a.explain)
        print(f"\n{r['label']} rules:")
        for p, t in r["rules"]:
            print(f"  {p}={t}")
        groups = {}
        for n, t in r["_effective"].items():
            key = (re.sub(r"blk\.\d+\.", "blk.N.", n), t, r["_why"][n])
            groups[key] = groups.get(key, 0) + 1
        print("\neffective types:")
        for (n, t, w), c in sorted(groups.items()):
            print(f"  {c:4d} {n:44s} {t:8s} {w}")

    if a.compare_log:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import quantlog
        label = a.label or (ladder["rungs"][0]["label"] if len(ladder["rungs"]) == 1 else None)
        r = next(x for x in ladder["rungs"] if x["label"] == label)
        log = quantlog.parse_file(a.compare_log)
        diffs = compare(r, log, a.skip)
        print(f"\ncompare {label} with {os.path.basename(a.compare_log)}: {len(diffs)} differences")
        for d in diffs[:40]:
            print("  %-44s want %-8s got %s" % d)

    if problems:
        print("\nREFUSED, llama-quantize would not do what the ladder says:")
        for label, errs in problems.items():
            print(f"  {label}: {len(errs)} problems")
            for e in errs[:8]:
                print(f"    {e}")
        return 2
    if a.out:
        write(ladder, a.out)
        print(f"\nwrote {len(ladder['rungs'])} rungs to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
