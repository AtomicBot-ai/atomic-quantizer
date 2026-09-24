#!/usr/bin/env python3
"""Did llama-quantize build what the ladder asked for? Exit 3 when it did not.

    verify_quant.py --inventory inventory.json --profile dense-hybrid.yaml --label AD-Q4_K_M \
                    --log quantize-AD-Q4_K_M.log [--commit SHA]

Checks every tensor's final type against the simulation, the exact set of
tensors that printed "applying manual override", that nothing fell back to
another type, the llama.cpp commit, and that the size is what the ladder
predicted. A file that fails is deleted by node_quant and never published.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ladder_gen  # noqa: E402
import quantlog  # noqa: E402


def verify(inventory, profile, label, log, commit=None):
    ladder, problems = ladder_gen.build(profile, inventory, only=[label])
    if not ladder["rungs"]:
        return [f"no rung {label} in profile {profile['name']}"]
    errors = [f"ladder refused: {e}" for e in problems.get(label, [])]
    rung = ladder["rungs"][0]
    for name, want, got in ladder_gen.compare(rung, log):
        errors.append(f"{name}: ladder says {want}, file has {got}")
    extra = set(log["overrides"]) - rung["_overrides"]
    missing = rung["_overrides"] - set(log["overrides"])
    if extra or missing:
        errors.append(f"override set differs: {len(extra)} unexpected, {len(missing)} missing "
                      f"(e.g. {sorted(extra)[:2]} {sorted(missing)[:2]})")
    if log["fallbacks"]:
        errors.append(f"{log['fallbacks']} tensors fell back to another type")
    if commit and log["commit"] and not (commit.startswith(log["commit"]) or log["commit"].startswith(commit)):
        errors.append(f"built by llama.cpp {log['commit']}, the run is pinned to {commit}")
    if log["quant_mib"] and abs(log["quant_mib"] * 2**20 / rung["predicted_bytes"] - 1) > 0.002:
        errors.append(f"size {log['quant_mib']} MiB, ladder predicted {rung['predicted_bytes'] / 2**20:.1f} MiB")
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--commit")
    a = ap.parse_args()
    with open(a.inventory) as f:
        inv = json.load(f)
    log = quantlog.parse_file(a.log)
    errors = verify(inv, ladder_gen.load_profile(a.profile), a.label, log, a.commit)
    if errors:
        print(f"{a.label}: {len(errors)} problems")
        for e in errors[:30]:
            print("  " + e)
        return 3
    print(f"{a.label}: ok, {len(log['tensors'])} tensors as planned, {len(log['overrides'])} overrides, "
          f"{log['quant_mib']} MiB {log['quant_bpw']} BPW, llama.cpp {log['commit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
