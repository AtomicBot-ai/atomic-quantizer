"""Parse llama-quantize logs.

Two uses: the tests read published logs to learn which type every tensor really
got, and node_quant reads its own log to check the build did what the ladder
asked for.
"""
import re

ROW = re.compile(r"^\[\s*(\d+)/\s*(\d+)\]\s+(\S+)\s+-\s+\[([^\]]*)\],\s+type\s+=\s+(\S+),\s*(.*)$")
CONVERT = re.compile(r"converting to (\S+)")
OVERRIDE = re.compile(r"llama_tensor_get_type: (\S+)\s+- applying manual override: (\S+) -> (\S+)")
FALLBACK = re.compile(r"-> falling back to\s+(\S+)")
BUILD = re.compile(r"build (\d+), commit ([0-9a-f]+)")
QUANTIZING = re.compile(r"quantizing '([^']+)' to '([^']+)' as (\S+)")
QSIZE = re.compile(r"quant size\s+=\s+([0-9.]+) MiB \(([0-9.]+) BPW\)")
MSIZE = re.compile(r"model size\s+=\s+([0-9.]+) MiB \(([0-9.]+) BPW\)")


def parse(text):
    """Return dict with tensors (name -> {src, dst, shape}), overrides, fallbacks and header facts."""
    lines = text.splitlines()
    tensors = {}
    i = 0
    while i < len(lines):
        m = ROW.match(lines[i])
        if not m:
            i += 1
            continue
        rest = m.group(6)
        j = i + 1
        # "did not find weights" splits one row over several lines
        while "converting to" not in rest and "size =" not in rest and j < len(lines) and not ROW.match(lines[j]):
            rest += " " + lines[j]
            j += 1
        c = CONVERT.search(rest)
        src = m.group(5).lower()
        shape = [int(x) for x in m.group(4).split(",")]
        tensors[m.group(3)] = {"src": src, "dst": (c.group(1) if c else src).lower(), "shape": shape}
        i = j

    out = {
        "tensors": tensors,
        "overrides": {n: b.lower() for n, a, b in OVERRIDE.findall(text)},
        "fallbacks": len(FALLBACK.findall(text)),
        "commit": None, "build": None, "ftype": None, "src": None, "dst": None,
        "quant_mib": None, "quant_bpw": None, "model_mib": None,
    }
    if (b := BUILD.search(text)):
        out["build"], out["commit"] = int(b.group(1)), b.group(2)
    if (q := QUANTIZING.search(text)):
        out["src"], out["dst"], out["ftype"] = q.group(1), q.group(2), q.group(3)
    if (s := QSIZE.search(text)):
        out["quant_mib"], out["quant_bpw"] = float(s.group(1)), float(s.group(2))
    if (s := MSIZE.search(text)):
        out["model_mib"] = float(s.group(1))
    return out


def parse_file(path):
    with open(path, errors="replace") as f:
        return parse(f.read())
