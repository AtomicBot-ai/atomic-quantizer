"""Run pipeline nodes on a rented box: copy the code and the token, start in tmux, follow the log.

Port of atomic-forge scripts/ablit_remote.sh. The node runs in tmux, so a dropped
ssh connection costs nothing; the follower reconnects and resumes at the byte it
had read. A node's result is its sentinel line (NODE_DONE / NODE_FAIL).
"""
import json
import os
import shlex
import subprocess
import time

PIPE_LOCAL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TOKEN_FILE = os.path.expanduser(os.environ.get("PIPELINE_HF_ENV", "~/.config/atomic-pipeline/hf_env"))


def _ssh_args(box, key):
    return ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30", "-i", key]


def ssh(box, key, cmd, timeout=120, check=True):
    r = subprocess.run(["ssh", *_ssh_args(box, key), "-p", str(box["port"]), f"root@{box['host']}", cmd],
                       capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"ssh {cmd[:60]!r}: {r.stderr.strip()[-300:]}")
    return r.stdout


def bootstrap(box, key):
    """Copy pipeline/ to /opt/pipeline and the token file to /root/.hf_env (0600)."""
    st = os.stat(TOKEN_FILE)
    if st.st_mode & 0o077:
        raise RuntimeError(f"{TOKEN_FILE} is readable by others, chmod 600 it")
    tar = subprocess.run(["tar", "-C", PIPE_LOCAL, "--exclude", "tests/.cache", "--exclude", "__pycache__",
                          "--exclude", "runs", "-czf", "-", "."], capture_output=True, check=True).stdout
    subprocess.run(["ssh", *_ssh_args(box, key), "-p", str(box["port"]), f"root@{box['host']}",
                    "rm -rf /opt/pipeline && mkdir -p /opt/pipeline && tar -xzf - -C /opt/pipeline"],
                   input=tar, check=True, capture_output=True, timeout=300)
    subprocess.run(["scp", *_ssh_args(box, key), "-P", str(box["port"]), TOKEN_FILE,
                    f"root@{box['host']}:/root/.hf_env"], check=True, capture_output=True, timeout=120)
    ssh(box, key, "chmod 600 /root/.hf_env; command -v tmux >/dev/null || "
                  "(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -yq tmux >/dev/null)", timeout=600)


def run_node(box, key, node, env, session=None, on_line=print, max_hours=12, stall_min=90):
    """Start nodes/<node>.sh in tmux and follow its log until the sentinel. Returns (ok, info)."""
    session = session or node
    log = f"/root/{session}.log"
    envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items() if v is not None)
    inner = f"set -o pipefail; env {envs} bash /opt/pipeline/nodes/{node}.sh 2>&1 | tee -a {log}; echo NODE_EXIT=$? >> {log}"
    ssh(box, key, f"tmux kill-session -t {session} 2>/dev/null; : > {log}; "
                  f"tmux new-session -d -s {session} {shlex.quote('bash -c ' + shlex.quote(inner))}")
    offset, t0, last_new, fails, pending = 0, time.time(), time.time(), 0, b""
    result = None
    while True:
        try:
            r = subprocess.run(["ssh", *_ssh_args(box, key), "-p", str(box["port"]), f"root@{box['host']}",
                                f"tail -c +{offset + 1} {log}"], capture_output=True, timeout=60)
            if r.returncode:
                raise RuntimeError(r.stderr.decode(errors="replace"))
            data = r.stdout
            fails = 0
        except (RuntimeError, subprocess.TimeoutExpired):
            fails += 1
            if fails > 60:
                return False, {"error": "lost the box"}
            time.sleep(30)
            continue
        if data:
            offset += len(data)
            last_new = time.time()
            data, pending = pending + data, b""
            if not data.endswith(b"\n"):  # keep a partial last line (and any split utf-8) for the next read
                data, _, pending = data.rpartition(b"\n")
            for line in data.decode(errors="replace").splitlines():
                on_line(line)
                if line.startswith("NODE_DONE ") or line.startswith("NODE_FAIL "):
                    result = line
                if line.startswith("NODE_EXIT="):
                    code = int(line.split("=")[1] or 1)
                    if result and result.startswith("NODE_DONE "):
                        payload = result.split(" ", 2)[2] if result.count(" ") >= 2 else "{}"
                        try:
                            return True, json.loads(payload)
                        except ValueError:
                            return True, {"raw": payload}
                    return False, {"exit": code, "line": result}
        if time.time() - t0 > max_hours * 3600:
            return False, {"error": f"over {max_hours} h"}
        if time.time() - last_new > stall_min * 60:
            return False, {"error": f"no output for {stall_min} min"}
        time.sleep(15)
