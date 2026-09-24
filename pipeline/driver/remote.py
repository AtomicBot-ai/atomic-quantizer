"""Run pipeline nodes on a box: copy the code and the token, start in tmux, follow the log.

Port of atomic-forge scripts/ablit_remote.sh. The node runs in tmux, so a dropped
connection costs nothing; the follower reconnects and resumes at the byte it had
read. A node's result is its sentinel line (NODE_DONE / NODE_FAIL).

A box is either a rented machine reached over ssh ({"kind": "ssh", host, port})
or a local container ({"kind": "docker", name}) for runs that cost nothing.
"""
import json
import os
import shlex
import subprocess
import time

PIPE_LOCAL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TOKEN_FILE = os.path.expanduser(os.environ.get("PIPELINE_HF_ENV", "~/.config/atomic-pipeline/hf_env"))


def _argv(box, key, cmd):
    if box.get("kind") == "docker":
        return ["docker", "exec", "-i", box["name"], "bash", "-c", cmd]
    return ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30", "-i", key, "-p", str(box["port"]), f"root@{box['host']}", cmd]


def run(box, key, cmd, timeout=120, input=None, check=True):
    """Run a shell command on the box. Returns stdout as bytes."""
    try:
        r = subprocess.run(_argv(box, key, cmd), input=input, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{cmd[:60]!r} on the box: no answer in {timeout} s (network?)") from None
    if check and r.returncode:
        raise RuntimeError(f"{cmd[:60]!r} on the box: {r.stderr.decode(errors='replace').strip()[-300:]}")
    return r.stdout


def token_bytes():
    if not os.path.exists(TOKEN_FILE):
        raise RuntimeError(f"no {TOKEN_FILE}: create it with `export HF_TOKEN=hf_...` and chmod 600")
    if os.stat(TOKEN_FILE).st_mode & 0o077:
        raise RuntimeError(f"{TOKEN_FILE} is readable by others, chmod 600 it")
    with open(TOKEN_FILE, "rb") as f:
        return f.read()


def bootstrap(box, key, with_token=True):
    """pipeline/ to /opt/pipeline, the token to /root/.hf_env (0600), tmux installed."""
    tar = subprocess.run(["tar", "-C", PIPE_LOCAL, "--exclude", "tests/.cache", "--exclude", "__pycache__",
                          "--exclude", "runs", "-czf", "-", "."], capture_output=True, check=True).stdout
    run(box, key, "rm -rf /opt/pipeline && mkdir -p /opt/pipeline && tar -xzf - -C /opt/pipeline", input=tar, timeout=300)
    if with_token:
        run(box, key, "umask 077 && cat > /root/.hf_env", input=token_bytes())
    # apt can hang for good on a connection opened during a network drop: bound each try, try three times
    run(box, key, "command -v tmux >/dev/null || for i in 1 2 3; do "
                  "timeout 240 apt-get -o Acquire::Retries=3 update -qq && "
                  "DEBIAN_FRONTEND=noninteractive timeout 240 apt-get install -yq tmux >/dev/null && break; "
                  "[ $i = 3 ] && exit 1; sleep 20; done", timeout=900)


def run_node(box, key, node, env, session=None, on_line=print, max_hours=12, stall_min=90):
    """Start nodes/<node>.sh in tmux and follow its log until the sentinel. Returns (ok, info)."""
    session = session or node
    log = f"/root/{session}.log"
    envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items() if v is not None)
    inner = f"set -o pipefail; env {envs} bash /opt/pipeline/nodes/{node}.sh 2>&1 | tee -a {log}; echo NODE_EXIT=$? >> {log}"
    run(box, key, f"tmux kill-session -t {session} 2>/dev/null; : > {log}; "
                  f"tmux new-session -d -s {session} {shlex.quote('bash -c ' + shlex.quote(inner))}")
    offset, t0, last_new, fails, pending = 0, time.time(), time.time(), 0, b""
    result, last_beat = None, None
    while True:
        try:
            data = run(box, key, f"tail -c +{offset + 1} {log}", timeout=60)
            fails = 0
        except (RuntimeError, subprocess.TimeoutExpired):
            fails += 1
            if fails > 60:
                return False, {"error": "lost the box"}
            time.sleep(30)
            continue
        if data:
            offset += len(data)
            data, pending = pending + data, b""
            if not data.endswith(b"\n"):  # keep a partial last line (and any split utf-8) for the next read
                data, _, pending = data.rpartition(b"\n")
            for line in data.decode(errors="replace").splitlines():
                on_line(line)
                # a heartbeat that repeats the same log tail is not progress (a hung download keeps beating)
                beat = line.split("] alive, ", 1)[1] if "] alive, " in line else None
                if beat is None or beat != last_beat:
                    last_new = time.time()
                last_beat = beat if beat is not None else last_beat
                if line.startswith("NODE_DONE ") or line.startswith("NODE_FAIL "):
                    result = line
                if line.startswith("NODE_EXIT="):
                    code = int(line.split("=")[1] or 1)
                    if code == 0 and result and result.startswith("NODE_DONE "):
                        payload = result.split(" ", 2)[2] if result.count(" ") >= 2 else "{}"
                        try:
                            return True, json.loads(payload)
                        except ValueError:
                            return True, {"raw": payload}
                    return False, {"exit": code, "line": result}
        if time.time() - t0 > max_hours * 3600:
            return False, {"error": f"over {max_hours} h"}
        if time.time() - last_new > stall_min * 60:
            return False, {"error": f"no progress for {stall_min} min"}
        time.sleep(5 if box.get("kind") == "docker" else 15)


def docker_box(name, hub_dir, image="ubuntu:24.04", cpus=None, memory=None):
    """A local container standing in for a rented box; the local hub is mounted at the same path."""
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    args = ["docker", "run", "-d", "--init", "--name", name, "-v", f"{hub_dir}:{hub_dir}"]   # --init reaps what tmux leaves
    if cpus:
        args += ["--cpus", str(cpus)]
    if memory:
        args += ["--memory", memory]
    subprocess.run(args + [image, "sleep", "infinity"], check=True, capture_output=True)
    return {"kind": "docker", "name": name, "iid": name, "gpu": "cpu (docker)", "dph": 0.0,
            "offer": {"num_gpus": 0}, "t_rented": time.time(), "host": "local"}


def docker_destroy(box):
    subprocess.run(["docker", "rm", "-f", box["name"]], capture_output=True)
