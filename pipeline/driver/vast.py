"""vast.ai boxes for the pipeline: search, race, probe, destroy. Port of atomic-forge src/rent_race.sh.

The lessons it keeps, all learned by losing money:
  - rent a few offers at once and keep the first that is really usable, destroy the rest;
  - the vast ssh gateway answers before the container runs, only an echoed marker counts;
  - probe the download speed from the HF CDN with 8 streams, one stream says nothing;
  - destroy on exit and on SIGINT/SIGTERM, and label every box so `release.py reap` finds strays.
"""
import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time

PROBE_URL = "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/model-00001-of-00004.safetensors"
IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
_alive = set()
_claimed = set()            # offers some rent() in this process already tried: several boxes are rented side by side
_claim_lock = threading.Lock()


def say(msg):
    """print that never raises: with the laptop's disk full, a failed write of the driver's log
    killed the driver before it destroyed its boxes (rehearsal 2026-09-29, three boxes left billing)"""
    try:
        print(msg, flush=True)
    except OSError:
        pass


def _vast(*args, raw=True, check=True):
    cmd = ["vastai", *args] + (["--raw"] if raw else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"vastai {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
    return json.loads(r.stdout) if raw and r.stdout.strip() else r.stdout


# $/GB of traffic. A box pulls the BF16 and the KLD reference (~145 GB on a 27B) before its first rung;
# on the rehearsal (2026-09-29) two hosts at $0.020 and $0.039/GB billed more for that than for the GPU.
MAX_GB_COST = 0.01


def search(query, disk_gb, limit=20):
    q = f"{query} disk_space>={disk_gb} inet_down>=1000 reliability>0.98 rentable=true"
    if "inet_down_cost" not in query:
        q += f" inet_down_cost<={MAX_GB_COST} inet_up_cost<={MAX_GB_COST}"
    offers = _vast("search", "offers", q, "-o", "dph", "--limit", str(limit))
    return offers or []


def destroy(iid, quiet=False):
    r = subprocess.run(["vastai", "destroy", "instance", str(iid), "-y"], capture_output=True, text=True)
    _alive.discard(iid)
    if not quiet:
        say(f"[vast] destroy {iid}: {'ok' if r.returncode == 0 else r.stderr.strip()}")


def _cleanup(*_):
    for iid in list(_alive):
        try:
            destroy(iid)
        except Exception:   # one box that will not go must not keep the others billing
            pass
    if _:
        sys.exit(130)


atexit.register(_cleanup)
signal.signal(signal.SIGINT, _cleanup)
signal.signal(signal.SIGTERM, _cleanup)


def labeled(prefix="release:"):
    insts = _vast("show", "instances") or []
    return [i for i in insts if str(i.get("label") or "").startswith(prefix)]


def _ssh(host, port, key, cmd, timeout=60):
    base = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=30", "-i", key, "-p", str(port), f"root@{host}", cmd]
    try:
        return subprocess.run(base, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def rent(query, disk_gb, label, key, race=3, min_mbps=100, window_s=3600, image=IMAGE):
    """Rent `race` offers, return the first that runs commands and downloads fast; destroy the others."""
    offers = search(query, disk_gb)
    if not offers:
        raise RuntimeError(f"no offers for: {query} disk>={disk_gb}")
    pub = subprocess.run(["ssh-keygen", "-y", "-f", key], capture_output=True, text=True, check=True).stdout.strip()
    onstart = ("{ [ -x /usr/sbin/sshd ] && command -v curl >/dev/null; } || { apt-get update -q && "
               "DEBIAN_FRONTEND=noninteractive apt-get install -yq openssh-server curl; }; "
               f"mkdir -p /run/sshd /root/.ssh; echo '{pub}' >> /root/.ssh/authorized_keys; "
               "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd")
    with _claim_lock:
        mine = [o for o in offers if o["id"] not in _claimed][:race]
        _claimed.update(o["id"] for o in mine)
    if not mine:
        raise RuntimeError(f"every offer for {query} is already being raced by another rental")
    racers = {}
    for o in mine:
        try:
            r = _vast("create", "instance", str(o["id"]), "--image", image, "--disk", str(disk_gb), "--ssh", "--direct",
                      "--onstart-cmd", onstart, "--label", label, "--cancel-unavail")
        except RuntimeError as e:
            say(f"[vast] offer {o['id']}: {e}")
            continue
        if not isinstance(r, dict):   # vastai prints some refusals as a bare JSON string
            say(f"[vast] offer {o['id']}: {str(r)[:200]}")
            continue
        iid = r.get("new_contract")
        if iid:
            racers[iid] = o
            _alive.add(iid)
            say(f"[vast] racing {iid}: {o.get('num_gpus')}x {o.get('gpu_name')}, {o.get('cpu_cores_effective')} cores, "
                  f"{o.get('cpu_ram', 0) / 1000:.0f} GB RAM, ${o.get('dph_total', o.get('dph', 0)):.2f}/h")
    if not racers:
        raise RuntimeError("no instance could be created")

    t0, tries = time.time(), {}
    while time.time() - t0 < window_s:
        for iid in list(racers):
            try:
                inst = _vast("show", "instance", str(iid))
            except RuntimeError:
                continue
            if (inst or {}).get("actual_status") != "running":
                continue
            url = _vast("ssh-url", str(iid), raw=False, check=False).strip()
            if not url.startswith("ssh://"):
                continue
            host, port = url[len("ssh://root@"):].rsplit(":", 1)
            probe = ("echo SH_OK; ( for i in 1 2 3 4 5 6 7 8; do curl -sL --max-time 20 -o /dev/null "
                     f"-w '%{{speed_download}}\\n' '{PROBE_URL}' & done; wait ) | awk '{{s+=$1}} END{{printf \"SPEED=%d\\n\", s}}'")
            r = _ssh(host, port, key, probe, timeout=90)
            out = r.stdout if r else ""
            if "SH_OK" not in out:
                continue  # gateway up, container not yet
            speed = int(next((l.split("=")[1] for l in out.splitlines() if l.startswith("SPEED=")), "0") or 0)
            if speed >= min_mbps * 1_000_000:
                for other in list(racers):
                    if other != iid:
                        destroy(other)
                o = racers[iid]
                say(f"[vast] winner {iid} at {host}:{port}, {speed // 1_000_000} MB/s")
                return {"iid": iid, "host": host, "port": int(port), "offer": o,
                        "dph": o.get("dph_total", o.get("dph")), "gpu": f"{o.get('num_gpus')}x {o.get('gpu_name')}",
                        "t_rented": time.time()}
            tries[iid] = tries.get(iid, 0) + 1
            say(f"[vast] {iid}: probe {tries[iid]} at {speed // 1_000_000} MB/s")
            if tries[iid] >= 20:
                destroy(iid)
                racers.pop(iid)
        if not racers:
            raise RuntimeError("every racer dropped out")
        time.sleep(10)
    raise RuntimeError(f"nobody passed the probe in {window_s // 60} minutes")


def keep(iid):
    """Do not destroy this box when the driver exits (for debugging on the box)."""
    _alive.discard(iid)


def whoami():
    return os.environ.get("USER", "unknown")
