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
import time

PROBE_URL = "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/model-00001-of-00004.safetensors"
IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
_alive = set()


def _vast(*args, raw=True, check=True):
    cmd = ["vastai", *args] + (["--raw"] if raw else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"vastai {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
    return json.loads(r.stdout) if raw and r.stdout.strip() else r.stdout


def search(query, disk_gb, limit=20):
    q = f"{query} disk_space>={disk_gb} inet_down>=1000 reliability>0.98 rentable=true"
    offers = _vast("search", "offers", q, "-o", "dph", "--limit", str(limit))
    return offers or []


def destroy(iid, quiet=False):
    r = subprocess.run(["vastai", "destroy", "instance", str(iid), "-y"], capture_output=True, text=True)
    _alive.discard(iid)
    if not quiet:
        print(f"[vast] destroy {iid}: {'ok' if r.returncode == 0 else r.stderr.strip()}", flush=True)


def _cleanup(*_):
    for iid in list(_alive):
        destroy(iid)
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
    racers = {}
    for o in offers[:race]:
        try:
            r = _vast("create", "instance", str(o["id"]), "--image", image, "--disk", str(disk_gb), "--ssh", "--direct",
                      "--onstart-cmd", onstart, "--label", label, "--cancel-unavail")
        except RuntimeError as e:
            print(f"[vast] offer {o['id']}: {e}", flush=True)
            continue
        iid = r.get("new_contract")
        if iid:
            racers[iid] = o
            _alive.add(iid)
            print(f"[vast] racing {iid}: {o.get('num_gpus')}x {o.get('gpu_name')}, {o.get('cpu_cores_effective')} cores, "
                  f"{o.get('cpu_ram', 0) / 1000:.0f} GB RAM, ${o.get('dph_total', o.get('dph', 0)):.2f}/h", flush=True)
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
                print(f"[vast] winner {iid} at {host}:{port}, {speed // 1_000_000} MB/s", flush=True)
                return {"iid": iid, "host": host, "port": int(port), "offer": o,
                        "dph": o.get("dph_total", o.get("dph")), "gpu": f"{o.get('num_gpus')}x {o.get('gpu_name')}",
                        "t_rented": time.time()}
            tries[iid] = tries.get(iid, 0) + 1
            print(f"[vast] {iid}: probe {tries[iid]} at {speed // 1_000_000} MB/s", flush=True)
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
