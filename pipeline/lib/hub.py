#!/usr/bin/env python3
"""Every read and write of the pipeline's own repos goes through here.

With LOCAL_HUB=/some/dir the repos under AtomicChat/ are plain folders
(/some/dir/AtomicChat--<name>/), so the whole chain can run on a laptop in a
container without a token and without touching the real hub. Upstream models and
calib-corpora are always read from Hugging Face.

With LOCAL_HUB=ssh://root@HOST:PORT/hub the same folders live on a rented box
and are reached over ssh (key: LOCAL_HUB_KEY, default ~/.ssh/id_ed25519). That
is how the driver keeps a run off the hub entirely: the nodes on the box see
LOCAL_HUB=/hub, the driver on the laptop sees the ssh form of the same place.

    hub.py has    REPO TYPE GLOB          exit 0 when a file matches
    hub.py ls     REPO TYPE               one path per line
    hub.py ensure REPO TYPE               create, private
    hub.py up     LOCAL REMOTE REPO TYPE  one file
    hub.py updir  LOCAL REMOTE REPO TYPE  a folder, one commit
    hub.py get    REPO TYPE REV GLOB DEST files matching GLOB into DEST/<path>
    hub.py sha    REPO TYPE [REV]         current commit (model id + revision on the real hub)
"""
import fnmatch
import io
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

OWN = "AtomicChat/"
SHARED = {"AtomicChat/calib-corpora"}   # inputs every run reads from the real hub


def local_root(repo):
    root = os.environ.get("LOCAL_HUB")
    if root and repo.startswith(OWN) and repo not in SHARED:
        return os.path.join(root, repo.replace("/", "--"))
    return None


def _remote(root):
    """(ssh argv, path on the box) for an ssh:// root, else None."""
    m = re.match(r"ssh://([^@]+)@([^:/]+):(\d+)(/.*)$", root or "")
    if not m:
        return None
    user, host, port, path = m.groups()
    key = os.path.expanduser(os.environ.get("LOCAL_HUB_KEY", "~/.ssh/id_ed25519"))
    argv = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30", "-i", key, "-p", port, f"{user}@{host}"]
    return argv, path


def _ssh(argv, cmd, input=None, timeout=600):
    r = subprocess.run(argv + [cmd], input=input, capture_output=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"hub over ssh, {cmd[:60]!r}: {r.stderr.decode(errors='replace').strip()[-300:]}")
    return r.stdout


def _remote_ls(argv, path):
    # portable (no -printf): the same command also runs under a local shell in the tests
    out = _retry(lambda: _ssh(argv, f"test -d {shlex.quote(path)} || exit 0; cd {shlex.quote(path)} && "
                                    "find . -type f | sed 's|^\\./||' | sort", timeout=120), f"ls {path}")
    return [l for l in out.decode().splitlines() if l]


def _tar_bytes(local):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(local, arcname=".")
    return buf.getvalue()


def _api():
    from huggingface_hub import HfApi
    return HfApi()


def _retry(fn, what, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # network and 5xx; the caller sees the last one
            if i == tries - 1:
                raise
            print(f"[hub] {what} failed ({type(e).__name__}: {str(e)[:120]}), retry {i + 1}", file=sys.stderr)
            time.sleep(20 * (i + 1))


def ls(repo, kind):
    root = local_root(repo)
    if root and _remote(root):
        return _remote_ls(*_remote(root))
    if root:
        if not os.path.isdir(root):
            return []
        return sorted(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
                      for d, _, fs in os.walk(root) for f in fs)
    try:
        return _api().list_repo_files(repo, repo_type=kind)
    except Exception:
        return []


def has(repo, kind, pattern):
    return any(fnmatch.fnmatch(f, pattern) for f in ls(repo, kind))


def ensure(repo, kind):
    root = local_root(repo)
    if root and _remote(root):
        argv, path = _remote(root)
        _retry(lambda: _ssh(argv, f"mkdir -p {shlex.quote(path)}", timeout=120), f"ensure {path}")
        return
    if root:
        os.makedirs(root, exist_ok=True)
        return
    _retry(lambda: _api().create_repo(repo, repo_type=kind, private=True, exist_ok=True), f"create {repo}")


def up(local, remote, repo, kind, message=None):
    root = local_root(repo)
    if root and _remote(root):
        argv, path = _remote(root)
        dest = shlex.quote(f"{path}/{remote}")
        with open(local, "rb") as f:
            data = f.read()
        _retry(lambda: _ssh(argv, f"mkdir -p $(dirname {dest}) && cat > {dest}.part && mv {dest}.part {dest}",
                            input=data, timeout=600), f"up {remote}")
        return
    if root:
        dest = os.path.join(root, remote)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(local, dest + ".part")
        os.replace(dest + ".part", dest)
        return
    _retry(lambda: _api().upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo, repo_type=kind,
                                      commit_message=message or f"pipeline: {remote}"), f"upload {remote}")


def updir(local, remote, repo, kind, message=None):
    root = local_root(repo)
    if root and _remote(root):
        argv, path = _remote(root)
        dest = shlex.quote(path if remote in ("", ".") else f"{path}/{remote}")
        _retry(lambda: _ssh(argv, f"mkdir -p {dest} && tar -xzf - -C {dest}", input=_tar_bytes(local), timeout=600),
               f"updir {remote}/")
        return
    if root:
        dest = os.path.join(root, remote) if remote not in ("", ".") else root
        shutil.copytree(local, dest, dirs_exist_ok=True)
        return
    _retry(lambda: _api().upload_folder(folder_path=local, path_in_repo=remote, repo_id=repo, repo_type=kind,
                                        commit_message=message or f"pipeline: {remote}/"), f"upload {remote}/")


def get(repo, kind, rev, pattern, dest):
    """Files matching pattern into dest, keeping their paths. Returns the local paths."""
    root = local_root(repo)
    if root and _remote(root):
        argv, path = _remote(root)
        names = [f for f in _remote_ls(argv, path) if fnmatch.fnmatch(f, pattern)]
        if not names:
            raise FileNotFoundError(f"{repo}: nothing matches {pattern}")
        data = _retry(lambda: _ssh(argv, f"cd {shlex.quote(path)} && tar -czf - --null -T -",
                                   input=b"\0".join(n.encode() for n in names) + b"\0", timeout=1800), f"get {pattern}")
        os.makedirs(dest, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
            t.extractall(dest)
        return [os.path.join(dest, n) for n in names]
    if root:
        out = []
        for f in ls(repo, kind):
            if fnmatch.fnmatch(f, pattern):
                d = os.path.join(dest, f)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copyfile(os.path.join(root, f), d)
                out.append(d)
        if not out:
            raise FileNotFoundError(f"{repo}: nothing matches {pattern}")
        return out
    _download(repo, kind, rev, pattern, dest)
    out = [os.path.join(dp, f) for dp, _, fs in os.walk(dest) for f in fs
           if fnmatch.fnmatch(os.path.relpath(os.path.join(dp, f), dest).replace(os.sep, "/"), pattern)]
    if not out:
        raise FileNotFoundError(f"{repo}: nothing matches {pattern}")
    return out


def _bytes_under(*dirs):
    return sum(os.path.getsize(os.path.join(dp, f)) for d in dirs if os.path.isdir(d)
               for dp, _, fs in os.walk(d) for f in fs if os.path.exists(os.path.join(dp, f)))


def _download(repo, kind, rev, pattern, dest, stall_s=300, tries=3):
    """snapshot_download in a child process, killed when no byte arrives for stall_s.

    A download can hang without an error (seen with the Xet backend on a flaky
    link: the file stays at 0 bytes forever while plain HTTP works), and a hung
    node costs box hours. The last try goes over plain HTTP.
    """
    xet_cache = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "xet")
    for i in range(tries):
        env = dict(os.environ)
        if i == tries - 1:
            env["HF_HUB_DISABLE_XET"] = "1"
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "_snapshot", repo, kind, rev or "-", pattern, dest],
                             env=env)
        last, since = -1, time.time()
        t0 = since
        while p.poll() is None:
            time.sleep(1 if time.time() - t0 < 15 else 10)   # small files return in a second
            now = _bytes_under(dest, xet_cache)
            if now != last:
                last, since = now, time.time()
            elif time.time() - since > stall_s:
                p.kill()
                p.wait()
                print(f"[hub] {repo} {pattern}: no progress for {stall_s} s, restarting the download", file=sys.stderr)
                break
        if p.returncode == 0:
            return
        if i < tries - 1:
            time.sleep(20 * (i + 1))
    raise RuntimeError(f"download of {repo} {pattern} failed {tries} times")


def _snapshot(repo, kind, rev, pattern, dest):
    from huggingface_hub import snapshot_download
    snapshot_download(repo, repo_type=kind, revision=None if rev in (None, "-") else rev,
                      allow_patterns=[pattern], local_dir=dest)


def sha(repo, kind, rev=None):
    if local_root(repo):
        return "local"
    info = _api().repo_info(repo, repo_type=kind, revision=None if rev in (None, "-") else rev)
    return info.sha


def main(argv):
    cmd, args = argv[1], argv[2:]
    if cmd == "has":
        return 0 if has(*args) else 1
    if cmd == "ls":
        print("\n".join(ls(*args)))
    elif cmd == "ensure":
        ensure(*args)
    elif cmd == "up":
        up(*args)
    elif cmd == "updir":
        updir(*args)
    elif cmd == "get":
        get(*args)
    elif cmd == "sha":
        print(sha(*args))
    elif cmd == "_snapshot":
        _snapshot(*args)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
