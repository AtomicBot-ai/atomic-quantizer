#!/usr/bin/env python3
"""Every read and write of the pipeline's own repos goes through here.

With LOCAL_HUB=/some/dir the repos under AtomicChat/ are plain folders
(/some/dir/AtomicChat--<name>/), so the whole chain can run on a laptop in a
container without a token and without touching the real hub. Upstream models and
calib-corpora are always read from Hugging Face.

    hub.py has    REPO TYPE GLOB          exit 0 when a file matches
    hub.py ls     REPO TYPE               one path per line
    hub.py ensure REPO TYPE               create, private
    hub.py up     LOCAL REMOTE REPO TYPE  one file
    hub.py updir  LOCAL REMOTE REPO TYPE  a folder, one commit
    hub.py get    REPO TYPE REV GLOB DEST files matching GLOB into DEST/<path>
    hub.py sha    REPO TYPE [REV]         current commit (model id + revision on the real hub)
"""
import fnmatch
import os
import shutil
import subprocess
import sys
import time

OWN = "AtomicChat/"
SHARED = {"AtomicChat/calib-corpora"}   # inputs every run reads from the real hub


def local_root(repo):
    root = os.environ.get("LOCAL_HUB")
    if root and repo.startswith(OWN) and repo not in SHARED:
        return os.path.join(root, repo.replace("/", "--"))
    return None


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
    if root:
        os.makedirs(root, exist_ok=True)
        return
    _retry(lambda: _api().create_repo(repo, repo_type=kind, private=True, exist_ok=True), f"create {repo}")


def up(local, remote, repo, kind, message=None):
    root = local_root(repo)
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
    if root:
        dest = os.path.join(root, remote) if remote not in ("", ".") else root
        shutil.copytree(local, dest, dirs_exist_ok=True)
        return
    _retry(lambda: _api().upload_folder(folder_path=local, path_in_repo=remote, repo_id=repo, repo_type=kind,
                                        commit_message=message or f"pipeline: {remote}/"), f"upload {remote}/")


def get(repo, kind, rev, pattern, dest):
    """Files matching pattern into dest, keeping their paths. Returns the local paths."""
    root = local_root(repo)
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
