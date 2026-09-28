"""hub.py over ssh: the same folders on a box, reached through a command channel.

No box here: the ssh channel is replaced by a local bash, so the commands hub.py
sends are executed as they are (find, tar, cat, mkdir), against a temp folder.
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
import hub  # noqa: E402


@pytest.fixture
def box(tmp_path, monkeypatch):
    """LOCAL_HUB in its ssh form, with the ssh replaced by a local shell."""
    root = tmp_path / "hub"
    monkeypatch.setenv("LOCAL_HUB", f"ssh://root@box.example:2222{root}")
    monkeypatch.setenv("LOCAL_HUB_KEY", str(tmp_path / "no-such-key"))

    def local_shell(argv, cmd, input=None, timeout=600):
        assert argv[0] == "ssh" and argv[-1] == "root@box.example" and "2222" in argv
        r = subprocess.run(["bash", "-c", cmd], input=input, capture_output=True, timeout=timeout)
        if r.returncode:
            raise RuntimeError(r.stderr.decode())
        return r.stdout

    monkeypatch.setattr(hub, "_ssh", local_shell)
    monkeypatch.setattr(hub.time, "sleep", lambda s: None)
    return root


REPO = "AtomicChat/Tiny-GGUF-metrics"


def test_remote_root_is_parsed(box):
    argv, path = hub._remote("ssh://root@1.2.3.4:40001/hub")
    assert path == "/hub" and argv[-1] == "root@1.2.3.4" and argv[argv.index("-p") + 1] == "40001"
    assert hub._remote("/plain/dir") is None
    assert hub.local_root(REPO).startswith("ssh://") and hub.local_root(REPO).endswith("/AtomicChat--Tiny-GGUF-metrics")
    assert hub.local_root("Qwen/Qwen3.5-2B") is None   # upstream stays on the real hub


def test_ls_of_a_missing_repo_is_empty(box):
    assert hub.ls(REPO, "dataset") == []
    assert not hub.has(REPO, "dataset", "*")


def test_up_updir_ls_get_round_trip(box, tmp_path):
    hub.ensure(REPO, "dataset")
    f = tmp_path / "results.json"
    f.write_text(json.dumps([{"name": "x"}]))
    hub.up(str(f), "results.json", REPO, "dataset")
    d = tmp_path / "ladder"
    (d / "sub").mkdir(parents=True)
    (d / "ladder.json").write_text("{}")
    (d / "sub" / "AD-Q4_K_M.types").write_text("ffn_down=q4_k\n")
    hub.updir(str(d), "ladder", REPO, "dataset")
    assert hub.ls(REPO, "dataset") == ["ladder/ladder.json", "ladder/sub/AD-Q4_K_M.types", "results.json"]
    assert hub.has(REPO, "dataset", "ladder/*.json") and not hub.has(REPO, "dataset", "kld/*")
    dest = tmp_path / "fetched"
    got = hub.get(REPO, "dataset", "-", "ladder/*", str(dest))
    assert sorted(os.path.relpath(p, dest) for p in got) == ["ladder/ladder.json", "ladder/sub/AD-Q4_K_M.types"]
    assert (dest / "ladder" / "sub" / "AD-Q4_K_M.types").read_text() == "ffn_down=q4_k\n"
    with pytest.raises(FileNotFoundError):
        hub.get(REPO, "dataset", "-", "nothing/*", str(dest))
    assert hub.sha(REPO, "dataset") == "local"
    # the folder on the "box" is the plain layout the nodes use with LOCAL_HUB=/hub
    assert (box / "AtomicChat--Tiny-GGUF-metrics" / "results.json").exists()
