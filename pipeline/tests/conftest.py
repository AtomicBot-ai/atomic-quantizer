"""Fixtures: published llama-quantize logs, fetched once at a pinned revision.

The logs are the ground truth of what the August releases actually contain, so
the ladder generator is tested against them rather than against its own idea
of the rules. They live on Hugging Face; nothing is copied into this repo.
"""
import os
import sys
import urllib.request

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
CACHE = os.environ.get("PIPELINE_FIXTURES", os.path.join(HERE, ".cache"))

DENSE = ("AtomicChat/Qwen3.8-27B-GGUF-metrics", "6d91a9e43743e4cae36354df810860b8c1eb9834")
MOE = ("AtomicChat/Ling-3.0-flash-GGUF-metrics", "c4e73e27bf86590fb30b9547509fde69d5f39510")


def fetch(repo, rev, path):
    dest = os.path.join(CACHE, repo.replace("/", "--"), rev[:12], path)
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
        req = urllib.request.Request(url, headers={"User-Agent": "atomic-pipeline-tests"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dest + ".part", dest)
    return dest


@pytest.fixture(scope="session")
def dense_log():
    return lambda name: fetch(*DENSE, f"logs/{name}")


@pytest.fixture(scope="session")
def moe_log():
    return lambda name: fetch(*MOE, f"logs/quant/{name}")


@pytest.fixture(scope="session")
def profiles():
    return os.path.join(HERE, "..", "profiles")
