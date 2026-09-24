#!/usr/bin/env python3
"""Release driver: from a hub model id to published AD GGUF, abliterated and NVFP4 builds.

    release.py status --model Qwen/Qwen3.8-27B
    release.py gguf   --model Qwen/Qwen3.8-27B --recipe qwen3.8-27b --profile dense-hybrid \
                      --llama-commit 1692f9e50bb2... [--repo-suffix -rehearsal] [--ladder-ok]
    release.py ladder --model ... --profile ...          # generate + review + upload ladder/ only
    release.py results --model ...                       # merge results/rows into results.json
    release.py card   --model ... [--card-overwrite]     # README draft from results.json
    release.py ablit  --model Qwen/Qwen3.8-27B            # heretic, then run gguf on the result
    release.py nvfp4  --model Qwen/Qwen3.8-27B --recipe qwen3.8-27b
    release.py reap   [--all]                            # destroy boxes labelled release:

State lives on the hub; every stage is skipped when its outputs exist, so a run
that died is resumed by running the same command again. Needs `vastai set
api-key` done once, ~/.config/atomic-pipeline/hf_env (0600, `export HF_TOKEN=...`)
and an ssh key (default ~/.ssh/id_ed25519).

A free local run, no token, no vast: --local-hub DIR --local-box keeps the
AtomicChat/ repos as folders under DIR and runs the nodes in a CPU container
(docker, ubuntu:24.04). With --ctx 512 --kld-chunks 16 --im-max-chunks 64 a
small model goes through every stage in well under an hour.
"""
import argparse
import fnmatch
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)
import hub  # noqa: E402
import remote  # noqa: E402
import vast  # noqa: E402

PROFILES = os.path.join(HERE, "..", "profiles")
RUNS = os.path.join(HERE, "..", "runs")


def load_token():
    if not os.path.exists(remote.TOKEN_FILE):
        sys.exit(f"no {remote.TOKEN_FILE}: create it (0600) with `export HF_TOKEN=hf_...`, or run with --local-hub")
    with open(remote.TOKEN_FILE) as f:
        m = re.search(r"HF_TOKEN=['\"]?(hf_[A-Za-z0-9]+)", f.read())
    if not m:
        sys.exit(f"no HF_TOKEN in {remote.TOKEN_FILE}")
    os.environ["HF_TOKEN"] = m.group(1)


def api():
    from huggingface_hub import HfApi
    return HfApi()


class Run:
    def __init__(self, a, stem=None, journal=True):
        self.model = a.model
        self.stem = stem or a.model.split("/")[-1]
        suffix = a.repo_suffix or ""
        self.main = f"AtomicChat/{self.stem}-GGUF{suffix}"
        self.metrics = f"AtomicChat/{self.stem}-GGUF-metrics{suffix}"
        self.suffix = suffix
        self.dir = os.path.join(RUNS, f"{self.stem}{suffix}-{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(self.dir, exist_ok=True)
        self.log = open(os.path.join(self.dir, "run.jsonl"), "a") if journal else None
        self.boxes = []

    def event(self, kind, **kw):
        rec = {"t": time.strftime("%FT%TZ", time.gmtime()), "event": kind, **kw}
        if self.log:
            self.log.write(json.dumps(rec) + "\n")
            self.log.flush()
        print(f"== {kind} " + " ".join(f"{k}={v}" for k, v in kw.items() if k not in ("offer",)), flush=True)

    def fetch(self, repo, kind, pattern):
        """Files from our repos into the run folder; returns their local paths."""
        return hub.get(repo, kind, "-", pattern, os.path.join(self.dir, "hub", repo.replace("/", "--")))


def stage_state(run, only=None):
    """What is on the hub. With `only` (--only-rungs) the quant stage is those rungs, not the whole ladder."""
    m = hub.ls(run.metrics, "dataset")
    g = hub.ls(run.main, "model")
    have = lambda files, pat: any(fnmatch.fnmatch(f, pat) for f in files)  # noqa: E731
    st = {
        "convert": have(m, "inventory.json") and have(m, f"bf16/{run.stem}-BF16*.gguf"),
        "base": have(m, "kld/base-neutral.manifest.txt"),
        "imatrix": have(m, "imatrix/imatrix.gguf"),
        "ladder": have(m, "ladder/ladder.json"),
        "results": have(m, "results.json"),
    }
    rungs = []
    if st["ladder"]:
        lad = json.load(open(run.fetch(run.metrics, "dataset", "ladder/ladder.json")[0]))
        for r in lad["rungs"]:
            if (only and r["label"] not in only) or (not only and r.get("control")):
                continue
            name = f"{run.stem}-{r['label']}"
            done = (have(g, f"{name}.gguf") or have(g, f"{name}-00001-of-*.gguf")) and \
                have(m, f"logs/kld-neutral--{name}.log")
            rungs.append((r["label"], done))
    st["quant"] = bool(rungs) and all(d for _, d in rungs)
    return st, rungs


def model_bytes(model):
    info = api().model_info(model, files_metadata=True)
    return sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors"))


def box_plan(size_b, gpu_query=None, disk_gb=None):
    """GPU box for convert, base, imatrix and quant of one model, from its BF16 size."""
    s = size_b / 1e9
    if gpu_query:
        q = gpu_query
    elif s <= 20:
        q = "gpu_name in [RTX_4090,RTX_5090] num_gpus>=2 cpu_cores_effective>=16 cpu_ram>=64"
    elif s <= 60:
        q = "gpu_name=RTX_5090 num_gpus>=4 cpu_cores_effective>=32 cpu_ram>=128"
    elif s <= 160:
        q = "gpu_ram>=80 num_gpus>=4 cpu_cores_effective>=64 cpu_ram>=384"
    else:
        q = "gpu_ram>=140 num_gpus>=8 cpu_cores_effective>=96 cpu_ram>=1000"
    q += " cuda_max_good>=13.0"
    # source + BF16 + its upload shards, then BF16 + KLD reference (~90 GB) + its parts, + the largest quant
    disk = disk_gb or int(max(3 * s, s + 200) + 0.6 * s + 40)
    return q, disk


def open_box(run, a, stage, query=None, disk=None, size_b=None):
    """A vast box for the stage, or with --local-box a local container. Released by close_boxes."""
    if a.local_box:
        name = re.sub(r"[^A-Za-z0-9_.-]", "-", f"release-{run.stem}-{stage}")
        run.event("box", box="docker", name=name)
        box = remote.docker_box(name, a.local_hub)
    else:
        if query is None:
            query, disk = box_plan(size_b, a.gpu_query, a.disk_gb)
        run.event("rent", stage=stage, query=query, disk_gb=disk)
        box = vast.rent(query, disk, f"release:{os.path.basename(run.dir)}:{stage}", a.ssh_key, race=a.race)
        run.event("box", iid=box["iid"], gpu=box["gpu"], dph=box["dph"], host=box["host"])
    run.boxes.append(box)
    remote.bootstrap(box, a.ssh_key, with_token=not a.local_hub)
    return box


def close_boxes(run, a):
    for b in run.boxes:
        hours = (time.time() - b["t_rented"]) / 3600
        run.event("release_box", iid=b["iid"], hours=round(hours, 2), cost=round(hours * float(b["dph"] or 0), 2))
        if a.keep_box:
            vast.keep(b["iid"]) if b.get("kind") != "docker" else print(f"   kept container {b['name']}")
        elif b.get("kind") == "docker":
            remote.docker_destroy(b)
        else:
            vast.destroy(b["iid"])
    run.boxes = []


def node(run, a, box, name, **env):
    env.update(STEM=run.stem, LLAMA_COMMIT=getattr(a, "llama_commit", None), LLAMA_REPO=getattr(a, "llama_repo", None),
               REPO_SUFFIX=run.suffix or None, FORCE="1" if a.force else None, LOCAL_HUB=a.local_hub,
               CTX=getattr(a, "ctx", None), KLD_CHUNKS=getattr(a, "kld_chunks", None),
               IM_MAX_CHUNKS=getattr(a, "im_max_chunks", None))
    run.event("node", node=name, **{k: v for k, v in env.items() if v is not None and k != "LLAMA_REPO"})
    t0 = time.time()
    ok, info = remote.run_node(box, a.ssh_key, name, env, on_line=lambda l: print(f"  [{name}] {l}", flush=True),
                               max_hours=a.max_hours)
    run.event("node_done" if ok else "node_failed", node=name, minutes=round((time.time() - t0) / 60, 1), info=info)
    if not ok:
        raise SystemExit(f"{name} failed: {info}")
    return info


def ladder_stage(run, a):
    import ladder_gen
    inv = json.load(open(run.fetch(run.metrics, "dataset", "inventory.json")[0]))
    profile = ladder_gen.load_profile(os.path.join(PROFILES, f"{a.profile}.yaml"))
    ladder, problems = ladder_gen.build(profile, inv)
    print(f"\n{profile['name']} on {inv['arch']}, {inv['block_count']} blocks, mtp {inv['mtp_blocks']}, bands {ladder['bands']}")
    for r in ladder["rungs"]:
        print(f"  {r['label']:22s} {r['ftype']:8s} {len(r['rules']):3d} rules {r['predicted_gib']:7.2f} GiB "
              f"{r['predicted_bpw']:5.2f} BPW{'  (control)' if r['control'] else ''}")
    if problems:
        for label, errs in problems.items():
            print(f"  REFUSED {label}: {errs[:3]}")
        raise SystemExit("the profile does not fit this model: add roles or adjust types, then rerun")
    if not a.ladder_ok and input("\nupload this ladder and quantize? [y/N] ").strip().lower() != "y":
        raise SystemExit("stopped at the ladder review")
    out = os.path.join(run.dir, "ladder")
    ladder_gen.write(ladder, out)
    hub.updir(out, "ladder", run.metrics, "dataset", message=f"ladder from profile {profile['name']}")
    run.event("ladder", profile=profile["name"], rungs=len(ladder["rungs"]))


def results_stage(run):
    import results
    rows = sorted(run.fetch(run.metrics, "dataset", "results/rows/*.json"))
    merged = results.merge(rows)
    path = os.path.join(run.dir, "results.json")
    with open(path, "w") as f:
        json.dump(merged, f, indent=1)
        f.write("\n")
    hub.up(path, "results.json", run.metrics, "dataset", message="results.json")
    print(f"\n{'file':40s} {'GB':>6s} {'BPW':>5s} {'mean KLD':>9s} {'top-1':>6s}")
    for r in merged:
        q = r["quality"].get("neutral", {})
        print(f"{r['name']:40s} {r['size_gb']:6.2f} {r['bpw'] or 0:5.2f} {q.get('mean_kld', 0):9.5f} {q.get('top1_pct', 0):6.2f}")
    run.event("results", rows=len(merged))


CARD_INPUTS = ["results.json", "inventory.json", "ladder/*", "kld/base-neutral.manifest.txt", "logs/base-neutral.log",
               "logs/env-node_quant.txt", "logs/env-node_base.txt", "logs/llama-version.txt",
               "imatrix/params.txt", "imatrix/corpus-manifest.json"]


def card_stage(run, a):
    """Draft README from the metrics repo. The main repo gets it only while it has no README of its own."""
    import make_card
    d = os.path.join(run.dir, "card")
    for pat in CARD_INPUTS:
        try:
            hub.get(run.metrics, "dataset", "-", pat, d)
        except FileNotFoundError:
            pass
    args = argparse.Namespace(dir=d, model=run.model, stem=run.stem, main=run.main, metrics=run.metrics,
                              main_files=hub.ls(run.main, "model"), recommended=getattr(a, "recommended", None),
                              evalset="neutral")
    card = make_card.Card(args)
    path = os.path.join(d, "README.md")
    with open(path, "w") as f:
        f.write(card.render())
    hub.up(path, "card/README.draft.md", run.metrics, "dataset", message="card draft")
    where = "card/README.draft.md in the metrics repo"
    if getattr(a, "card_overwrite", False) or not hub.has(run.main, "model", "README.md"):
        hub.up(path, "README.md", run.main, "model", message="model card from results.json")
        where += " and README.md in the main repo"
    run.event("card", recommended=card.rec, todo=len(card.todo), written=where)
    for t in card.todo:
        print(f"   still for a person: {t}")


def cmd_status(a):
    run = Run(a, a.stem, journal=False)
    st, rungs = stage_state(run)
    print(f"{run.model} -> {run.main} / {run.metrics}")
    for k, v in st.items():
        print(f"  {k:8s} {'done' if v else '-'}")
    for label, done in rungs:
        print(f"    {label:22s} {'done' if done else '-'}")


def cmd_gguf(a):
    run = Run(a, a.stem)
    st, _ = stage_state(run, a.only_rungs)
    run.event("start", model=run.model, main=run.main, metrics=run.metrics, state=st)
    need_box = not (st["convert"] and st["base"] and st["imatrix"] and st["quant"])
    box = None
    try:
        if need_box:
            box = open_box(run, a, "gguf", size_b=None if a.local_box else model_bytes(run.model))
        if not st["convert"]:
            node(run, a, box, "node_convert", MODEL=run.model, KEEP_MTP=a.keep_mtp)
        if not st["base"]:
            node(run, a, box, "node_base")
        if not st["imatrix"]:
            ngpu = int(box["offer"].get("num_gpus") or 1)
            shards = max(1, min(a.im_shards or ngpu // 2, ngpu))
            node(run, a, box, "node_imatrix", RECIPE=a.recipe, IM_TOTAL=shards,
                 IM_INDEX=" ".join(map(str, range(shards))))
        if not stage_state(run)[0]["ladder"]:
            ladder_stage(run, a)
        if not stage_state(run, a.only_rungs)[0]["quant"]:
            node(run, a, box, "node_quant", PROFILE=a.profile, RUNGS=" ".join(a.only_rungs or []) or None)
        results_stage(run)
        card_stage(run, a)
    finally:
        close_boxes(run, a)


def cmd_ladder(a):
    ladder_stage(Run(a, a.stem), a)


def cmd_results(a):
    results_stage(Run(a, a.stem))


def cmd_card(a):
    card_stage(Run(a, a.stem), a)


def cmd_ablit(a):
    run = Run(a)
    target = f"AtomicChat/{run.stem}-abliterated{a.repo_suffix or ''}"
    q = a.gpu_query or "gpu_name in [H100_SXM,H100_NVL,H200] num_gpus=1 cpu_ram>=128 cuda_max_good>=13.0"
    disk = a.disk_gb or int(2.5 * model_bytes(run.model) / 1e9 + 80)
    try:
        box = open_box(run, a, "ablit", q, disk)
        env = dict(MODEL=run.model, TARGET=target, N_TRIALS=a.n_trials, TRIAL_INDEX=a.trial_index,
                   MAX_REFUSALS=a.max_refusals, MAX_KL=a.max_kl, RESUME="1" if a.resume else None, STEM=run.stem,
                   FORCE="1" if a.force else None)
        ok, info = remote.run_node(box, a.ssh_key, "node_abliterate", env,
                                   on_line=lambda l: print(f"  [ablit] {l}", flush=True), max_hours=a.max_hours)
        run.event("node_done" if ok else "node_failed", node="node_abliterate", info=info)
        if not ok:
            raise SystemExit(f"abliteration failed or gate not passed: {info}. Pick another TRIAL_INDEX and rerun with --resume --force")
    finally:
        close_boxes(run, a)
    print(f"\nnext: release.py gguf --model {target} --recipe <same recipe> --profile <same profile> --llama-commit ...")


def cmd_nvfp4(a):
    run = Run(a)
    target = f"AtomicChat/{run.stem}-NVFP4{a.repo_suffix or ''}"
    q = a.gpu_query or "gpu_name in [H100_SXM,H100_NVL,H200,RTX_PRO_6000_WS] num_gpus=1 cpu_ram>=128 cuda_max_good>=13.0"
    disk = a.disk_gb or int(1.4 * model_bytes(run.model) / 1e9 + 60)
    try:
        box = open_box(run, a, "nvfp4", q, disk)
        env = dict(MODEL=run.model, TARGET_NVFP4=target, RECIPE=a.recipe, STEM=run.stem, FORCE="1" if a.force else None)
        ok, info = remote.run_node(box, a.ssh_key, "node_nvfp4", env,
                                   on_line=lambda l: print(f"  [nvfp4] {l}", flush=True), max_hours=a.max_hours)
        run.event("node_done" if ok else "node_failed", node="node_nvfp4", info=info)
        if not ok:
            raise SystemExit(f"nvfp4 failed: {info}")
    finally:
        close_boxes(run, a)


def cmd_reap(a):
    boxes = vast.labeled("release:")
    if not boxes:
        print("no release: boxes")
    for b in boxes:
        age_h = (time.time() - (b.get("start_date") or time.time())) / 3600
        print(f"{b['id']}  {b.get('label')}  {b.get('actual_status')}  {age_h:.1f} h  ${b.get('dph_total', 0):.2f}/h")
        if a.all or input("  destroy? [y/N] ").strip().lower() == "y":
            vast.destroy(b["id"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, model=True):
        if model:
            p.add_argument("--model", required=True)
        p.add_argument("--stem", help="published name, default: the part of --model after the slash")
        p.add_argument("--repo-suffix", default="", help="e.g. -rehearsal, to publish next to the real repos")
        p.add_argument("--ssh-key", default=os.path.expanduser("~/.ssh/id_ed25519"))
        p.add_argument("--gpu-query")
        p.add_argument("--disk-gb", type=int)
        p.add_argument("--race", type=int, default=3)
        p.add_argument("--max-hours", type=float, default=12)
        p.add_argument("--keep-box", action="store_true", help="leave the box running on exit (debugging)")
        p.add_argument("--force", action="store_true")
        p.add_argument("--local-hub", help="keep the AtomicChat/ repos as folders here instead of on the hub")
        p.add_argument("--local-box", action="store_true", help="run the nodes in a local CPU container (needs --local-hub)")

    p = sub.add_parser("status"); common(p)
    p = sub.add_parser("gguf"); common(p)
    p.add_argument("--recipe", required=True, help="calib-corpora build, e.g. qwen3.8-27b")
    p.add_argument("--profile", required=True, help="profiles/<name>.yaml")
    p.add_argument("--llama-commit", required=True)
    p.add_argument("--llama-repo", default="https://github.com/ggml-org/llama.cpp")
    p.add_argument("--keep-mtp", default="1")
    p.add_argument("--im-shards", type=int)
    p.add_argument("--only-rungs", nargs="*")
    p.add_argument("--ladder-ok", action="store_true", help="skip the interactive ladder review")
    p.add_argument("--ctx", type=int, help="KLD context (default 4096, the published protocol)")
    p.add_argument("--kld-chunks", type=int, help="KLD chunks (default all); a smaller number is for test runs")
    p.add_argument("--im-max-chunks", type=int, help="cap the imatrix at this many chunks (test runs only)")
    p = sub.add_parser("ladder"); common(p)
    p.add_argument("--profile", required=True)
    p.add_argument("--ladder-ok", action="store_true")
    p = sub.add_parser("results"); common(p)
    p = sub.add_parser("card"); common(p)
    p.add_argument("--recommended", help="the rung the card runs in its examples, default AD-Q4_K_M")
    p.add_argument("--card-overwrite", action="store_true", help="replace the README the main repo already has")
    p = sub.add_parser("ablit"); common(p)
    p.add_argument("--n-trials", default="200")
    p.add_argument("--trial-index", default="0")
    p.add_argument("--max-refusals", default="5")
    p.add_argument("--max-kl", default="0.10")
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("nvfp4"); common(p)
    p.add_argument("--recipe", required=True)
    p = sub.add_parser("reap"); common(p, model=False)
    p.add_argument("--all", action="store_true")

    a = ap.parse_args()
    if a.local_hub:
        a.local_hub = os.path.abspath(os.path.expanduser(a.local_hub))
        os.makedirs(a.local_hub, exist_ok=True)
        os.environ["LOCAL_HUB"] = a.local_hub
    if a.local_box and not a.local_hub:
        sys.exit("--local-box needs --local-hub: the container can not write to the real hub without a token")
    if a.local_hub and a.cmd in ("gguf", "ablit", "nvfp4") and not a.local_box:
        sys.exit("--local-hub needs --local-box: a rented box can not see a folder on this machine")
    if not a.local_hub:
        load_token()
    {"status": cmd_status, "gguf": cmd_gguf, "ladder": cmd_ladder, "results": cmd_results, "card": cmd_card,
     "ablit": cmd_ablit, "nvfp4": cmd_nvfp4, "reap": cmd_reap}[a.cmd](a)


if __name__ == "__main__":
    main()
