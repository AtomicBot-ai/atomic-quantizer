#!/usr/bin/env python3
"""Release driver: from a hub model id to published AD GGUF, abliterated and NVFP4 builds.

    release.py status --model Qwen/Qwen3.8-27B
    release.py gguf   --model Qwen/Qwen3.8-27B --recipe qwen3.8-27b --profile dense-hybrid \
                      --llama-commit 1692f9e50bb2... [--repo-suffix -rehearsal] [--ladder-ok]
    release.py ladder --model ... --profile ...          # generate + review + upload ladder/ only
    release.py results --model ...                       # merge results/rows into results.json
    release.py ablit  --model Qwen/Qwen3.8-27B            # heretic, then run gguf on the result
    release.py nvfp4  --model Qwen/Qwen3.8-27B --recipe qwen3.8-27b
    release.py reap   [--all]                            # destroy boxes labelled release:

State lives on the hub; every stage is skipped when its outputs exist, so a run
that died is resumed by running the same command again. Needs `vastai set
api-key` done once, ~/.config/atomic-pipeline/hf_env (0600, `export HF_TOKEN=...`)
and an ssh key (default ~/.ssh/id_ed25519).
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
import remote  # noqa: E402
import vast  # noqa: E402

PROFILES = os.path.join(HERE, "..", "profiles")
RUNS = os.path.join(HERE, "..", "runs")


def load_token():
    with open(remote.TOKEN_FILE) as f:
        m = re.search(r"HF_TOKEN=['\"]?(hf_[A-Za-z0-9]+)", f.read())
    if not m:
        sys.exit(f"no HF_TOKEN in {remote.TOKEN_FILE}")
    os.environ["HF_TOKEN"] = m.group(1)


def api():
    from huggingface_hub import HfApi
    return HfApi()


class Run:
    def __init__(self, a, stem=None):
        self.model = a.model
        self.stem = stem or a.model.split("/")[-1]
        suffix = a.repo_suffix or ""
        self.main = f"AtomicChat/{self.stem}-GGUF{suffix}"
        self.metrics = f"AtomicChat/{self.stem}-GGUF-metrics{suffix}"
        self.suffix = suffix
        self.dir = os.path.join(RUNS, f"{self.stem}{suffix}-{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(self.dir, exist_ok=True)
        self.log = open(os.path.join(self.dir, "run.jsonl"), "a")
        self.boxes = []

    def event(self, kind, **kw):
        rec = {"t": time.strftime("%FT%TZ", time.gmtime()), "event": kind, **kw}
        self.log.write(json.dumps(rec) + "\n")
        self.log.flush()
        print(f"== {kind} " + " ".join(f"{k}={v}" for k, v in kw.items() if k not in ("offer",)), flush=True)

    def files(self, repo, kind):
        try:
            return api().list_repo_files(repo, repo_type=kind)
        except Exception:
            return []

    def has(self, repo, kind, pattern):
        return any(fnmatch.fnmatch(f, pattern) for f in self.files(repo, kind))


def stage_state(run, profile=None):
    m = run.files(run.metrics, "dataset")
    g = run.files(run.main, "model")
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
        from huggingface_hub import hf_hub_download
        lad = json.load(open(hf_hub_download(run.metrics, "ladder/ladder.json", repo_type="dataset")))
        for r in lad["rungs"]:
            if r.get("control"):
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


def rent(run, a, stage, size_b):
    q, disk = box_plan(size_b, a.gpu_query, a.disk_gb)
    run.event("rent", stage=stage, query=q, disk_gb=disk)
    box = vast.rent(q, disk, f"release:{os.path.basename(run.dir)}:{stage}", a.ssh_key, race=a.race)
    run.event("box", iid=box["iid"], gpu=box["gpu"], dph=box["dph"], host=box["host"])
    remote.bootstrap(box, a.ssh_key)
    run.boxes.append(box)
    return box


def node(run, a, box, name, **env):
    env.update(STEM=run.stem, LLAMA_COMMIT=a.llama_commit, LLAMA_REPO=a.llama_repo,
               REPO_SUFFIX=run.suffix or None, FORCE="1" if a.force else None)
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
    from huggingface_hub import hf_hub_download
    inv_path = hf_hub_download(run.metrics, "inventory.json", repo_type="dataset", local_dir=run.dir)
    inv = json.load(open(inv_path))
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
    api().upload_folder(folder_path=out, path_in_repo="ladder", repo_id=run.metrics, repo_type="dataset",
                        commit_message=f"ladder from profile {profile['name']}")
    run.event("ladder", profile=profile["name"], rungs=len(ladder["rungs"]))


def results_stage(run):
    import results
    from huggingface_hub import snapshot_download
    d = snapshot_download(run.metrics, repo_type="dataset", allow_patterns=["results/rows/*.json"], local_dir=run.dir)
    rows = sorted(os.path.join(d, "results", "rows", f) for f in os.listdir(os.path.join(d, "results", "rows")))
    merged = results.merge(rows)
    path = os.path.join(run.dir, "results.json")
    with open(path, "w") as f:
        json.dump(merged, f, indent=1)
        f.write("\n")
    api().upload_file(path_or_fileobj=path, path_in_repo="results.json", repo_id=run.metrics, repo_type="dataset",
                      commit_message="results.json")
    print(f"\n{'file':40s} {'GB':>6s} {'BPW':>5s} {'mean KLD':>9s} {'top-1':>6s}")
    for r in merged:
        q = r["quality"].get("neutral", {})
        print(f"{r['name']:40s} {r['size_gb']:6.2f} {r['bpw'] or 0:5.2f} {q.get('mean_kld', 0):9.5f} {q.get('top1_pct', 0):6.2f}")
    run.event("results", rows=len(merged))


def cmd_status(a):
    run = Run(a)
    st, rungs = stage_state(run)
    print(f"{run.model} -> {run.main} / {run.metrics}")
    for k, v in st.items():
        print(f"  {k:8s} {'done' if v else '-'}")
    for label, done in rungs:
        print(f"    {label:22s} {'done' if done else '-'}")


def cmd_gguf(a):
    run = Run(a, a.stem)
    st, _ = stage_state(run)
    run.event("start", model=run.model, main=run.main, metrics=run.metrics, state=st)
    need_box = not (st["convert"] and st["base"] and st["imatrix"] and st["quant"])
    box = None
    try:
        if need_box:
            box = rent(run, a, "gguf", model_bytes(run.model))
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
        if not stage_state(run)[0]["quant"]:
            node(run, a, box, "node_quant", PROFILE=a.profile, RUNGS=" ".join(a.only_rungs or []) or None)
        results_stage(run)
    finally:
        for b in run.boxes:
            hours = (time.time() - b["t_rented"]) / 3600
            run.event("release_box", iid=b["iid"], hours=round(hours, 2), cost=round(hours * float(b["dph"] or 0), 2))
            if a.keep_box:
                vast.keep(b["iid"])
            else:
                vast.destroy(b["iid"])


def cmd_ladder(a):
    ladder_stage(Run(a, a.stem), a)


def cmd_results(a):
    results_stage(Run(a, a.stem))


def cmd_ablit(a):
    run = Run(a)
    target = f"AtomicChat/{run.stem}-abliterated{a.repo_suffix or ''}"
    size = model_bytes(run.model)
    q = a.gpu_query or "gpu_name in [H100_SXM,H100_NVL,H200] num_gpus=1 cpu_ram>=128 cuda_max_good>=13.0"
    disk = a.disk_gb or int(2.5 * size / 1e9 + 80)
    run.event("rent", stage="ablit", query=q, disk_gb=disk)
    box = vast.rent(q, disk, f"release:{os.path.basename(run.dir)}:ablit", a.ssh_key, race=a.race)
    run.boxes.append(box)
    try:
        remote.bootstrap(box, a.ssh_key)
        env = dict(MODEL=run.model, TARGET=target, N_TRIALS=a.n_trials, TRIAL_INDEX=a.trial_index,
                   MAX_REFUSALS=a.max_refusals, MAX_KL=a.max_kl, RESUME="1" if a.resume else None, STEM=run.stem,
                   FORCE="1" if a.force else None)
        ok, info = remote.run_node(box, a.ssh_key, "node_abliterate", env,
                                   on_line=lambda l: print(f"  [ablit] {l}", flush=True), max_hours=a.max_hours)
        run.event("node_done" if ok else "node_failed", node="node_abliterate", info=info)
        if not ok:
            raise SystemExit(f"abliteration failed or gate not passed: {info}. Pick another TRIAL_INDEX and rerun with --resume --force")
    finally:
        hours = (time.time() - box["t_rented"]) / 3600
        run.event("release_box", iid=box["iid"], hours=round(hours, 2), cost=round(hours * float(box["dph"] or 0), 2))
        vast.keep(box["iid"]) if a.keep_box else vast.destroy(box["iid"])
    print(f"\nnext: release.py gguf --model {target} --recipe <same recipe> --profile <same profile> --llama-commit ...")


def cmd_nvfp4(a):
    run = Run(a)
    target = f"AtomicChat/{run.stem}-NVFP4{a.repo_suffix or ''}"
    size = model_bytes(run.model)
    q = a.gpu_query or "gpu_name in [H100_SXM,H100_NVL,H200,RTX_PRO_6000_WS] num_gpus=1 cpu_ram>=128 cuda_max_good>=13.0"
    disk = a.disk_gb or int(1.4 * size / 1e9 + 60)
    run.event("rent", stage="nvfp4", query=q, disk_gb=disk)
    box = vast.rent(q, disk, f"release:{os.path.basename(run.dir)}:nvfp4", a.ssh_key, race=a.race)
    run.boxes.append(box)
    try:
        remote.bootstrap(box, a.ssh_key)
        env = dict(MODEL=run.model, TARGET_NVFP4=target, RECIPE=a.recipe, STEM=run.stem, FORCE="1" if a.force else None)
        ok, info = remote.run_node(box, a.ssh_key, "node_nvfp4", env,
                                   on_line=lambda l: print(f"  [nvfp4] {l}", flush=True), max_hours=a.max_hours)
        run.event("node_done" if ok else "node_failed", node="node_nvfp4", info=info)
        if not ok:
            raise SystemExit(f"nvfp4 failed: {info}")
    finally:
        hours = (time.time() - box["t_rented"]) / 3600
        run.event("release_box", iid=box["iid"], hours=round(hours, 2), cost=round(hours * float(box["dph"] or 0), 2))
        vast.keep(box["iid"]) if a.keep_box else vast.destroy(box["iid"])


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
    p = sub.add_parser("ladder"); common(p)
    p.add_argument("--profile", required=True)
    p.add_argument("--ladder-ok", action="store_true")
    p = sub.add_parser("results"); common(p)
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
    load_token()
    {"status": cmd_status, "gguf": cmd_gguf, "ladder": cmd_ladder, "results": cmd_results,
     "ablit": cmd_ablit, "nvfp4": cmd_nvfp4, "reap": cmd_reap}[a.cmd](a)


if __name__ == "__main__":
    main()
