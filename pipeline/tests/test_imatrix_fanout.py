"""imatrix shards over several boxes: the plan, the two phases, a failing box."""
import os
import sys
import threading
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "driver"))
import release  # noqa: E402


def box(iid, gpus):
    return {"iid": iid, "offer": {"num_gpus": gpus}}


def test_one_shard_per_gpu_pair():
    plan, total = release.shard_plan([box("b0", 4), box("b1", 4), box("b2", 8)])
    assert total == 8
    assert [(b["iid"], idx) for b, idx in plan] == [("b0", [0, 1]), ("b1", [2, 3]), ("b2", [4, 5, 6, 7])]


def test_fixed_total_is_shared_in_proportion():
    plan, total = release.shard_plan([box("b0", 4), box("b1", 4)], total=5)
    assert total == 5 and [len(idx) for _, idx in plan] == [3, 2] and plan[0][1][0] == 0


def test_fewer_shards_than_boxes_leaves_boxes_out():
    plan, total = release.shard_plan([box("b0", 2), box("b1", 2), box("b2", 2)], total=2)
    assert total == 2 and [b["iid"] for b, _ in plan] == ["b0", "b1"]


class Run:
    def __init__(self, boxes):
        self.events, self.boxes, self.lock = [], list(boxes), threading.Lock()

    def event(self, kind, **kw):
        self.events.append((kind, kw))


def stage(monkeypatch, boxes, fail_on=None, calls=None, **opts):
    calls, closed = ([] if calls is None else calls), []

    def node(run, a, b, name, **env):
        calls.append((b["iid"], env.get("IM_INDEX"), env.get("IM_MERGE"), env["IM_TOTAL"]))
        if b["iid"] == fail_on:
            raise SystemExit("node_imatrix failed: box lost")

    monkeypatch.setattr(release, "node", node)
    monkeypatch.setattr(release, "close_box", lambda run, a, b: closed.append(b["iid"]))
    a = types.SimpleNamespace(recipe="r", im_shards=None, ladder_ok=opts.get("ladder_ok", False),
                              quant_boxes=opts.get("quant_boxes", 1))
    run = Run(boxes)
    kept = release.imatrix_stage(run, a, boxes[0], boxes[1:])
    return calls, closed, kept


def test_shards_side_by_side_then_merge_on_box_0(monkeypatch):
    boxes = [box("b0", 4), box("b1", 4)]
    calls, closed, kept = stage(monkeypatch, boxes)
    phase1, merge = calls[:2], calls[2:]
    assert sorted(phase1) == [("b0", "0 1", 0, 4), ("b1", "2 3", 0, 4)]
    assert merge == [("b0", None, "only", 4)]
    assert closed == ["b1"] and kept == []


def test_extra_boxes_stay_for_the_quant_stage(monkeypatch):
    boxes = [box("b0", 4), box("b1", 4), box("b2", 4)]
    _, closed, kept = stage(monkeypatch, boxes, ladder_ok=True, quant_boxes=4)
    assert closed == [] and [b["iid"] for b in kept] == ["b1", "b2"]


def test_a_failed_box_stops_before_the_merge(monkeypatch):
    boxes = [box("b0", 4), box("b1", 4)]
    calls = []
    with pytest.raises(SystemExit, match="rerun to compute only the missing shards"):
        stage(monkeypatch, boxes, fail_on="b1", calls=calls)
    assert len(calls) == 2 and all(m == 0 for _, _, m, _ in calls)   # both boxes ran, no merge started


def test_single_box_is_the_old_path(monkeypatch):
    calls, _, kept = stage(monkeypatch, [box("b0", 4)])
    assert calls == [("b0", "0 1", None, 2)] and kept == []
