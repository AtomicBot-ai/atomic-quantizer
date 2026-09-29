"""quant_fanout: rungs over several boxes without a box or a hub.

The boxes are dicts, a rung's work is a sleep, rental is a callable that may fail.
What is checked: every rung runs exactly once, extra boxes join late and are
released when the queue is dry, a failed rental costs nothing but its share, and
the first failed rung stops new rungs from starting.
"""
import os
import sys
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "driver"))
import release  # noqa: E402


class FakeRun:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def event(self, kind, **kw):
        with self.lock:
            self.events.append((kind, kw))


def fanout(rungs, extra, open_extra, work, first={"iid": "first"}):
    run, released = FakeRun(), []
    failed = release.quant_fanout(run, types.SimpleNamespace(), first, [(r, 0) for r in rungs], extra,
                                  open_extra, work, released.append)
    return run, failed, released


def test_every_rung_once_and_extra_boxes_released():
    done, lock = [], threading.Lock()

    def work(box, label):
        time.sleep(0.02)
        with lock:
            done.append((box["iid"], label))

    def open_extra(i):
        time.sleep(0.03 * i)   # rentals finish at different times
        return {"iid": f"x{i}"}

    rungs = [f"R{i}" for i in range(12)]
    _, failed, released = fanout(rungs, 3, open_extra, work)
    assert failed == []
    assert sorted(l for _, l in done) == sorted(rungs)
    assert {b for b, _ in done} >= {"first", "x1"}
    assert sorted(b["iid"] for b in released) == ["x1", "x2", "x3"]   # the first box is the caller's


def test_queue_order_is_kept():
    order = []
    _, failed, _ = fanout(["Q8_0", "AD-Q4_K_M", "AD-IQ1_M"], 0, None, lambda b, l: order.append(l))
    assert failed == [] and order == ["Q8_0", "AD-Q4_K_M", "AD-IQ1_M"]


def test_failed_rental_costs_only_its_share():
    def open_extra(i):
        if i == 1:
            raise RuntimeError("no offers")
        return {"iid": f"x{i}"}

    done = []
    run, failed, released = fanout([f"R{i}" for i in range(6)], 2, open_extra,
                                   lambda b, l: (time.sleep(0.01), done.append(l)))
    assert failed == [] and len(done) == 6
    assert [b["iid"] for b in released] == ["x2"]
    assert any(k == "quant_box_failed" for k, _ in run.events)


def test_first_failure_stops_new_rungs():
    started = []

    def work(box, label):
        started.append(label)
        if label == "R1":
            raise SystemExit("R1: verify refused")
        time.sleep(0.01)

    _, failed, _ = fanout([f"R{i}" for i in range(10)], 0, None, work)
    assert failed == [("R1", "R1: verify refused")]
    assert started == ["R0", "R1"]


def test_no_box_at_all_reports_the_rungs():
    def open_extra(i):
        raise RuntimeError("no offers")
    _, failed, released = fanout(["R0", "R1"], 2, open_extra, lambda b, l: None, first=None)
    assert failed and failed[0][1] == "no box was left to take it" and released == []


def test_quant_box_plan_keeps_the_card_and_fits_the_reference():
    q, disk = release.quant_box_plan(55.6e9)
    assert "RTX_5090" in q and "cuda_max_good" in q
    assert disk >= 55 + 2 * 88 + 30   # BF16, the reference while its parts are joined, Q8_0


def test_batches_go_to_one_node_call():
    calls, lock = [], threading.Lock()

    def work(box, labels):
        time.sleep(0.01)
        with lock:
            calls.append((box["iid"], labels))

    rungs = [f"R{i}" for i in range(7)]
    run, released = FakeRun(), []
    failed = release.quant_fanout(run, types.SimpleNamespace(), {"iid": "first"}, [(r, 0) for r in rungs], 0,
                                  None, work, released.append, batch=3)
    assert failed == []
    assert [len(l.split()) for _, l in calls] == [3, 3, 1]
    assert sorted(x for _, l in calls for x in l.split()) == sorted(rungs)


def test_ready_boxes_start_at_once_and_are_released():
    done, lock = [], threading.Lock()

    def work(box, label):
        time.sleep(0.02)
        with lock:
            done.append((box["iid"], label))

    rented = []
    run, released = FakeRun(), []
    failed = release.quant_fanout(run, types.SimpleNamespace(), {"iid": "first"}, [(f"R{i}", 0) for i in range(6)], 0,
                                  lambda i: rented.append(i), work, released.append, ready=[{"iid": "kept1"}])
    assert failed == [] and rented == []
    assert {b for b, _ in done} == {"first", "kept1"}
    assert [b["iid"] for b in released] == ["kept1"]
