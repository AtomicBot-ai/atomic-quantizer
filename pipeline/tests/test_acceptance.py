"""The acceptance script passes the August release against itself and catches a drift."""
import acceptance_qwen38 as acc


def rows_from(ref):
    return [{"label": k, "size_bytes": v["size_bytes"],
             "quality": {"neutral": {"mean_kld": v["mean_kld"], "top1_pct": v["top1_pct"]}}} for k, v in ref.items()]


def test_august_against_itself_passes():
    ref = acc.august()
    assert abs(ref["AD-Q4_K_M"]["mean_kld"] - 0.011262) < 1e-6
    assert abs(ref["AD-IQ1_M"]["mean_kld"] - 0.342121) < 1e-6
    lines, bad = acc.compare(rows_from(ref), ref)
    assert not bad
    assert len(lines) == 1 + len(acc.RUNGS)


def test_drift_and_gaps_fail():
    ref = acc.august()
    rows = rows_from(ref)
    for r in rows:
        if r["label"] == "AD-Q4_K_M":
            r["quality"]["neutral"]["mean_kld"] *= 1.2
        if r["label"] == "AD-IQ2_XS":
            r["size_bytes"] *= 1.05
    rows = [r for r in rows if r["label"] != "AD-IQ1_M"]
    _, bad = acc.compare(rows, ref)
    assert any(b.startswith("AD-Q4_K_M: KLD") for b in bad)
    assert any(b.startswith("AD-IQ2_XS: size") for b in bad)
    assert any(b.startswith("AD-IQ1_M: not in") for b in bad)


def test_q8_0_noise_floor():
    """0.00064 +15 % is 0.0001 absolute: noise at this level, not a regression."""
    ref = acc.august()
    rows = rows_from(ref)
    for r in rows:
        if r["label"] == "Q8_0":
            r["quality"]["neutral"]["mean_kld"] *= 1.15
    _, bad = acc.compare(rows, ref)
    assert not bad
