"""Synthetic llama-perplexity --kl-divergence logs, in the layout of the real ones."""


def kld_log(kld, top1, ppl_base=5.421271, n_ctx=4096):
    """A finished log whose per chunk KLD and top-1 are the given lists (printed as running means)."""
    lines = [f"0.04.522.763 I kl_divergence: computing over {len(kld)} chunks, n_ctx={n_ctx}, batch_size=2048, n_seq=1",
             "",
             "chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p"]
    sk = st = 0.0
    for i, (k, t) in enumerate(zip(kld, top1), 1):
        sk += k
        st += t
        lines.append(f"{i:4d}      5.4728 ±    0.0502       0.00946 ±    0.00069       {sk / i:.5f} ±    0.00010"
                     f"     3.361 ±  0.035 %    {st / i:.3f} ±  0.089 %")
    lines += ["====== Perplexity statistics ======",
              f"Mean PPL(base)                :   {ppl_base:.6f} ±   0.049605",
              "",
              "====== KL divergence statistics ======",
              f"Mean    KLD:   {sk / len(kld):.6f} ±   0.000101",
              f"Same top p: {st / len(kld):.3f} ± 0.089 %"]
    return "\n".join(lines) + "\n"


def write_kld_log(path, kld, top1=None, **kw):
    with open(path, "w") as f:
        f.write(kld_log(kld, top1 or [95.0] * len(kld), **kw))
    return str(path)
