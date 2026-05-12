#!/usr/bin/env python3
"""Compute bootstrap CI and paired sign-test p-values for Phase G/H results."""
import json
import math
import random
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results_fixed"

def proportion(lst):
    return sum(lst) / len(lst) if lst else 0.0

def percentile(sv, q):
    pos = (len(sv)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi: return sv[lo]
    return sv[lo]*(1-(pos-lo)) + sv[hi]*(pos-lo)

def bootstrap_diff_ci(baseline, compare, samples=10000, seed=42):
    rng = random.Random(seed)
    n = len(baseline)
    diffs = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(proportion([compare[i] for i in idx]) - proportion([baseline[i] for i in idx]))
    diffs.sort()
    return percentile(diffs, 0.025), percentile(diffs, 0.975)

def binomial_pvalue(successes, trials):
    if trials == 0: return 1.0
    tp = 0.5
    obs_p = math.comb(trials, successes)*(tp**successes)*((1-tp)**(trials-successes))
    return min(1.0, sum(
        math.comb(trials, k)*(tp**k)*((1-tp)**(trials-k))
        for k in range(trials+1)
        if math.comb(trials,k)*(tp**k)*((1-tp)**(trials-k)) <= obs_p*(1+1e-10)
    ))

def sign_test_pval(baseline, compare):
    imp = sum(1 for b,c in zip(baseline,compare) if c and not b)
    wor = sum(1 for b,c in zip(baseline,compare) if b and not c)
    d = imp + wor
    return binomial_pvalue(max(imp,wor), d)

def fmt_p(p):
    if p >= 0.999: return "$1.000$"
    if p < 1e-10: return f"$< 10^{{{math.floor(math.log10(p))}}}$"
    if p < 0.001:
        e = math.floor(math.log10(p))
        return f"${p/(10**e):.1f} \\\\times 10^{{{e}}}$"
    return f"${p:.3f}$"

def analyze(path, label):
    d = json.load(open(path))
    res = d["results"]
    def sc(lst): return [x["correct"] for x in sorted(lst, key=lambda x: x["id"])]
    base = sc(res["baseline"])
    base_acc = proportion(base)
    print(f"\n=== {label} ===")
    print(f"n={len(base)}, baseline={base_acc:.3f}")
    print("Appendix rows:")
    for tgt in ["middle","prefix","suffix"]:
        cmp = sc(res[tgt])
        delta = proportion(cmp) - base_acc
        lo, hi = bootstrap_diff_ci(base, cmp)
        p = sign_test_pval(base, cmp)
        s = lambda v: f"${'+' if v>=0 else ''}{v:+.3f}$"
        print(f"  & {tgt.capitalize():8s}  & {s(delta)} & [{s(lo)}, {s(hi)}] & {fmt_p(p)} \\\\")
    print("Accuracies:")
    for tgt in ["baseline","middle","prefix","suffix"]:
        print(f"  {tgt}: {proportion(sc(res[tgt])):.3f}")

if __name__ == "__main__":
    orig = RESULTS_DIR / "phaseG_qwen3b_orig_matched.json"
    ns =   RESULTS_DIR / "phaseG_qwen3b_ns_matched.json"
    if orig.exists(): analyze(orig, "Phase G (GSM8K-v1 orig, Qwen-3B matched)")
    else: print(f"Not yet: {orig}")
    if ns.exists(): analyze(ns, "Phase H (GSM8K-neutral-stripped, Qwen-3B matched)")
    else: print(f"Not yet: {ns}")
