#!/usr/bin/env python3
"""Compute within-stable-subset analysis for 7B N=300 experiments."""
import json
import sys
from scipy.stats import binomtest

def load_results(path):
    with open(path) as f:
        return json.load(f)

def analyze_within_stable(orig_results, strip_results):
    """Match examples by ID and find within-stable subset."""
    
    # Build lookup by example ID from runs structure
    orig_by_id = {}
    strip_by_id = {}
    
    for run in orig_results.get('runs', []):
        target = run['summary']['corruption_target']
        orig_by_id[target] = {}
        for trial in run['trials']:
            eid = trial['example_id']
            orig_by_id[target][eid] = trial
    
    for run in strip_results.get('runs', []):
        target = run['summary']['corruption_target']
        strip_by_id[target] = {}
        for trial in run['trials']:
            # Use original_id if available (stripped datasets use different ID prefix)
            eid = trial.get('example_metadata', {}).get('original_id', trial['example_id'])
            strip_by_id[target][eid] = trial
    
    # For within-stable, we need examples that get baseline correct in BOTH formats
    # Use suffix target since that's the primary comparison
    orig_suffix = orig_by_id.get('suffix', {})
    strip_suffix = strip_by_id.get('suffix', {})
    
    # Find matched IDs
    matched_ids = set(orig_suffix.keys()) & set(strip_suffix.keys())
    print(f"\nMatched examples (suffix target): {len(matched_ids)}")
    
    # Within-stable: correct baseline in BOTH conditions
    stable_ids = []
    for eid in matched_ids:
        orig_correct = orig_suffix[eid].get('baseline_correct', False)
        strip_correct = strip_suffix[eid].get('baseline_correct', False)
        if orig_correct and strip_correct:
            stable_ids.append(eid)
    
    print(f"Within-stable subset: {len(stable_ids)} (correct in both formats)")
    
    if not stable_ids:
        print("ERROR: No stable examples found. Check data format.")
        return
    
    # Compute suffix sensitivity on stable subset
    orig_suffix_degrade = 0
    orig_suffix_improve = 0
    orig_suffix_same = 0
    strip_suffix_degrade = 0
    strip_suffix_improve = 0
    strip_suffix_same = 0
    
    orig_baseline_sum = 0
    orig_corrupt_sum = 0
    strip_baseline_sum = 0
    strip_corrupt_sum = 0
    
    for eid in stable_ids:
        # Original format
        o = orig_suffix[eid]
        ob = 1 if o.get('baseline_correct', False) else 0
        oc = 1 if o.get('corrupted_correct', False) else 0
        orig_baseline_sum += ob
        orig_corrupt_sum += oc
        if oc < ob: orig_suffix_degrade += 1
        elif oc > ob: orig_suffix_improve += 1
        else: orig_suffix_same += 1
        
        # Stripped format
        s = strip_suffix[eid]
        sb = 1 if s.get('baseline_correct', False) else 0
        sc = 1 if s.get('corrupted_correct', False) else 0
        strip_baseline_sum += sb
        strip_corrupt_sum += sc
        if sc < sb: strip_suffix_degrade += 1
        elif sc > sb: strip_suffix_improve += 1
        else: strip_suffix_same += 1
    
    N = len(stable_ids)
    orig_base_acc = orig_baseline_sum / N
    orig_corr_acc = orig_corrupt_sum / N
    orig_delta = orig_corr_acc - orig_base_acc
    strip_base_acc = strip_baseline_sum / N
    strip_corr_acc = strip_corrupt_sum / N
    strip_delta = strip_corr_acc - strip_base_acc
    
    # Sign test for original
    if orig_suffix_degrade + orig_suffix_improve > 0:
        orig_p = binomtest(orig_suffix_degrade, orig_suffix_degrade + orig_suffix_improve, 0.5).pvalue
    else:
        orig_p = 1.0
    
    # Sign test for stripped
    if strip_suffix_degrade + strip_suffix_improve > 0:
        strip_p = binomtest(strip_suffix_degrade, strip_suffix_degrade + strip_suffix_improve, 0.5).pvalue
    else:
        strip_p = 1.0
    
    print(f"\n=== WITHIN-STABLE SUBSET (N={N}) ===")
    print(f"\nOriginal format (suffix corruption):")
    print(f"  Baseline acc: {orig_base_acc:.3f}")
    print(f"  Corrupted acc: {orig_corr_acc:.3f}")
    print(f"  Delta_suffix: {orig_delta:+.3f}")
    print(f"  Sign test: {orig_suffix_degrade} degradations, {orig_suffix_improve} improvements")
    print(f"  p-value: {orig_p:.2e}")
    
    print(f"\nStripped format (suffix corruption):")
    print(f"  Baseline acc: {strip_base_acc:.3f}")
    print(f"  Corrupted acc: {strip_corr_acc:.3f}")
    print(f"  Delta_suffix: {strip_delta:+.3f}")
    print(f"  Sign test: {strip_suffix_degrade} degradations, {strip_suffix_improve} improvements")
    print(f"  p-value: {strip_p:.2e}")
    
    if abs(orig_delta) > 0 and abs(strip_delta) > 0:
        attenuation = abs(orig_delta) / abs(strip_delta)
        print(f"\nAttenuation ratio: {attenuation:.1f}x")
    elif abs(strip_delta) == 0:
        print(f"\nAttenuation: infinite (stripped delta = 0)")
    
    # Also compute FW for both
    if orig_base_acc > 0:
        orig_fw = orig_corr_acc / orig_base_acc
        print(f"\nOriginal FW: {orig_fw:.3f}")
    if strip_base_acc > 0:
        strip_fw = strip_corr_acc / strip_base_acc
        print(f"Stripped FW: {strip_fw:.3f}")
    
    # Also do prefix analysis on stable subset
    print("\n=== PREFIX ON STABLE SUBSET ===")
    for target in ['prefix', 'middle']:
        orig_t = orig_by_id.get(target, {})
        strip_t = strip_by_id.get(target, {})
        
        o_deg = o_imp = o_same = 0
        s_deg = s_imp = s_same = 0
        o_base = o_corr = s_base = s_corr = 0
        
        for eid in stable_ids:
            if eid in orig_t:
                ob = 1 if orig_t[eid].get('baseline_correct', False) else 0
                oc = 1 if orig_t[eid].get('corrupted_correct', False) else 0
                o_base += ob; o_corr += oc
                if oc < ob: o_deg += 1
                elif oc > ob: o_imp += 1
                else: o_same += 1
            if eid in strip_t:
                sb = 1 if strip_t[eid].get('baseline_correct', False) else 0
                sc = 1 if strip_t[eid].get('corrupted_correct', False) else 0
                s_base += sb; s_corr += sc
                if sc < sb: s_deg += 1
                elif sc > sb: s_imp += 1
                else: s_same += 1
        
        print(f"\n{target.upper()} target:")
        print(f"  Original: baseline={o_base/N:.3f} corrupted={o_corr/N:.3f} delta={((o_corr-o_base)/N):+.3f} ({o_deg} deg, {o_imp} imp)")
        print(f"  Stripped: baseline={s_base/N:.3f} corrupted={s_corr/N:.3f} delta={((s_corr-s_base)/N):+.3f} ({s_deg} deg, {s_imp} imp)")
    
    # Aggregate results (all N=300)
    print("\n\n=== AGGREGATE RESULTS (ALL N=300) ===")
    for target in ['middle', 'prefix', 'suffix']:
        orig_exs = orig_by_id.get(target, {})
        strip_exs = strip_by_id.get(target, {})
        
        for label, exs in [('Original', orig_exs), ('Stripped', strip_exs)]:
            n = len(exs)
            if n == 0: continue
            base_c = sum(1 for e in exs.values() if e.get('baseline_correct', False))
            corr_c = sum(1 for e in exs.values() if e.get('corrupted_correct', False))
            deg = sum(1 for e in exs.values() if not e.get('corrupted_correct', False) and e.get('baseline_correct', False))
            imp = sum(1 for e in exs.values() if e.get('corrupted_correct', False) and not e.get('baseline_correct', False))
            base_acc = base_c / n
            corr_acc = corr_c / n
            delta = corr_acc - base_acc
            print(f"  {label} {target}: N={n} baseline={base_acc:.3f} corrupted={corr_acc:.3f} delta={delta:+.3f} ({deg} deg, {imp} imp)")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <original_results.json> <stripped_results.json>")
        sys.exit(1)
    
    orig = load_results(sys.argv[1])
    strip = load_results(sys.argv[2])
    analyze_within_stable(orig, strip)
