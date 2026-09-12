"""
Layer-wise probe sweep across the residual stream.

Fits linear probes at each layer of the model to evaluate where surface-observable
variables (`exists_*`) versus memory-only variables (`locked_*`) are most decodable.

Uses episode-level splits consistent with probe.py and computes bootstrap confidence
intervals for the best-performing layers.

Usage:
    python wm/layer_sweep.py                    # evaluate all layers
    python wm/layer_sweep.py --stride 2         # evaluate every other layer
"""

from __future__ import annotations

import argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from wm.env import VARS_LOCKED, VARS_EXISTS
from wm.probe import last_assertion, build_text_features
from wm.stats import boot_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="data/acts.npz")
    ap.add_argument("--probes", default="data/probe_results.json")
    ap.add_argument("--out", default="data/layer_sweep.json")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    d = np.load(args.acts, allow_pickle=True)
    X = d["X"].astype(np.float32)
    meta = json.loads(d["meta"].item())
    P = json.load(open(args.probes))
    te = set(P["test_episodes"])

    ep_id = np.array([m["episode"] for m in meta])
    is_te = np.array([e in te for e in ep_id])
    N, L, D = X.shape
    layers = list(range(0, L, args.stride))
    print(f"[sweep] {N} turns, {L} layers, evaluating {len(layers)} of them")

    _, hists = build_text_features(meta)
    results = {"n_layers": L, "layers": layers, "vars": {}}

    for var in VARS_LOCKED + VARS_EXISTS:
        y = np.array([m["state_before"][var] for m in meta])
        if min(y.mean(), 1 - y.mean()) < 0.08:
            print(f"[sweep] skip {var} (base rate too extreme)")
            continue

        accs, aucs = [], []
        for li in layers:
            Xl = X[:, li, :]
            mu, sd = Xl[~is_te].mean(0), Xl[~is_te].std(0) + 1e-6
            Z = (Xl - mu) / sd
            clf = LogisticRegression(max_iter=2000, C=args.C,
                                     class_weight="balanced")
            clf.fit(Z[~is_te], y[~is_te])
            accs.append(float(balanced_accuracy_score(
                y[is_te], clf.predict(Z[is_te]))))
            aucs.append(float(roc_auc_score(
                y[is_te], clf.decision_function(Z[is_te]))))

        best_i = int(np.argmax(accs))
        best_layer = layers[best_i]

        # bootstrap CI at the best layer, resampling EPISODES not turns
        Xl = X[:, best_layer, :]
        mu, sd = Xl[~is_te].mean(0), Xl[~is_te].std(0) + 1e-6
        Z = (Xl - mu) / sd
        clf = LogisticRegression(max_iter=2000, C=args.C,
                                 class_weight="balanced")
        clf.fit(Z[~is_te], y[~is_te])
        pt, lo, hi = boot_ci(y[is_te], clf.predict(Z[is_te]),
                             balanced_accuracy_score, n_boot=1000,
                             groups=ep_id[is_te])

        entry = {"acc_by_layer": accs, "auc_by_layer": aucs,
                 "best_layer": best_layer, "best_acc": accs[best_i],
                 "best_acc_ci": [lo, hi],
                 "base_rate": float(max(y.mean(), 1 - y.mean())),
                 "kind": "locked" if var in VARS_LOCKED else "exists"}

        if var in VARS_LOCKED:
            fn = var[len("locked_"):]
            base = entry["base_rate"]
            r = np.array([
                (lambda v: v if v is not None else int(base >= 0.5))(
                    last_assertion(hists[i], fn)[0])
                for i in np.where(is_te)[0]])
            rp, rlo, rhi = boot_ci(y[is_te], r, balanced_accuracy_score,
                                   n_boot=1000, groups=ep_id[is_te])
            entry["regex"] = rp
            entry["regex_ci"] = [rlo, rhi]

        results["vars"][var] = entry
        extra = (f" regex={entry['regex']:.3f}" if "regex" in entry else "")
        print(f"[sweep] {var:<20} best L{best_layer:<3} "
              f"acc={accs[best_i]:.3f} [{lo:.3f},{hi:.3f}]{extra}")

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n[sweep] wrote {args.out}")
    print("Expect: exists_* peaking early (~L8), locked_* peaking late (L20-30).")


if __name__ == "__main__":
    main()
