"""
Within-condition probe evaluation.

Evaluates whether the model maintains world state in its internal activations
under different scratchpad conditions (thinking, no-thinking, terse), trained and
tested within each condition independently.

Compares decodability of:
  - Surface-observable variables (`exists_*`): repeatedly observable in context via `ls`.
  - Memory-only variables (`locked_*`): knowable only by integrating historical events.

Uses episode-level splits to prevent leakage across turns.

Usage:
    python wm/within_condition.py
"""

from __future__ import annotations

import argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from wm.env import VARS_LOCKED, VARS_EXISTS
from wm.probe import last_assertion, build_text_features
from wm.stats import boot_ci

CONDS = [("thinking", "data/acts.npz"),
         ("nothink", "data/acts_nothink.npz"),
         ("terse", "data/acts_terse.npz")]


def run_condition(name, path, layers, test_frac=0.3):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    meta = json.loads(d["meta"].item())
    ep_id = np.array([m["episode"] for m in meta])
    uniq = np.unique(ep_id)
    rng = np.random.RandomState(0)
    rng.shuffle(uniq)
    te_eps = set(uniq[: max(1, int(len(uniq) * test_frac))].tolist())
    is_te = np.array([e in te_eps for e in ep_id])
    _, hists = build_text_features(meta)
    print(f"\n=== {name}: {len(meta)} turns, {len(uniq)} episodes "
          f"({is_te.sum()} test) ===")

    out = {}
    for var in VARS_LOCKED + VARS_EXISTS:
        y = np.array([m["state_before"][var] for m in meta])
        if min(y.mean(), 1 - y.mean()) < 0.08:
            continue
        best = None
        for li in layers:
            if li >= X.shape[1]:
                continue
            Xl = X[:, li, :]
            mu, sd = Xl[~is_te].mean(0), Xl[~is_te].std(0) + 1e-6
            Z = (Xl - mu) / sd
            clf = LogisticRegression(max_iter=2000, C=1.0,
                                     class_weight="balanced")
            clf.fit(Z[~is_te], y[~is_te])
            auc = roc_auc_score(y[is_te], clf.decision_function(Z[is_te]))
            acc = balanced_accuracy_score(y[is_te], clf.predict(Z[is_te]))
            if best is None or auc > best["auroc"]:
                best = {"layer": li, "auroc": float(auc), "bal_acc": float(acc),
                        "_Z": Z, "_clf": clf}
        if best is None:
            continue
        _, lo, hi = boot_ci(y[is_te],
                            best["_clf"].decision_function(best["_Z"][is_te]),
                            roc_auc_score, n_boot=800, groups=ep_id[is_te])
        entry = {k: v for k, v in best.items() if not k.startswith("_")}
        entry.update({"auroc_ci": [lo, hi],
                      "base_rate": float(max(y.mean(), 1 - y.mean())),
                      "kind": "locked" if var in VARS_LOCKED else "exists"})
        if var in VARS_LOCKED:
            fn = var[len("locked_"):]
            b = entry["base_rate"]
            r = np.array([(lambda v: v if v is not None else int(b >= 0.5))(
                last_assertion(hists[i], fn)[0]) for i in np.where(is_te)[0]])
            entry["regex"] = float(balanced_accuracy_score(y[is_te], r))
        out[var] = entry
        print(f"  {var:<20} L{entry['layer']:<3} AUROC={entry['auroc']:.3f} "
              f"[{lo:.3f},{hi:.3f}]  bal-acc={entry['bal_acc']:.3f}"
              + (f"  regex={entry['regex']:.3f}" if "regex" in entry else ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="*",
                    default=[1, 2, 4, 8, 16, 20, 22, 26, 30])
    ap.add_argument("--out", default="data/within_condition.json")
    args = ap.parse_args()

    import os
    res = {}
    for name, path in CONDS:
        if not os.path.exists(path):
            print(f"[wc] missing {path} - skipping {name}")
            continue
        res[name] = run_condition(name, path, args.layers)

    json.dump(res, open(args.out, "w"), indent=2)

    print("\n=== THE COMPARISON THAT MATTERS (mean AUROC within condition) ===")
    print(f"{'condition':<12}{'exists_* (surface)':>22}{'locked_* (memory)':>22}{'gap':>8}")
    for name, r in res.items():
        e = np.mean([v["auroc"] for v in r.values() if v["kind"] == "exists"])
        l = np.mean([v["auroc"] for v in r.values() if v["kind"] == "locked"])
        print(f"{name:<12}{e:>22.3f}{l:>22.3f}{e-l:>8.3f}")
    print(f"\n[wc] wrote {args.out}")
    print("""
HOW TO READ IT
  If terse exists_* stays high while terse locked_* falls toward 0.5, the model
  without a scratchpad still reads what is in front of it but no longer carries
  integrated state -- which is the compensation result, measured cleanly and
  without any cross-condition comparison.
  If terse locked_* is also high, the terse model DOES hold latent state and
  its behavioural failures are about using it, not having it. Say that instead.""")


if __name__ == "__main__":
    main()
