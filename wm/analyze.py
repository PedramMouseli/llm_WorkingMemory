"""
Stage 4b: Statistical controls and robustness checks for probe findings.

Evaluates three primary controls:
  1. Regularization tuning: Sweeps C with GroupKFold (grouped by episode) on
     the training episodes only, tuning hyperparameters without test leakage.
  2. Conditional information test: Tests whether the probe carries independent
     information when the last-assertion regex baseline is wrong, evaluated via
     McNemar's test.
  3. Positional confound control: Tests whether the probe merely tracks the passage
     of time by evaluating a turn-index-only baseline and turn-stratified probe
     accuracy.

Outputs data/analysis.json.
"""

from __future__ import annotations

import argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

from wm.env import VARS_LOCKED, VARS_EXISTS
from wm.probe import last_assertion, build_text_features

CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def mcnemar(a_correct, b_correct):
    """Two-sided exact-ish McNemar on paired correctness vectors."""
    from math import comb
    b = int(np.sum(a_correct & ~b_correct))   # probe right, regex wrong
    c = int(np.sum(~a_correct & b_correct))   # regex right, probe wrong
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return b, c, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="data/acts.npz")
    ap.add_argument("--probes", default="data/probe_results.json")
    ap.add_argument("--out", default="data/analysis.json")
    args = ap.parse_args()

    d = np.load(args.acts, allow_pickle=True)
    X = d["X"].astype(np.float32)
    meta = json.loads(d["meta"].item())
    P = json.load(open(args.probes))
    te_eps = set(P["test_episodes"])

    texts, hists = build_text_features(meta)
    ep_id = np.array([m["episode"] for m in meta])
    turn = np.array([m["turn"] for m in meta])
    is_te = np.array([e in te_eps for e in ep_id])
    print(f"[an] {len(meta)} turns | train {(~is_te).sum()} test {is_te.sum()}")

    out = {}
    for var in VARS_LOCKED + VARS_EXISTS:
        if var not in P["probes"]:
            continue
        y = np.array([m["state_before"][var] for m in meta])
        li = P["probes"][var]["best_layer"]
        Xl = X[:, li, :]
        mu, sd = Xl[~is_te].mean(0), Xl[~is_te].std(0) + 1e-6
        Z = (Xl - mu) / sd

        # --- OBJECTION 1: tune C on TRAIN only, grouped by episode -----------
        best_c, best_cv = None, -1
        gkf = GroupKFold(n_splits=4)
        for C in CS:
            accs = []
            for tr, va in gkf.split(Z[~is_te], y[~is_te], groups=ep_id[~is_te]):
                clf = LogisticRegression(max_iter=3000, C=C,
                                         class_weight="balanced")
                clf.fit(Z[~is_te][tr], y[~is_te][tr])
                accs.append(balanced_accuracy_score(
                    y[~is_te][va], clf.predict(Z[~is_te][va])))
            m = float(np.mean(accs))
            if m > best_cv:
                best_cv, best_c = m, C
        clf = LogisticRegression(max_iter=3000, C=best_c,
                                 class_weight="balanced")
        clf.fit(Z[~is_te], y[~is_te])
        p_pred = clf.predict(Z[is_te])
        p_acc = balanced_accuracy_score(y[is_te], p_pred)

        entry = {"layer": li, "best_C": best_c, "cv_acc": best_cv,
                 "probe_tuned": float(p_acc),
                 "probe_untuned": float(P["probes"][var]["best_probe"]),
                 "tfidf": float(P["probes"][var]["tfidf"]),
                 "base_rate": float(P["probes"][var]["base_rate"])}

        # --- OBJECTION 3: is it just position? -------------------------------
        T = turn.reshape(-1, 1).astype(float)
        clf_t = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf_t.fit(T[~is_te], y[~is_te])
        entry["turn_index_only"] = float(
            balanced_accuracy_score(y[is_te], clf_t.predict(T[is_te])))
        by_turn = {}
        for t in sorted(set(turn[is_te])):
            m = is_te & (turn == t)
            if m.sum() > 25 and len(set(y[m])) > 1:
                by_turn[int(t)] = float(balanced_accuracy_score(
                    y[m], clf.predict(Z[m])))
        entry["probe_by_turn"] = by_turn

        # --- OBJECTION 2: the decisive conditional test (lock vars only) -----
        if var in VARS_LOCKED:
            f = var[len("locked_"):]
            base = entry["base_rate"]
            r_pred = np.array([
                (lambda v: v if v is not None else int(base >= 0.5))(
                    last_assertion(hists[i], f)[0])
                for i in np.where(is_te)[0]])
            yt = y[is_te]
            entry["regex"] = float(balanced_accuracy_score(yt, r_pred))

            r_ok = r_pred == yt
            p_ok = p_pred == yt
            b, c, pval = mcnemar(p_ok, r_ok)
            entry["mcnemar"] = {"probe_only_right": b, "regex_only_right": c,
                                "p": pval}
            if (~r_ok).sum() > 20:
                sub = ~r_ok
                entry["probe_where_regex_wrong"] = {
                    "n": int(sub.sum()),
                    "acc": float(balanced_accuracy_score(yt[sub], p_pred[sub]))
                    if len(set(yt[sub])) > 1 else None}

        out[var] = entry
        msg = (f"[an] {var:20s} base={entry['base_rate']:.2f} "
               f"probe={p_acc:.3f}(C={best_c:g}, was {entry['probe_untuned']:.3f}) "
               f"tfidf={entry['tfidf']:.3f} turn_only={entry['turn_index_only']:.3f}")
        if "regex" in entry:
            pw = entry.get("probe_where_regex_wrong", {})
            msg += (f" regex={entry['regex']:.3f}"
                    f" | probe|regex-wrong={pw.get('acc')}"
                    f" (n={pw.get('n')}) McNemar p={entry['mcnemar']['p']:.3g}")
        print(msg, flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[an] wrote {args.out}")
    print("\nREAD THIS: if `probe|regex-wrong` is ~0.5 for the lock variables,")
    print("the probe carries no information the transcript doesn't already state.")
    print("That IS the result. Report it as the finding, not as a failure.")


if __name__ == "__main__":
    main()
