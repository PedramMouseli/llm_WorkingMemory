"""
Cross-condition probe evaluation and calibration.

Evaluates how linear probes trained in one generation condition (e.g. thinking)
transfer to another (e.g. terse or no-thinking).

Metrics evaluated:
  1. AUROC: Threshold-free ranking metric invariant to intercept/distribution shift.
  2. Balanced accuracy (original threshold): Evaluates the probe using the original
     decision boundary.
  3. Balanced accuracy (recalibrated): Retains the probe direction vector w and fits
     scale and bias parameters on a calibration split of episodes, evaluated on held-out
     episodes.

Usage:
    python wm/eval_transfer.py
    python wm/eval_transfer.py --terse-acts data/acts_nothink.npz --tag nothink
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
    ap.add_argument("--th-acts", default="data/acts.npz")
    ap.add_argument("--terse-acts", default="data/acts_terse.npz")
    ap.add_argument("--probes", default="data/probe_results.json")
    ap.add_argument("--tag", default="terse")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or f"data/transfer_{args.tag}.json"

    d_th = np.load(args.th_acts, allow_pickle=True)
    X_th = d_th["X"].astype(np.float32)
    meta_th = json.loads(d_th["meta"].item())

    d_te = np.load(args.terse_acts, allow_pickle=True)
    X_te = d_te["X"].astype(np.float32)
    meta_te = json.loads(d_te["meta"].item())

    P = json.load(open(args.probes))
    th_test = set(P["test_episodes"])
    th_ep = np.array([m["episode"] for m in meta_th])
    is_th_te = np.array([e in th_test for e in th_ep])
    is_th_tr = ~is_th_te

    te_ep = np.array([m["episode"] for m in meta_te])
    # split TARGET episodes in half for honest recalibration
    uniq = np.unique(te_ep)
    rng = np.random.RandomState(0)
    rng.shuffle(uniq)
    calib_eps = set(uniq[: len(uniq) // 2].tolist())
    is_calib = np.array([e in calib_eps for e in te_ep])

    print(f"[tr] thinking {len(meta_th)} turns  |  {args.tag} {len(meta_te)} turns")
    print(f"[tr] recalibration split: {is_calib.sum()} calib / {(~is_calib).sum()} eval\n")

    _, hists_te = build_text_features(meta_te)
    results = {}

    hdr = (f"{'variable':<20}{'L':>4}{'th-test':>9}{'AUROC':>8}{'AUROC 95%CI':>18}"
           f"{'bal-acc':>9}{'recal':>8}{'maj-frac':>10}")
    print(hdr)
    print("-" * len(hdr))

    for var in VARS_LOCKED + VARS_EXISTS:
        if var not in P["probes"]:
            continue
        li = P["probes"][var]["best_layer"]
        y_th = np.array([m["state_before"][var] for m in meta_th])
        y_te = np.array([m["state_before"][var] for m in meta_te])
        if len(set(y_te)) < 2:
            print(f"{var:<20} skipped - degenerate labels in {args.tag}")
            continue

        # ---- fit on thinking train, exactly as probe.py did ----
        Xl = X_th[:, li, :]
        mu = Xl[is_th_tr].mean(0)
        sd = Xl[is_th_tr].std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit((Xl[is_th_tr] - mu) / sd, y_th[is_th_tr])
        acc_th = balanced_accuracy_score(
            y_th[is_th_te], clf.predict((Xl[is_th_te] - mu) / sd))
        auc_th = roc_auc_score(
            y_th[is_th_te], clf.decision_function((Xl[is_th_te] - mu) / sd))

        # ---- apply to the target condition ----
        Z_te = (X_te[:, li, :] - mu) / sd
        score = clf.decision_function(Z_te)        # threshold-free
        pred = clf.predict(Z_te)                   # original boundary

        auc, auc_lo, auc_hi = boot_ci(y_te, score, roc_auc_score,
                                      n_boot=1000, groups=te_ep)
        acc_raw = balanced_accuracy_score(y_te, pred)
        maj_frac = float(max(np.mean(pred == 0), np.mean(pred == 1)))

        # ---- recalibrate scale+bias on half the target episodes ----
        cal = LogisticRegression(max_iter=1000, class_weight="balanced")
        cal.fit(score[is_calib].reshape(-1, 1), y_te[is_calib])
        acc_recal = balanced_accuracy_score(
            y_te[~is_calib], cal.predict(score[~is_calib].reshape(-1, 1)))

        entry = {
            "layer": li,
            "n_target_turns": int(len(y_te)),
            "base_rate_target": float(max(y_te.mean(), 1 - y_te.mean())),
            "probe_th_test_bal_acc": float(acc_th),
            "probe_th_test_auroc": float(auc_th),
            "transfer_auroc": float(auc),
            "transfer_auroc_ci": [float(auc_lo), float(auc_hi)],
            "transfer_bal_acc_original_boundary": float(acc_raw),
            "transfer_bal_acc_recalibrated": float(acc_recal),
            "majority_pred_fraction": maj_frac,
            "constant_predictor": bool(maj_frac > 0.995),
        }

        if var in VARS_LOCKED:
            f = var[len("locked_"):]
            base = entry["base_rate_target"]
            r = np.array([
                (lambda v: v if v is not None else int(base >= 0.5))(
                    last_assertion(hists_te[i], f)[0])
                for i in range(len(meta_te))])
            entry["regex_target"] = float(balanced_accuracy_score(y_te, r))

        results[var] = entry
        flag = "  <- CONSTANT" if entry["constant_predictor"] else ""
        print(f"{var:<20}{li:>4}{acc_th:>9.3f}{auc:>8.3f}"
              f"{f'[{auc_lo:.3f},{auc_hi:.3f}]':>18}"
              f"{acc_raw:>9.3f}{acc_recal:>8.3f}{maj_frac:>10.3f}{flag}")

    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n[tr] wrote {out_path}")
    print("""
HOW TO READ THIS
  AUROC ~= 0.5 with CI spanning 0.5   -> the direction genuinely does not
      transfer. The latent representation is absent in this condition.
  AUROC clearly > 0.5 but bal-acc = 0.5 and maj-frac = 1.0
      -> pure intercept shift. The direction transfers; only the threshold
      moved. The old "collapse to chance" conclusion was an artefact.
  Watch exists_* especially: that information IS in the target context, so a
  low AUROC there means the pipeline is still wrong, not the model.""")


if __name__ == "__main__":
    main()
