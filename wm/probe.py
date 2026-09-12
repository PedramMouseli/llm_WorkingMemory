"""
Stage 3: linear probes and text baselines.

Invariants:
  1. TRAIN/TEST SPLIT IS BY EPISODE, never by turn. Turns within an episode are
     correlated; splitting by turn causes data leakage and inflates performance.
  2. Every probe number is evaluated against FOUR baselines:
       - Shuffled labels          (empirical chance baseline given the split)
       - Majority class           (base rate)
       - TF-IDF on prompt text    (surface lexical baseline)
       - Last-assertion regex     (recency-based transcript baseline)
     A probe must demonstrate an advantage over surface text baselines.
  3. Metric is BALANCED accuracy, reported alongside base rates.

Outputs data/probe_results.json.
"""

from __future__ import annotations

import argparse, json, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score

from wm.env import FILES, VARS, VARS_LOCKED, VARS_EXISTS

LOCK_RE = {
    f: re.compile(rf"(?:\b{re.escape(f)}\b[^.\n]{{0,60}}?\b(un)?locked\b)"
                  rf"|(?:\b(un)?locked\b[^.\n]{{0,60}}?\b{re.escape(f)}\b)", re.I)
    for f in FILES
}


def last_assertion(history: list[str], f: str):
    """Scan backwards for the most recent textual assertion about f's lock state.

    Returns (value, turns_ago) where value is 1 (locked) / 0 (unlocked) / None.
    """
    for back, text in enumerate(reversed(history)):
        m = None
        for m in LOCK_RE[f].finditer(text):
            pass  # keep the LAST match within this turn
        if m:
            un = (m.group(1) or m.group(2))
            return (0 if un else 1), back
    return None, len(history)


def verbalized(text: str, f: str) -> bool:
    return LOCK_RE[f].search(text) is not None


def load():
    d = np.load("data/acts.npz", allow_pickle=True)
    X = d["X"].astype(np.float32)          # [N, L, d]
    meta = json.loads(d["meta"].item())
    return X, meta


def build_text_features(meta):
    """Per-turn: all observations + thinking up to and including this turn."""
    by_ep = {}
    for i, m in enumerate(meta):
        by_ep.setdefault(m["episode"], []).append((m["turn"], i))
    texts = [None] * len(meta)
    hists = [None] * len(meta)
    for ep, items in by_ep.items():
        items.sort()
        acc = []
        for turn, i in items:
            acc.append(meta[i]["obs"] + "\n" + meta[i]["think"])
            texts[i] = "\n".join(acc)
            hists[i] = list(acc)
    return texts, hists


def fit_eval(Xtr, ytr, Xte, yte, C=1.0):
    """ALWAYS returns (accuracy, classifier). accuracy is nan / clf is None
    when a split is degenerate. Callers unpack two values unconditionally."""
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return float("nan"), None
    clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return balanced_accuracy_score(yte, clf.predict(Xte)), clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="default: a spread of 5 layers")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--out", default="data/probe_results.json")
    args = ap.parse_args()

    X, meta = load()
    N, L, D = X.shape
    layers = args.layers or sorted({int(L * f) for f in (0.25, 0.5, 0.65, 0.8, 0.95)} - {L})
    texts, hists = build_text_features(meta)

    # ---- split BY EPISODE ----
    eps = sorted({m["episode"] for m in meta})
    rng = np.random.RandomState(0)
    rng.shuffle(eps)
    n_te = max(1, int(len(eps) * args.test_frac))
    te_eps = set(eps[:n_te])
    is_te = np.array([m["episode"] in te_eps for m in meta])
    print(f"[probe] {N} turns, {len(eps)} episodes -> {is_te.sum()} test turns")

    results = {}
    for var in VARS:
        y = np.array([m["state_before"][var] for m in meta])
        base = max(y.mean(), 1 - y.mean())
        if min(y.mean(), 1 - y.mean()) < 0.08:
            print(f"[probe] SKIP {var}: base rate {base:.2f} too extreme")
            continue

        entry = {"base_rate": float(base), "n_pos": int(y.sum()), "layers": {}}

        # --- baseline: shuffled labels (at the best layer, later) ---
        # --- baseline: TF-IDF on surface text ---
        vec = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
        Ttr = vec.fit_transform([t for t, m in zip(texts, is_te) if not m])
        Tte = vec.transform([t for t, m in zip(texts, is_te) if m])
        acc_tfidf, _ = fit_eval(Ttr, y[~is_te], Tte, y[is_te])
        entry["tfidf"] = float(acc_tfidf)

        # --- baseline: last-assertion regex (lock variables only) ---
        if var in VARS_LOCKED:
            f = var[len("locked_"):]
            preds, gold = [], []
            for i in np.where(is_te)[0]:
                v, _ = last_assertion(hists[i], f)
                preds.append(v if v is not None else int(base >= 0.5))
                gold.append(y[i])
            entry["last_assertion"] = float(balanced_accuracy_score(gold, preds))

        # --- probes, per layer ---
        for li in layers:
            if li >= X.shape[1]:
                continue
            Xl = X[:, li, :]
            mu, sd = Xl[~is_te].mean(0), Xl[~is_te].std(0) + 1e-6
            Z = (Xl - mu) / sd
            acc, clf = fit_eval(Z[~is_te], y[~is_te], Z[is_te], y[is_te])
            if clf is None:
                continue
            yshuf = y.copy()
            rng2 = np.random.RandomState(1); rng2.shuffle(yshuf)
            acc_shuf, _ = fit_eval(Z[~is_te], yshuf[~is_te], Z[is_te], yshuf[is_te])
            entry["layers"][str(li)] = {
                "probe": float(acc), "shuffled": float(acc_shuf),
                "w": clf.coef_[0].tolist(),
            }
        if not entry["layers"]:
            print(f"[probe] SKIP {var}: no usable layer")
            continue
        best = max(entry["layers"], key=lambda k: entry["layers"][k]["probe"])
        entry["best_layer"] = int(best)
        entry["best_probe"] = entry["layers"][best]["probe"]
        results[var] = entry
        la = entry.get("last_assertion")
        print(f"[probe] {var:22s} base={base:.2f} probe={entry['best_probe']:.3f}"
              f" (L{best}) tfidf={acc_tfidf:.3f}"
              + (f" lastassert={la:.3f}" if la is not None else ""))

    # ---- Experiment 2: verbalized vs not, and recency ----
    exp2 = {}
    for var in [v for v in VARS_LOCKED if v in results]:
        f = var[len("locked_"):]
        li = results[var]["best_layer"]
        Xl = X[:, li, :]
        mu, sd = Xl[~is_te].mean(0), Xl[~is_te].std(0) + 1e-6
        Z = (Xl - mu) / sd
        y = np.array([m["state_before"][var] for m in meta])
        _, clf = fit_eval(Z[~is_te], y[~is_te], Z[is_te], y[is_te])
        if clf is None:
            continue
        pred = clf.predict(Z)
        ago = np.array([last_assertion(hists[i], f)[1] for i in range(N)])
        verb = ago == 0
        rows = {}
        for name, mask in [("verbalized_this_turn", verb & is_te),
                           ("not_verbalized", (~verb) & is_te)]:
            if mask.sum() > 20 and len(set(y[mask])) > 1:
                rows[name] = {"n": int(mask.sum()),
                              "acc": float(balanced_accuracy_score(y[mask], pred[mask]))}
        by_ago = {}
        for k in range(0, 6):
            mask = (ago == k) & is_te
            if mask.sum() > 15 and len(set(y[mask])) > 1:
                by_ago[k] = {"n": int(mask.sum()),
                             "acc": float(balanced_accuracy_score(y[mask], pred[mask]))}
        exp2[var] = {"split": rows, "by_turns_since_mention": by_ago, "layer": li}
        print(f"[exp2] {var}: {rows}")

    json.dump({"probes": results, "exp2": exp2,
               "test_episodes": sorted(te_eps)},
              open(args.out, "w"), indent=2)
    print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
