"""
Publication figures for world-model state evaluation.

Figures:
  F0: Behavioral false-belief rates across conditions and recency strata.
  F1: Latent fidelity over episode turns and recency decay dissociation.
  F2: Balanced accuracy of linear probes vs text baselines (TF-IDF, regex, base rate).
  F3: Layer-wise decodability heatmap across residual stream layers.
  F4: Forest plot of net directional causal effects (token corruption vs latent steering).

Usage:
    python wm/plots_final.py
"""

from __future__ import annotations

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
TOK, LAT, GREY = "#a25c1f", "#1f6478", "#9aa5ab"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fig1():
    A = _load("data/analysis.json")
    P = _load("data/probe_results.json")
    if not A:
        print("  skip F1 (no analysis.json)"); return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)

    ax = axes[0]
    for kind, col in (("locked_", TOK), ("exists_", LAT)):
        series = [v["probe_by_turn"] for k, v in A.items()
                  if k.startswith(kind) and v.get("probe_by_turn")]
        if not series:
            continue
        ts = sorted({int(t) for s in series for t in s})
        for s in series:
            ax.plot(ts, [s.get(str(t), np.nan) for t in ts], color=col,
                    alpha=.22, lw=1)
        mean = [np.nanmean([s.get(str(t), np.nan) for s in series]) for t in ts]
        ax.plot(ts, mean, color=col, lw=2.4, marker="o", ms=4,
                label=f"{kind}* ({'memory-only' if kind=='locked_' else 'surface-observable'})")
    ax.axhline(.5, color="r", ls=":", lw=.9)
    ax.set_xlabel("turn index within episode")
    ax.set_ylabel("probe balanced accuracy")
    ax.set_title("Latent fidelity over the episode", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    ax = axes[1]
    e2 = (P or {}).get("exp2", {})
    for var, d in e2.items():
        ks = sorted(int(k) for k in d.get("by_turns_since_mention", {}))
        if not ks:
            continue
        acc = [d["by_turns_since_mention"][str(k)]["acc"] for k in ks]
        n = [d["by_turns_since_mention"][str(k)]["n"] for k in ks]
        ax.plot(ks, acc, color=TOK, alpha=.5, lw=1.2, marker="o", ms=3)
        for k, a, nn in zip(ks, acc, n):
            ax.annotate(f"{nn}", (k, a), fontsize=5.5, color=GREY,
                        xytext=(0, -9), textcoords="offset points", ha="center")
    ax.axhline(.5, color="r", ls=":", lw=.9)
    ax.set_xlabel("turns since value last asserted in text")
    ax.set_title("Latent fidelity vs text recency (n per point)", fontsize=10)
    fig.tight_layout(); fig.savefig("figs/F1_dissociation.png")
    print("  wrote figs/F1_dissociation.png")


def fig2():
    A = _load("data/analysis.json")
    if not A:
        print("  skip F2"); return
    lock = [(k, v) for k, v in A.items() if k.startswith("locked_")]
    ex = [(k, v) for k, v in A.items() if k.startswith("exists_")]
    lock.sort(key=lambda kv: -(kv[1].get("regex", 0) - kv[1]["probe_tuned"]))
    ex.sort(key=lambda kv: -(kv[1]["probe_tuned"] - kv[1]["tfidf"]))
    order = lock + ex
    x = np.arange(len(order)); w = .2
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.bar(x - 1.5*w, [v["probe_tuned"] for _, v in order], w,
           color=LAT, label="linear probe (tuned)")
    ax.bar(x - .5*w, [v["tfidf"] for _, v in order], w,
           color="#7fa8b4", label="TF-IDF on transcript")
    ax.bar(x + .5*w, [v.get("regex", np.nan) for _, v in order], w,
           color=TOK, label="last-assertion regex")
    ax.bar(x + 1.5*w, [v["base_rate"] for _, v in order], w,
           color=GREY, label="majority class")
    ax.axvline(len(lock) - .5, color="k", lw=.9, ls="--")
    ax.axhline(.5, color="r", ls=":", lw=.9)
    ax.set_xticks(x)
    ax.set_xticklabels([k for k, _ in order], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(.4, .95)
    ax.set_title("memory-only (test)      |      surface-observable (control)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, ncol=4,
              loc="upper center", bbox_to_anchor=(.5, -.34))
    fig.tight_layout(); fig.savefig("figs/F2_probe_vs_baselines.png",
                                    bbox_inches="tight")
    print("  wrote figs/F2_probe_vs_baselines.png")


def fig3():
    S = _load("data/layer_sweep.json")
    if not S:
        print("  skip F3 (run wm/layer_sweep.py first)"); return
    names = ([k for k, v in S["vars"].items() if v["kind"] == "locked"] +
             [k for k, v in S["vars"].items() if v["kind"] == "exists"])
    n_lock = sum(1 for k in names if S["vars"][k]["kind"] == "locked")
    M = np.array([S["vars"][k]["acc_by_layer"] for k in names])
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=.5,
                   vmax=max(.75, float(M.max())))
    for i, k in enumerate(names):
        j = S["layers"].index(S["vars"][k]["best_layer"])
        ax.plot(j, i, marker="o", ms=5, mfc="none", mec="white", mew=1.4)
    ax.axhline(n_lock - .5, color="white", lw=1.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks(range(0, len(S["layers"]), 4))
    ax.set_xticklabels(S["layers"][::4])
    ax.set_xlabel("layer")
    ax.set_title("Where each variable becomes linearly decodable "
                 "(circle = argmax)", fontsize=10)
    fig.colorbar(im, ax=ax, label="balanced accuracy", pad=.015)
    fig.tight_layout(); fig.savefig("figs/F3_layer_heatmap.png")
    print("  wrote figs/F3_layer_heatmap.png")


def fig4(tag="beta_log"):
    """Forest plot of NET directional effect.

    The flip rate alone is misleading: an intervention that breaks 67 correct
    beliefs while fixing 49 wrong ones has a high flip rate and no directional
    effect at all - it is randomising the answer, not steering it. Net =
    broke - fixed is what separates control from damage, so plot that, and
    colour by whether McNemar says the direction is real.
    """
    R = _load(f"data/intervene_{tag}.json")
    if not R:
        print(f"  skip F4 (no data/intervene_{tag}.json)"); return
    S = R["summary"]
    rows = [("token corruption", "token_corrupt")]
    for nm in ("probe", "random", "other_probe"):
        rows += [(k.replace("steer_", "").replace("_", " "), k)
                 for k in sorted([k for k in S if k.startswith(f"steer_{nm}_a")],
                                 key=lambda k: float(k.split("a")[-1]))]
    fig, ax = plt.subplots(figsize=(7.6, .28*len(rows) + 1.6))
    for i, (lab, key) in enumerate(rows):
        s_ = S[key]
        mc = s_["mcnemar"]
        broke, fixed = mc["a_only"], mc["b_only"]
        n = max(s_["n_paired"], 1)
        net = (broke - fixed) / n
        sig = mc["p"] < 0.01 and net > 0
        col = TOK if key == "token_corrupt" else (LAT if sig else GREY)
        se = np.sqrt(max(broke + fixed, 1)) / n
        ax.plot([net - 1.96*se, net + 1.96*se], [i, i], color=col, lw=2,
                solid_capstyle="round")
        ax.plot(net, i, "o", color=col, ms=5)
        ax.annotate(f"{broke}br/{fixed}fx  p={mc['p']:.1g}", (net + 1.96*se, i),
                    fontsize=6, color=GREY, xytext=(5, -2),
                    textcoords="offset points")
    ax.axvline(0, color="k", lw=.9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("NET directional effect = (beliefs broken - beliefs fixed) / n\n"
                  "teal = directional (McNemar p<0.01); grey = scrambling, not steering")
    ax.set_title(f"Which channel CONTROLS the belief?  target = {R['target']}",
                 fontsize=10)
    fig.tight_layout(); fig.savefig("figs/F4_causal_forest.png")
    print("  wrote figs/F4_causal_forest.png")


def fig0():
    B = _load("data/behav.json")
    if not B:
        print("  skip F0"); return
    rows, names = B["rows"], B["names"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    x = np.arange(len(rows))
    ax = axes[0]
    ax.bar(x, [r["fb_rate"] for r in rows], .55, color=[GREY, GREY, TOK][:len(rows)])
    ax.errorbar(x, [r["fb_rate"] for r in rows],
                yerr=[[r["fb_rate"]-r["fb_lo"] for r in rows],
                      [r["fb_hi"]-r["fb_rate"] for r in rows]],
                fmt="none", ecolor="k", capsize=3, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("false-belief rate per mv/rm attempt")
    ax.set_title("Overall (95% Wilson CI)", fontsize=10)
    ax = axes[1]
    w = .35
    for j, (k, col) in enumerate((("fresh", LAT), ("stale", TOK))):
        v = [r["strata"][k]["rate"] for r in rows]
        ax.bar(x + (j - .5)*w, v, w, color=col, label=k)
        ax.errorbar(x + (j - .5)*w, v,
                    yerr=[[r["strata"][k]["rate"]-r["strata"][k]["lo"] for r in rows],
                          [r["strata"][k]["hi"]-r["strata"][k]["rate"] for r in rows]],
                    fmt="none", ecolor="k", capsize=2.5, lw=.9)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_title("Split by text recency: memory or competence?", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig("figs/F0_false_beliefs.png")
    print("  wrote figs/F0_false_beliefs.png")


if __name__ == "__main__":
    fig0(); fig1(); fig2(); fig3(); fig4()
    print("done -> figs/")
