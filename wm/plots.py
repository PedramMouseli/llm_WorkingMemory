"""Stage 5: the three figures. Plain matplotlib, labelled so a reader with zero
context can read them. Every axis says what it is; every baseline is on the plot."""

from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
P = json.load(open("data/probe_results.json"))


def fig1():
    probes = P["probes"]
    order = ([k for k in probes if k.startswith("locked_")] +
             [k for k in probes if k.startswith("exists_")] +
             [k for k in probes if k == "cwd"])
    x = np.arange(len(order)); w = 0.2
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - 1.5 * w, [probes[k]["best_probe"] for k in order], w, label="linear probe")
    ax.bar(x - 0.5 * w, [probes[k]["tfidf"] for k in order], w, label="TF-IDF on transcript")
    ax.bar(x + 0.5 * w, [probes[k].get("last_assertion", np.nan) for k in order], w,
           label="last-assertion regex")
    ax.bar(x + 1.5 * w, [probes[k]["base_rate"] for k in order], w,
           label="majority class", color="0.7")
    n_lock = sum(k.startswith("locked_") for k in order)
    ax.axvline(n_lock - 0.5, color="k", lw=1, ls="--")
    ax.text(n_lock / 2 - 0.5, 1.02, "memory-only (test)", ha="center")
    ax.text(n_lock + (len(order) - n_lock) / 2 - 0.5, 1.02,
            "surface-observable (control)", ha="center")
    ax.axhline(0.5, color="r", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("balanced accuracy (held-out episodes)")
    ax.set_ylim(0.35, 1.08); ax.legend(ncol=2, fontsize=9)
    ax.set_title("Where is the agent's world state recoverable from?")
    fig.tight_layout(); fig.savefig("figs/fig1_probe_vs_baselines.png", dpi=160)


def fig2():
    exp2 = P.get("exp2", {})
    if not exp2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for var, d in exp2.items():
        ks = sorted(int(k) for k in d["by_turns_since_mention"])
        if not ks:
            continue
        ax.plot(ks, [d["by_turns_since_mention"][str(k)]["acc"] for k in ks],
                marker="o", label=var)
    ax.axhline(0.5, color="r", ls=":", lw=0.8)
    ax.set_xlabel("turns since the value was last asserted in text")
    ax.set_ylabel("probe balanced accuracy")
    ax.set_title("Does the latent state survive when the text stops mentioning it?")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig("figs/fig2_recency.png", dpi=160)


def fig3():
    if not os.path.exists("data/intervene_results.json"):
        return
    R = json.load(open("data/intervene_results.json"))["conditions"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tc = R.get("token_corrupt", {}).get("follows_corruption")
    if tc is not None:
        ax.axhline(tc, color="C3", lw=2,
                   label=f"token corruption ({tc:.2f})")
    for name, c in [("probe", "C0"), ("random", "C7"), ("other_probe", "C2")]:
        pts = sorted((float(k.split("a")[-1]), v["follows_corruption"])
                     for k, v in R.items()
                     if k.startswith(f"steer_{name}_a") and "follows_corruption" in v)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=c,
                    label=f"latent steering: {name}")
    ax.set_xlabel("steering magnitude alpha (signed)")
    ax.set_ylabel("fraction of answers that flip away from ground truth")
    ax.set_title("Which channel controls the agent's stated belief?")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig("figs/fig3_causal.png", dpi=160)


fig1(); fig2(); fig3()
print("wrote figs/")
