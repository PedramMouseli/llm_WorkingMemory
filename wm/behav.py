"""
Behavioral analysis of scratchpad removal: testing memory vs competence accounts.

Quantifies agent failure modes (actions based on false beliefs about file lock states)
when the scratchpad is removed or constrained.

Disambiguating memory vs. competence:
  - Memory account: Without a scratchpad, the model cannot carry state forward across
    turns, failing selectively when facts are stale (last asserted >= 2 turns ago).
  - Competence account: Without deliberation, the model is uniformly worse across
    all turns regardless of fact recency.

False-belief rates are stratified into FRESH (asserted in the current or preceding turn)
and STALE (asserted >= 2 turns ago, or never), evaluating the interaction ratio.

Usage:
    python wm/behav.py data/transcripts.jsonl \
                       data/transcripts_nothink.jsonl \
                       data/transcripts_terse.jsonl
"""

from __future__ import annotations

import json, re, sys
import numpy as np

from wm.env import FILES
from wm.probe import last_assertion
from wm.stats import wilson, fisher_exact_2x2, rate_ratio

CMD_LINE = re.compile(r"^\s*CMD:.*$", re.M | re.I)
THINK_TAG = re.compile(r"</?think>")


def scratchpad_chars(turn: dict) -> int:
    """Everything the model WROTE this turn that is not the command itself.

    Works for all three conditions: thinking (prose inside <think>), no-think
    (prose before CMD:), terse (nothing).
    """
    raw = turn.get("raw", "")
    body = CMD_LINE.sub("", raw)
    body = THINK_TAG.sub("", body)
    return len(body.strip())


def episode_histories(ep: dict) -> list[list[str]]:
    """Cumulative visible text per turn: observations + whatever the model wrote.

    Mirrors probe.build_text_features so the staleness measure is the same one
    the regex baseline uses.
    """
    acc, hists = [], []
    for t in ep["turns"]:
        acc.append(t["obs"] + "\n" + t.get("think", ""))
        hists.append(list(acc))
    return hists


def analyse(path: str) -> dict:
    eps = [json.loads(l) for l in open(path)]
    n_turns = n_err = 0
    pad = []
    # attempts[(stratum)] = [n_attempts, n_false_belief]
    strata = {"fresh": [0, 0], "stale": [0, 0]}
    n_att = n_fb = 0
    per_ep_fb = []
    ago_hist = []

    for ep in eps:
        hists = episode_histories(ep)
        ef = 0
        for i, t in enumerate(ep["turns"]):
            n_turns += 1
            pad.append(scratchpad_chars(t))
            if t["obs"].startswith("error"):
                n_err += 1

            parts = t["cmd"].strip().lower().split()
            if not parts or parts[0] not in ("mv", "rm"):
                continue
            target = parts[1] if len(parts) > 1 else None
            if target not in FILES:
                continue
            if i + 1 >= len(ep["turns"]):
                continue          # outcome never observed
            outcome = ep["turns"][i + 1]["obs"]
            if "is not in /logs" in outcome:
                continue          # a different kind of error, not a lock belief

            n_att += 1
            failed = "is locked" in outcome
            n_fb += failed
            ef += failed

            _, ago = last_assertion(hists[i], target)
            ago_hist.append(ago)
            key = "fresh" if ago <= 1 else "stale"
            strata[key][0] += 1
            strata[key][1] += failed

        per_ep_fb.append(ef)

    p, lo, hi = wilson(n_fb, n_att)
    out = {
        "file": path,
        "episodes": len(eps),
        "turns": n_turns,
        "mean_scratchpad_chars": float(np.mean(pad)),
        "median_scratchpad_chars": float(np.median(pad)),
        "lock_attempts": n_att,
        "false_beliefs": n_fb,
        "fb_rate": p, "fb_lo": lo, "fb_hi": hi,
        "fb_per_100_turns": 100.0 * n_fb / max(n_turns, 1),
        "attempt_rate_per_turn": n_att / max(n_turns, 1),
        "obs_error_rate": n_err / max(n_turns, 1),
        "per_ep_mean": float(np.mean(per_ep_fb)),
        "median_turns_since_assertion": float(np.median(ago_hist)) if ago_hist else None,
        "strata": {},
    }
    for k, (na, nf) in strata.items():
        pp, ll, hh = wilson(nf, na)
        out["strata"][k] = {"attempts": na, "false_beliefs": nf,
                            "rate": pp, "lo": ll, "hi": hh}
    return out


def label(path: str) -> str:
    """Name the condition. Do NOT use str.strip() here - it strips a CHARACTER
    SET, so "transcripts_nothink.jsonl" became "think", the single most
    confusing possible label for the no-thinking condition."""
    s = path.split("/")[-1]
    for suf in (".jsonl", ".json"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = s.replace("transcripts", "").lstrip("_")
    return s or "thinking"


if __name__ == "__main__":
    paths = sys.argv[1:] or ["data/transcripts.jsonl"]
    rows = [analyse(p) for p in paths]
    names = [label(p) for p in paths]

    print("\n=== OVERALL " + "=" * 62)
    print(f"{'condition':<12}{'turns':>7}{'scratchpad':>12}{'attempts':>10}"
          f"{'FB':>5}{'FB rate (95% CI)':>24}{'err rate':>10}")
    for n, r in zip(names, rows):
        ci = f"{100*r['fb_rate']:.2f}% [{100*r['fb_lo']:.2f}, {100*r['fb_hi']:.2f}]"
        print(f"{n:<12}{r['turns']:>7}{r['mean_scratchpad_chars']:>12.0f}"
              f"{r['lock_attempts']:>10}{r['false_beliefs']:>5}{ci:>24}"
              f"{r['obs_error_rate']:>10.3f}")

    print("\n=== MEMORY VS COMPETENCE STRATIFICATION " + "=" * 32)
    print("FRESH = lock state asserted in text this turn or last turn")
    print("STALE = asserted >=2 turns ago, or never\n")
    print(f"{'condition':<12}{'FRESH n':>9}{'FRESH rate':>22}"
          f"{'STALE n':>9}{'STALE rate':>22}{'stale/fresh':>13}")
    for n, r in zip(names, rows):
        fr, st = r["strata"]["fresh"], r["strata"]["stale"]
        f_ci = f"{100*fr['rate']:.2f}% [{100*fr['lo']:.1f},{100*fr['hi']:.1f}]"
        s_ci = f"{100*st['rate']:.2f}% [{100*st['lo']:.1f},{100*st['hi']:.1f}]"
        rr = rate_ratio(st["false_beliefs"], st["attempts"],
                        fr["false_beliefs"], fr["attempts"])
        print(f"{n:<12}{fr['attempts']:>9}{f_ci:>22}"
              f"{st['attempts']:>9}{s_ci:>22}{rr['rr']:>13.2f}")

    if len(rows) >= 2:
        print("\n=== PAIRWISE (terse vs each other condition) " + "=" * 30)
        terse = [r for r, n in zip(rows, names) if "terse" in n]
        if terse:
            T = terse[0]
            for n, r in zip(names, rows):
                if "terse" in n:
                    continue
                p = fisher_exact_2x2(
                    T["false_beliefs"], T["lock_attempts"] - T["false_beliefs"],
                    r["false_beliefs"], r["lock_attempts"] - r["false_beliefs"])
                rr = rate_ratio(T["false_beliefs"], T["lock_attempts"],
                                r["false_beliefs"], r["lock_attempts"])
                print(f"  terse vs {n:<10} rate ratio {rr['rr']:.2f} "
                      f"[{rr['lo']:.2f}, {rr['hi']:.2f}]   Fisher p = {p:.3g}")
            # the interaction that decides memory vs competence
            for n, r in zip(names, rows):
                if "terse" in n:
                    continue
                rr_f = rate_ratio(T["strata"]["fresh"]["false_beliefs"],
                                  T["strata"]["fresh"]["attempts"],
                                  r["strata"]["fresh"]["false_beliefs"],
                                  r["strata"]["fresh"]["attempts"])
                rr_s = rate_ratio(T["strata"]["stale"]["false_beliefs"],
                                  T["strata"]["stale"]["attempts"],
                                  r["strata"]["stale"]["false_beliefs"],
                                  r["strata"]["stale"]["attempts"])
                print(f"  terse/{n}: FRESH ratio {rr_f['rr']:.2f}, "
                      f"STALE ratio {rr_s['rr']:.2f}")

    json.dump({"rows": rows, "names": names}, open("data/behav.json", "w"), indent=2)
    print("""
HOW TO READ THE INTERACTION
  STALE ratio >> FRESH ratio        -> MEMORY. Terse fails specifically where
      the value had to be carried forward. This supports the claim.
  STALE ratio ~= FRESH ratio, both high -> COMPETENCE. Terse is uniformly worse
      and the false-belief gap is not evidence about memory. Say so plainly.
  Also check `attempt_rate_per_turn`: if terse attempts far more actions per
  turn, some of the gap is recklessness, and that belongs in the limitations.

wrote data/behav.json""")
