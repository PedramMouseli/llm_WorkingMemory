"""
Randomly selected raw turns for the write-up. Not cherry-picked: fixed seed,
uniform draw over all turns, printed whether flattering or not.

    python wm/qualitative.py --n 5 > examples.txt
"""

from __future__ import annotations

import argparse, json, random, textwrap

from wm.env import FILES
from wm.probe import last_assertion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default="data/transcripts.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-think", type=int, default=600)
    args = ap.parse_args()

    eps = [json.loads(l) for l in open(args.transcripts)]
    pool = [(ei, ti) for ei, e in enumerate(eps) for ti in range(len(e["turns"]))]
    picks = random.Random(args.seed).sample(pool, args.n)
    picks.sort()

    print(f"{args.n} turns drawn uniformly at random from {len(pool)} "
          f"(seed={args.seed}). Not selected for interest.\n")

    for k, (ei, ti) in enumerate(picks, 1):
        t = eps[ei]["turns"][ti]
        st = t["state_before"]
        locks = " ".join(f"{f}={'L' if st[f'locked_{f}'] else 'U'}" for f in FILES)
        print("=" * 74)
        print(f"EXAMPLE {k}  |  episode seed {eps[ei]['seed']}, turn {ti}")
        print("=" * 74)
        print("\nWHAT THE AGENT SAW:")
        print(textwrap.indent(t["obs"], "    "))
        think = t.get("think", "").strip()
        if len(think) > args.max_think:
            think = think[:args.max_think] + " [...truncated for the write-up]"
        print("\nWHAT IT WROTE" + (" (none - terse condition)" if not think else "") + ":")
        if think:
            print(textwrap.indent(textwrap.fill(think, 70), "    "))
        print(f"\nCOMMAND ISSUED:  {t['cmd']}")
        print(f"TRUE LOCK STATE: {locks}")
        nxt = eps[ei]["turns"][ti + 1]["obs"] if ti + 1 < len(eps[ei]["turns"]) else "(episode ended)"
        print(f"WHAT HAPPENED:   {nxt.splitlines()[0]}")
        if t.get("truncated"):
            print("NOTE:            reply hit the token budget; command repaired by gen.py")
        print()


if __name__ == "__main__":
    main()
