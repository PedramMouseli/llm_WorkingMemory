"""
Stage 4: Causal interventions on agent beliefs.

Compares two intervention channels:
  1. Transcript token corruption: Rewriting assertions about target file state
     in the preceding context.
  2. Latent residual-stream steering: Adding or subtracting probe vectors at
     the target layer across various magnitudes (alpha), with controls for
     random directions and other-probe directions.

Stores per-item predictions to evaluate paired statistics via McNemar's test,
measuring the conditional flip rate (fraction of previously correct beliefs broken
by the intervention) and net directional effect.

Usage:
    python wm/intervene.py --n-items 300 --corrupt-mode all
    python wm/intervene.py --target locked_report.txt --tag report
"""

from __future__ import annotations

import argparse, json, random, re
import numpy as np
from wm.env import FILES
from wm.probe import LOCK_RE
from wm.util import chat
from wm.stats import mcnemar_exact, wilson

QUESTION = ("Is {f} currently locked? Ignore the task for one turn and answer "
            "with exactly one word: LOCKED or UNLOCKED.")


def parse_answer(text: str):
    body = text.split("</think>")[-1].upper()
    u = "UNLOCKED" in body
    l = "LOCKED" in body.replace("UNLOCKED", "")
    if u == l:
        return None
    return 1 if l else 0


def _flip(seg: str) -> str:
    if re.search(r"\bunlocked\b", seg, re.I):
        return re.sub(r"\bunlocked\b", "locked", seg, flags=re.I)
    return re.sub(r"\blocked\b", "unlocked", seg, flags=re.I)


def corrupt_text(s: str, f: str, mode: str = "last") -> tuple[str, int]:
    """Flip lock assertions about f. Returns (new_s, n_flipped)."""
    matches = list(LOCK_RE[f].finditer(s))
    if not matches:
        return s, 0
    if mode == "last":
        matches = matches[-1:]
    out, prev, n = [], 0, 0
    for m in matches:
        out.append(s[prev:m.start()])
        out.append(_flip(m.group(0)))
        prev = m.end()
        n += 1
    out.append(s[prev:])
    return "".join(out), n


def build_msgs(ep, upto_turn, f, corrupt=False, mode="last"):
    msgs = [{"role": "system", "content": ep["system"]}]
    for t in ep["turns"][:upto_turn + 1]:
        msgs.append({"role": "user", "content": t["obs"]})
        msgs.append({"role": "assistant", "content": t["raw"]})
    n_flipped = 0
    if corrupt:
        if mode == "all":
            for i in range(1, len(msgs)):
                new, n = corrupt_text(msgs[i]["content"], f, "all")
                if n:
                    msgs[i] = {**msgs[i], "content": new}
                    n_flipped += n
        else:
            for i in range(len(msgs) - 1, 0, -1):
                new, n = corrupt_text(msgs[i]["content"], f, "last")
                if n:
                    msgs[i] = {**msgs[i], "content": new}
                    n_flipped = n
                    break
    msgs.append({"role": "user", "content": QUESTION.format(f=f)})
    return msgs, n_flipped


def pick_target(P, how: str) -> str:
    lock = {k: v for k, v in P["probes"].items() if k.startswith("locked_")}
    if how in lock:
        return how
    if how == "auto-strongest":
        return max(lock, key=lambda k: lock[k]["best_probe"])
    # auto-story: where the transcript most outperforms the residual stream
    return max(lock, key=lambda k: lock[k].get("last_assertion", 0)
               - lock[k]["best_probe"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--transcripts", default="data/transcripts.jsonl")
    ap.add_argument("--probes", default="data/probe_results.json")
    ap.add_argument("--acts", default="data/acts.npz")
    ap.add_argument("--target", default="auto-story",
                    help="a locked_* name, 'auto-story' (max regex-probe margin), "
                         "or 'auto-strongest' (max probe)")
    ap.add_argument("--corrupt-mode", choices=["last", "all"], default="all")
    ap.add_argument("--steer-at", choices=["all", "last"], default="all")
    ap.add_argument("--n-items", type=int, default=300)
    ap.add_argument("--alphas", type=float, nargs="*",
                    default=[0, 4, 8, 16, 32, 64])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--think-readout", action="store_true",
                    help="let the model think before answering (default: no)")
    args = ap.parse_args()

    import torch
    import wm  # CUDA preload side-effects
    from transformers import AutoTokenizer, AutoModelForCausalLM

    P = json.load(open(args.probes))
    eps = [json.loads(l) for l in open(args.transcripts)]
    te = set(P["test_episodes"])
    Xacts = np.load(args.acts, allow_pickle=True)["X"].astype(np.float32)

    target = pick_target(P, args.target)
    tag = args.tag or target.replace("locked_", "").replace(".", "_")
    out_path = f"data/intervene_{tag}.json"
    f = target[len("locked_"):]
    lock = P["probes"][target]
    li = lock["best_layer"]
    w = np.array(lock["layers"][str(li)]["w"], dtype=np.float32)
    w /= np.linalg.norm(w) + 1e-9
    resid_norm = float(np.linalg.norm(Xacts[:, li, :], axis=1).mean())
    print(f"[iv] target={target} (probe {lock['best_probe']:.3f} vs regex "
          f"{lock.get('last_assertion', float('nan')):.3f})  layer={li}  "
          f"|resid|~{resid_norm:.1f}  corrupt={args.corrupt_mode}  steer_at={args.steer_at}")

    rng = np.random.RandomState(0)
    w_rand = rng.randn(len(w)).astype(np.float32)
    w_rand /= np.linalg.norm(w_rand)
    others = [k for k in P["probes"]
              if k.startswith("locked_") and k != target
              and str(li) in P["probes"][k]["layers"]]
    w_other = None
    if others:
        k2 = max(others, key=lambda k: P["probes"][k]["best_probe"])
        w_other = np.array(P["probes"][k2]["layers"][str(li)]["w"], dtype=np.float32)
        w_other /= np.linalg.norm(w_other) + 1e-9
        print(f"[iv] other-probe control = {k2}")

    items = []
    for ei, ep in enumerate(eps):
        if ei not in te:
            continue
        for t in ep["turns"]:
            if t["turn"] >= 2:
                items.append((ei, t["turn"], t["state_before"][target]))
    random.Random(0).shuffle(items)
    items = items[:args.n_items]
    truths = np.array([t for _, _, t in items])
    print(f"[iv] {len(items)} items, base rate {max(truths.mean(), 1-truths.mean()):.3f}")

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()

    steer = {"vec": None, "mode": args.steer_at}

    def hook(_m, _i, out):
        if steer["vec"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        v = steer["vec"].to(h.dtype).to(h.device)
        if steer["mode"] == "last":
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + v
        else:
            h = h + v
        return (h,) + out[1:] if isinstance(out, tuple) else h

    model.model.layers[li].register_forward_hook(hook)

    def run(msg_list):
        outs = []
        for b in range(0, len(msg_list), args.batch):
            chunk = msg_list[b:b + args.batch]
            prompts = [chat(tok, m, thinking=args.think_readout) for m in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=8192).to(args.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for j in range(len(chunk)):
                outs.append(tok.decode(g[j][enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True))
        return outs

    preds_by_cond = {}

    # ---------------- clean ----------------
    msgs = [build_msgs(eps[ei], tn, f)[0] for ei, tn, _ in items]
    steer["vec"] = None
    clean = np.array([parse_answer(o) if parse_answer(o) is not None else -1
                      for o in run(msgs)])
    preds_by_cond["clean"] = clean.tolist()
    valid = clean >= 0
    clean_ok = (clean == truths) & valid
    print(f"[iv] clean accuracy {clean_ok.sum()}/{valid.sum()} = "
          f"{clean_ok.sum()/max(valid.sum(),1):.4f}  (unparseable {int((~valid).sum())})")

    # ---------------- token corruption ----------------
    pairs = [build_msgs(eps[ei], tn, f, corrupt=True, mode=args.corrupt_mode)
             for ei, tn, _ in items]
    n_flip = np.array([n for _, n in pairs])
    corr = np.full(len(items), -1)
    did = n_flip > 0
    if did.any():
        got = run([pairs[i][0] for i in np.where(did)[0]])
        for i, o in zip(np.where(did)[0], got):
            a = parse_answer(o)
            corr[i] = -1 if a is None else a
    preds_by_cond["token_corrupt"] = corr.tolist()
    print(f"[iv] corruption applied to {int(did.sum())}/{len(items)} items, "
          f"mean {n_flip[did].mean():.2f} assertions flipped" if did.any() else
          "[iv] corruption applied to 0 items")

    # ---------------- steering + controls ----------------
    d = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    vecs = [("probe", w), ("random", w_rand)]
    if w_other is not None:
        vecs.append(("other_probe", w_other))
    for name, vec in vecs:
        for a in args.alphas:
            for sign in ([1] if a == 0 else [1, -1]):
                key = f"steer_{name}_a{sign*a:+g}"
                steer["vec"] = (None if a == 0 else
                                torch.tensor(sign * a * resid_norm / np.sqrt(d) * vec))
                p = np.array([parse_answer(o) if parse_answer(o) is not None else -1
                              for o in run(msgs)])
                preds_by_cond[key] = p.tolist()
                v = p >= 0
                acc = ((p == truths) & v).sum() / max(v.sum(), 1)
                print(f"[iv] {key:26s} acc={acc:.4f} unparseable={int((~v).sum())}",
                      flush=True)
                if a == 0:
                    break
        steer["vec"] = None

    # ---------------- paired analysis ----------------
    summary = {}
    for key, plist in preds_by_cond.items():
        if key == "clean":
            continue
        p = np.array(plist)
        both = valid & (p >= 0)
        a_ok = (clean == truths) & both        # clean correct
        b_ok = (p == truths) & both            # intervened correct
        mc = mcnemar_exact(a_ok[both], b_ok[both])
        broke = int(np.sum(a_ok & ~b_ok))
        n_was_right = int(np.sum(a_ok))
        rate, lo, hi = wilson(broke, n_was_right)
        summary[key] = {
            "n_paired": int(both.sum()),
            "acc_clean": float((a_ok & both).sum() / max(both.sum(), 1)),
            "acc_intervened": float((b_ok & both).sum() / max(both.sum(), 1)),
            "delta_acc": float(((b_ok & both).sum() - (a_ok & both).sum())
                               / max(both.sum(), 1)),
            "broke_correct_beliefs": broke,
            "of_n_correct": n_was_right,
            "conditional_flip_rate": rate,
            "flip_ci": [lo, hi],
            "mcnemar": mc,
        }

    out = {
        "target": target, "file": f, "layer": li,
        "corrupt_mode": args.corrupt_mode, "steer_at": args.steer_at,
        "n_items": len(items),
        "resid_norm": resid_norm,
        "truths": truths.tolist(),
        "preds": preds_by_cond,
        "summary": summary,
    }
    json.dump(out, open(out_path, "w"), indent=2)

    print(f"\n{'condition':<26}{'acc':>8}{'Δacc':>8}{'broke':>8}"
          f"{'flip rate (95% CI)':>26}{'McNemar p':>12}")
    for k, s in summary.items():
        ci = (f"{100*s['conditional_flip_rate']:.1f}% "
              f"[{100*s['flip_ci'][0]:.1f},{100*s['flip_ci'][1]:.1f}]")
        print(f"{k:<26}{s['acc_intervened']:>8.4f}{s['delta_acc']:>+8.4f}"
              f"{s['broke_correct_beliefs']:>4}/{s['of_n_correct']:<3}"
              f"{ci:>26}{s['mcnemar']['p']:>12.3g}")
    print(f"\n[iv] wrote {out_path}")
    print("""
HOW TO READ THIS
  `conditional flip rate` is the honest headline: of the beliefs the model held
  CORRECTLY when clean, what fraction did this intervention break? A random
  direction should sit near zero. Compare token_corrupt against the probe
  steering rows at matched effort.
  McNemar p is the paired test. Every steering row is expected to be null; the
  question is whether token corruption separates from them.""")


if __name__ == "__main__":
    main()
