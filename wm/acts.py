"""
Stage 2: cache residual-stream activations at each turn's probe point.

The probe point is the FINAL token of the prompt that was fed to the model for
that turn, i.e. the last token of the observation, immediately before the model
begins to think. That position holds the model's state of knowledge entering
the turn.

We use forward hooks that keep only the last position, so memory stays flat.
Left padding is required for [:, -1, :] to be the last real token.

Writes data/acts.npz:
  X      float16 [n_turns, n_layers, d_model]
  layers int32   [n_layers]
  meta   object  aligned list of {episode, turn, state_before, ...}
"""

from __future__ import annotations

import argparse, json
import numpy as np
import torch
import wm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--transcripts", default="data/transcripts.jsonl")
    ap.add_argument("--out", default="data/acts.npz")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(args.device)
    model.eval()

    layers = model.model.layers
    n_layers = len(layers)
    d_model = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    print(f"[acts] {n_layers} layers, d_model={d_model}")

    buf = {}

    def mk_hook(idx):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            buf[idx] = h[:, -1, :].detach().float().cpu()
        return hook

    handles = [l.register_forward_hook(mk_hook(i)) for i, l in enumerate(layers)]

    rows, meta = [], []
    with open(args.transcripts) as fh:
        eps = [json.loads(l) for l in fh]

    flat = []
    for ei, ep in enumerate(eps):
        for t in ep["turns"]:
            if "prompt" not in t:
                continue
            flat.append((ei, ep["seed"], t))

    print(f"[acts] {len(flat)} turns to encode")
    for b in range(0, len(flat), args.batch):
        chunk = flat[b:b + args.batch]
        enc = tok([c[2]["prompt"] for c in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=8192).to(args.device)
        with torch.no_grad():
            model(**enc)
        stacked = torch.stack([buf[i] for i in range(n_layers)], dim=1)  # [B, L, d]
        rows.append(stacked.half().numpy())
        for ei, seed, t in chunk:
            meta.append({
                "episode": ei, "seed": seed, "turn": t["turn"],
                "state_before": t["state_before"],
                "think": t["think"], "cmd": t["cmd"], "obs": t["obs"],
                "prompt_chars": len(t["prompt"]),
            })
        if (b // args.batch) % 25 == 0:
            print(f"[acts] {b + len(chunk)}/{len(flat)}", flush=True)

    for h in handles:
        h.remove()

    X = np.concatenate(rows, axis=0)
    print("[acts] X", X.shape, X.dtype)
    np.savez_compressed(
        args.out, X=X, layers=np.arange(n_layers, dtype=np.int32),
        meta=np.array(json.dumps(meta), dtype=object),
    )
    print(f"[acts] wrote {args.out}")


if __name__ == "__main__":
    main()
