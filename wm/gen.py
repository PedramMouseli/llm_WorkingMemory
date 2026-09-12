"""
Stage 1: generate agent rollouts with vLLM.

All episodes run IN LOCKSTEP (turn 0 for every episode, then turn 1, ...) so that
each vLLM call is one big batch. 300 episodes x 8 turns = 8 batched calls.

Writes data/transcripts.jsonl. One line per episode:
  {seed, system, turns:[{turn, obs, think, cmd, raw, state_before, prompt}]}

`prompt` is the EXACT string that was fed to the model for that turn. acts.py
re-tokenizes it and takes the final position; that is the probe point.
Do not change how prompts are built without changing acts.py too.
"""

from __future__ import annotations

import argparse, json, os
from wm import env as E
from wm.util import chat, assert_thinking


def build_prompt(tok, system: str, turns: list, obs: str, enable_thinking=True) -> str:
    msgs = [{"role": "system", "content": system}]
    for t in turns:
        msgs.append({"role": "user", "content": t["obs"]})
        msgs.append({"role": "assistant", "content": t["raw"]})
    msgs.append({"role": "user", "content": obs})
    return chat(tok, msgs, thinking=enable_thinking)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/transcripts.jsonl")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-frac", type=float, default=0.80)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--terse", action="store_true")
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    if args.terse:
        args.no_thinking = True
        args.max_tokens = 12
    thinking = not args.no_thinking

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=8192,
        dtype="bfloat16",
        attention_backend="TRITON_ATTN",
        mm_encoder_attn_backend="TORCH_SDPA",
    )
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=args.max_tokens)

    eps = [E.make_episode(args.seed0 + i) for i in range(args.n)]
    for ep in eps:
        if args.terse:
            ep["system"] += "\nOutput ONLY the command line, with no explanation."
        elif thinking:
            ep["system"] += (
                "\nReasoning guidelines:\n"
                "- Limit your reasoning to at most 3 short sentences.\n"
                "- Close thinking with </think>.\n"
                "- Provide exactly one command on the final line: CMD: <command>"
            )
    live = list(range(len(eps)))

    for turn in range(E.N_TURNS):
        if not live:
            break
        prompts = [
            build_prompt(tok, eps[i]["system"], eps[i]["turns"], eps[i]["obs"],
                         enable_thinking=thinking)
            for i in live
        ]
        outs = llm.generate(prompts, sp)
        raws = [o.outputs[0].text for o in outs]
        if thinking:
            need = [k for k, r in enumerate(raws) if "</think>" not in r]
            suffix, joiner = "\n</think>\n\nCMD:", "\n</think>\n\nCMD: "
        else:
            need = [k for k, r in enumerate(raws) if "CMD:" not in r.upper()]
            suffix, joiner = "\nCMD:", "\nCMD: "
        if need:
            sp_cmd = SamplingParams(temperature=0.0, max_tokens=16, stop=["\n"])
            fixes = llm.generate(
                [prompts[k] + raws[k] + suffix for k in need], sp_cmd)
            for k, fo in zip(need, fixes):
                txt = fo.outputs[0].text.strip()
                raws[k] = raws[k] + joiner + (txt.splitlines()[0].strip() if txt else "ls")
            print(f"[gen] turn {turn}: repaired {len(need)}/{len(raws)} truncated replies")

        still = []
        for i, p, raw in zip(live, prompts, raws):
            alive = E.step_episode(eps[i], raw, turn, thinking=thinking)
            eps[i]["turns"][-1]["prompt"] = p
            eps[i]["turns"][-1]["truncated"] = (list(live).index(i) in need)
            if alive:
                still.append(i)
        live = still
        print(f"[gen] turn {turn} done, {len(live)} episodes still live", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for ep in eps:
            fh.write(json.dumps({
                "seed": ep["seed"],
                "system": ep["system"],
                "turns": ep["turns"],
            }) + "\n")
    n_turns = sum(len(e["turns"]) for e in eps)
    print(f"[gen] wrote {len(eps)} episodes / {n_turns} turns -> {args.out}")


if __name__ == "__main__":
    main()
