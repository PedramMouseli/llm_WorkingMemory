"""
Sanity check: verify numerical agreement between vLLM and Hugging Face backends.

Transcripts are generated via vLLM with Triton attention. Activations and steering
interventions are evaluated via Hugging Face Transformers. If the backends diverge
significantly on greedy completions, activation analysis can become decoupled from
the generated rollouts.

Pass criterion: greedy continuations agree on initial tokens across prompts.

Usage:
    python wm/check_backend.py --transcripts data/smoke.jsonl
"""

from __future__ import annotations

import argparse, json
import torch
import wm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--transcripts", default="data/smoke.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--new-tokens", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from vllm import LLM, SamplingParams

    eps = [json.loads(l) for l in open(args.transcripts)]
    prompts = []
    for ep in eps:
        for t in ep["turns"]:
            if "prompt" in t:
                prompts.append(t["prompt"])
    prompts = prompts[:args.n]
    print(f"[check] {len(prompts)} prompts")

    tok = AutoTokenizer.from_pretrained(args.model)

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.7,
              max_num_seqs=64,
              attention_backend="TRITON_ATTN",
              mm_encoder_attn_backend="TORCH_SDPA")
    vout = llm.generate(prompts, SamplingParams(temperature=0.0,
                                                max_tokens=args.new_tokens))
    v_texts = [o.outputs[0].text for o in vout]
    del llm
    torch.cuda.empty_cache()

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else getattr(cfg, "text_config", cfg)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=text_cfg, torch_dtype=torch.bfloat16).to(args.device).eval()

    agree = 0
    for p, vt in zip(prompts, v_texts):
        enc = tok(p, return_tensors="pt").to(args.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=args.new_tokens,
                               do_sample=False, pad_token_id=tok.eos_token_id)
        ht = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        v_ids = tok(vt, add_special_tokens=False).input_ids
        h_ids = tok(ht, add_special_tokens=False).input_ids
        n_match = 0
        for a, b in zip(v_ids, h_ids):
            if a != b:
                break
            n_match += 1
        ok = n_match >= min(10, len(v_ids))
        agree += ok
        print(f"[check] {'PASS' if ok else 'FAIL'} matched {n_match} tokens")
        if not ok:
            print("   vLLM:", repr(vt[:160]))
            print("   HF  :", repr(ht[:160]))

    print(f"\n[check] {agree}/{len(prompts)} prompts agree.")
    if agree < len(prompts) * 0.6:
        raise SystemExit(
            "[check] BACKENDS DISAGREE. Do not trust probe/steering results "
            "computed on HF activations from vLLM transcripts until resolved. "
            "Try attention_backend='TORCH_SDPA' in gen.py and re-run."
        )
    print("[check] backends agree well enough to proceed.")


if __name__ == "__main__":
    main()
