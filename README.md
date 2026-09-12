# Where Does an LLM Agent Keep the State of the World?

Does a reasoning LLM agent maintain the state of an evolving environment in its **internal activations**, or does it write the state into its **own generated text** and read it back from the transcript?

This repository evaluates this question using **Qwen3.5-9B** in a deterministic shell environment with exact ground truth across 8-turn episodes.

---

## Experimental Framework

The environment tracks four files across two directories (`/logs` and `/archive`):
- **Surface-observable control (`exists_<file>`):** Visible repeatedly via `ls`.
- **Memory-only test variable (`locked_<file>`):** Never reported by `ls`; modified via toggles rather than explicit value assignments, requiring integration across turns.

### Experiments

- **Experiment 0: Behavioral Analysis (`wm/behav.py`)**  
  Measures action failure rates when removing or constraining the scratchpad, stratifying by whether target file state was *fresh* ($\le 1$ turn ago) or *stale* ($\ge 2$ turns ago).
- **Experiments 1 & 2: Probing & Baselines (`wm/probe.py`, `wm/analyze.py`, `wm/layer_sweep.py`)**  
  Linear probes trained on residual stream activations at the prompt-final position entering each turn. Evaluated with strictly held-out episodes and compared against shuffled labels, base rate, TF-IDF on prompt text, and last-assertion regex.
- **Experiment 3: Causal Interventions (`wm/intervene.py`)**  
  Direct comparison between transcript token corruption and residual-stream activation steering across magnitudes, evaluated on paired items with exact McNemar tests.
- **Experiment 4: Within-Condition Representations (`wm/within_condition.py`)**  
  Evaluates state decodability within thinking, non-thinking, and terse regimes independently to assess whether representations persist without scratchpad reasoning.

---

## Key Takeaways

1. **Transcript baselines match or outperform residual-stream probes:**
   For three out of four memory-only variables, a simple regex tracking the most recent textual assertion performed slightly better than a tuned linear probe of the residual stream (balanced accuracy **0.79** vs. **0.75**). On instances where the regex failed, the probe recovered minimal residual information (**0.54–0.61** balanced accuracy).
2. **Latent representations decay with recency:**
   For memory-only variables, probe accuracy decayed from **1.00** on turn 0 down toward chance by turn 7 as time passed since the last assertion. In contrast, surface-observable variables (`exists_*`) became easier to decode over time as observations accumulated.
3. **Layer specialization:**
   Surface-observable information peaked early (layers 1–2), whereas memory-only state peaked late in the network (layers 20–30).
4. **Context dominates causal belief:**
   Rewriting textual assertions in the transcript broke **151 of 215** previously correct beliefs (net effect **0.68**, $p = 1.6 \times 10^{-38}$). Latent residual steering along probe directions also affected beliefs (**0.05** at $\alpha=+4$ to **0.15** at $\alpha=+64$), but the latent causal effect was roughly five times weaker than text intervention.
5. **The scratchpad enables computation, not storage:**
   Removing the scratchpad caused an eightfold surge in false-belief actions (**2.7%** $\to$ **21.7%**), concentrated on stale facts. However, latent decodability of world state remained intact within the terse condition (AUROC **0.835** vs. **0.786** with full reasoning), indicating that the scratchpad is used for serial reasoning rather than passive state retention.