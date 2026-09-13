# Keel: an event-sourced runtime for measuring the economics of recovery in LLM agent workflows

Artifact for the paper. The runtime is called Keel in the paper; the Python package is `ledger`. Pure standard library, no third-party dependencies.

## Layout

```
ledger/
  providers.py          one call() over 13 OpenAI-compatible endpoints plus Gemini and Cohere; token-aware pacer; 429/5xx as first-class outcomes
  pricing.py            list prices (USD per million tokens) used to convert tokens to cost
  events.py             append-only JSONL event log (ModelResponded, RetryScheduled, CheckpointTaken, RolledBack, Escalated, BudgetExhausted, StepCommitted, ...)
  tasks_t1.py           T1 integer-ledger task family: generator, prompts, programmatic verifier
  tasks_t2.py           T2 warehouse-order task family
  pilot.py              Experiment 1: teacher-forced failure points and retry perturbations
  frontier.py           Experiment 2: free-running policies (none, retry, reset, rollback, escalate, budget, escalate_step)
  replay.py             Experiment 3: log indexing and replay of policy changes
  stacks.py             Experiment 3: same weights on different serving stacks
  analyze_pilot.py      Exp 1 tables; analyze_hetero.py fits the Beta-geometric null (Eq. 2)
  analyze_frontier.py   Exp 2 tables; recost.py recomputes cost from events
  analyze_escalation.py target-only / escalate / escalate-and-return comparison (paper Table IV)
  analyze_stacks.py     per-stack accuracy and pairwise divergence (paper Table VI); stacks_rebuild.py rebuilds a stacks JSON from its event log
  figures.py            paper figures
runs/                   raw event logs (*.events.jsonl) and per-run summaries (*.rows.json / *.summary.json)
results/                analysis outputs (JSON), one file per experiment cell or analysis
logs/                   stdout of every run
check_keys.py           probes each provider key and prints measured limits
```

## Data

Every model call in the study is one `ModelResponded` record carrying provider, model, reasoning setting, token counts, latency, HTTP status, error class, raw text, and list-price cost. The paper's call count (about 41,000) is the number of `ModelResponded` records under `runs/` excluding the directories `*_discarded`, `*_old`, and `smoke*`, which hold aborted or superseded runs and are kept only for provenance.

Prompts for both task families are the string constants in `ledger/tasks_t1.py` and `ledger/tasks_t2.py`.

## Reproducing the tables

```
python3 -m ledger.analyze_hetero --out results/19-heterogeneity.md                       # Table II, Figure 1
python3 -m ledger.analyze_frontier runs/frontier/*.rows.json --out results/08-frontier.md # Table III
python3 -m ledger.analyze_escalation runs/frontier_escstep_k0/*.events.jsonl             # Table IV rows (nemotron target)
python3 -m ledger.analyze_escalation runs/frontier_target_amd_retry/*.events.jsonl --base-provider none
python3 -m ledger.analyze_stacks runs/stacks/stacks_deepseek_v4_flash_*.json              # Table VI rows (DeepSeek)
python3 -m ledger.figures --out figures
```

## Running new cells

API keys are read from `.env` (variables such as `MISTRAL_API_KEY`, `NVIDIA_API_KEY`, `GEMINI_API_KEY`, `AMD_API_KEY`); no keys are included. Example:

```
python3 -m ledger.frontier --task t1 --provider mistral --model ministral-8b-latest \
  --switch-provider nvidia --switch-model nvidia/nemotron-3-super-120b-a12b \
  --policies escalate_step --k 0 --n-traj 20 --n-steps 16 --ops 3 --out runs/new_cell
```

All experiments in the paper ran on free API tiers; `results/00-provider-limits.md` records every measured limit and outage.
