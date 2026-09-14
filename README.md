# Keel: an event-sourced runtime for measuring the economics of recovery in LLM agent workflows

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22754512.svg)](https://doi.org/10.5281/zenodo.22754512)

Artifact for the paper *The Economics of Recovery: Measuring What Retry, Rollback, and Escalation Buy in Long-Horizon LLM Agent Workflows* (Zhouran Song, Independent Researcher, 2026; under review). The runtime is called Keel in the paper; the Python package is `ledger`. Pure standard library, no third-party dependencies, except `ledger/figures.py`, which needs matplotlib.

## Author and contact

Zhouran Song, Independent Researcher, Santa Clara, California. ORCID [0009-0001-2985-0012](https://orcid.org/0009-0001-2985-0012). Contact: songzhouran@gmail.com. Issues and pull requests on this repository are welcome.

## Citing

Until the paper appears, please cite the artifact and the preprint:

```bibtex
@misc{song2026recovery,
  author = {Song, Zhouran},
  title  = {The Economics of Recovery: Measuring What Retry, Rollback, and Escalation Buy in Long-Horizon {LLM} Agent Workflows},
  year   = {2026},
  note   = {Manuscript under review. Artifact: \url{https://github.com/szrusca/keel-recovery-artifact}}
}

@software{song2026keel,
  author    = {Song, Zhouran},
  title     = {Keel: event-sourced runtime and logs for the economics of recovery in {LLM} agent workflows},
  year      = {2026},
  version   = {v1.0.1},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22754512},
  url       = {https://doi.org/10.5281/zenodo.22754512}
}
```

The archived snapshot of this repository is Zenodo record 10.5281/zenodo.22754512 (all versions: 10.5281/zenodo.22754511).

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
  analyze_escalation.py target-only / escalate / escalate-and-return comparison
  analyze_stacks.py     per-stack accuracy and pairwise divergence; stacks_rebuild.py rebuilds a stacks JSON from its event log
  figures.py            paper figures (the only module that needs a third-party package: matplotlib)
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
python3 -m ledger.analyze_hetero --out results/19-heterogeneity.md                       # failure-point heterogeneity table and Figure 1
python3 -m ledger.analyze_frontier runs/frontier/*.rows.json --out results/08-frontier.md # policy cost-frontier table
python3 -m ledger.analyze_escalation runs/frontier_escstep_k0/*.events.jsonl             # escalation-variant table (nemotron target)
python3 -m ledger.analyze_escalation runs/frontier_target_amd_retry/*.events.jsonl --base-provider none
python3 -m ledger.analyze_stacks runs/stacks/stacks_deepseek_v4_flash_*.json              # serving-stack divergence table (DeepSeek)
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

## License

Code (`ledger/`, `check_keys.py`) is under the MIT License (`LICENSE`); data and logs (`runs/`, `results/`, `logs/`) are under CC BY 4.0 (`LICENSE-DATA.md`). Citation metadata is in `CITATION.cff`.
