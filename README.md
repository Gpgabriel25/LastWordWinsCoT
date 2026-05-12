# LastWordWinsCoT

**The Last Word Often Wins: A Format Confound in Chain-of-Thought Corruption Studies**

Gabriel Garcia · Independent Researcher · [gpgabriel25@gmail.com](mailto:gpgabriel25@gmail.com)

[arXiv:2605.10799](https://arxiv.org/abs/2605.10799) · [`paper/main.pdf`](paper/main.pdf)

---

This repo has  the paper, code, datasets, and result JSONs for a project that
started as trying to analyze just how much of CoT is rationalization, and a discovvery
that overshadowed the original paper and became  about how we measure chain-of-thought 
(CoT) faithfulness.

The short version: most corruption-based faithfulness studies (the kind where
you mess with a reasoning step and watch accuracy drop) are partly
measuring something they didn't intend to. When a chain ends with an explicit
"the answer is X" line (which is how almost every standard benchmark formats
its CoTs), corrupting positions near the end of the chain doesn't really tell
you those positions were doing computation. It tells you that's where the
answer text sits. Move the answer line, and the "load-bearing" location moves
with it.

That's the whole thesis, and it has a few uncomfortable consequences for the
literature. The rest of the paper is me trying very hard to break my own
finding and not succeeding.

## What's actually in here

- A within-dataset format ablation on GSM8K that collapses suffix-corruption
  sensitivity ~19× at 3B (N=300, p=0.022). Same reasoning steps, just no
  "the answer is X" line at the end.
- Conflicting-answer experiments across five model families at 7B —
  basically, give the model a correct chain but a wrong final answer and
  watch the wrong answer win. CC accuracy collapses to ≤0.02. (At larger
  scales the effect attenuates, which is itself interesting.)
- The same protocol applied to suffix-free chains inverts: now the
  *prefix* looks load-bearing (Δ=−0.77, p<10⁻¹²). Same models, same kind of
  prompt, opposite conclusion (which is exactly the diagnostic you'd want
  if the format itself was driving the original signal)
- Generation-time probes that try to rule out the "the model decided the
  answer before writing anything" explanation: <5% early commitment. So the
  effect isn't living in the *reasoning*, it's living in the
  *readout* — i.e. consumption time, not generation time.
- A three-step protocol I'm proposing as a minimum for any future
  corruption-based faithfulness paper: a question-only control, a format
  characterization, and an all-position sweep (not just the suffix).

The paper has the details and the JSONs in `results/summary/` have the 
raw numbers if you want to look at them directly.

## Repository layout

```
LastWordWinsCoT/
├── paper/                 # Final paper PDF + LaTeX source
│   ├── main.pdf
│   ├── main.tex
│   └── neurips_2026.sty
├── figures/               # The 11 figure PDFs used in the paper
├── data/                  # Curated experimental JSONL datasets (11 files)
├── src/
│   ├── datasets/          # build_*.py and generate_*.py — dataset construction
│   ├── inference/         # run_*_tpu.py — model evaluation on TPU (JAX)
│   ├── analysis/          # analyze_*.py, compute_*.py, integrate_*.py
│   └── figures/           # generate_figures.py, plot_*.py — figure builders
├── results/
│   └── summary/           # 81 per-experiment result JSONs cited in the paper
├── requirements.txt
├── CITATION.cff
├── LICENSE                # MIT — applies to all code (src/, requirements)
└── LICENSE-CONTENT        # CC BY 4.0 — applies to paper/, figures/, data/, results/
```

## License (dual)

- **Code** (`src/` and root config files): [MIT](LICENSE).
- **Paper, figures, datasets, and results** (`paper/`, `figures/`, `data/`,
  `results/`): [CC BY 4.0](LICENSE-CONTENT). Attribution required; commercial
  use and modification are fine with attribution.

## Reproducing the experiments

### What you'll need

- A Cloud TPU v5e-64 (or comparable; the inference scripts target JAX on TPU).
  The work was done on v5e — other hardware should be straightforward to port
  but I haven't tried.
- Python ≥ 3.11
- JAX ≥ 0.9.1, `libtpu` ≥ 0.37.0
- `pip install -r requirements.txt`

### Pipeline

The pipeline is roughly: build datasets -> run inference -> analyze ->
regenerate figures. Steps 1–2 are the expensive bits; analysis and figures
are cheap.

```bash
# 1. Build the root datasets from HuggingFace (or just use the JSONLs in data/)
python src/datasets/build_gsm8k_benchmark_v1.py --split test --limit 1000
python src/datasets/generate_conflicting_answer_dataset.py
python src/datasets/build_math_benchmark_v1.py
python src/datasets/generate_commonsense_dataset.py

# 2. Build experiment-specific variants
python src/datasets/build_gsm8k_stripped_neutral_v1.py
python src/datasets/build_factorial_dataset.py
python src/datasets/build_counterbalanced_placement_data.py

# 3. Run experiments on TPU (5–20 min each on v5e-64)
python src/inference/run_matched_format_ablation_tpu.py    --model-id Qwen/Qwen2.5-3B-Instruct
python src/inference/run_factorial_inference_tpu.py        --model-id Qwen/Qwen2.5-3B-Instruct
python src/inference/run_conflicting_answer_tpu.py         --model-id Qwen/Qwen2.5-7B-Instruct
python src/inference/run_counterbalanced_placement_tpu.py  --model-id Qwen/Qwen2.5-7B-Instruct
python src/inference/run_prefix_branch_probe_tpu.py        --model-id Qwen/Qwen2.5-3B-Instruct
python src/inference/run_neutral_filler_suffix_tpu.py      --model-id Qwen/Qwen2.5-3B-Instruct
python src/inference/run_early_stop_probe_tpu.py           --model-id Qwen/Qwen2.5-3B-Instruct

# 4. Analyze
python src/analysis/analyze_within_stable.py
python src/analysis/integrate_qwen3_results.py
python src/analysis/compute_phaseG_stats.py

# 5. Rebuild figures
python src/figures/generate_figures.py        --results-dir results/summary --output-dir figures
python src/figures/generate_fw_scale_figure.py
python src/figures/plot_protocol_schematic.py
python src/figures/plot_scale_conditioning_summary.py
```

Random seeds are fixed and statistical tests are exact paired tests where
applicable, so the numbers should land on the same values reported in the
paper. If they don't, please open an issue — I'd genuinely like to know.

`data/` ships with the 11 curated JSONLs that back every primary result in
the paper, and `results/summary/` ships with the 81 result JSONs (raw model
outputs plus accuracy / FW tables) used in the analyses and figures. So
you don't actually need a TPU to re-derive the tables and plots — only to
re-run inference from scratch.

## Datasets included

| File | What it is |
|---|---|
| `phase0_position_control_gsm8k_v1.jsonl` | GSM8K-v1 (N=100, suffix-bearing benchmark) |
| `phase0_position_control_gsm8k_v1_n300.jsonl` | GSM8K-v1 (N=300, matched ablation pair) |
| `phase0_position_control_gsm8k_stripped_neutral_v1.jsonl` | GSM8K-stripped (answer line removed) |
| `phase0_conflicting_answer_gsm8k_v2.jsonl` | GSM8K-conflict-v2 (N=500, direct causal test) |
| `phase0_conflicting_answer_commonsense_v1.jsonl` | Commonsense conflict (N=150, 5 domains) |
| `phase0_factorial_2x2_gsm8k.jsonl` | 2×2 reasoning × answer-line factorial |
| `phase0_counterbalanced_placement_gsm8k_v1.jsonl` | Counterbalanced placement controls |
| `phase0_position_control_hard_v3.jsonl` | Hard-v3 (synthetic, suffix-free, N=60) |
| `phase0_bidirectional_ablation_hard_v3.jsonl` | Bidirectional ablation (4-condition) |
| `phase0_position_control_math_v1.jsonl` | MATH benchmark (N=100) |
| `phase0_position_control_math_stripped_neutral_v1.jsonl` | MATH-stripped (N=100) |

## Citing

If this is useful, please cite the arXiv version:

```bibtex
@misc{garcia2026wordwinsformatconfound,
  title         = {The Last Word Often Wins: A Format Confound in Chain-of-Thought Corruption Studies},
  author        = {Gabriel Garcia},
  year          = {2026},
  eprint        = {2605.10799},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.10799}
}
```

GitHub's "Cite this repository" button also works (it reads
[`CITATION.cff`](CITATION.cff)).

## Contact

Gabriel Garcia · [gpgabriel25@gmail.com](mailto:gpgabriel25@gmail.com)
