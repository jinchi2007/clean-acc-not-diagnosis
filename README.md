# Clean Accuracy Is Not a Diagnosis

A noise-stress re-evaluation of eight lightweight 1D architectures for bearing fault
diagnosis. Companion code, data splits and per-seed result files for the manuscript.

**Repository:** <https://github.com/jinchi2007/clean-acc-not-diagnosis>

```bash
git clone git@github.com:jinchi2007/clean-acc-not-diagnosis.git
```

**The one-sentence result:** on the standard CWRU benchmark every architecture is
saturated above 97% clean accuracy, so clean accuracy cannot rank architectures; adding
test noise separates the same models by 55 points and reverses the ranking, and the
separation is mostly repaired by a one-line per-batch noise augmentation rather than by
architecture.

## What is in here

| Path | Contents |
|---|---|
| `code/` | data pipelines (CWRU, Paderborn), 8 model definitions, the single training entry point |
| `experiments/` | one script per experiment E0--E11, plus `stats.py`, `figures.py`, `check_tables.py` |
| `results/` | aggregated tables (`*.json`, `*.md`) and every raw run under `results/runs/<grid>/<run_id>.json` |
| `figures/` | vector PDFs and PNGs used in the paper |
| `paper/` | LaTeX sources (`_p1.tex _p2.tex _p3.tex`), bibliography, compiled `main.pdf` |
| `setup/` | environment install / activation / verification scripts |

## Results at a glance

| Experiment | What it measures | Headline |
|---|---|---|
| E0 saturation (64 runs) | clean accuracy under 4 split protocols | 8 architectures span 2.5 points; overlapping-window leakage inflates at most 0.63 points |
| E0b few-shot (16 runs) | same, 20 windows/class | leakage at most 0.9 points, so the contamination is in the protocol, not in leakage |
| E2 noise matrix (24 runs) | clean-trained models under test noise | 10 dB accuracy spans 0.43--0.98 (55 points); clean rank does not predict noisy rank (Spearman -0.38) |
| E4 augmentation (24 runs) | per-batch SNR ~ U[0,15] dB | mean 10 dB accuracy 0.699 -> 0.967 (Wilcoxon p < 1e-6), worst model 0.43 -> 0.95, clean cost 5.4 points |
| E5 wide-kernel stem (13 configs + 48 runs) | learnable multi-channel front end | +13 to +52 points at 10 dB for ~1% more parameters; channels, not kernel width, is the knob |
| E3 open set (48 runs) | unknown-class rejection | AUROC 0.90, TPR at 5% FPR 0.12--0.87; anti-correlated with noise robustness (rho = -0.64) |
| E6 noise families (24 runs) | harmonic / impulsive / drift noise | the four families induce four different rankings; robustness without naming the perturbation is unfalsifiable |
| E7 efficiency | parameters and latency | 3,018--747,722 parameters, 3.7k--115k windows/s on Apple M5 (MPS) |
| E8 Paderborn (16 runs) | second rig, two-class task | the inversion reproduces (Spearman rho = 0) and disappears under augmentation |
| E9 open set + augmentation (32 runs) | rejection with augmented training | the recipe that buys robustness loses rejection: mean AUROC 0.899 -> 0.797, TPR at 5% FPR 0.570 -> 0.388 (paired Wilcoxon p = 5e-7 / 6e-3) |
| E10 decision-layer scores (64 runs) | rejection scored three ways on the same networks | the rejection loss is mostly the softmax read-out: mean AUROC 0.953 clean / 0.940 augmented with a logit-space Mahalanobis score, so augmentation costs 0.013 instead of 0.103 |
| E11 penultimate-feature scores (64 runs) | the same distance measured in feature space | 0.952 clean / 0.938 augmented, indistinguishable from the logit-space values (p = 0.95 / 0.64): the gain comes from measuring distance, not from logit coordinates |

## Reproducing

Datasets are not redistributed; both are public.

```bash
# 1. environment (Python 3.12 + PyTorch; Apple Silicon MPS is used automatically)
bash setup/install.sh
source setup/env.sh

# 2. data: CWRU mat files, then build the four split protocols
python -m code.data.download_cwru
python -m code.data.cwru

# 3. experiments (each is resumable; raw runs land in results/runs/<grid>/)
python -m experiments.grid --grid e0_sat --seeds 0,1 --epochs 20
python -m experiments.grid --grid e2_noise_clean --seeds 0,1,2 --epochs 20
python -m experiments.grid --grid e4_aug --seeds 0,1,2 --epochs 20
python -m experiments.grid --grid e3_openset --seeds 0,1,2 --epochs 20
python experiments/e0_fewshot.py --seeds 0,1
python experiments/stem_probe.py
python experiments/e5_combo.py
python experiments/e6_noise_types.py
python experiments/e7_efficiency.py
python experiments/e8_paderborn.py
python experiments/e9_openset_aug.py --seeds 0,1 --epochs 20
python experiments/e10_scores.py --seeds 0,1 --epochs 20
python experiments/e11_penultimate.py --seeds 0,1 --epochs 20

# 4. tables, figures, and a cell-by-cell check of the manuscript tables
python -m experiments.stats
python -m experiments.figures
python experiments/check_tables.py

# 5. paper
bash paper/build.sh
```

A single training run takes 8--84 s on a laptop-class accelerator (Apple M5, MPS
backend); the full matrix is about 430 runs, i.e. a few hours.

The paper needs [tectonic](https://tectonic-typesetting.github.io/) (it finds `$TECTONIC`,
then `tectonic` on `PATH`, then `../tools/tectonic`, and falls back to `xelatex`); the first
build downloads the TeX bundle, later builds take a few seconds. All build products stay
under `paper/` (`main.tex`, `.tcache`, `.texhome`), and rebuilding rewrites `main.pdf`
with the same content but fresh PDF timestamps. `experiments/check_tables.py` works
before the paper is built: it falls back to the three `_p*.tex` sources.

## Submission bundle

bash pkg_submission.sh assembles `dist/mst_submission/` -- the concatenated
`main.tex`, `refs.bib`, only the six figure PDFs the manuscript includes, the
compiled `main.pdf` and the cover letter -- and zips it as
`dist/mst_submission.zip` for ScholarOne-style upload. The bundle is verified to
compile standalone (20 pages, 0 Overfull, 0 LaTeX/BibTeX warnings with tectonic
0.15.0). Re-run it after editing the sources so the shipped PDF matches.

## Hardware and software

All numbers in the paper come from one machine: Apple M5, 16 GB, macOS 26.5,
Python 3.12.16, PyTorch 2.14.1, MPS backend.
`setup/verify_env.py` reproduces the environment check and `setup/bench_mps.py` the
throughput numbers.

## Data

* CWRU: https://engineering.case.edu/bearingdatacenter -- cite Smith & Randall (2015).
* Paderborn: https://groups.uni-paderborn.de/kat/BearingDataCenter/ -- CC BY-NC 4.0,
  cite Lessmeier et al. (2016). The subset used here is the 18-bearing KAt set
  (6 healthy + 12 artificially damaged outer races).

## License

Code: MIT (see `LICENSE`). Manuscript text and figures: CC BY 4.0.
The datasets keep their own licenses.

## Citation

See `CITATION.cff`. If you use the noise-stress protocol, please cite the paper and
the two dataset papers.
