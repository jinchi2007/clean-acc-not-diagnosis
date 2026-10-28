# What is in results/

Two kinds of files:

* `results/runs/<grid>/<run_id>.json` — **one file per training run**, the ground truth for
  every number in the paper. `run_id` encodes the configuration
  (`<model>_<split>_seed<N>_ep<E>[_hold369][_aug0-15][_stem<KxS>]`). Written incrementally,
  so a grid can be resumed; nothing in the paper is hand-typed from these files.
* `results/<grid>.json` + `results/<grid>.md` — the aggregate that `stats.py` / `figures.py` /
  `check_tables.py` read, produced by each experiment script together with a markdown table.

## Raw run schema

```
{ cfg: {...}, data_meta: {...}, n_params, n_train,
  test_acc, best_test_acc, train_time_s, history: [...],
  noise: {kind, acc_by_snr},              # if noise_snr / noise_snrs was set
  openset: {auroc, tpr_at_95fpr, ...},    # if holdout_classes was set
  openset_scores: {softmax|energy|mahalanobis: {auroc, tpr95}}  # E10 only
}
```

## Aggregated schemas the manuscript tables are checked against

| file | keys used by `experiments/check_tables.py` |
|---|---|
| `e0_sat.json` | `table[model][split] = [mean, std]` over seeds (leaky, safe, crossload, filedisjoint) |
| `e0_fewshot.json` | `table[model][split]` with `n_per_class = 20` |
| `e2_noise_clean.json` | `table[model]["+10dB" / "+5dB" / "+0dB" / "-5dB" / "clean"] = [mean, std]` |
| `e4_aug.json` | same shape, model trained with SNR ~ U[0, 15] per batch |
| `e3_openset.json` | `table[model][hold369|hold147] = {auroc, tpr95, known_acc}` |
| `e9_openset_aug.json` | same shape for augmented training, plus `paired_vs_e3.metrics` (32 paired runs) |
| `e10_scores.json` | `table[model][clean|aug][score] = {auroc, tpr95}`; `stats.score_comparison` holds the paired Wilcoxon |
| `e11_penultimate.json` | same shape with scores `softmax`, `mahalanobis`, `mahalanobis_feat`; per-model feature width in `table[model][mode][\"feat_dim\"]` |
| `stem_ablation.json` | `rows[] = {label, backbone, k, s, c, clean, by_snr{"10.0","5.0","0.0"}}` |
| `e5_combo.json` | `table[model][condition] = {clean, "10.0", "5.0", "0.0"}`, plus `deltas_vs_plain` |
| `e6_noise_types.json` | `tables[model][kind][snr] = [mean, std]`, `gaussian_reference` (from E2) |
| `e7_efficiency.json` | `rows[] = {model, n_params, lat_b1_ms, lat_b256_ms, lat_b256_per_sample_ms, thr_b256}` |
| `e8_paderborn.json` | `table["model:plain"|"model:aug"] = {clean{mean,std}, snr10, snr5, snr0}`, `rank` |
| `stats.json` | `tables.T1_clean_trained`, `tables.T2_aug_vs_clean[key] = {clean_trained, aug, delta, p_holm}` |

## Regenerating

```bash
source setup/env.sh
python -m experiments.grid --grid e2_noise_clean --seeds 0,1,2 --epochs 20   # then the other grids
python -m experiments.stats            # results/stats.json
python -m experiments.figures          # figures/*.pdf and *.png
python experiments/check_tables.py     # every table cell against the JSON above
```

The datasets (`data/`) are not distributed; `code/data/download_cwru.py` fetches CWRU and
`code/data/paderborn.py` parses the Paderborn KAt archive (see the top-level README).

## Unused runs

results/runs/e0_sat/ holds the 64 runs that enter results/e0_sat.json (seeds 0 and 1).
Nine seed-2 runs produced by an abandoned job are kept in results/extra_runs/e0_sat/
and are deliberately excluded from every aggregation.
