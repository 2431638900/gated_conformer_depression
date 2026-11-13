# Experiment log (mapping to the paper)

All paths below refer to the original project folder `conformer_depression_v2` unless noted. Copies of the corresponding log files are kept in this `results/` directory.

## DAIC-WOZ official split

- Participants: 107 train / 35 development / 47 test (AVEC 2017).
- Segments reported in the paper and in both English training logs: 4,589 / 983 / 984.
- Test-set class counts in the paper: 679 HC / 305 MDD.
- Official participant lists: `splits/avec2017/`.

The later `data/augmented_stratified` tree in the original folder (5 May 2026) is a **random 132 / 28 / 29 resplit** (4,680 / 882 / 994) and must not be used to reproduce the paper. This release writes the official split to `data/official_split/`.

## Table of reported English test metrics

Source: `test_results_20260518_102206_ft0.json` (18 May 2026, frozen HuBERT, 18.44M trainable parameters).

| Metric | Paper | Log |
| --- | --- | --- |
| Accuracy | 94.92% | 94.918699... |
| Precision | 0.9293 | 0.9292929... |
| Recall | 0.9049 | 0.904918... |
| F1 | 0.9169 | 0.9169435... |
| AUC | 0.9874 | 0.9874357... |
| Confusion | 658 / 21 / 29 / 276 | same |
| N | 984 | 984 |

Matching training log: `train_20260518_015803_ft0.log` (best val F1@0.5 = 0.9264, Acc 95.52%, AUC 0.9837 at epoch 41; early stop at epoch 51; seed 42; lr \(2\times 10^{-4}\)).

The 1 April 2026 test file `test_results_20260401_185847.json` does **not** match the paper (Acc 94.82%, F1 0.9134, AUC 0.9869, CM 664/15/36/269). That run fine-tuned the last five HuBERT layers (53.88M trainable parameters).

## Ablation and representation tables (validation)

Source (seed 42, first repeat only, not the five-seed mean):

- `ablation_study_20260409_042341.json` / `.csv`
- `representation_ablation_20260413_092411.json` / `.csv`

The “Full Model” / “WLA” row (F1 0.9140, Acc 94.81%, AUC 0.9837) is the 1 April 2026 last-5-layer checkpoint, evaluated once. Variant rows match the paper to the reported rounding.

## Reproduced baselines (validation, seed 42)

Source: `baseline_comparison_20260409_203636.json` (the file was truncated during the significance-test section, but the single-run `part1` block is complete).

| Method | Acc | F1 | AUC |
| --- | --- | --- | --- |
| CNN | 57.17 | 0.4718 | 0.6115 |
| BiLSTM+Attention | 88.50 | 0.8144 | 0.9204 |
| HuBERT+Linear | 86.47 | 0.7925 | 0.9330 |
| HuBERT+Transformer | 93.39 | 0.8918 | 0.9805 |
| Ours (April full model) | 94.81 | 0.9140 | 0.9837 |

## MODMA (Table 5)

Source: `cross_lingual_20260410_113704.json`.

The JSON also contains Freeze-7 and LoRA (rank=8) runs that are not reported in the paper. The Language Adapter has 199,042 parameters; downstream full fine-tuning (HuBERT frozen) has 18,440,855 parameters. This transfer experiment was logged on 10 April, before the 18 May frozen English retraining, so the English source weights are those of the April checkpoint.

## Files not used as paper evidence

Older `train_*.log`, `history_*.json`, and intermediate checkpoints from March 2026 are exploratory runs and are not copied here.
