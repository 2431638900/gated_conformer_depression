# Gated Multi-Scale Conformer for Speech-Based Depression Detection

Code for the paper *Hierarchical Self-supervised Speech Representation Learning for Depression Detection Based on Gated Multi-Scale Conformer Framework*.

The model uses a frozen HuBERT backbone, learnable hierarchical layer aggregation, a multi-scale depression feature extractor, a gated multi-scale Conformer (6 layers, width 256, 8 heads), and statistical attention pooling. A lightweight language adapter is used for subject-level transfer to Chinese MODMA.

This repository was cleaned from the original project so that a public release can follow the official AVEC 2017 split and the frozen-HuBERT training setup described in the paper.

## What this release can and cannot reproduce

The reported **test-set** numbers on DAIC-WOZ (segment-level Acc 94.92%, Precision 0.9293, Recall 0.9049, Specificity 0.9691, F1 0.9169, AUC 0.9874; confusion matrix 658/21/29/276) come from the **18 May 2026 frozen-HuBERT** run (18.44M trainable parameters, seed 42). Placing that checkpoint in `checkpoints/best_model.pth` and evaluating on the official test split reproduces those figures.

The **validation** full-model row in the ablation / representation tables (F1 0.9140, Acc 94.81%, AUC 0.9837) comes from a **1 April 2026** run that fine-tuned the last five HuBERT layers (53.88M trainable parameters). The frozen-HuBERT run's own validation score was F1 0.9264, Acc 95.52%, AUC 0.9837; that row is not the one printed in those tables. Re-training with the default frozen setting will therefore not regenerate the 0.9140 validation entry. See `results/EXPERIMENT_LOG.md`.

DAIC-WOZ and MODMA audio are not included. You must obtain them under their original licenses.

## Setup

Python 3.9+ and a CUDA GPU are recommended (the original runs used PyTorch 2.1 with CUDA 12).

```bash
pip install -r requirements.txt
# if you do not already have a local HuBERT checkpoint:
#   export HUBERT_MODEL=facebook/hubert-base-ls960
# or point to a local folder:
#   export HUBERT_MODEL=/path/to/hubert-base-ls960
```

## Prepare the official DAIC-WOZ split

```bash
python scripts/prepare_official_split.py --daic-woz-root /path/to/DAIC-WOZ
```

This uses the official 107 / 35 / 47 participant lists in `splits/avec2017/` and writes segments to `data/official_split/`. A complete run should yield 4,589 / 983 / 984 segments. Do not run the legacy random resplit in `scripts/legacy/`.

## Train and test (frozen HuBERT, paper default)

```bash
# train + test
bash scripts/train_freeze.sh

# or separately
python scripts/02_train_model.py --finetune-layers 0
python scripts/03_test_model.py --checkpoint best_model.pth
```

The last-five-layer setting used only for the April validation / ablation full-model numbers is:

```bash
python scripts/02_train_model.py --finetune-layers 5
```

## Other experiments

```bash
python scripts/04_ablation_study.py
python scripts/05_baseline_comparison.py
python scripts/09_prepare_modma.py --modma-root /path/to/audio_lanzhou_2015
python scripts/10_cross_lingual.py
python scripts/11_train_language_adapter.py
```

Logged results that match the paper tables are in `results/`.

## Default training settings

AdamW, learning rate \(2\times 10^{-4}\), weight decay \(10^{-3}\), 8-epoch warmup, at most 80 epochs, early stopping patience 10, Focal loss \(\gamma=2\) with MDD class weight 1.70, label smoothing 0.1, Mixup \(\alpha=0.2\), SpecAugment 4\(\times\)35, gradient accumulation 4, grad clip 1.0, EMA 0.999, dropout 0.35, DropPath 0.1, seed 42.

## License

Code is released under the MIT License. DAIC-WOZ and MODMA remain subject to their own data-use agreements.
