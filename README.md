# Gated Multi-Scale Conformer for Speech-Based Depression Detection

Code for the paper *Hierarchical Self-supervised Speech Representation Learning for Depression Detection Based on Gated Multi-Scale Conformer Framework*.

The model uses a frozen HuBERT backbone, learnable hierarchical layer aggregation, a multi-scale depression feature extractor, a gated multi-scale Conformer (6 layers, width 256, 8 heads), and statistical attention pooling. A lightweight language adapter is used for subject-level transfer to Chinese MODMA.

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

DAIC-WOZ and MODMA audio are not included. You must obtain them under their original licenses.

```bash
python scripts/prepare_official_split.py --daic-woz-root /path/to/DAIC-WOZ
```

This uses the official AVEC 2017 participant lists in `splits/avec2017/` (107 / 35 / 47) and writes segments to `data/official_split/`. A complete run should yield 4,589 / 983 / 984 segments.

## Train and test (frozen HuBERT)

```bash
# train + test
bash scripts/train_freeze.sh

# or evaluate an existing checkpoint
python scripts/03_test_model.py --checkpoint best_model.pth
```

Place the trained weights at `checkpoints/best_model.pth` before testing. The default training setting freezes all HuBERT parameters (`--finetune-layers 0`).

## Other experiments

```bash
python scripts/04_ablation_study.py
python scripts/05_baseline_comparison.py
python scripts/09_prepare_modma.py --modma-root /path/to/audio_lanzhou_2015
python scripts/10_cross_lingual.py
python scripts/11_train_language_adapter.py
```

Logged results corresponding to the paper tables are in `results/`.

## Default training settings

AdamW, learning rate \(2\times 10^{-4}\), weight decay \(10^{-3}\), 8-epoch warmup, at most 80 epochs, early stopping patience 10, Focal loss \(\gamma=2\) with MDD class weight 1.70, label smoothing 0.1, Mixup \(\alpha=0.2\), SpecAugment 4\(\times\)35, gradient accumulation 4, grad clip 1.0, EMA 0.999, dropout 0.35, DropPath 0.1, seed 42.

## License

Code is released under the MIT License. DAIC-WOZ and MODMA remain subject to their own data-use agreements.
