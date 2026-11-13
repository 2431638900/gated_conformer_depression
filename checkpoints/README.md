# Checkpoints

Trained weights are not stored in git (each English DAIC-WOZ run is several hundred megabytes).

To evaluate the paper's main test result (segment-level Acc 94.92%, F1 0.9169, AUC 0.9874), copy the 18 May 2026 **frozen-HuBERT** checkpoint into this folder as `best_model.pth`:

```text
<original project>/checkpoints/best_model_ft0_20260518_015803_w1.70_f1at050.9264.pth
  ->  checkpoints/best_model.pth
```

`best_model_ft0.pth` from the same run is an equivalent copy. Do **not** use the April 2026 `best_model.pth` from the old folder: that file comes from the last-5-layer fine-tuning run (53.88M trainable parameters) and does not match the reported test numbers.

Then run:

```bash
python scripts/03_test_model.py --checkpoint best_model.pth
```

To train from scratch instead (frozen HuBERT, official split, seed=42):

```bash
bash scripts/train_freeze.sh
```
