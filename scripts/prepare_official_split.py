#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the official DAIC-WOZ / AVEC 2017 split used in the paper.

This script does NOT randomly re-split participants. It keeps the official
lists (107 train / 35 development / 47 test) and applies the same
transcript-based segmentation used in the original project:

  - keep only Participant turns (drop Ellie)
  - group every 5 transcript sentences into one 30 s segment

Expected paper segment counts after a complete, successful run:
  train 4,589 / val 983 / test 984

The raw DAIC-WOZ corpus is not redistributed. Provide it with --daic-woz-root
(or the DAIC_WOZ_ROOT environment variable). The directory should contain
participant folders such as 303_P/303_AUDIO.wav and 303_P/303_TRANSCRIPT.csv.
Official split CSVs from this repository (splits/avec2017/) are used by
default; you may also place the original AVEC files in the corpus root.

Test-set labels are taken from full_test_split.csv (PHQ_Binary / PHQ_Score).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import Config, config
from preprocessing.transcript_augmentor import TranscriptBasedAugmentor


SPLIT_DIR = PROJECT_ROOT / "splits" / "avec2017"
EXPECTED = {"train": 4589, "val": 983, "test": 984}


def _first_existing(paths):
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path)
    return None


def load_official_lists(daic_root: Path) -> dict:
    """Load official AVEC participant lists and attach test labels."""
    train_csv = _first_existing([
        SPLIT_DIR / "train_split_Depression_AVEC2017.csv",
        daic_root / "train_split_Depression_AVEC2017.csv",
    ])
    dev_csv = _first_existing([
        SPLIT_DIR / "dev_split_Depression_AVEC2017.csv",
        daic_root / "dev_split_Depression_AVEC2017.csv",
    ])
    test_ids_csv = _first_existing([
        SPLIT_DIR / "test_split_Depression_AVEC2017.csv",
        daic_root / "test_split_Depression_AVEC2017.csv",
    ])
    test_labels_csv = _first_existing([
        SPLIT_DIR / "full_test_split.csv",
        daic_root / "full_test_split.csv",
    ])
    missing = [name for name, path in [
        ("train", train_csv), ("dev", dev_csv),
        ("test ids", test_ids_csv), ("test labels", test_labels_csv),
    ] if path is None]
    if missing:
        raise FileNotFoundError(
            "Missing official split files: " + ", ".join(missing)
            + f"\nLooked in {SPLIT_DIR} and {daic_root}"
        )

    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv)
    test_df = pd.read_csv(test_ids_csv)
    labels = pd.read_csv(test_labels_csv)

    # Harmonize column names
    for df in (train_df, dev_df, test_df, labels):
        rename = {}
        for col in df.columns:
            key = col.strip().lower().replace(" ", "_")
            if key in {"participant_id", "participantid"}:
                rename[col] = "Participant_ID"
            elif key in {"phq8_binary", "phq_binary"}:
                rename[col] = "PHQ8_Binary"
            elif key in {"phq8_score", "phq_score"}:
                rename[col] = "PHQ8_Score"
        if rename:
            df.rename(columns=rename, inplace=True)

    test_df = test_df.merge(
        labels[["Participant_ID", "PHQ8_Binary", "PHQ8_Score"]],
        on="Participant_ID",
        how="left",
    )
    if test_df["PHQ8_Binary"].isna().any():
        missing_ids = test_df.loc[test_df["PHQ8_Binary"].isna(), "Participant_ID"].tolist()
        raise ValueError(f"Test labels missing for participants: {missing_ids}")

    return {"train": train_df, "val": dev_df, "test": test_df}


def process_split(name: str, df: pd.DataFrame, daic_root: Path,
                  output_root: Path, augmentor: TranscriptBasedAugmentor) -> pd.DataFrame:
    out_dir = output_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"Processing official {name} split: {len(df)} participants")
    print("=" * 60)
    aug_df = augmentor.process_dataset(data_df=df, data_dir=daic_root, output_dir=out_dir)
    # Store portable relative paths inside the official-split directory
    if "Audio_Path" in aug_df.columns:
        aug_df = aug_df.copy()
        aug_df["Audio_Path"] = aug_df["Audio_Path"].map(
            lambda p: str(Path(name) / Path(str(p)).name)
        )
    csv_path = out_dir / f"augmented_{name}.csv"
    aug_df.to_csv(csv_path, index=False)
    print(f"Wrote {len(aug_df)} segments to {csv_path}")
    return aug_df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the official DAIC-WOZ (AVEC 2017) split for this paper."
    )
    parser.add_argument(
        "--daic-woz-root",
        type=str,
        default=None,
        help="Directory that contains 303_P/, train_split_*.csv, etc. "
             "Defaults to $DAIC_WOZ_ROOT or config.DATA_DIR.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/official_split).",
    )
    args = parser.parse_args()

    daic_root = Path(args.daic_woz_root).expanduser() if args.daic_woz_root else Config.DATA_DIR
    output_root = Path(args.output_dir).expanduser() if args.output_dir else Config.OFFICIAL_SPLIT_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    if not daic_root.exists():
        raise FileNotFoundError(
            f"DAIC-WOZ root does not exist: {daic_root}\n"
            "Pass --daic-woz-root or set DAIC_WOZ_ROOT."
        )

    lists = load_official_lists(daic_root)
    print(
        "Official participant counts: "
        f"train={len(lists['train'])}, val={len(lists['val'])}, test={len(lists['test'])}"
    )

    augmentor = TranscriptBasedAugmentor(
        sample_rate=config.SAMPLE_RATE,
        sentences_per_sample=config.SENTENCES_PER_SAMPLE,
        target_duration=config.AUDIO_DURATION,
        padding=0.1,
    )

    results = {}
    for name in ("train", "val", "test"):
        results[name] = process_split(name, lists[name], daic_root, output_root, augmentor)

    info = {
        "split_method": "official_avec2017",
        "train_participants": int(lists["train"]["Participant_ID"].nunique()),
        "val_participants": int(lists["val"]["Participant_ID"].nunique()),
        "test_participants": int(lists["test"]["Participant_ID"].nunique()),
        "train_participant_ids": [int(x) for x in lists["train"]["Participant_ID"].tolist()],
        "val_participant_ids": [int(x) for x in lists["val"]["Participant_ID"].tolist()],
        "test_participant_ids": [int(x) for x in lists["test"]["Participant_ID"].tolist()],
        "train_samples": int(len(results["train"])),
        "val_samples": int(len(results["val"])),
        "test_samples": int(len(results["test"])),
        "train_depression_count": int((results["train"]["PHQ8_Binary"] == 1).sum()),
        "val_depression_count": int((results["val"]["PHQ8_Binary"] == 1).sum()),
        "test_depression_count": int((results["test"]["PHQ8_Binary"] == 1).sum()),
        "expected_segment_counts": EXPECTED,
    }
    with open(output_root / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60)
    print("Official split preparation finished")
    print("=" * 60)
    ok = True
    for name in ("train", "val", "test"):
        got = len(results[name])
        exp = EXPECTED[name]
        flag = "OK" if got == exp else "MISMATCH"
        if got != exp:
            ok = False
        print(f"  {name:5s}: {got:5d} segments (paper expects {exp}) [{flag}]")
    print(f"Output: {output_root}")
    if not ok:
        print(
            "\nSegment counts do not match the paper. Check that every official "
            "participant folder is present and complete (AUDIO + TRANSCRIPT)."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
