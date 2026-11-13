# Data (not redistributed)

DAIC-WOZ and MODMA are licensed corpora and are not included here.

1. Obtain DAIC-WOZ from the official distributors and point `DAIC_WOZ_ROOT` (or `--daic-woz-root`) at the directory that contains folders such as `303_P/`.
2. Official AVEC 2017 participant lists are already in `splits/avec2017/`.
3. Build the paper split (107 / 35 / 47 participants; expected 4,589 / 983 / 984 segments):

```bash
python scripts/prepare_official_split.py --daic-woz-root /path/to/DAIC-WOZ
```

Segments are written to `data/official_split/`. Do not run `scripts/legacy/01_resplit_dataset.py`: that script created a random 132/28/29 split and will not reproduce the paper.

For MODMA:

```bash
python scripts/09_prepare_modma.py --modma-root /path/to/audio_lanzhou_2015
```
