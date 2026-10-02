# Single-Channel Speaker Distance Estimation: Initial work

> ### ℹ️ These results assume time calibration
>
> The synthetic benchmark used here presents every clip aligned to the true
> propagation delay, with a fixed source level. That amounts to assuming the
> microphone and the loudspeaker are **synchronised**, so the time of flight is
> directly observable.
>
> Under that assumption the task is considerably easier, and the reported figures
> reflect it. In the wild, synchronisation between source and receiver is rarely
> available: a recording seldom starts at a known instant relative to when the talker
> began speaking.
>
> **For results that do not assume synchronisation**, see
> [`../rir-analysis/`](../rir-analysis/) and [`../calibration/`](../calibration/),
> which remove the timing and level cues by construction and measure what remains.

This folder is kept so the two published papers stay reproducible and so the
progression of the work remains legible.

- **The architecture.** The CRNN here is the same network used in all later stages,
  now maintained once in [`src/speaker_distance/models/`](../../src/speaker_distance/models/).
- **The problem framing.** Single-channel, single-microphone distance regression on
  reverberant speech.
- **The real-corpus evaluation protocol.** VoiceHome-2, STARSS23 and QMULTIMIT, with
  the splits now published as annotations in [`data/labels/`](../../data/labels/).

## How to read the numbers

They describe performance in the **time-calibrated** setting.

## Contents

- 📄 `model.py` — the original network and Lightning wrapper
- 📄 `synthetic.py`, `QMULTIMIT.py`, `STARS23.py`, `VoiceHome.py` — dataset loaders
- 📓 `training_synthetic.py`, `training_realData.py`, `training_QMULTIMIT.py`

## Running it

```bash
pip install -e .
export SPEAKER_DISTANCE_DATA=/path/to/real/corpora     # VoiceHome2, STARS23, QMULTIMIT
export SPEAKER_DISTANCE_NOISE=/path/to/noise           # WHAM!-derived splits
python papers/estimation-2023-2024/training_realData.py
```

Set `SPEAKER_DISTANCE_LOGGER=wandb` for Weights & Biases instead of CSV. Check what
resolved where with `python -m speaker_distance.paths`.

### The datasets these need

**The real-corpus scripts work today.** `training_realData.py` and
`training_QMULTIMIT.py` read directories of WAV files, so point
`SPEAKER_DISTANCE_DATA` at your copies of VoiceHome-2, STARSS23 and QMULTIMIT and
they run.

**`training_synthetic.py` uses an old version of the dataset. Use [`../rir-analysis/`](../rir-analysis/) instead. Its `data.py` reads the current corpus, and `synthetic_baseline` gives the same dataset.

| want | use |
|---|---|
| time- and level-calibrated | `synthetic_baseline` |
| no timing or level cue | `synthetic_both` |
| no timing cue | `synthetic_onset_randomized` |
| no level cue | `synthetic_gain_var` |

All four ship in the Zenodo record linked from the
[root README](../../README.md#datasets).

> The exact repository state as published with these papers is tagged
> [`v1.0-taslp2024`](https://github.com/michaelneri/audio-distance-estimation/tree/v1.0-taslp2024). Code there expects the
> old `dataset_wav/` layout.

## Citation

```bibtex
@ARTICLE{Neri_TASLP_2024,
  author={Neri, Michael and Politis, Archontis and Krause, Daniel A. and Carli, Marco and Virtanen, Tuomas},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  title={{Speaker Distance Estimation in Enclosures from Single-Channel Audio}},
  year={2024},
  volume={32},
  pages={2242-2254},
  doi={10.1109/TASLP.2024.3382504}
}

@INPROCEEDINGS{Neri_WASPAA_2023,
  author={Neri, Michael and Politis, Archontis and Krause, Daniel A. and Carli, Marco and Virtanen, Tuomas},
  booktitle={2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  title={{Single-Channel Speaker Distance Estimation in Reverberant Environments}},
  year={2023},
  pages={1-5},
  doi={10.1109/WASPAA58266.2023.10248087}
}
```
