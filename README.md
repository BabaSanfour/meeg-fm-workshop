# Do foundation models beat classical EEG decoding?

Hands-on notebooks for a workshop on M/EEG foundation models. One small EEG dataset, left hand against right hand, decoded four ways on the same trials and the same train/test split: two classical decoders, a pretrained model (REVE) used frozen, and the same model fine-tuned.

## Notebooks

They are in `notebooks/`. Run them in order: each one saves its scores in `notebooks/results/`, and notebooks 3 to 5 read the scores of the earlier ones.

| | Notebook | Content |
|---|---|---|
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/meeg-fm-workshop/blob/main/notebooks/00_setup.ipynb) | `00_setup` | Check the computer, download the data |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/meeg-fm-workshop/blob/main/notebooks/01_data.ipynb) | `01_data` | The experiment, mu and beta rhythms, hand-made features and a first model |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/meeg-fm-workshop/blob/main/notebooks/02_classical.ipynb) | `02_classical` | CSP + LDA, tangent space + logistic regression |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/meeg-fm-workshop/blob/main/notebooks/03_fm_frozen.ipynb) | `03_fm_frozen` | Load REVE, embeddings and a linear probe, three other pretrained models |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/meeg-fm-workshop/blob/main/notebooks/04_fm_finetune.ipynb) | `04_fm_finetune` | How a network learns, fine-tuning REVE |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/meeg-fm-workshop/blob/main/notebooks/05_compare.ipynb) | `05_compare` | One table, per-person scores, confidence intervals and tests |

The notebooks are saved with their outputs, so they can be read without running anything.

## Run

**Colab**: click a badge, then *Runtime › Run all*. Start with `00_setup`.

**Your computer** (Python 3.10 or newer):

```bash
git clone https://github.com/BabaSanfour/meeg-fm-workshop.git
cd meeg-fm-workshop
pip install -e ".[jupyter]"      # or: uv venv && uv pip install -e ".[jupyter]"
jupyter lab
```

A laptop is enough. Fine-tuning (notebook 4) needs a GPU, so its runs are saved in the package and read by default; set `RUN_FINETUNE = True` to redo one.

## Data and model

- **Data**: BCI Competition IV, data set 2a (BNCI 2014-001): 9 people, 22 EEG channels, two sessions on different days. Downloaded by notebook 0 with [MOABB](https://moabb.neurotechx.com), about 780 MB. Tangermann et al. (2012), https://doi.org/10.3389/fnins.2012.00055
- **Model**: [REVE](https://huggingface.co/brain-bzh/reve-base) through [Braindecode](https://braindecode.org). El Ouahidi et al. (2025), https://arxiv.org/abs/2510.21585

## Layout

- `notebooks/`: the six notebooks, and `results/` with the scores written by notebooks 2 to 4.
- `meeg_fm/`: the helper code (data, figures, scoring, other models, fine-tuning), the two scripts that produced the saved fine-tuning runs, the cached REVE embeddings (`cache/`) and those saved runs (`precomputed/`).
- `figures/`: the concept figures embedded in the notebooks, and the script that draws them.

## Licence

Code: BSD 3-Clause, see `LICENSE`. The dataset and the REVE weights keep their own licences.
