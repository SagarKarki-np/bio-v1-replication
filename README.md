# Independent Python reimplementation of the Ruslim et al. (2025) spiking V1 model

Code and summary results for the preprint

> Karki S. *To what extent do the reported emergent properties of a biologically grounded spiking V1 model reproduce across its released demonstration and published parameterizations, and how sensitive are those properties to ambiguities in homeostatic calibration and receptive-field estimation?* bioRxiv (preprint, not peer reviewed). [DOI to be added]

This repository is an **independent** reimplementation in Python of the model of

> Ruslim MA, Spencer MJ, Hogendoorn H, Meffin H, Lian Y, Burkitt AN. Emergence of sparse coding, balance and decorrelation from a biologically-grounded spiking neural network model of learning in the primary visual cortex. *PLOS Computational Biology* 21(11): e1013644 (2025). https://doi.org/10.1371/journal.pcbi.1013644

The original authors' code is at https://github.com/marko-ruslim/bio-spiking-v1-learning. This repository is not affiliated with or endorsed by the original authors.

## Status and caveats (please read)

- Research-code snapshot, not a polished package. There is no single command that reproduces every figure.
- The original MATLAB code was **read, not executed**. This is not a bit-exact port, and statements about MATLAB behaviour in the preprint come from code reading.
- Every condition is a **single run** (network seed 3002; training seed 303002 for the paper-protocol runs). Results are descriptive.
- "matlab_demo" in some folder names means the *demo-parameterized* branch (numerical settings read from the released demonstration).
- Parts of the code, analysis scripts and text were written with help from an AI assistant (Claude, Anthropic); the author reviewed them and is responsible for all content. Please run the scripts yourself before relying on any number.

## Relationship to the original code and licensing

This repository contains **no files from the original authors' repository** (no MATLAB code, data or figures). It is Python code written for this study. The original repository, at the time of writing, does not include a licence file, so this repository does not claim any rights over it and does not redistribute it. Where the preprint describes the original demonstration code, it is describing what the author read there. If the original authors state a preferred licence or attribution, it will be added here.

## Contents

| Path | What it is |
|---|---|
| `config.py`, `figure10_config.py` | Parameters (demo-parameterized and paper-derived) |
| `lgn_spikes.py`, `preprocessing.py`, `paper_image_dataset.py` | Image preprocessing and ON/OFF LGN spike generation |
| `network.py`, `simulation.py`, `plasticity.py`, `learning_batch.py` | Network, dynamics, triplet/symmetric STDP, learning batch |
| `train_v1.py` | Training / resume entry point (`python train_v1.py --help`) |
| `download_paper_images.py`, `preprocess_paper_*.py` | Download and preprocess a 1,000-image van Hateren sample |
| `analysis/` | Calibration, sparseness, STA, Gabor fitting, correlations, E/I balance, variant experiments |
| `preprint_figures/` | Scripts, tables and figures added for the preprint (A_d experiment, rate/sparseness bootstraps, one-synapse Poisson sweep) |
| `results/` | Small metadata and summary files only (no checkpoints or response archives) |
| `data/paper_van_hateren/metadata/selection_manifest.json` | Which images were sampled (seed 2027) and the source archive URL |

## Setup

```
python -m venv .venv
# activate it, then:
pip install -r requirements.txt
```

Python version used by the author: [to be filled in]. The code was developed on Windows.

## Data

Images are **not** included. `download_paper_images.py` fetches the van Hateren archive listed in the manifest and samples 1,000 images (seed 2027). The preprint notes that this sample is not verified to be identical to the original authors' 1,000 images. Check the dataset's terms of use before redistributing images.

## Large files

Checkpoints, response archives and STA outputs are large and are not in this repository. [Archive location/DOI to be added, e.g. Zenodo.]

## Seeds used

Network 3002; calibration 103002; Track B continuation 203002; training 303002; evaluations 2029 (rates, E/I), 2027 (Fig. 6 evaluations, firing-rate sweep, sparseness), 2028 (population balance); STA 2041; Gabor 2705 + neuron index.

## Known issues

- Some analysis scripts read from fixed relative `results/...` paths that exist only when the large local outputs are present.
- `analysis/firing_rates.py` and `analysis/sparseness.py` accept an optional `V1_ORIGINAL_PROJECT` environment variable for earlier archived data.
- Several scripts are one-off drivers for specific conditions (e.g. `figure10_*`).

## Citation

See the preprint above. [BibTeX to be added after posting.]

## License

MIT (see `LICENSE`).
