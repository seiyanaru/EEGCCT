# EEGCCT - Compact Convolutional Transformer for MI EEG-based BCIs
PyTorch implementation of "Compact Convolutional Transformer for Subject-Independent Motor Imagery EEG-based BCIs"

## Abstract
![image](https://github.com/user-attachments/assets/6afeffc4-459b-4551-ad33-e868a453ab25)
![image](https://github.com/user-attachments/assets/591da44d-0ce5-455a-ab83-81424d51dc05)
This paper introduces two versions of EEGCCT, an adaptation of the Compact Convolutional Transformer (CCT) model for EEG analysis in motor imagery tasks. The EEGCCT model distinguishes itself in several key aspects:
1. Hybrid Model Structure: EEGCCT combines the global, long-range perspective provided by Transformers with the local feature extraction capabilities of CNNs.
2. Subject Independence: EEGCCT emphasizes its ability to generalize across several subjects. This makes EEGCCT especially well-suited for a variety of BCI applications in which subject-specific training data may be scarce, particularly when assessed through the application of the LOSO approach.
3. Handling Limited Data: Enhancement in performance with a smaller parameter size is a major advantage of EEGCCT over models such as Conformer, Hybrid s-CViT, and Hybrid t-CViT.

## Requirements:
* Python 3.8.0
* Pytorch 1.11.0
* torchvision=0.12.0
* pandas=1.5.2
* numpy=1.19.5
* cudatoolkit=11.3.1

### Environment setup
We recommend creating and activating the provided `eegcct` conda environment
before running any scripts or notebooks:

```bash
conda activate eegcct
```

The environment includes all required Python packages such as `torch`.

## Datasets:
The datasets used during the current study are available in the BCI Competition IV repository. The specific datasets used are 2a[1] and 2b[2], which can be accessed at \url{https://www.bbci.de/competition/iv/}.

[1] Tangermann, M. et al. Review of the BCI competition IV. Front. Neurosci. 6, DOI: 10.3389/fnins.2012.00055 (2012).
[2] Leeb, R. et al. Brain–computer communication: Motivation, aim, and impact of exploring a virtual apartment. IEEE
Transactions on Neural Syst. Rehabil. Eng. 15, 473–482, DOI: 10.1109/TNSRE.2007.906956 (2007).

## Usage

### Quick Start

Run a single-subject experiment:

```bash
python run_mamba_example.py --test-subject 0 --val-subject 1
```

To run all nine folds of LOSO evaluation:

```bash
python run_mamba.py
```

For a full leave-one-subject-out evaluation across all subjects:

```bash
python utils/run_training.py
```

### Jupyter Notebooks

Notebook files are grouped under the `notebooks/` directory by model type:

- `notebooks/stmamba/` – STMambaCCT experiments
- `notebooks/cct/` – Baseline CCT implementations
- `notebooks/mb_performer/` – Performer variants
- `notebooks/cct_experiments/` – Additional CCT experiments

You can run these notebooks individually for step-by-step exploration.

## Project Structure

```
EEGCCT/
├── utils/                        # Data loading and training helpers
│   ├── config.py                 # Centralized parameters
│   ├── data_utils.py             # Loading/augmentation routines
│   ├── training_utils.py         # Train/test helpers
│   └── run_training.py           # Example LOSO experiment script
├── model/                        # Core model implementations
├── notebooks/                    # Jupyter notebook experiments
├── run_mamba_example.py          # Single-subject training example
├── run_mamba.py                  # LOSO evaluation wrapper
├── data/                         # Raw dataset files
├── pickles/                      # Preprocessed data
├── example_usage.py              # Usage examples
└── loso_example_simple.py        # Minimal LOSO wrapper
```

## Citation
If our code was helpful to your research, we kindly ask that you cite our paper:
```
Not published yet. In Peer Review.
```
