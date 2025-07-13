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

## Datasets:
The datasets used during the current study are available in the BCI Competition IV repository. The specific datasets used are 2a[1] and 2b[2], which can be accessed at \url{https://www.bbci.de/competition/iv/}.

[1] Tangermann, M. et al. Review of the BCI competition IV. Front. Neurosci. 6, DOI: 10.3389/fnins.2012.00055 (2012).
[2] Leeb, R. et al. Brain–computer communication: Motivation, aim, and impact of exploring a virtual apartment. IEEE
Transactions on Neural Syst. Rehabil. Eng. 15, 473–482, DOI: 10.1109/TNSRE.2007.906956 (2007).

## Usage

### Option 1: Modular Package (Recommended)

We provide a clean, modular package structure for easy experimentation and development:

```python
# Option A: Simple train/test split
from src.data.preprocessing import load_and_preprocess_bci_data
from src.models.stmamba_utils import create_complete_setup
from src.training.trainer import train_model
from src.utils.visualization import create_comprehensive_report

train_loader, test_loader, metadata = load_and_preprocess_bci_data()
setup = create_complete_setup()
results = train_model(setup, train_loader, test_loader)
create_comprehensive_report(results, setup['model_config'], setup['training_config'])

# Option B: LOSO evaluation (recommended for research)
from src.training.trainer import run_loso_evaluation
from src.utils.visualization import save_loso_report

# Complete LOSO evaluation in 3 lines!
results = run_loso_evaluation(n_classes=4, save_results=True)
save_loso_report(results, "loso_results")
print(f"Average accuracy: {results['avg_accuracy']:.2f}%")
```

**Features:**
- **LOSO Evaluation**: Subject-independent performance assessment (9-fold cross-validation)
- **Memory-efficient**: Automatic GPU/CPU fallback, gradient accumulation
- **Easy experimentation**: Simple configuration management
- **Comprehensive analysis**: Automated visualization and reporting
- **Reusable**: Clean modular structure for other projects
- **4-class & 2-class support**: Motor imagery classification (left, right, feet, tongue)

See `src/README.md` for detailed documentation and `example_usage.py` for complete examples.

### Option 2: Jupyter Notebooks

Individual notebook files are also available for step-by-step experimentation:
- `cct_bci2a_stmamba.ipynb` - Main STMambaCCT implementation
- `cct_bci2a_no_augmentation.ipynb` - Baseline implementation
- Various experiment notebooks in `cct_experiments/`

## Project Structure

```
EEGCCT/
├── src/                          # Modular package (recommended)
│   ├── data/                     # Data processing utilities
│   ├── models/                   # Model configurations
│   ├── training/                 # Training utilities
│   └── utils/                    # Visualization and analysis
├── model/                        # Core model implementations
├── notebooks/                    # Jupyter notebook experiments
├── data/                         # Raw dataset files
├── pickles/                      # Preprocessed data
└── example_usage.py              # Usage examples
```

## Citation
If our code was helpful to your research, we kindly ask that you cite our paper:
```
Not published yet. In Peer Review.
```
