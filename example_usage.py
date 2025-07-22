"""
Example usage of the MambaCCT utilities.

This script demonstrates two ways to train the model:
1. A quick single-subject experiment using ``run_mamba_example``.
2. A full leave-one-subject-out evaluation via ``utils.run_training``.
"""

from run_mamba_example import main as run_single_subject
from utils.run_training import main as run_loso


if __name__ == "__main__":
    # Run a simple single-subject experiment
    run_single_subject()

    # Uncomment the next line to perform a full LOSO evaluation
    # run_loso()
