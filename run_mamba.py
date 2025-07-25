"""Run LOSO evaluation for the MambaCCT model.

This script follows the training logic used in the Jupyter notebooks
under ``notebooks/cct``. It repeatedly calls the single-split
``main`` function from ``run_mamba_example.py`` for each subject,
tracking results and printing a summary at the end.
"""

import time
import random
import numpy as np
import pandas as pd
import torch

from utils.config import NUM_SUBJECTS
from run_mamba_example import main as run_single_split


def run_loso(seed: int | None = None) -> None:
    """Run leave-one-subject-out evaluation for all subjects."""
    results_df = pd.DataFrame(
        columns=["Test Subject", "Val Subject", "Test Acc", "Seed"]
    )
    all_acc: list[float] = []
    all_loss: list[float] = []

    for test_sub in range(NUM_SUBJECTS):
        start_time = time.time()
        fold_seed = np.random.randint(2021) if seed is None else seed + test_sub

        print(f"seed is {fold_seed}")
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)

        val_sub = (test_sub + 1) % NUM_SUBJECTS
        print(f"Val Subject {val_sub + 1}:")

        acc, loss = run_single_split(test_sub, val_sub, fold_seed)
        all_acc.append(acc)
        all_loss.append(loss)

        time_elapsed = time.time() - start_time
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print("\n======================================")

        results_df.loc[len(results_df)] = {
            "Test Subject": test_sub + 1,
            "Val Subject": val_sub + 1,
            "Test Acc": acc,
            "Seed": fold_seed,
        }

    avg_acc = np.mean(all_acc)
    avg_loss = np.mean(all_loss)
    print(f"Average Test Accuracy: {avg_acc:.2f}%")
    print(f"Average Test Loss: {avg_loss:.4f}")
    print(results_df)


if __name__ == "__main__":
    run_loso()
