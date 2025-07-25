"""Run leave-one-subject-out evaluation for the MambaCCT model.

This script mirrors the training routine used in ``notebooks/cct`` but can be
executed from the command line. It loads the preprocessed EEG data via the
``utils`` package and trains the :class:`MambaCCT` model for each subject in
turn.
"""

from __future__ import annotations

import argparse
import logging
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from utils.config import MODEL_PARAMS, TRAINING_PARAMS, NUM_SUBJECTS, DEVICE
from utils.data_utils import get_source_data, prepare_dataloaders
from utils.training_utils import EarlyStopping, train_model, test_model
from model import MambaCCT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_single_fold(test_subject: int, val_subject: int, seed: int) -> tuple[float, float]:
    """Train and evaluate one fold.

    Parameters
    ----------
    test_subject : int
        Index of the subject held out for testing (0-based).
    val_subject : int
        Index of the subject used for validation.
    seed : int
        Random seed for reproducibility.
    """

    logger.info(
        "Running fold: test_subject=%d, val_subject=%d, seed=%d",
        test_subject,
        val_subject,
        seed,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = get_source_data(test_subject, val_subject)
    train_loader, val_loader = prepare_dataloaders(
        X_train, y_train, X_val, y_val, TRAINING_PARAMS["batch_size"]
    )
    test_loader = prepare_dataloaders(
        X_test, y_test, X_test, y_test, TRAINING_PARAMS["batch_size"]
    )[1]

    # Model, loss and optimizer
    model = MambaCCT(**MODEL_PARAMS).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAINING_PARAMS["lr"],
        betas=(TRAINING_PARAMS["b1"], TRAINING_PARAMS["b2"]),
    )

    early_stopping = EarlyStopping(patience=10, min_delta=0.01)
    trained_model, *_ = train_model(
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        TRAINING_PARAMS,
        X_train,
        y_train,
        early_stopping,
    )

    test_accuracy, test_loss = test_model(trained_model, loss_fn, test_loader)
    logger.info("Fold result - accuracy: %.2f%%, loss: %.4f", test_accuracy, test_loss)

    return test_accuracy, test_loss


def run_loso(seed: int | None = None) -> list[dict[str, float]]:
    """Run LOSO evaluation across all subjects."""

    results: list[dict[str, float]] = []
    accs: list[float] = []
    losses: list[float] = []

    for test_sub in range(NUM_SUBJECTS):
        start_time = time.time()
        fold_seed = np.random.randint(2021) if seed is None else seed + test_sub
        val_sub = (test_sub + 1) % NUM_SUBJECTS

        acc, loss = run_single_fold(test_sub, val_sub, fold_seed)
        accs.append(acc)
        losses.append(loss)

        elapsed = time.time() - start_time
        logger.info("Fold %d finished in %dm %ds", test_sub + 1, int(elapsed // 60), int(elapsed % 60))

        results.append(
            {
                "Test Subject": test_sub + 1,
                "Val Subject": val_sub + 1,
                "Test Acc": acc,
                "Seed": fold_seed,
            }
        )
        logger.info("======================================")

    avg_acc = np.mean(accs)
    avg_loss = np.mean(losses)
    logger.info("Average Test Accuracy: %.2f%%", avg_acc)
    logger.info("Average Test Loss: %.4f", avg_loss)
    logger.info("\n%s", pd.DataFrame(results))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOSO evaluation for MambaCCT")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed. Different subjects will add their index to this value.",
    )
    args = parser.parse_args()

    run_loso(args.seed)


if __name__ == "__main__":
    main()

