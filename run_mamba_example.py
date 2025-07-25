"""Example training script for the :class:`MambaCCT` model.

This program loads EEG data using the utility functions in ``utils`` and trains
the ``MambaCCT`` model for one train/validation/test split.

You can specify the test/validation subject as command line arguments.
"""

import argparse
import random
import logging
import numpy as np
import torch
from torch import nn, optim

from utils.config import MODEL_PARAMS, TRAINING_PARAMS, DEVICE, NUM_SUBJECTS
from utils.data_utils import get_source_data, prepare_dataloaders
from utils.training_utils import EarlyStopping, train_model, test_model
from model import MambaCCT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(test_subject: int = 0, val_subject: int = 1, seed: int = 42):
    """Train and evaluate ``MambaCCT`` using the utility modules.

    Parameters
    ----------
    test_subject : int, default=0
        Index of the subject used for testing.
    val_subject : int, default=1
        Index of the subject used for validation.
    seed : int, default=42
        Random seed for reproducibility.
    """
    logger.info(
        f"Starting run: test_subject={test_subject}, val_subject={val_subject}, seed={seed}"
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = get_source_data(
        test_subject, val_subject
    )
    train_loader, val_loader = prepare_dataloaders(
        X_train, y_train, X_val, y_val, TRAINING_PARAMS['batch_size']
    )
    test_loader = prepare_dataloaders(
        X_test, y_test, X_test, y_test, TRAINING_PARAMS['batch_size']
    )[1]

    # Model, loss and optimizer
    model = MambaCCT(**MODEL_PARAMS).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAINING_PARAMS['lr'],
        betas=(TRAINING_PARAMS['b1'], TRAINING_PARAMS['b2'])
    )

    # Train
    early_stopping = EarlyStopping(patience=10, min_delta=0.01)
    trained_model, *_ = train_model(
        model, optimizer, loss_fn, train_loader, val_loader,
        TRAINING_PARAMS, X_train, y_train, early_stopping
    )

    # Evaluate
    test_accuracy, test_loss = test_model(trained_model, loss_fn, test_loader)
    logger.info(f"Final Test Accuracy: {test_accuracy:.2f}%")
    logger.info(f"Final Test Loss: {test_loss:.4f}")

    return test_accuracy, test_loss


def run_loso(seed: int = 42):
    """Run leave-one-subject-out evaluation over all subjects."""
    all_acc = []
    all_loss = []

    for test_sub in range(NUM_SUBJECTS):
        val_sub = (test_sub + 1) % NUM_SUBJECTS
        current_seed = np.random.randint(2021)
        logger.info(
            f"LOSO fold -- test_subject={test_sub}, val_subject={val_sub}, seed={current_seed}"
        )
        acc, loss = main(test_sub, val_sub, current_seed)
        all_acc.append(acc)
        all_loss.append(loss)

    avg_acc = np.mean(all_acc)
    avg_loss = np.mean(all_loss)
    logger.info(f"Average Test Accuracy: {avg_acc:.2f}%")
    logger.info(f"Average Test Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MambaCCT on one split or run LOSO evaluation")
    parser.add_argument("--test-subject", type=int, default=0,
                        help="index of subject used for testing")
    parser.add_argument("--val-subject", type=int, default=1,
                        help="index of subject used for validation")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--loso", action="store_true",
                        help="run leave-one-subject-out evaluation")
    args = parser.parse_args()

    if args.loso:
        run_loso(args.seed)
    else:
        main(args.test_subject, args.val_subject, args.seed)
