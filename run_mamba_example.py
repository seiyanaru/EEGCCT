"""Example training script for the :class:`MambaCCT` model.

This program loads EEG data using the utility functions in ``utils`` and trains
the ``MambaCCT`` model for one train/validation/test split.

You can specify the test/validation subject as command line arguments.
"""

import argparse
import random
import numpy as np
import torch
from torch import nn, optim

from utils.config import MODEL_PARAMS, TRAINING_PARAMS, DEVICE
from utils.data_utils import get_source_data, prepare_dataloaders
from utils.training_utils import EarlyStopping, train_model, test_model
from model import MambaCCT


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = get_source_data(test_subject, val_subject)
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
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MambaCCT on one split")
    parser.add_argument("--test-subject", type=int, default=0,
                        help="index of subject used for testing")
    parser.add_argument("--val-subject", type=int, default=1,
                        help="index of subject used for validation")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    main(args.test_subject, args.val_subject, args.seed)
