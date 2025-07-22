import time
import random
import numpy as np
import torch
import pandas as pd
from torch import optim, nn

from .config import MODEL_PARAMS, TRAINING_PARAMS, NUM_SUBJECTS, DEVICE
from .data_utils import get_source_data, prepare_dataloaders
from .training_utils import EarlyStopping, train_model, test_model
from model import MambaCCT


def main():
    results_df = pd.DataFrame(columns=['Test Subject', 'Val Subject', 'Test Acc', 'Seed'])
    all_test_accuracies = []
    all_test_losses = []

    for test_sub in range(NUM_SUBJECTS):
        start_time = time.time()
        seed_n = np.random.randint(2021)
        print('seed is', seed_n)
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        val_sub = (test_sub + 1) % NUM_SUBJECTS
        print(f"Val Subject {val_sub + 1}:")
        model = MambaCCT(**MODEL_PARAMS).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=TRAINING_PARAMS['lr'], betas=(TRAINING_PARAMS['b1'], TRAINING_PARAMS['b2']))

        X_train, y_train, X_val, y_val, X_test, y_test = get_source_data(test_sub, val_sub)
        train_loader, val_loader = prepare_dataloaders(X_train, y_train, X_val, y_val, TRAINING_PARAMS['batch_size'])
        test_loader = prepare_dataloaders(X_test, y_test, X_test, y_test, TRAINING_PARAMS['batch_size'])[1]

        early_stopping = EarlyStopping(patience=10, min_delta=0.01)
        trained_model, *_ = train_model(model, optimizer, loss_fn, train_loader, val_loader, TRAINING_PARAMS, X_train, y_train, early_stopping)

        test_accuracy, test_loss = test_model(trained_model, loss_fn, test_loader)
        all_test_accuracies.append(test_accuracy)
        all_test_losses.append(test_loss)

        time_elapsed = time.time() - start_time
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        results_df.loc[len(results_df)] = {'Test Subject': test_sub + 1, 'Val Subject': val_sub + 1, 'Test Acc': test_accuracy, 'Seed': seed_n}

    avg_acc = np.mean(all_test_accuracies)
    avg_loss = np.mean(all_test_losses)
    print(f"Average Test Accuracy: {avg_acc:.2f}%")
    print(f"Average Test Loss: {avg_loss:.4f}")
    print(results_df)


if __name__ == '__main__':
    main()
