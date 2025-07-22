import glob
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import DEVICE, DATA_DIR, NUM_SUBJECTS


def load_data(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


def get_source_data(test_sub, val_sub):
    files = sorted(glob.glob(f"{DATA_DIR}/A*.pkl"))
    all_data = [load_data(fn) for fn in files]

    test_d = all_data[test_sub]['train']
    val_d = all_data[val_sub]['train']
    train_idxs = [i for i in range(NUM_SUBJECTS) if i not in (test_sub, val_sub)]
    train_ds = [all_data[i]['train'] for i in train_idxs]

    X_train = np.concatenate([d['X'] for d in train_ds], axis=0)
    y_train = np.concatenate([d['y'] for d in train_ds], axis=0)
    X_val = val_d['X']
    y_val = val_d['y']
    X_test = test_d['X']
    y_test = test_d['y']

    mask_tr = np.isin(y_train, [0, 1])
    X_train, y_train = X_train[mask_tr], y_train[mask_tr]
    mask_val = np.isin(y_val, [0, 1])
    X_val, y_val = X_val[mask_val], y_val[mask_val]
    mask_te = np.isin(y_test, [0, 1])
    X_test, y_test = X_test[mask_te], y_test[mask_te]

    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)

    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    mu, sigma = X_train.mean(), X_train.std()
    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma
    X_test = (X_test - mu) / sigma

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    train_data = torch.from_numpy(X_train).float().to(DEVICE)
    train_labels = torch.from_numpy(y_train).long().to(DEVICE)
    val_data = torch.from_numpy(X_val).float().to(DEVICE)
    val_labels = torch.from_numpy(y_val).long().to(DEVICE)

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def augment_data(X, y, batch_size, n_segments=3):
    n_trials, _, n_ch, n_t = X.shape
    half = batch_size // 2
    bounds = [int(round(i * n_t / n_segments)) for i in range(n_segments + 1)]

    aug_data = np.zeros((half, 1, n_ch, n_t), dtype=X.dtype)
    aug_label = np.zeros(half, dtype=y.dtype)
    classes = [0, 1]

    for i in range(half):
        lbl = np.random.choice(classes)
        aug_label[i] = lbl
        idxs = np.where(y == lbl)[0]
        picks = np.random.choice(idxs, size=n_segments, replace=True)
        segments = []
        for s in range(n_segments):
            st, ed = bounds[s], bounds[s + 1]
            segments.append(X[picks[s], 0, :, st:ed])
        new_trial = np.concatenate(segments, axis=-1)
        aug_data[i, 0] = new_trial

    tdata = torch.from_numpy(aug_data).float().to(DEVICE)
    tlabels = torch.from_numpy(aug_label).long().to(DEVICE)
    return tdata, tlabels
