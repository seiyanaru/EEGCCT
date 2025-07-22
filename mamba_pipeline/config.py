import torch

DATA_DIR = "pickles"
NUM_SUBJECTS = 9

MODEL_PARAMS = {
    'kernel_sizes': [(22, 1), (1, 24)],
    'stride': (1, 1),
    'padding': (0, 0),
    'pooling_kernel_size': (3, 3),
    'pooling_stride': (1, 1),
    'pooling_padding': (0, 0),
    'n_conv_layers': 2,
    'n_input_channels': 1,
    'in_planes': 64,
    'activation': None,
    'max_pool': False,
    'conv_bias': False,
    'dim': 64,
    'num_layers': 2,
    'num_classes': 2,
    'dropout': 0.1,
    'positional_emb': 'learnable',
}

TRAINING_PARAMS = {
    'batch_size': 32,
    'n_epochs': 100,
    'lr': 3e-5,
    'b1': 0.9,
    'b2': 0.999,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
