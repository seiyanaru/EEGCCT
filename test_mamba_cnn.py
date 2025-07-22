#!/usr/bin/env python3
"""Simple test script for the MambaCCT model"""
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import MambaCCT
    print("Successfully imported MambaCCT model")
except ImportError as e:
    print("Import error:", e)
    sys.exit(1)


def test_model():
    """Test the MambaCCT model creation and forward pass"""
    print("Testing MambaCCT model...")

    try:
        model = MambaCCT(
            kernel_sizes=[(22, 1), (1, 24)],
            stride=(1, 1),
            padding=(0, 0),
            pooling_kernel_size=(3, 3),
            pooling_stride=(1, 1),
            pooling_padding=(0, 0),
            n_conv_layers=2,
            n_input_channels=1,
            in_planes=64,
            activation=None,
            max_pool=False,
            conv_bias=False,
            dim=64,
            num_layers=2,
            num_classes=4,
        )
        print("Model created successfully!")
    except Exception as e:
        print("Error creating model:", e)
        return False

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    try:
        x = torch.randn(2, 1, 22, 1000)
        print("Input shape:", x.shape)
        with torch.no_grad():
            output = model(x)
        print("Output shape:", output.shape)
        expected = (2, 4)
        if output.shape == expected:
            print("Output shape is correct!")
            return True
        else:
            print("Expected", expected, "got", output.shape)
            return False
    except Exception as e:
        print("Error during forward pass:", e)
        return False


if __name__ == "__main__":
    success = test_model()
    if success:
        print("All tests passed! MambaCCT model is working correctly.")
    else:
        print("Tests failed! Please check the model implementation.")
        sys.exit(1)
