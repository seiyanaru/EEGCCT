#!/usr/bin/env python3
"""
Simple test script for STMambaCCT model
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import STMambaCCT, create_stmamba_cct
    print("Successfully imported STMambaCCT model")
except ImportError as e:
    print("Import error:", e)
    sys.exit(1)

def test_model():
    """Test the STMambaCCT model creation and forward pass"""
    print("Testing STMambaCCT model...")
    
    # Create model
    try:
        model = create_stmamba_cct(
            n_input_channels=22, 
            sequence_length=1000, 
            num_classes=4
        )
        print("Model created successfully!")
    except Exception as e:
        print("Error creating model:", e)
        return False
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    
    # Test forward pass
    try:
        x = torch.randn(2, 22, 1000)  # (batch_size, channels, time_points)
        print("Input shape:", x.shape)
        
        with torch.no_grad():
            output = model(x)
        
        print("Output shape:", output.shape)
        print("Forward pass successful!")
        
        # Check output dimensions
        expected_shape = (2, 4)  # (batch_size, num_classes)
        if output.shape == expected_shape:
            print("Output shape is correct!")
            return True
        else:
            print("Expected output shape", expected_shape, "got", output.shape)
            return False
            
    except Exception as e:
        print("Error during forward pass:", e)
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("All tests passed! STMambaCCT model is working correctly.")
    else:
        print("Tests failed! Please check the model implementation.")
        sys.exit(1) 
