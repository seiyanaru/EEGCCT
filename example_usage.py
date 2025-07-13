"""
Example usage of the modularized STMambaCCT training pipeline.

This script demonstrates how to use the src modules for clean, 
efficient STMambaCCT training on BCI Competition IV-2a data.
"""

import warnings
warnings.filterwarnings('ignore')

# Import our modularized utilities
from src.data.preprocessing import load_and_preprocess_bci_data
from src.models.stmamba_utils import create_complete_setup
from src.training.trainer import train_model
from src.utils.visualization import create_comprehensive_report


def main():
    """Main training pipeline demonstration."""
    
    print("="*60)
    print("STMambaCCT Training Pipeline Example")
    print("="*60)
    
    # Step 1: Data Loading and Preprocessing
    print("\n1. Loading and preprocessing BCI-IV-2a dataset...")
    train_loader, test_loader, metadata = load_and_preprocess_bci_data(
        data_dir="pickles",
        test_size=0.2,
        batch_size=4,  # Memory-efficient batch size
        random_state=42
    )
    
    print(f"✓ Data loading complete!")
    print(f"  - Number of subjects: {metadata['n_subjects']}")
    print(f"  - Label mapping: {metadata['subject_info']['label_map']}")
    
    # Step 2: Model Configuration and Setup
    print("\n2. Setting up STMambaCCT model...")
    
    # Model configuration (easily customizable)
    model_config = {
        'n_input_channels': 22,
        'sequence_length': 1000,
        'num_classes': 4,
        'dim': 64,              # Small for memory efficiency
        'num_layers': 2,        # Minimal layers
        'd_state': 4,           # Small state dimension
        'd_conv': 4,
        'expand_factor': 2,
        'dropout': 0.1
    }
    
    # Training configuration (easily customizable)
    training_config = {
        'batch_size': 4,
        'learning_rate': 1e-3,
        'num_epochs': 10,       # Small number for demonstration
        'weight_decay': 1e-5,
        'accumulation_steps': 8  # Effective batch size = 32
    }
    
    # Create complete setup
    setup = create_complete_setup(model_config, training_config)
    print("✓ Model and training setup complete!")
    
    # Step 3: Training
    print("\n3. Starting training...")
    results = train_model(
        setup=setup,
        train_loader=train_loader,
        test_loader=test_loader,
        save_path='best_stmamba_example.pth'
    )
    
    print("✓ Training completed successfully!")
    print(f"  - Best accuracy achieved: {results['best_test_acc']:.2f}%")
    
    # Step 4: Results Analysis and Visualization
    print("\n4. Generating comprehensive analysis...")
    create_comprehensive_report(
        results=results,
        model_config=setup['model_config'],
        training_config=setup['training_config'],
        save_dir="example_results"
    )
    
    print("✓ Analysis complete!")
    print("  - Check 'example_results' folder for detailed reports")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)


def quick_experiment_example():
    """
    Example of how to quickly run different experiments 
    with different configurations.
    """
    print("\n" + "="*60)
    print("Quick Experiment Example")
    print("="*60)
    
    # Load data once
    train_loader, test_loader, metadata = load_and_preprocess_bci_data(
        data_dir="pickles", batch_size=4
    )
    
    # Experiment configurations
    experiments = {
        'tiny_model': {
            'model_config': {'dim': 32, 'num_layers': 1, 'd_state': 2},
            'training_config': {'num_epochs': 5, 'learning_rate': 1e-3}
        },
        'small_model': {
            'model_config': {'dim': 64, 'num_layers': 2, 'd_state': 4},
            'training_config': {'num_epochs': 5, 'learning_rate': 1e-3}
        }
    }
    
    results_comparison = []
    
    for name, configs in experiments.items():
        print(f"\nRunning experiment: {name}")
        
        # Create setup
        setup = create_complete_setup(
            configs['model_config'], 
            configs['training_config']
        )
        
        # Train
        results = train_model(
            setup, train_loader, test_loader, 
            save_path=f'model_{name}.pth'
        )
        
        results_comparison.append((name, results['best_test_acc']))
        print(f"✓ {name} completed: {results['best_test_acc']:.2f}%")
    
    # Compare results
    print(f"\nExperiment Comparison:")
    print("-" * 30)
    for name, acc in results_comparison:
        print(f"{name:15s}: {acc:6.2f}%")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Uncomment to run quick experiments
    # quick_experiment_example() 
