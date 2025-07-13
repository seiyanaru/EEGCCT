"""
Simple LOSO evaluation example using modular STMambaCCT

This script demonstrates how easy it is to run LOSO evaluation 
with our modular approach - just a few lines of code!
"""

from src.training.trainer import run_loso_evaluation
from src.utils.visualization import save_loso_report

# Simple LOSO evaluation with default settings
print("Running STMambaCCT LOSO Evaluation...")
print("=" * 40)

# Run LOSO evaluation (everything in one function call!)
results = run_loso_evaluation(
    n_classes=4,           # 4-class motor imagery
    save_results=True,     # Save detailed results
    results_dir="simple_loso_results"
)

# Generate comprehensive report
save_loso_report(results, "simple_loso_results")

# Print final summary
print(f"\n🎉 LOSO Evaluation Complete!")
print(f"📊 Average Accuracy: {results['avg_accuracy']:.2f}% ± {results['std_accuracy']:.2f}%")
print(f"📁 Results saved to: simple_loso_results/")

print(f"\n📈 Per-subject results:")
for i, acc in enumerate(results['individual_accuracies']):
    print(f"   Subject {i+1}: {acc:.2f}%") 
