"""
src/train.py — §3.6 (script, not a library — `python -m src.train`)
"""

import argparse
import json
import sys
import io

# Fix stdout encoding to prevent Unicode errors on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sklearn.model_selection import train_test_split

from src.baseline import RuleBasedClassifier
from src.feature_extractor import FEATURE_NAMES
from src.model import (
    DEFAULT_MODEL_PATH, evaluate_model, save_model, train_classifier,
)
from src.synthetic_data import add_benchmark_examples, generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Train the GPUGate profitability model.")
    parser.add_argument('--evaluate', action='store_true',
                         help="Print evaluation metrics on the held-out test split.")
    parser.add_argument('--compare-baseline', action='store_true',
                         help="Also train/evaluate RuleBasedClassifier and print the comparison.")
    parser.add_argument('--n-samples', type=int, default=5000,
                         help="Number of synthetic rows to generate (default: 5000).")
    parser.add_argument('--output', type=str, default=DEFAULT_MODEL_PATH,
                         help=f"Where to save the trained model (default: {DEFAULT_MODEL_PATH}).")
    args = parser.parse_args()

    print(f"Generating {args.n_samples} synthetic samples...")
    df = generate_dataset(n_samples=args.n_samples)
    df = add_benchmark_examples(df)
    print(f"Total rows after hand-crafted anchors: {len(df)}")
    print(f"Class balance -> profitable=1: {int(df['profitable'].sum())}, "
          f"profitable=0: {int((df['profitable'] == 0).sum())}")

    X = df[FEATURE_NAMES]
    y = df['profitable']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    print(f"\nTraining XGBClassifier (scale_pos_weight={scale_pos_weight:.1f})...")
    
    model = train_classifier(X_train, y_train, scale_pos_weight=scale_pos_weight)

    if args.evaluate:
        metrics = evaluate_model(model, X_test, y_test)
        print("\n=== GPUGate model — held-out synthetic test metrics ===")
        print(json.dumps(metrics, indent=2))

        if args.compare_baseline:
            print("\nTraining/evaluating RuleBasedClassifier baseline...")
            baseline = RuleBasedClassifier()
            baseline_metrics = evaluate_model(baseline, X_test, y_test)
            print("\n=== RuleBasedClassifier baseline — same test split ===")
            print(json.dumps(baseline_metrics, indent=2))

            print("\n=== Comparison (model - baseline) ===")
            for key in ('accuracy', 'precision', 'recall', 'f1', 'auc_roc'):
                delta = metrics[key] - baseline_metrics[key]
                print(f"  {key:12s}: {metrics[key]:.4f} vs {baseline_metrics[key]:.4f}  (Delta {delta:+.4f})")

    save_model(model, args.output)
    print(f"\nModel saved to {args.output}")

if __name__ == '__main__':
    main()