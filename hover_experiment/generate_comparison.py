import json
import argparse
import time
from pathlib import Path

def load_stats(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # When running evaluate_test_set.py with --prompt_file, the target prompt
    # is evaluated as the "optimized" prompt.
    return data.get("optimized", {})

def main():
    parser = argparse.ArgumentParser(description="Generate comparison JSON from individual evaluation results")
    parser.add_argument("--baseline", required=True, help="Path to baseline result JSON")
    parser.add_argument("--optimized", required=True, help="Path to GEPA optimized result JSON")
    parser.add_argument("--fewshot_opt", required=True, help="Path to Few-shot optimized result JSON")
    parser.add_argument("--output", required=True, help="Path to save comparison JSON")
    args = parser.parse_args()

    baseline_stats = load_stats(args.baseline)
    optimized_stats = load_stats(args.optimized)
    fs_opt_stats = load_stats(args.fewshot_opt)

    # Helper to calculate deltas
    def add_deltas(stats, baseline):
        acc = stats.get("accuracy", 0.0)
        base_acc = baseline.get("accuracy", 0.0)
        stats["delta_absolute"] = acc - base_acc
        stats["delta_percent"] = ((acc - base_acc) / base_acc * 100.0) if base_acc else 0.0
        stats["additional_correct"] = stats.get("correct", 0) - baseline.get("correct", 0)
        return stats

    comparison = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "test_size": baseline_stats.get("total", 500),
        "baseline": baseline_stats,
        "optimized": add_deltas(optimized_stats, baseline_stats),
        "fewshot_optimized": add_deltas(fs_opt_stats, baseline_stats)
    }

    with open(args.output, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Comparison saved to {args.output}")

if __name__ == "__main__":
    main()
