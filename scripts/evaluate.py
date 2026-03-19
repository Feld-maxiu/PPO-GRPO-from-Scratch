"""Standalone evaluation script.

Usage:
    python scripts/evaluate.py --model_path outputs/checkpoints/ppo/iter_10
    python scripts/evaluate.py --model_path <path_to_model> --dataset_path <path> --num_samples 200
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data import build_eval_dataset
from src.evaluation import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on GSM8K test set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_path", type=str,
                        default=r"C:\Users\曾\Desktop\84Post-training of LLMs\L7\gsm8k",
                        help="Path to GSM8K dataset")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of eval samples")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max generation length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    system_prompt = (
        "Please solve the problem and answer in the format of 'The answer is...', no process required. "
        "Always include the final numeric answer inside \\boxed{}."
    )

    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Building eval dataset ({args.num_samples} samples)...")
    eval_dataset = build_eval_dataset(args.dataset_path, system_prompt, args.num_samples)

    print("Evaluating...")
    results = evaluate_model(
        model, tokenizer, eval_dataset,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=device,
    )

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Samples:     {results['num_samples']}")
    print(f"  Accuracy:    {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
    print(f"  Format Rate: {results['format_rate']:.4f} ({results['format_rate']*100:.1f}%)")
    print("=" * 60)

    # Show some sample predictions
    print("\nSample predictions:")
    for i, detail in enumerate(results["details"][:5]):
        status = "CORRECT" if detail["correct"] else "WRONG"
        print(f"\n  [{i+1}] [{status}] GT: {detail['ground_truth']}")
        print(f"      Response: {detail['response'][:120]}...")


if __name__ == "__main__":
    main()
