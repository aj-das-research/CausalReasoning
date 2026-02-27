"""
test_and_visualize.py - Test a trained model and produce evaluation visualizations.

Responsibilities:
  1. Load a trained checkpoint and the test split.
  2. Run inference on the full test set.
  3. Compute per-modality and overall metrics (accuracy, format compliance, concept F1).
  4. Generate visualizations: confusion matrix, reward curves, per-modality bars,
     and example predictions with images.
  5. Save all results and figures to the outputs/visualizations directory.

Usage:
  python test_and_visualize.py --checkpoint outputs/checkpoints/best
  python test_and_visualize.py --checkpoint outputs/checkpoints/final --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from PIL import Image
from tqdm import tqdm

from config import (
    CONCEPT_MAP,
    EVAL_CFG,
    LOG_DIR,
    MODALITIES,
    MODEL_ID,
    SYSTEM_PROMPT,
    TRAIN_CFG,
    VIS_DIR,
)
from data_prep_and_viewer import PREPARED_DATA_DIR
from model_loader_and_checker import load_model_and_processor
from train_eval import accuracy_reward, concept_reward, format_reward


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model,
    processor,
    dataset,
    device: str,
    cfg=EVAL_CFG,
    max_samples: int | None = None,
) -> list[dict]:
    """Run inference on a dataset and return detailed results per sample."""
    model.eval()
    n = min(max_samples or len(dataset), len(dataset))
    modalities_str = ", ".join(MODALITIES)
    results = []

    for i in tqdm(range(n), desc="Running inference"):
        sample = dataset[i]

        # Prepare image
        img = sample["image"]
        if isinstance(img, str):
            img = Image.open(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Build prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            f"{sample['problem']}\n\n"
                            f"Identify the modality from: {modalities_str}\n"
                            "Respond with <think><modality>...</modality>"
                            "<concepts>[...]</concepts>reasoning</think>"
                            "<answer>LETTER</answer>"
                        ),
                    },
                ],
            },
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.temperature > 0,
        )

        prompt_len = inputs["input_ids"].shape[1]
        completion = processor.batch_decode(
            generated_ids[:, prompt_len:], skip_special_tokens=True
        )[0]

        # Parse ground truth
        gt_letter = sample.get("answer_letter", "")
        gt_concepts = sample.get("concepts", "[]")
        if isinstance(gt_concepts, str):
            gt_concepts = json.loads(gt_concepts)
        gt_modality = sample.get("modality", "unknown")

        # Parse prediction
        pred_letter = ""
        ans_match = re.search(r"<answer>\s*\(?([A-J])\)?\s*</answer>", completion, re.IGNORECASE)
        if ans_match:
            pred_letter = ans_match.group(1).upper()

        pred_modality = ""
        mod_match = re.search(r"<modality>(.*?)</modality>", completion, re.DOTALL)
        if mod_match:
            pred_modality = mod_match.group(1).strip()

        pred_concepts = []
        con_match = re.search(r"<concepts>\s*\[(.*?)\]\s*</concepts>", completion, re.DOTALL)
        if con_match:
            pred_concepts = [c.strip().strip("'\"") for c in con_match.group(1).split(",") if c.strip()]

        # Compute individual rewards
        fmt = format_reward([completion])[0]
        acc = accuracy_reward([completion], [gt_letter])[0]
        con = concept_reward([completion], [gt_concepts])[0]

        results.append({
            "index": i,
            "problem": sample["problem"][:200],
            "gt_letter": gt_letter,
            "gt_modality": gt_modality,
            "gt_concepts": gt_concepts,
            "pred_letter": pred_letter,
            "pred_modality": pred_modality,
            "pred_concepts": pred_concepts,
            "completion": completion,
            "format_reward": fmt,
            "accuracy_reward": acc,
            "concept_reward": con,
            "correct": pred_letter == gt_letter.upper(),
            "format_ok": fmt == 1.0,
        })

    return results


# ── Metrics Computation ──────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate and per-modality metrics."""
    n = len(results)
    if n == 0:
        return {}

    # Overall
    overall = {
        "total_samples": n,
        "accuracy": sum(r["correct"] for r in results) / n,
        "format_compliance": sum(r["format_ok"] for r in results) / n,
        "mean_accuracy_reward": sum(r["accuracy_reward"] for r in results) / n,
        "mean_concept_f1": sum(r["concept_reward"] for r in results) / n,
        "mean_total_reward": sum(
            TRAIN_CFG.format_reward_weight * r["format_reward"]
            + TRAIN_CFG.accuracy_reward_weight * r["accuracy_reward"]
            + TRAIN_CFG.concept_reward_weight * r["concept_reward"]
            for r in results
        ) / n,
    }

    # Per modality
    by_modality = defaultdict(list)
    for r in results:
        by_modality[r["gt_modality"]].append(r)

    per_modality = {}
    for mod, mod_results in by_modality.items():
        m = len(mod_results)
        per_modality[mod] = {
            "count": m,
            "accuracy": sum(r["correct"] for r in mod_results) / m,
            "format_compliance": sum(r["format_ok"] for r in mod_results) / m,
            "concept_f1": sum(r["concept_reward"] for r in mod_results) / m,
        }

    # Answer distribution
    pred_dist = defaultdict(int)
    gt_dist = defaultdict(int)
    for r in results:
        pred_dist[r["pred_letter"]] += 1
        gt_dist[r["gt_letter"]] += 1

    return {
        "overall": overall,
        "per_modality": dict(per_modality),
        "pred_answer_distribution": dict(pred_dist),
        "gt_answer_distribution": dict(gt_dist),
    }


# ── Visualizations ───────────────────────────────────────────────────────────

def plot_per_modality_accuracy(metrics: dict, save_path: Path) -> None:
    """Bar chart of accuracy per modality."""
    per_mod = metrics.get("per_modality", {})
    if not per_mod:
        return

    modalities = list(per_mod.keys())
    accuracies = [per_mod[m]["accuracy"] for m in modalities]
    counts = [per_mod[m]["count"] for m in modalities]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(modalities, accuracies, color="steelblue", edgecolor="black")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"n={count}", ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per Modality")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=metrics["overall"]["accuracy"], color="red", linestyle="--",
               label=f"Overall: {metrics['overall']['accuracy']:.3f}")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_metrics_radar(metrics: dict, save_path: Path) -> None:
    """Radar chart of overall metrics."""
    overall = metrics.get("overall", {})
    labels = ["Accuracy", "Format\nCompliance", "Concept F1", "Total\nReward"]
    values = [
        overall.get("accuracy", 0),
        overall.get("format_compliance", 0),
        overall.get("mean_concept_f1", 0),
        overall.get("mean_total_reward", 0) / 2.5,  # normalize to 0-1 range
    ]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, "o-", linewidth=2, color="steelblue")
    ax.fill(angles, values, alpha=0.25, color="steelblue")
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Overview", pad=20)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_answer_distribution(metrics: dict, save_path: Path) -> None:
    """Side-by-side bar chart of predicted vs ground-truth answer distributions."""
    pred_dist = metrics.get("pred_answer_distribution", {})
    gt_dist = metrics.get("gt_answer_distribution", {})

    all_letters = sorted(set(list(pred_dist.keys()) + list(gt_dist.keys())))
    if not all_letters:
        return

    x = np.arange(len(all_letters))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, [gt_dist.get(l, 0) for l in all_letters],
           width, label="Ground Truth", color="steelblue")
    ax.bar(x + width / 2, [pred_dist.get(l, 0) for l in all_letters],
           width, label="Predicted", color="coral")

    ax.set_xlabel("Answer Letter")
    ax.set_ylabel("Count")
    ax.set_title("Answer Distribution: Ground Truth vs Predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(all_letters)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_training_curves(log_path: Path, save_path: Path) -> None:
    """Plot training loss and reward curves from the training log."""
    if not log_path.exists():
        print(f"  No training log found at {log_path}, skipping curves")
        return

    with open(log_path) as f:
        log_data = json.load(f)

    # Separate training and eval entries
    train_entries = [e for e in log_data if "step" in e and "eval" not in e]
    eval_entries = [e.get("eval", e) for e in log_data if "eval" in e]

    if not train_entries:
        return

    steps = [e["step"] for e in train_entries]
    losses = [e["loss"] for e in train_entries]
    rewards = [e["mean_reward"] for e in train_entries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(steps, losses, "b-", alpha=0.7, linewidth=1)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("GRPO Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    # Reward curve
    ax2.plot(steps, rewards, "g-", alpha=0.7, linewidth=1, label="Mean Reward")
    if eval_entries:
        eval_steps = [e.get("step", 0) for e in eval_entries]
        eval_rewards = [e.get("mean_reward", 0) for e in eval_entries]
        ax2.plot(eval_steps, eval_rewards, "ro-", markersize=5, label="Eval Reward")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Reward")
    ax2.set_title("Reward Progression")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_example_predictions(
    results: list[dict],
    dataset,
    save_path: Path,
    n: int = 8,
) -> None:
    """Show example predictions with images, ground truth, and model output."""
    # Pick a mix of correct and incorrect
    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]

    n_correct = min(n // 2, len(correct))
    n_incorrect = min(n - n_correct, len(incorrect))
    n_correct = min(n - n_incorrect, len(correct))  # fill remaining with correct
    selected = correct[:n_correct] + incorrect[:n_incorrect]

    if not selected:
        return

    cols = min(4, len(selected))
    rows = math.ceil(len(selected) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, result in enumerate(selected):
        row, col = divmod(idx, cols)
        ax = axes[row, col]

        # Get image from dataset
        sample = dataset[result["index"]]
        img = sample["image"]
        if isinstance(img, str):
            img = Image.open(img)

        ax.imshow(img)
        status = "CORRECT" if result["correct"] else "WRONG"
        color = "green" if result["correct"] else "red"
        ax.set_title(f"[{status}] GT:({result['gt_letter']}) Pred:({result['pred_letter']})",
                     fontsize=9, color=color, fontweight="bold")
        ax.axis("off")

        # Info text
        info = (
            f"Mod: {result['gt_modality']}\n"
            f"GT concepts: {result['gt_concepts'][:3]}\n"
            f"Pred concepts: {result['pred_concepts'][:3]}\n"
            f"Fmt:{result['format_reward']:.0f} Acc:{result['accuracy_reward']:.1f} "
            f"Con:{result['concept_reward']:.2f}"
        )
        ax.text(
            0.5, -0.02, info,
            transform=ax.transAxes, fontsize=6,
            verticalalignment="top", horizontalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    # Hide unused axes
    for idx in range(len(selected), rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    plt.suptitle("Example Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test and visualize MedVLM-R1 model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained checkpoint directory",
    )
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples")
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation"])
    parser.add_argument("--output-dir", type=str, default=str(VIS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, processor, device = load_model_and_processor(
        model_id=args.model_id,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Load test data
    splits = DatasetDict.load_from_disk(str(PREPARED_DATA_DIR))
    test_dataset = splits[args.split]
    print(f"Test set: {len(test_dataset)} samples")

    # Run inference
    results = run_inference(
        model, processor, test_dataset, device,
        max_samples=args.max_samples,
    )

    # Compute metrics
    metrics = compute_metrics(results)

    # Print results
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Overall Accuracy:      {metrics['overall']['accuracy']:.4f}")
    print(f"  Format Compliance:     {metrics['overall']['format_compliance']:.4f}")
    print(f"  Mean Concept F1:       {metrics['overall']['mean_concept_f1']:.4f}")
    print(f"  Mean Total Reward:     {metrics['overall']['mean_total_reward']:.4f}")

    print(f"\n  Per-Modality Accuracy:")
    for mod, mod_metrics in metrics["per_modality"].items():
        print(f"    {mod:20s}: {mod_metrics['accuracy']:.3f} (n={mod_metrics['count']})")
    print(f"{'='*60}")

    # Save results JSON
    results_path = out_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump({"metrics": metrics, "predictions": results}, f, indent=2, default=str)
    print(f"\nSaved results to {results_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_per_modality_accuracy(metrics, out_dir / "per_modality_accuracy.png")
    plot_metrics_radar(metrics, out_dir / "metrics_radar.png")
    plot_answer_distribution(metrics, out_dir / "answer_distribution.png")
    plot_training_curves(LOG_DIR / "training_log.json", out_dir / "training_curves.png")
    plot_example_predictions(results, test_dataset, out_dir / "example_predictions.png")

    print(f"\nAll visualizations saved to {out_dir}")


if __name__ == "__main__":
    main()
