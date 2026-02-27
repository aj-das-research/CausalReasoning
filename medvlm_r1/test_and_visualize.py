"""
test_and_visualize.py - Test a trained model and produce evaluation visualizations.

Supports both methods (--method baseline / --method ours) and includes:
  1. Standard evaluation: accuracy, format compliance, concept F1, modality accuracy.
  2. Per-modality breakdown and cross-domain leave-one-out evaluation.
  3. Perturbation faithfulness testing (mask image regions, check if reasoning changes).
  4. Method comparison visualizations.
  5. Training curve plots from logs.

Usage:
  python test_and_visualize.py --checkpoint outputs/checkpoints/best --method ours
  python test_and_visualize.py --checkpoint outputs/checkpoints/best --method baseline
  python test_and_visualize.py --checkpoint <path> --cross-domain   # leave-one-out eval
  python test_and_visualize.py --checkpoint <path> --faithfulness   # perturbation test
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
from datasets import DatasetDict
from PIL import Image, ImageDraw
from tqdm import tqdm

from config import (
    CONCEPT_MAP,
    EVAL_CFG,
    LOG_DIR,
    MODALITIES,
    MODEL_ID,
    OUTPUT_DIR,
    TRAIN_CFG,
    VIS_DIR,
    Method,
    get_prompts,
)
from data_prep_and_viewer import PREPARED_DATA_DIR, create_cross_domain_splits
from model_loader_and_checker import load_model_and_processor
from train_eval import (
    accuracy_reward,
    build_prompt_messages,
    concept_reward,
    format_reward,
    get_reward_weights,
    modality_reward,
)


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model,
    processor,
    dataset,
    device: str,
    method: Method = Method.OURS,
    cfg=EVAL_CFG,
    max_samples: int | None = None,
) -> list[dict]:
    """Run inference on a dataset and return detailed results per sample."""
    model.eval()
    n = min(max_samples or len(dataset), len(dataset))
    results = []

    for i in tqdm(range(n), desc="Running inference"):
        sample = dataset[i]

        img = sample["image"]
        if isinstance(img, str):
            img = Image.open(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        messages = build_prompt_messages(sample["problem"], method)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=[text], images=[img], return_tensors="pt", padding=True,
        )
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.temperature > 0,
        )

        prompt_len = inputs["input_ids"].shape[1]
        completion = processor.batch_decode(
            generated_ids[:, prompt_len:], skip_special_tokens=True,
        )[0]

        # Parse ground truth
        gt_letter = sample.get("answer_letter", "")
        gt_concepts = sample.get("concepts", "[]")
        if isinstance(gt_concepts, str):
            gt_concepts = json.loads(gt_concepts)
        gt_modality = sample.get("modality", "unknown")

        # Parse prediction
        pred_letter = ""
        ans_match = re.search(
            r"<answer>\s*\(?([A-J])\)?\s*</answer>", completion, re.IGNORECASE,
        )
        if ans_match:
            pred_letter = ans_match.group(1).upper()

        pred_modality = ""
        mod_match = re.search(r"<modality>(.*?)</modality>", completion, re.DOTALL)
        if mod_match:
            pred_modality = mod_match.group(1).strip()

        pred_concepts = []
        con_match = re.search(
            r"<concepts>\s*\[(.*?)\]\s*</concepts>", completion, re.DOTALL,
        )
        if con_match:
            pred_concepts = [
                c.strip().strip("'\"")
                for c in con_match.group(1).split(",") if c.strip()
            ]

        # Compute individual rewards
        fmt = format_reward([completion], method)[0]
        acc = accuracy_reward([completion], [gt_letter])[0]
        con = concept_reward([completion], [gt_concepts])[0] if method == Method.OURS else 0.0
        mod = modality_reward([completion], [gt_modality])[0] if method == Method.OURS else 0.0

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
            "modality_reward": mod,
            "correct": pred_letter == gt_letter.upper(),
            "format_ok": fmt == 1.0,
        })

    return results


# ── Metrics Computation ──────────────────────────────────────────────────────

def compute_metrics(results: list[dict], method: Method = Method.OURS) -> dict:
    """Compute aggregate and per-modality metrics."""
    n = len(results)
    if n == 0:
        return {}

    w = get_reward_weights(TRAIN_CFG)

    overall = {
        "total_samples": n,
        "accuracy": sum(r["correct"] for r in results) / n,
        "format_compliance": sum(r["format_ok"] for r in results) / n,
        "mean_accuracy_reward": sum(r["accuracy_reward"] for r in results) / n,
    }

    total_reward = sum(
        w["format"] * r["format_reward"] + w["accuracy"] * r["accuracy_reward"]
        for r in results
    )

    if method == Method.OURS:
        overall["mean_concept_f1"] = sum(r["concept_reward"] for r in results) / n
        overall["mean_modality_accuracy"] = sum(r["modality_reward"] for r in results) / n
        total_reward += sum(
            w["concept"] * r["concept_reward"] + w["modality"] * r["modality_reward"]
            for r in results
        )

    overall["mean_total_reward"] = total_reward / n

    # Per modality
    by_modality = defaultdict(list)
    for r in results:
        by_modality[r["gt_modality"]].append(r)

    per_modality = {}
    for mod, mod_results in by_modality.items():
        m = len(mod_results)
        entry = {
            "count": m,
            "accuracy": sum(r["correct"] for r in mod_results) / m,
            "format_compliance": sum(r["format_ok"] for r in mod_results) / m,
        }
        if method == Method.OURS:
            entry["concept_f1"] = sum(r["concept_reward"] for r in mod_results) / m
            entry["modality_accuracy"] = sum(r["modality_reward"] for r in mod_results) / m
        per_modality[mod] = entry

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


# ── Perturbation Faithfulness Test ───────────────────────────────────────────

def create_masked_image(
    image: Image.Image,
    mask_fraction: float = 0.5,
) -> Image.Image:
    """Create a version of the image with a large region masked out.

    Masks a centered rectangle covering mask_fraction of the image area.
    """
    img = image.copy()
    w, h = img.size
    mask_w = int(w * math.sqrt(mask_fraction))
    mask_h = int(h * math.sqrt(mask_fraction))
    x0 = (w - mask_w) // 2
    y0 = (h - mask_h) // 2

    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x0 + mask_w, y0 + mask_h], fill=(128, 128, 128))
    return img


@torch.no_grad()
def faithfulness_test(
    model,
    processor,
    dataset,
    device: str,
    method: Method = Method.OURS,
    cfg=EVAL_CFG,
    max_samples: int = 50,
    mask_fraction: float = 0.5,
) -> dict:
    """Test reasoning faithfulness by comparing outputs on original vs masked images.

    If the model's reasoning is faithful (actually uses the image), masking
    significant image regions should change the reasoning and/or the answer.
    Unfaithful reasoning would remain identical regardless of image content.

    Returns dict with faithfulness metrics.
    """
    model.eval()
    n = min(max_samples, len(dataset))

    reasoning_changed = 0
    answer_changed = 0
    concept_changed = 0

    for i in tqdm(range(n), desc="Faithfulness test"):
        sample = dataset[i]

        img = sample["image"]
        if isinstance(img, str):
            img = Image.open(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        masked_img = create_masked_image(img, mask_fraction)

        # Run on both original and masked
        completions = []
        for test_img in [img, masked_img]:
            messages = build_prompt_messages(sample["problem"], method)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = processor(
                text=[text], images=[test_img],
                return_tensors="pt", padding=True,
            )
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=0.01,  # near-deterministic for comparison
                do_sample=True,
            )
            prompt_len = inputs["input_ids"].shape[1]
            comp = processor.batch_decode(
                generated_ids[:, prompt_len:], skip_special_tokens=True,
            )[0]
            completions.append(comp)

        orig_comp, masked_comp = completions

        # Compare reasoning
        orig_think = re.search(r"<think>(.*?)</think>", orig_comp, re.DOTALL)
        masked_think = re.search(r"<think>(.*?)</think>", masked_comp, re.DOTALL)

        if orig_think and masked_think:
            orig_text = orig_think.group(1).strip()
            masked_text = masked_think.group(1).strip()
            if orig_text != masked_text:
                reasoning_changed += 1

        # Compare answers
        orig_ans = re.search(r"<answer>(.*?)</answer>", orig_comp, re.DOTALL)
        masked_ans = re.search(r"<answer>(.*?)</answer>", masked_comp, re.DOTALL)
        if orig_ans and masked_ans:
            if orig_ans.group(1).strip() != masked_ans.group(1).strip():
                answer_changed += 1

        # Compare concepts (ours only)
        if method == Method.OURS:
            orig_con = re.search(
                r"<concepts>\s*\[(.*?)\]\s*</concepts>", orig_comp, re.DOTALL,
            )
            masked_con = re.search(
                r"<concepts>\s*\[(.*?)\]\s*</concepts>", masked_comp, re.DOTALL,
            )
            if orig_con and masked_con:
                if orig_con.group(1).strip() != masked_con.group(1).strip():
                    concept_changed += 1

    result = {
        "samples_tested": n,
        "mask_fraction": mask_fraction,
        "reasoning_change_rate": reasoning_changed / n if n else 0,
        "answer_change_rate": answer_changed / n if n else 0,
    }
    if method == Method.OURS:
        result["concept_change_rate"] = concept_changed / n if n else 0

    return result


# ── Visualizations ───────────────────────────────────────────────────────────

def plot_per_modality_accuracy(
    metrics: dict, save_path: Path, method_label: str = "",
) -> None:
    """Bar chart of accuracy per modality."""
    per_mod = metrics.get("per_modality", {})
    if not per_mod:
        return

    modalities = list(per_mod.keys())
    accuracies = [per_mod[m]["accuracy"] for m in modalities]
    counts = [per_mod[m]["count"] for m in modalities]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(modalities, accuracies, color="steelblue", edgecolor="black")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"n={count}", ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel("Accuracy")
    title = f"Accuracy per Modality"
    if method_label:
        title += f" ({method_label})"
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.axhline(
        y=metrics["overall"]["accuracy"], color="red", linestyle="--",
        label=f"Overall: {metrics['overall']['accuracy']:.3f}",
    )
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_metrics_radar(
    metrics: dict, save_path: Path, method: Method = Method.OURS,
) -> None:
    """Radar chart of overall metrics."""
    overall = metrics.get("overall", {})

    if method == Method.OURS:
        labels = ["Accuracy", "Format", "Concept F1", "Modality", "Reward"]
        values = [
            overall.get("accuracy", 0),
            overall.get("format_compliance", 0),
            overall.get("mean_concept_f1", 0),
            overall.get("mean_modality_accuracy", 0),
            min(overall.get("mean_total_reward", 0) / 2.8, 1.0),
        ]
    else:
        labels = ["Accuracy", "Format\nCompliance", "Total\nReward"]
        values = [
            overall.get("accuracy", 0),
            overall.get("format_compliance", 0),
            min(overall.get("mean_total_reward", 0) / 2.0, 1.0),
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
    """Plot training loss, reward, and curriculum stage transitions."""
    if not log_path.exists():
        print(f"  No training log found at {log_path}, skipping curves")
        return

    with open(log_path) as f:
        log_data = json.load(f)

    train_entries = [e for e in log_data if "step" in e and "eval" not in e]
    eval_entries = [e.get("eval", e) for e in log_data if "eval" in e]

    if not train_entries:
        return

    steps = [e["step"] for e in train_entries]
    losses = [e["loss"] for e in train_entries]
    rewards = [e["mean_reward"] for e in train_entries]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curve
    axes[0].plot(steps, losses, "b-", alpha=0.7, linewidth=1)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("GRPO Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    # Reward curve
    axes[1].plot(steps, rewards, "g-", alpha=0.7, linewidth=1, label="Train")
    if eval_entries:
        eval_steps = [e.get("step", 0) for e in eval_entries]
        eval_rewards = [e.get("mean_reward", 0) for e in eval_entries]
        axes[1].plot(eval_steps, eval_rewards, "ro-", markersize=5, label="Eval")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Reward Progression")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Per-component rewards (if available)
    component_keys = [k for k in train_entries[0] if k.startswith("mean_") and k.endswith("_reward")]
    if component_keys:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for idx, key in enumerate(component_keys):
            label = key.replace("mean_", "").replace("_reward", "")
            vals = [e.get(key, 0) for e in train_entries]
            axes[2].plot(
                steps, vals, alpha=0.7, linewidth=1,
                color=colors[idx % len(colors)], label=label,
            )
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Reward")
        axes[2].set_title("Per-Component Rewards")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis("off")

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
    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]

    n_correct = min(n // 2, len(correct))
    n_incorrect = min(n - n_correct, len(incorrect))
    n_correct = min(n - n_incorrect, len(correct))
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

        sample = dataset[result["index"]]
        img = sample["image"]
        if isinstance(img, str):
            img = Image.open(img)

        ax.imshow(img)
        status = "CORRECT" if result["correct"] else "WRONG"
        color = "green" if result["correct"] else "red"
        ax.set_title(
            f"[{status}] GT:({result['gt_letter']}) Pred:({result['pred_letter']})",
            fontsize=9, color=color, fontweight="bold",
        )
        ax.axis("off")

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

    for idx in range(len(selected), rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    plt.suptitle("Example Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_faithfulness(faith_results: dict, save_path: Path) -> None:
    """Bar chart of faithfulness test change rates."""
    keys = ["reasoning_change_rate", "answer_change_rate"]
    labels = ["Reasoning", "Answer"]
    if "concept_change_rate" in faith_results:
        keys.append("concept_change_rate")
        labels.append("Concepts")

    values = [faith_results.get(k, 0) for k in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["steelblue", "coral", "seagreen"][:len(labels)])

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.1%}", ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel("Change Rate")
    ax.set_title(
        f"Faithfulness Test (mask={faith_results.get('mask_fraction', 0.5):.0%})\n"
        "Higher = more faithful (reasoning depends on image)",
    )
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cross_domain(cross_results: dict, save_path: Path) -> None:
    """Bar chart of cross-domain (leave-one-out) accuracy."""
    if not cross_results:
        return

    modalities = list(cross_results.keys())
    accuracies = [cross_results[m]["accuracy"] for m in modalities]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(modalities, accuracies, color="mediumpurple", edgecolor="black")

    for bar, mod in zip(bars, modalities):
        n = cross_results[mod].get("count", 0)
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"n={n}", ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-Domain Evaluation (Leave-One-Modality-Out)")
    ax.set_ylim(0, 1.1)
    if accuracies:
        mean_acc = sum(accuracies) / len(accuracies)
        ax.axhline(y=mean_acc, color="red", linestyle="--",
                    label=f"Mean: {mean_acc:.3f}")
        ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Cross-Domain Evaluation ─────────────────────────────────────────────────

def run_cross_domain_eval(
    model,
    processor,
    device: str,
    method: Method,
    cfg=EVAL_CFG,
    max_samples_per_mod: int = 50,
) -> dict:
    """Run leave-one-modality-out cross-domain evaluation.

    For each modality, test the model on ONLY that modality's samples,
    simulating the scenario where the model was not trained on that domain.
    """
    splits = DatasetDict.load_from_disk(str(PREPARED_DATA_DIR))
    test_dataset = splits.get("test", splits.get("validation"))

    cross_results = {}
    for mod in MODALITIES:
        # Filter test set to this modality
        mod_indices = [
            i for i in range(len(test_dataset))
            if test_dataset[i].get("modality", "") == mod
        ]
        if not mod_indices:
            continue

        mod_indices = mod_indices[:max_samples_per_mod]
        mod_dataset = test_dataset.select(mod_indices)

        results = run_inference(
            model, processor, mod_dataset, device, method, cfg,
            max_samples=max_samples_per_mod,
        )

        n = len(results)
        cross_results[mod] = {
            "count": n,
            "accuracy": sum(r["correct"] for r in results) / n if n else 0,
            "format_compliance": sum(r["format_ok"] for r in results) / n if n else 0,
        }
        if method == Method.OURS:
            cross_results[mod]["concept_f1"] = (
                sum(r["concept_reward"] for r in results) / n if n else 0
            )

        print(f"  {mod}: acc={cross_results[mod]['accuracy']:.3f} (n={n})")

    return cross_results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test and visualize MedVLM-R1 model",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained checkpoint directory",
    )
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "mps", "cpu"])
    parser.add_argument("--method", type=str, default="ours",
                        choices=["baseline", "ours"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "validation"])
    parser.add_argument("--output-dir", type=str, default=str(VIS_DIR))
    parser.add_argument("--faithfulness", action="store_true",
                        help="Run perturbation faithfulness test")
    parser.add_argument("--cross-domain", action="store_true",
                        help="Run cross-domain leave-one-out evaluation")
    args = parser.parse_args()

    method = Method(args.method)
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
    print(f"Method:   {method.value}")

    # Run inference
    results = run_inference(
        model, processor, test_dataset, device, method,
        max_samples=args.max_samples,
    )

    # Compute metrics
    metrics = compute_metrics(results, method)

    # Print results
    print(f"\n{'='*60}")
    print(f"TEST RESULTS ({method.value})")
    print(f"{'='*60}")
    print(f"  Overall Accuracy:      {metrics['overall']['accuracy']:.4f}")
    print(f"  Format Compliance:     {metrics['overall']['format_compliance']:.4f}")
    if method == Method.OURS:
        print(f"  Mean Concept F1:       {metrics['overall']['mean_concept_f1']:.4f}")
        print(f"  Modality Accuracy:     {metrics['overall']['mean_modality_accuracy']:.4f}")
    print(f"  Mean Total Reward:     {metrics['overall']['mean_total_reward']:.4f}")

    print(f"\n  Per-Modality Accuracy:")
    for mod, mod_m in metrics["per_modality"].items():
        print(f"    {mod:20s}: {mod_m['accuracy']:.3f} (n={mod_m['count']})")
    print(f"{'='*60}")

    # Save results JSON
    results_path = out_dir / f"test_results_{method.value}.json"
    with open(results_path, "w") as f:
        json.dump(
            {"metrics": metrics, "predictions": results},
            f, indent=2, default=str,
        )
    print(f"\nSaved results to {results_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    suffix = f"_{method.value}"
    plot_per_modality_accuracy(
        metrics, out_dir / f"per_modality_accuracy{suffix}.png", method.value,
    )
    plot_metrics_radar(metrics, out_dir / f"metrics_radar{suffix}.png", method)
    plot_answer_distribution(metrics, out_dir / f"answer_distribution{suffix}.png")
    plot_training_curves(LOG_DIR / "training_log.json", out_dir / "training_curves.png")
    plot_example_predictions(
        results, test_dataset, out_dir / f"example_predictions{suffix}.png",
    )

    # Faithfulness test
    if args.faithfulness:
        print("\nRunning faithfulness test...")
        faith_results = faithfulness_test(
            model, processor, test_dataset, device, method,
            max_samples=min(50, args.max_samples or 50),
        )
        print(f"  Reasoning change rate: {faith_results['reasoning_change_rate']:.1%}")
        print(f"  Answer change rate:    {faith_results['answer_change_rate']:.1%}")
        if "concept_change_rate" in faith_results:
            print(f"  Concept change rate:   {faith_results['concept_change_rate']:.1%}")

        plot_faithfulness(faith_results, out_dir / f"faithfulness{suffix}.png")

        with open(out_dir / f"faithfulness{suffix}.json", "w") as f:
            json.dump(faith_results, f, indent=2)

    # Cross-domain evaluation
    if args.cross_domain:
        print("\nRunning cross-domain evaluation...")
        cross_results = run_cross_domain_eval(
            model, processor, device, method,
        )
        plot_cross_domain(cross_results, out_dir / f"cross_domain{suffix}.png")

        with open(out_dir / f"cross_domain{suffix}.json", "w") as f:
            json.dump(cross_results, f, indent=2)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
