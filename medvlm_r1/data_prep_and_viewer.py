"""
data_prep_and_viewer.py - Dataset loading, parsing, augmentation, and visualization.

Responsibilities:
  1. Load a HuggingFace dataset (images + MCQ problems + solutions).
  2. Parse the custom solution format.
  3. Convert each sample into the R1-style conversation format for training.
  4. Split into train / val / test sets.
  5. Generate cross-domain (leave-one-modality-out) splits for OOD evaluation.
  6. Prompt augmentation: rephrase questions to prevent shortcut learning.
  7. Concept perturbation: shuffle concept list order during training.
  8. Provide viewer utilities to inspect samples.

Usage:
  python data_prep_and_viewer.py                          # prepare + show 5 samples
  python data_prep_and_viewer.py --cross-domain            # generate leave-one-out splits
  python data_prep_and_viewer.py --view-only 10 --stats    # view + stats
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image

from config import (
    CONCEPT_MAP,
    CONCEPT_TO_MODALITY,
    DATASET_ID,
    MODALITIES,
    MODALITY_TAG_MAP,
    OUTPUT_DIR,
    QUESTION_REPHRASINGS,
    QUESTION_TEMPLATE,
    SEED,
    SYSTEM_PROMPT,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    Method,
    get_prompts,
)

PREPARED_DATA_DIR = OUTPUT_DIR / "prepared_dataset"
CROSS_DOMAIN_DIR = OUTPUT_DIR / "cross_domain_splits"


# ── Solution Parsing ─────────────────────────────────────────────────────────

def parse_solution(solution: str) -> dict[str, Any]:
    """Parse the dataset's solution string into structured components.

    Expected format:
        "<X_RAY><CONCEPT>['Enlarged cardiomediastinum']</CONCEPT>\\n(A) no covid-19"

    Returns dict with modality_tag, modality, concepts, answer_letter, answer_text, raw.
    """
    result: dict[str, Any] = {
        "modality_tag": "",
        "modality": "",
        "concepts": [],
        "answer_letter": "",
        "answer_text": "",
        "raw": solution,
    }

    # Extract modality tag:  <X_RAY>, <DERM>, etc.
    mod_match = re.search(r"<([A-Z_]+)>", solution)
    if mod_match:
        tag = mod_match.group(1)
        result["modality_tag"] = tag
        if tag != "CONCEPT":
            result["modality"] = MODALITY_TAG_MAP.get(tag, tag)

    # Extract concepts:  <CONCEPT>['c1', 'c2']</CONCEPT>
    concept_match = re.search(r"<CONCEPT>(.*?)</CONCEPT>", solution, re.DOTALL)
    if concept_match:
        raw_concepts = concept_match.group(1).strip()
        try:
            result["concepts"] = ast.literal_eval(raw_concepts)
        except (ValueError, SyntaxError):
            result["concepts"] = [
                c.strip().strip("'\"") for c in raw_concepts.strip("[]").split(",")
            ]

    # Extract answer:  (A) answer text
    ans_match = re.search(r"\(([A-Z])\)\s*(.*)", solution)
    if ans_match:
        result["answer_letter"] = ans_match.group(1)
        result["answer_text"] = ans_match.group(2).strip()

    # Infer modality from concepts if tag didn't match
    if not result["modality"] and result["concepts"]:
        for concept in result["concepts"]:
            if concept in CONCEPT_TO_MODALITY:
                result["modality"] = CONCEPT_TO_MODALITY[concept]
                break

    return result


# ── Format Conversion ────────────────────────────────────────────────────────

def build_expected_output(parsed: dict[str, Any], method: Method = Method.OURS) -> str:
    """Build the target R1-style output string for a sample."""
    letter = parsed["answer_letter"]
    text = parsed["answer_text"]
    answer_str = f"({letter}) {text}" if text else letter

    if method == Method.BASELINE:
        # MedVLM-R1 style: free-text reasoning
        return f"<think>\n</think>\n<answer>{answer_str}</answer>"

    # Ours: structured reasoning
    modality = parsed["modality"] or "unknown"
    concepts = parsed["concepts"]
    return (
        f"<think>\n"
        f"<modality>{modality}</modality>\n"
        f"<concepts>{concepts}</concepts>\n"
        f"</think>\n"
        f"<answer>{answer_str}</answer>"
    )


def build_conversation(
    problem: str,
    method: Method = Method.OURS,
    modality: str = "",
) -> list[dict]:
    """Build the chat-style conversation input for the model."""
    sys_prompt, q_template = get_prompts(method)
    modalities_str = ", ".join(MODALITIES)

    if method == Method.BASELINE:
        user_text = q_template.format(question=problem)
    else:
        user_text = q_template.format(question=problem, modalities=modalities_str)

    return [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def process_sample(example: dict) -> dict:
    """Process a single dataset sample into training-ready format."""
    parsed = parse_solution(example["solution"])
    conversation = build_conversation(example["problem"])
    expected_output = build_expected_output(parsed)

    return {
        "image": example["image"],
        "problem": example["problem"],
        "solution": example["solution"],
        "modality": parsed["modality"],
        "concepts": json.dumps(parsed["concepts"]),
        "answer_letter": parsed["answer_letter"],
        "expected_output": expected_output,
        "conversation": json.dumps(conversation),
    }


# ── Prompt Augmentation ─────────────────────────────────────────────────────
# Ref: Strategy doc Section 2.5. Rephrase questions to prevent shortcut
# learning on surface-level question patterns.

def extract_question_core(problem: str) -> str:
    """Extract the core question without the leading instruction prefix.

    E.g., "Select the most likely diagnosis from the following list.\\n
           options: (A) no covid-19, (B) covid-19"
      -> "select the most likely diagnosis from the following list.
          options: (A) no covid-19, (B) covid-19"
    """
    # Find the first line that looks like an instruction
    lines = problem.strip().split("\n")
    # Keep everything — just lowercase the first sentence for rephrasing
    core = problem.strip()
    # Try to split instruction from options
    for prefix in ["Select ", "What ", "Which ", "Identify ", "Choose "]:
        if core.startswith(prefix):
            core = core[0].lower() + core[1:]
            break
    return core


def augment_question(problem: str, rng: random.Random | None = None) -> str:
    """Generate a rephrased version of the question.

    Returns the original with probability 0.5, or a rephrased version otherwise.
    This ensures the model sees both original and augmented prompts.
    """
    rng = rng or random.Random()
    if rng.random() < 0.5:
        return problem  # keep original half the time

    core = extract_question_core(problem)
    template = rng.choice(QUESTION_REPHRASINGS)
    return template.format(question_core=core)


# ── Concept Perturbation ────────────────────────────────────────────────────
# Ref: Strategy doc Section 2.5. Shuffle concept list order to prevent
# the model from memorizing positional patterns.

def get_shuffled_concept_list(modality: str, rng: random.Random | None = None) -> list[str]:
    """Return the concept list for a modality in shuffled order."""
    rng = rng or random.Random()
    concepts = CONCEPT_MAP.get(modality, [])
    if not concepts:
        return concepts
    shuffled = concepts.copy()
    rng.shuffle(shuffled)
    return shuffled


# ── Dataset Loading & Splitting ──────────────────────────────────────────────

def load_and_prepare(
    dataset_id: str = DATASET_ID,
    force_reload: bool = False,
) -> DatasetDict:
    """Load the HF dataset, process samples, and split into train/val/test."""
    if PREPARED_DATA_DIR.exists() and not force_reload:
        print(f"Loading prepared dataset from {PREPARED_DATA_DIR}")
        return DatasetDict.load_from_disk(str(PREPARED_DATA_DIR))

    print(f"Loading raw dataset from HuggingFace: {dataset_id}")
    raw = load_dataset(dataset_id)

    if isinstance(raw, DatasetDict):
        if "train" in raw and len(raw) == 1:
            ds = raw["train"]
        elif "train" in raw and "test" in raw:
            processed = DatasetDict()
            for split_name, split_ds in raw.items():
                processed[split_name] = split_ds.map(
                    process_sample, remove_columns=[], num_proc=1
                )
            processed.save_to_disk(str(PREPARED_DATA_DIR))
            print(f"Saved prepared dataset to {PREPARED_DATA_DIR}")
            return processed
        else:
            ds = raw[list(raw.keys())[0]]
    else:
        ds = raw

    print(f"Processing {len(ds)} samples...")
    ds = ds.map(process_sample, remove_columns=[], num_proc=1)

    train_test = ds.train_test_split(test_size=(1 - TRAIN_RATIO), seed=SEED)
    val_test = train_test["test"].train_test_split(
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), seed=SEED
    )

    splits = DatasetDict({
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })

    splits.save_to_disk(str(PREPARED_DATA_DIR))
    print(f"Saved prepared dataset to {PREPARED_DATA_DIR}")
    print(f"  train:      {len(splits['train']):,}")
    print(f"  validation: {len(splits['validation']):,}")
    print(f"  test:       {len(splits['test']):,}")

    return splits


# ── Cross-Domain Splits ──────────────────────────────────────────────────────
# Leave-one-modality-out splits for OOD generalization evaluation.

def create_cross_domain_splits(
    dataset: Dataset,
    save: bool = True,
) -> dict[str, DatasetDict]:
    """Create leave-one-modality-out splits.

    For each modality M:
      train = all samples NOT from modality M
      test  = all samples from modality M

    Returns dict mapping held-out modality name -> DatasetDict(train, test).
    """
    cross_splits = {}

    for held_out_mod in MODALITIES:
        train_indices = []
        test_indices = []

        for i in range(len(dataset)):
            mod = dataset[i].get("modality", "")
            if mod == held_out_mod:
                test_indices.append(i)
            else:
                train_indices.append(i)

        if not test_indices:
            print(f"  Skipping {held_out_mod}: no samples found")
            continue

        cross_splits[held_out_mod] = DatasetDict({
            "train": dataset.select(train_indices),
            "test": dataset.select(test_indices),
        })
        print(f"  {held_out_mod}: train={len(train_indices)}, test={len(test_indices)}")

    if save:
        CROSS_DOMAIN_DIR.mkdir(parents=True, exist_ok=True)
        for mod_name, split_dict in cross_splits.items():
            save_path = CROSS_DOMAIN_DIR / mod_name
            split_dict.save_to_disk(str(save_path))
        print(f"Saved cross-domain splits to {CROSS_DOMAIN_DIR}")

    return cross_splits


# ── Viewer & Statistics ──────────────────────────────────────────────────────

def view_samples(
    dataset: Dataset,
    n: int = 5,
    save_path: Path | None = None,
) -> None:
    """Display dataset samples: image + problem + parsed solution."""
    n = min(n, len(dataset))
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        sample = dataset[i]
        img = sample["image"]
        if isinstance(img, str):
            img = Image.open(img)

        ax.imshow(img)
        ax.set_title(f"Sample {i}", fontsize=10, fontweight="bold")
        ax.axis("off")

        modality = sample.get("modality", "?")
        concepts = sample.get("concepts", "[]")
        if isinstance(concepts, str):
            concepts = json.loads(concepts)
        answer = sample.get("answer_letter", "?")

        info = (
            f"Modality: {modality}\n"
            f"Concepts: {concepts}\n"
            f"Answer: ({answer})\n"
            f"Problem: {sample['problem'][:80]}..."
        )
        ax.text(
            0.5, -0.02, info,
            transform=ax.transAxes, fontsize=7,
            verticalalignment="top", horizontalalignment="center",
            wrap=True, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved viewer image to {save_path}")
    else:
        plt.show()


def print_sample_details(dataset: Dataset, idx: int = 0) -> None:
    """Print full details of a single sample to stdout."""
    sample = dataset[idx]
    print(f"\n{'='*70}")
    print(f"SAMPLE {idx}")
    print(f"{'='*70}")
    print(f"Problem:\n  {sample['problem']}")
    print(f"\nRaw Solution:\n  {sample['solution']}")
    print(f"\nModality:      {sample.get('modality', '?')}")
    concepts = sample.get("concepts", "[]")
    if isinstance(concepts, str):
        concepts = json.loads(concepts)
    print(f"Concepts:      {concepts}")
    print(f"Answer Letter: {sample.get('answer_letter', '?')}")
    print(f"\nExpected Output:")
    print(f"  {sample.get('expected_output', 'N/A')}")
    print(f"{'='*70}\n")


def dataset_statistics(dataset: Dataset) -> dict:
    """Compute basic statistics about the dataset."""
    modalities: dict[str, int] = {}
    concept_counts: dict[str, int] = {}
    answer_dist: dict[str, int] = {}

    for sample in dataset:
        mod = sample.get("modality", "unknown")
        modalities[mod] = modalities.get(mod, 0) + 1

        concepts = sample.get("concepts", "[]")
        if isinstance(concepts, str):
            concepts = json.loads(concepts)
        for c in concepts:
            concept_counts[c] = concept_counts.get(c, 0) + 1

        ans = sample.get("answer_letter", "?")
        answer_dist[ans] = answer_dist.get(ans, 0) + 1

    return {
        "total_samples": len(dataset),
        "modality_distribution": dict(sorted(modalities.items(), key=lambda x: -x[1])),
        "answer_distribution": dict(sorted(answer_dist.items())),
        "top_concepts": dict(sorted(concept_counts.items(), key=lambda x: -x[1])[:20]),
        "unique_concepts": len(concept_counts),
    }


def print_statistics(stats: dict) -> None:
    """Pretty-print dataset statistics."""
    print(f"\n{'='*50}")
    print("DATASET STATISTICS")
    print(f"{'='*50}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Unique concepts: {stats['unique_concepts']}")

    print(f"\nModality Distribution:")
    for mod, count in stats["modality_distribution"].items():
        bar = "#" * (count * 40 // max(stats["total_samples"], 1))
        print(f"  {mod:15s} | {count:5d} | {bar}")

    print(f"\nAnswer Distribution:")
    for ans, count in stats["answer_distribution"].items():
        bar = "#" * (count * 40 // max(stats["total_samples"], 1))
        print(f"  ({ans})            | {count:5d} | {bar}")

    print(f"\nTop 10 Concepts:")
    for concept, count in list(stats["top_concepts"].items())[:10]:
        print(f"  {concept:40s} | {count:5d}")
    print(f"{'='*50}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare and view MedVLM-R1 dataset")
    parser.add_argument("--dataset-id", type=str, default=DATASET_ID)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--view-only", type=int, nargs="?", const=5)
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test"])
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--save-viewer", type=str, default=None)
    parser.add_argument("--cross-domain", action="store_true",
                        help="Generate leave-one-modality-out splits")
    args = parser.parse_args()

    if args.view_only is not None and PREPARED_DATA_DIR.exists():
        splits = DatasetDict.load_from_disk(str(PREPARED_DATA_DIR))
    else:
        splits = load_and_prepare(
            dataset_id=args.dataset_id, force_reload=args.force_reload
        )

    ds = splits[args.split]
    n = args.view_only if args.view_only is not None else 5

    for i in range(min(3, n)):
        print_sample_details(ds, i)

    if args.stats or args.view_only is None:
        stats = dataset_statistics(ds)
        print_statistics(stats)

    if args.cross_domain:
        print("\nGenerating cross-domain splits...")
        # Use the full dataset (all splits merged) or just train
        create_cross_domain_splits(ds)

    save_path = Path(args.save_viewer) if args.save_viewer else None
    view_samples(ds, n=n, save_path=save_path)


if __name__ == "__main__":
    main()
