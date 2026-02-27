"""
data_prep_and_viewer.py - Dataset loading, parsing, formatting, and visualization.

Responsibilities:
  1. Load a HuggingFace dataset (images + MCQ problems + solutions).
  2. Parse the custom solution format:  <MODALITY><CONCEPT>[...]</CONCEPT>\n(X) answer
  3. Convert each sample into the R1-style conversation format for training.
  4. Split into train / val / test sets.
  5. Provide viewer utilities to inspect samples (image + prompt + expected output).

Usage:
  python data_prep_and_viewer.py                    # prepare + show 5 samples
  python data_prep_and_viewer.py --view-only 10     # show 10 samples from saved data
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
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
    QUESTION_TEMPLATE,
    SEED,
    SYSTEM_PROMPT,
    TEST_RATIO,
    TRAIN_CFG,
    TRAIN_RATIO,
    VAL_RATIO,
)

PREPARED_DATA_DIR = OUTPUT_DIR / "prepared_dataset"


# ── Solution Parsing ─────────────────────────────────────────────────────────

def parse_solution(solution: str) -> dict[str, Any]:
    """Parse the dataset's solution string into structured components.

    Expected format examples:
        "<X_RAY><CONCEPT>['Enlarged cardiomediastinum']</CONCEPT>\n(A) no covid-19"
        "<DERM><CONCEPT>['pigment network', 'streaks']</CONCEPT>\n(B) melanoma"

    Returns:
        {
            "modality_tag": "X_RAY",
            "modality": "corda",               # canonical name
            "concepts": ["Enlarged cardiomediastinum"],
            "answer_letter": "A",
            "answer_text": "no covid-19",
            "raw": "<X_RAY><CONCEPT>['Enlarged ...']</CONCEPT>\n(A) no covid-19"
        }
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
        # If it's "CONCEPT", skip — that's the sub-tag
        if tag != "CONCEPT":
            result["modality"] = MODALITY_TAG_MAP.get(tag, tag)

    # Extract concepts:  <CONCEPT>['c1', 'c2']</CONCEPT>
    concept_match = re.search(r"<CONCEPT>(.*?)</CONCEPT>", solution, re.DOTALL)
    if concept_match:
        raw_concepts = concept_match.group(1).strip()
        try:
            result["concepts"] = ast.literal_eval(raw_concepts)
        except (ValueError, SyntaxError):
            # Fallback: split by comma
            result["concepts"] = [
                c.strip().strip("'\"") for c in raw_concepts.strip("[]").split(",")
            ]

    # Extract answer:  (A) answer text
    ans_match = re.search(r"\(([A-Z])\)\s*(.*)", solution)
    if ans_match:
        result["answer_letter"] = ans_match.group(1)
        result["answer_text"] = ans_match.group(2).strip()

    # If modality wasn't found from the tag, try to infer from concepts
    if not result["modality"] and result["concepts"]:
        for concept in result["concepts"]:
            if concept in CONCEPT_TO_MODALITY:
                result["modality"] = CONCEPT_TO_MODALITY[concept]
                break

    return result


# ── Format Conversion ────────────────────────────────────────────────────────

def build_expected_output(parsed: dict[str, Any]) -> str:
    """Build the target R1-style output string for a sample.

    Returns:
        "<think>\n<modality>corda</modality>\n<concepts>['Enlarged cardiomediastinum']
         </concepts>\n</think>\n<answer>(A) no covid-19</answer>"
    """
    modality = parsed["modality"] or "unknown"
    concepts = parsed["concepts"]
    letter = parsed["answer_letter"]
    text = parsed["answer_text"]

    answer_str = f"({letter}) {text}" if text else letter

    return (
        f"<think>\n"
        f"<modality>{modality}</modality>\n"
        f"<concepts>{concepts}</concepts>\n"
        f"</think>\n"
        f"<answer>{answer_str}</answer>"
    )


def build_conversation(problem: str, modality: str = "") -> list[dict]:
    """Build the chat-style conversation input for the model.

    Returns a list of messages: [system, user].
    The user message contains an image placeholder and the formatted question.
    """
    # Build the available modalities string for the prompt
    modalities_str = ", ".join(MODALITIES)

    user_text = QUESTION_TEMPLATE.format(
        question=problem,
        modalities=modalities_str,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def process_sample(example: dict) -> dict:
    """Process a single dataset sample into training-ready format.

    Input keys: image, problem, solution
    Output keys: image, problem, solution, parsed_solution, conversation,
                 expected_output, modality, concepts, answer_letter
    """
    parsed = parse_solution(example["solution"])
    conversation = build_conversation(example["problem"], parsed["modality"])
    expected_output = build_expected_output(parsed)

    return {
        "image": example["image"],
        "problem": example["problem"],
        "solution": example["solution"],
        "modality": parsed["modality"],
        "concepts": json.dumps(parsed["concepts"]),  # serialize list for Arrow
        "answer_letter": parsed["answer_letter"],
        "expected_output": expected_output,
        "conversation": json.dumps(conversation),      # serialize for Arrow
    }


# ── Dataset Loading & Splitting ──────────────────────────────────────────────

def load_and_prepare(
    dataset_id: str = DATASET_ID,
    force_reload: bool = False,
) -> DatasetDict:
    """Load the HF dataset, process samples, and split into train/val/test.

    If already prepared on disk, loads from cache unless force_reload=True.
    """
    if PREPARED_DATA_DIR.exists() and not force_reload:
        print(f"Loading prepared dataset from {PREPARED_DATA_DIR}")
        return DatasetDict.load_from_disk(str(PREPARED_DATA_DIR))

    print(f"Loading raw dataset from HuggingFace: {dataset_id}")
    raw = load_dataset(dataset_id)

    # Flatten if DatasetDict with a single split
    if isinstance(raw, DatasetDict):
        if "train" in raw and len(raw) == 1:
            ds = raw["train"]
        elif "train" in raw and "test" in raw:
            # Already split — process each
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

    # Process
    print(f"Processing {len(ds)} samples...")
    ds = ds.map(process_sample, remove_columns=[], num_proc=1)

    # Split
    train_test = ds.train_test_split(
        test_size=(1 - TRAIN_RATIO), seed=SEED
    )
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


# ── Viewer ───────────────────────────────────────────────────────────────────

def view_samples(
    dataset: Dataset,
    n: int = 5,
    save_path: Path | None = None,
) -> None:
    """Display dataset samples: image + problem + parsed solution + expected output."""
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

        # Build info text
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
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            horizontalalignment="center",
            wrap=True,
            family="monospace",
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
    print(f"\nConversation (prompt):")
    conv = sample.get("conversation", "[]")
    if isinstance(conv, str):
        conv = json.loads(conv)
    for msg in conv:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = [c["text"] for c in content if c.get("type") == "text"]
            content = " ".join(texts)
        print(f"  [{role}]: {str(content)[:200]}...")
    print(f"{'='*70}\n")


# ── Statistics ───────────────────────────────────────────────────────────────

def dataset_statistics(dataset: Dataset) -> dict:
    """Compute basic statistics about the dataset."""
    modalities = {}
    concept_counts = {}
    answer_dist = {}

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
        bar = "#" * (count * 40 // stats["total_samples"])
        print(f"  {mod:15s} | {count:5d} | {bar}")

    print(f"\nAnswer Distribution:")
    for ans, count in stats["answer_distribution"].items():
        bar = "#" * (count * 40 // stats["total_samples"])
        print(f"  ({ans})            | {count:5d} | {bar}")

    print(f"\nTop 10 Concepts:")
    for concept, count in list(stats["top_concepts"].items())[:10]:
        print(f"  {concept:40s} | {count:5d}")
    print(f"{'='*50}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare and view MedVLM-R1 dataset")
    parser.add_argument(
        "--dataset-id", type=str, default=DATASET_ID,
        help=f"HuggingFace dataset identifier (default: {DATASET_ID})",
    )
    parser.add_argument(
        "--force-reload", action="store_true",
        help="Force re-download and re-processing of the dataset",
    )
    parser.add_argument(
        "--view-only", type=int, nargs="?", const=5,
        help="Only view N samples from already-prepared data (default: 5)",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "validation", "test"],
        help="Which split to view (default: train)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print dataset statistics",
    )
    parser.add_argument(
        "--save-viewer", type=str, default=None,
        help="Save viewer image to this path instead of displaying",
    )
    args = parser.parse_args()

    # Load or prepare
    if args.view_only is not None and PREPARED_DATA_DIR.exists():
        splits = DatasetDict.load_from_disk(str(PREPARED_DATA_DIR))
    else:
        splits = load_and_prepare(
            dataset_id=args.dataset_id,
            force_reload=args.force_reload,
        )

    ds = splits[args.split]
    n = args.view_only if args.view_only is not None else 5

    # Print detailed view of first few samples
    for i in range(min(3, n)):
        print_sample_details(ds, i)

    # Show stats
    if args.stats or args.view_only is None:
        stats = dataset_statistics(ds)
        print_statistics(stats)

    # Visual viewer
    save_path = Path(args.save_viewer) if args.save_viewer else None
    view_samples(ds, n=n, save_path=save_path)


if __name__ == "__main__":
    main()
