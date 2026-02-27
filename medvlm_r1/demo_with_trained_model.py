"""
demo_with_trained_model.py - Interactive demo for inference with a trained model.

Supports both methods (--method baseline / --method ours):
  - Baseline: expects <think>...</think><answer>...</answer>
  - Ours: expects <think><modality>...</modality><concepts>...</concepts>...</think><answer>...</answer>

Usage:
  # Single image (ours method)
  python demo_with_trained_model.py --checkpoint outputs/checkpoints/best \
      --image path/to/xray.png \
      --question "Is there evidence of pneumonia?\noptions: (A) yes (B) no"

  # Baseline method
  python demo_with_trained_model.py --checkpoint outputs/checkpoints/best \
      --method baseline --image path/to/xray.png --question "..."

  # Batch mode
  python demo_with_trained_model.py --checkpoint outputs/checkpoints/best \
      --image-dir path/to/images/ --question "..."

  # Interactive loop
  python demo_with_trained_model.py --checkpoint outputs/checkpoints/best --interactive
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image

from config import (
    CONCEPT_MAP,
    EVAL_CFG,
    MODALITIES,
    MODEL_ID,
    Method,
    get_prompts,
)
from model_loader_and_checker import load_model_and_processor
from train_eval import build_prompt_messages


# ── Response Parsing ─────────────────────────────────────────────────────────

def parse_response(text: str, method: Method = Method.OURS) -> dict:
    """Parse a model response into structured components.

    Returns dict with: modality, concepts, reasoning, answer_letter,
                       answer_text, format_valid, raw
    """
    result = {
        "modality": "",
        "concepts": [],
        "reasoning": "",
        "answer_letter": "",
        "answer_text": "",
        "format_valid": False,
        "raw": text,
    }

    # Check format validity
    if method == Method.BASELINE:
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    else:
        pattern = (
            r"<think>.*?<modality>.*?</modality>.*?<concepts>.*?</concepts>"
            r".*?</think>\s*<answer>.*?</answer>"
        )
    result["format_valid"] = bool(re.fullmatch(pattern, text.strip(), re.DOTALL))

    # Extract modality (ours only, but parse if present)
    mod_match = re.search(r"<modality>(.*?)</modality>", text, re.DOTALL)
    if mod_match:
        result["modality"] = mod_match.group(1).strip()

    # Extract concepts
    con_match = re.search(r"<concepts>\s*\[(.*?)\]\s*</concepts>", text, re.DOTALL)
    if con_match:
        result["concepts"] = [
            c.strip().strip("'\"")
            for c in con_match.group(1).split(",") if c.strip()
        ]

    # Extract reasoning
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1)
        reasoning = re.sub(r"<modality>.*?</modality>", "", reasoning, flags=re.DOTALL)
        reasoning = re.sub(r"<concepts>.*?</concepts>", "", reasoning, flags=re.DOTALL)
        result["reasoning"] = reasoning.strip()

    # Extract answer
    ans_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if ans_match:
        answer_raw = ans_match.group(1).strip()
        letter_match = re.search(r"\(?([A-J])\)?", answer_raw)
        if letter_match:
            result["answer_letter"] = letter_match.group(1)
        result["answer_text"] = answer_raw

    return result


# ── Inference ────────────────────────────────────────────────────────────────

def predict(
    model,
    processor,
    image: Image.Image,
    question: str,
    device: str,
    method: Method = Method.OURS,
    temperature: float = EVAL_CFG.temperature,
    max_new_tokens: int = EVAL_CFG.max_new_tokens,
) -> dict:
    """Run inference on a single image + question.

    Returns parsed response dict.
    """
    messages = build_prompt_messages(question, method)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True,
    )
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=EVAL_CFG.top_p,
            do_sample=temperature > 0,
        )

    prompt_len = inputs["input_ids"].shape[1]
    completion = processor.batch_decode(
        generated_ids[:, prompt_len:], skip_special_tokens=True,
    )[0]

    return parse_response(completion, method)


# ── Display ──────────────────────────────────────────────────────────────────

def display_result(
    result: dict, image_path: str = "", method: Method = Method.OURS,
) -> None:
    """Pretty-print a prediction result."""
    fmt_status = "VALID" if result["format_valid"] else "INVALID"
    fmt_marker = "" if result["format_valid"] else " [!]"

    print(f"\n{'='*60}")
    if image_path:
        print(f"Image:  {image_path}")
    print(f"Method: {method.value}")
    print(f"Format: {fmt_status}{fmt_marker}")
    print(f"{'─'*60}")

    if method == Method.OURS:
        print(f"Modality:  {result['modality'] or '(not detected)'}")
        print(f"Concepts:  {result['concepts'] or '(none detected)'}")

        # Validate concepts against known list
        if result["modality"] and result["modality"] in CONCEPT_MAP:
            valid_concepts = set(CONCEPT_MAP[result["modality"]])
            for c in result["concepts"]:
                if c not in valid_concepts:
                    print(f"           ^ '{c}' is not in the "
                          f"{result['modality']} concept list")
        print(f"{'─'*60}")

    if result["reasoning"]:
        print("Reasoning:")
        for line in result["reasoning"].split("\n"):
            if line.strip():
                print(f"  {line.strip()}")
    print(f"{'─'*60}")
    print(f"Answer:    ({result['answer_letter']}) {result['answer_text']}")
    print(f"{'='*60}\n")


# ── Batch Mode ───────────────────────────────────────────────────────────────

def run_batch(
    model,
    processor,
    image_dir: str,
    question: str,
    device: str,
    method: Method = Method.OURS,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff"),
) -> list[dict]:
    """Run inference on all images in a directory."""
    image_dir = Path(image_dir)
    image_files = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in extensions
    )

    if not image_files:
        print(f"No images found in {image_dir}")
        return []

    print(f"Found {len(image_files)} images in {image_dir}")
    results = []

    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        result = predict(model, processor, img, question, device, method)
        result["image_path"] = str(img_path)
        results.append(result)
        display_result(result, str(img_path), method)

    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY ({len(results)} images, method={method.value})")
    print(f"{'='*60}")
    print(f"  Format valid: {sum(r['format_valid'] for r in results)}/{len(results)}")

    answer_dist = {}
    for r in results:
        letter = r["answer_letter"] or "?"
        answer_dist[letter] = answer_dist.get(letter, 0) + 1
    print(f"  Answer distribution: {answer_dist}")

    if method == Method.OURS:
        modality_dist = {}
        for r in results:
            m = r["modality"] or "unknown"
            modality_dist[m] = modality_dist.get(m, 0) + 1
        print(f"  Modality distribution: {modality_dist}")
    print(f"{'='*60}")

    return results


# ── Interactive Mode ─────────────────────────────────────────────────────────

def interactive_loop(
    model, processor, device: str, method: Method = Method.OURS,
) -> None:
    """Interactive loop: user provides image path and question, gets prediction."""
    print(f"\n{'='*60}")
    print(f"MedVLM-R1 Interactive Demo (method={method.value})")
    print("Type 'quit' or 'exit' to stop.")
    print(f"{'='*60}\n")

    while True:
        image_path = input("Image path (or 'quit'): ").strip()
        if image_path.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not Path(image_path).exists():
            print(f"  File not found: {image_path}")
            continue

        print("Enter question (end with empty line):")
        lines = []
        while True:
            line = input("  ")
            if not line:
                break
            lines.append(line)

        if not lines:
            print("  No question provided, skipping.")
            continue

        question = "\n".join(lines)

        try:
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            result = predict(model, processor, img, question, device, method)
            display_result(result, image_path, method)
        except Exception as e:
            print(f"  Error: {e}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Demo with trained MedVLM-R1 model",
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

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", type=str, help="Path to a single image file")
    group.add_argument("--image-dir", type=str, help="Directory of images")
    group.add_argument("--interactive", action="store_true", help="Interactive mode")

    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=EVAL_CFG.temperature)
    parser.add_argument("--max-tokens", type=int, default=EVAL_CFG.max_new_tokens)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    method = Method(args.method)

    # Load model
    model, processor, device = load_model_and_processor(
        model_id=args.model_id,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    if args.interactive:
        interactive_loop(model, processor, device, method)

    elif args.image:
        if not args.question:
            parser.error("--question is required with --image")

        img = Image.open(args.image)
        if img.mode != "RGB":
            img = img.convert("RGB")

        result = predict(
            model, processor, img, args.question, device, method,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
        )
        display_result(result, args.image, method)

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Saved result to {args.output_json}")

    elif args.image_dir:
        if not args.question:
            parser.error("--question is required with --image-dir")

        results = run_batch(
            model, processor, args.image_dir, args.question, device, method,
        )

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Saved {len(results)} results to {args.output_json}")

    else:
        interactive_loop(model, processor, device, method)


if __name__ == "__main__":
    main()
