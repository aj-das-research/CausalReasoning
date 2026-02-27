"""
model_loader_and_checker.py - Load VLM, verify architecture, and run sanity checks.

Supports Qwen2-VL (2B) and Qwen2.5-VL (3B/7B) model families.

Usage:
  python model_loader_and_checker.py                      # full check with default model
  python model_loader_and_checker.py --model-id Qwen/Qwen2-VL-2B-Instruct
  python model_loader_and_checker.py --checkpoint <path>  # check a fine-tuned checkpoint
  python model_loader_and_checker.py --method baseline    # check with baseline prompts
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from config import (
    EVAL_CFG,
    MODEL_ID,
    TRAIN_CFG,
    Method,
    get_prompts,
)


# ── Model Loading ────────────────────────────────────────────────────────────

def detect_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_processor(
    model_id: str = MODEL_ID,
    checkpoint_path: str | None = None,
    device: str | None = None,
    max_pixels: int = TRAIN_CFG.max_pixels,
    min_pixels: int = TRAIN_CFG.min_pixels,
    dtype: torch.dtype | None = None,
) -> tuple:
    """Load the VLM model and processor.

    Automatically selects the correct model class based on the model_id:
      - "qwen2.5-vl" -> Qwen2_5_VLForConditionalGeneration
      - "qwen2-vl"   -> Qwen2VLForConditionalGeneration

    Returns (model, processor, device_str).
    """
    device = device or detect_device()
    load_path = checkpoint_path or model_id

    if dtype is None:
        if device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

    print(f"Loading model from: {load_path}")
    print(f"  Device: {device}")
    print(f"  Dtype:  {dtype}")

    # Select model class based on model ID string
    model_id_lower = model_id.lower()
    if "qwen2.5-vl" in model_id_lower:
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        model_cls = Qwen2VLForConditionalGeneration

    model = model_cls.from_pretrained(
        load_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="flash_attention_2" if device == "cuda" else "eager",
    )

    if device != "cuda":
        model = model.to(device)

    model.eval()

    # Load processor — always from the base model_id (not checkpoint)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.image_processor.max_pixels = max_pixels
    processor.image_processor.min_pixels = min_pixels

    return model, processor, device


# ── Architecture Summary ─────────────────────────────────────────────────────

def print_model_summary(model) -> dict:
    """Print model architecture summary and return stats."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb = param_bytes / (1024 ** 2)

    print(f"\n{'='*60}")
    print("MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"  Architecture:     {model.__class__.__name__}")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Frozen params:    {frozen_params:,}")
    print(f"  Parameter memory: {param_mb:.1f} MB")
    print(f"  Dtype:            {next(model.parameters()).dtype}")

    module_counts: dict[str, int] = {}
    for _name, module in model.named_modules():
        cls_name = module.__class__.__name__
        module_counts[cls_name] = module_counts.get(cls_name, 0) + 1

    print(f"\n  Top module types:")
    for cls_name, count in sorted(module_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {cls_name:30s} x {count}")
    print(f"{'='*60}\n")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_memory_mb": param_mb,
    }


# ── Sanity Checks ────────────────────────────────────────────────────────────

def create_dummy_image(size: tuple[int, int] = (224, 224)) -> Image.Image:
    """Create a simple dummy image for testing."""
    return Image.new("RGB", size, color=(128, 128, 128))


def check_forward_pass(model, processor, device: str, method: Method = Method.OURS) -> bool:
    """Run a forward pass with dummy inputs and verify no errors."""
    print(f"Running forward pass check (method={method.value})...")

    sys_prompt, _ = get_prompts(method)
    dummy_image = create_dummy_image()
    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What do you see?\noptions: (A) normal (B) abnormal"},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[dummy_image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    print(f"  Forward pass OK. Logits shape: {logits.shape}, Vocab size: {logits.shape[-1]}")
    return True


def check_generation(model, processor, device: str, method: Method = Method.OURS) -> str:
    """Generate a short completion and return the text."""
    print(f"Running generation check (method={method.value})...")

    sys_prompt, _ = get_prompts(method)
    dummy_image = create_dummy_image()

    if method == Method.BASELINE:
        user_text = (
            "What type of medical image is this?\n"
            "options: (A) X-ray (B) MRI (C) CT scan (D) Ultrasound\n\n"
            "Respond with <think>...</think><answer>LETTER</answer>"
        )
    else:
        user_text = (
            "What type of medical image is this?\n"
            "options: (A) X-ray (B) MRI (C) CT scan (D) Ultrasound\n\n"
            "Respond with <think><modality>...</modality>"
            "<concepts>[...]</concepts>reasoning</think>"
            "<answer>LETTER</answer>"
        )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[dummy_image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=EVAL_CFG.max_new_tokens,
            temperature=EVAL_CFG.temperature, top_p=EVAL_CFG.top_p, do_sample=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_text = processor.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)[0]
    print(f"  Generation OK. Length: {len(generated_text)} chars")
    print(f"  Output preview:\n    {generated_text[:300]}...")
    return generated_text


def check_tokenizer(processor) -> None:
    """Verify the tokenizer handles our special tags correctly."""
    print("Checking tokenizer for R1 tags...")

    test_strings = [
        "<think>", "</think>", "<answer>", "</answer>",
        "<modality>", "</modality>", "<concepts>", "</concepts>",
    ]
    for s in test_strings:
        tokens = processor.tokenizer.encode(s, add_special_tokens=False)
        decoded = processor.tokenizer.decode(tokens)
        match = "OK" if s in decoded else "MISMATCH"
        print(f"  {s:20s} -> {len(tokens)} tokens -> '{decoded}' [{match}]")


def check_image_processing(processor) -> None:
    """Verify image processing works with different sizes and modes."""
    print("Checking image processing...")

    test_cases = [
        ("RGB 224x224", Image.new("RGB", (224, 224), "red")),
        ("RGBA 512x512", Image.new("RGBA", (512, 512), "blue")),
        ("L (grayscale) 256x256", Image.new("L", (256, 256), 128)),
        ("RGB 1024x1024", Image.new("RGB", (1024, 1024), "green")),
    ]
    for name, img in test_cases:
        try:
            if img.mode != "RGB":
                img = img.convert("RGB")
            dummy_text = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "test"}]}],
                tokenize=False, add_generation_prompt=True,
            )
            result = processor(text=[dummy_text], images=[img], return_tensors="pt")
            pv = result.get("pixel_values", result.get("pixel_values_videos"))
            if pv is not None:
                print(f"  {name:30s} -> pixel_values shape: {pv.shape}")
            else:
                print(f"  {name:30s} -> processed (no pixel_values key)")
        except Exception as e:
            print(f"  {name:30s} -> ERROR: {e}")


# ── Full Check Pipeline ─────────────────────────────────────────────────────

def run_all_checks(
    model_id: str = MODEL_ID,
    checkpoint_path: str | None = None,
    device: str | None = None,
    method: Method = Method.OURS,
) -> dict:
    """Run all sanity checks and return results."""
    results = {"model_id": model_id, "method": method.value, "checks": {}}

    model, processor, device = load_model_and_processor(
        model_id=model_id, checkpoint_path=checkpoint_path, device=device,
    )
    stats = print_model_summary(model)
    results["model_stats"] = stats

    checks = [
        ("tokenizer", lambda: check_tokenizer(processor)),
        ("image_processing", lambda: check_image_processing(processor)),
        ("forward_pass", lambda: check_forward_pass(model, processor, device, method)),
        ("generation", lambda: check_generation(model, processor, device, method)),
    ]

    for name, check_fn in checks:
        try:
            check_fn()
            results["checks"][name] = "PASSED"
            print(f"  [{name}] PASSED\n")
        except Exception as e:
            results["checks"][name] = f"FAILED: {e}"
            print(f"  [{name}] FAILED: {e}\n")

    passed = sum(1 for v in results["checks"].values() if v == "PASSED")
    total = len(results["checks"])
    print(f"\n{'='*40}")
    print(f"CHECK SUMMARY: {passed}/{total} passed")
    print(f"{'='*40}")
    for name, status in results["checks"].items():
        icon = "PASS" if status == "PASSED" else "FAIL"
        print(f"  [{icon}] {name}: {status}")
    print()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Load and check MedVLM-R1 model")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--method", type=str, default="ours", choices=["baseline", "ours"])
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    method = Method(args.method)

    if args.summary_only:
        model, processor, device = load_model_and_processor(
            model_id=args.model_id, checkpoint_path=args.checkpoint, device=args.device,
        )
        print_model_summary(model)
    else:
        run_all_checks(
            model_id=args.model_id, checkpoint_path=args.checkpoint,
            device=args.device, method=method,
        )


if __name__ == "__main__":
    main()
