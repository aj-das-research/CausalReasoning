#!/usr/bin/env python3
"""
investigation_h1_h2_h3.py - Investigation script for paper:
"Concept-Based Reasoning in Medical VLMs"

Covers three hypotheses:
  H1: Free-text reasoning exhibits Visual Reasoning Collapse (unfaithfulness)
      - Perturbation test: mask/blur image, check if reasoning changes
      - Metrics: reasoning change rate, answer flip rate, concept change ratio
      Ref: Turpin et al., NeurIPS 2023 (CoT unfaithfulness)

  H2: Concept bottleneck reasoning is more faithful/robust than free-text CoT
      - Compare free-text vs concept-guided prompting on same model
      - Metrics: accuracy, concept F1/precision/recall, hallucination rate
      Ref: Koh et al., ICML 2020 (CBM); Oikarinen et al., ICLR 2024

  H3: Concept-aware rewards improve GRPO (reward decomposition analysis)
      - Multi-generation rollouts, compute per-component reward variance
      - Detect vanishing advantages, compare MO-GRPO vs answer-only
      Ref: MO-GRPO (arXiv:2509.22047), VL-Rethinker (arXiv:2504.08837),
           Lightman et al., ICLR 2024 (PRM)

Models tested:
  - Qwen2-VL-2B-Instruct      (base, no medical fine-tuning)
  - Qwen2.5-VL-3B-Instruct    (our base model)
  - MedVLM-R1                  (GRPO-trained, free-text <think>)
  - Lingshu-7B                 (medical MLLM with CoT)
  - Hulu-Med-7B / 4B           (medical MLLM)

Usage:
  # Run all hypotheses with default model
  python investigation_h1_h2_h3.py \\
    --dataset_path abhijitdas/medvlm-r1-dataset \\
    --models qwen25vl \\
    --hypotheses h1,h2,h3 --max_samples 100

  # Compare multiple models
  python investigation_h1_h2_h3.py \\
    --dataset_path ./test_data.json \\
    --models medvlm_r1,qwen25vl \\
    --hypotheses h1,h2 --max_samples 50

  # Custom model paths
  python investigation_h1_h2_h3.py \\
    --dataset_path abhijitdas/medvlm-r1-dataset \\
    --models qwen25vl,medvlm_r1 \\
    --model_paths Qwen/Qwen2.5-VL-3B-Instruct,JZPeterPan/MedVLM-R1 \\
    --hypotheses h1,h2,h3 --max_samples 200

  # Use a fine-tuned checkpoint
  python investigation_h1_h2_h3.py \\
    --dataset_path abhijitdas/medvlm-r1-dataset \\
    --models ours_ckpt \\
    --model_paths ./outputs/checkpoints/best \\
    --hypotheses h1,h2,h3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image, ImageDraw, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Imports from our codebase ────────────────────────────────────────────────

from config import (
    CONCEPT_MAP,
    MODALITIES,
    MODALITY_TAG_MAP,
    MODEL_ID,
    Method,
    get_prompts,
)
from data_prep_and_viewer import parse_solution


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATHS: Dict[str, str] = {
    "qwen2vl":    "Qwen/Qwen2-VL-2B-Instruct",
    "qwen25vl":   "Qwen/Qwen2.5-VL-3B-Instruct",
    "medvlm_r1":  "JZPeterPan/MedVLM-R1",
    "lingshu":    "lingshu-medical-mllm/Lingshu-7B",
    "hulumed":    "ZJU-AI4H/Hulu-Med-7B",
    "hulumed_4b": "ZJU-AI4H/Hulu-Med-4B",
}


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str) -> logging.Logger:
    """Create a logger with file + console handlers."""
    logger = logging.getLogger("investigation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(os.path.join(output_dir, "investigation.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ── Text Extraction Helpers ──────────────────────────────────────────────────

def extract_answer_letter(text: str) -> str:
    """Extract the answer letter from model output or solution string."""
    # Check <answer> tag first
    m = re.search(r"<answer>\s*\(?([A-J])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: look for (A), (B), etc.
    m = re.search(r"\(([A-J])\)", text)
    if m:
        return m.group(1).upper()
    # Last resort: standalone letter at end
    m = re.search(r"\b([A-J])\s*$", text.strip())
    if m:
        return m.group(1).upper()
    return ""


def extract_think_content(text: str) -> str:
    """Extract the reasoning text inside <think>...</think> tags."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def extract_concepts_from_text(
    text: str, valid_concepts: List[str],
) -> List[str]:
    """Find which valid concepts are mentioned in a text (case-insensitive)."""
    text_lower = text.lower()
    found = []
    for c in valid_concepts:
        if c.lower() in text_lower:
            found.append(c)
    return found


# ── Metric Functions ─────────────────────────────────────────────────────────

def compute_concept_f1(
    predicted: List[str], ground_truth: List[str],
) -> Dict[str, float]:
    """Compute precision, recall, F1 for concept sets (case-insensitive)."""
    pred_set = set(c.lower() for c in predicted)
    gt_set = set(c.lower() for c in ground_truth)

    if not pred_set and not gt_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set or not gt_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_hallucination_rate(
    predicted: List[str], ground_truth: List[str],
) -> Dict[str, float]:
    """Fraction of predicted concepts not in ground truth."""
    pred_set = set(c.lower() for c in predicted)
    gt_set = set(c.lower() for c in ground_truth)

    if not pred_set:
        return {"hallucination_rate": 0.0, "hallucinated": 0, "total_predicted": 0}

    hallucinated = len(pred_set - gt_set)
    return {
        "hallucination_rate": hallucinated / len(pred_set),
        "hallucinated": hallucinated,
        "total_predicted": len(pred_set),
    }


def word_level_jaccard_distance(text_a: str, text_b: str) -> float:
    """Jaccard distance between word sets of two texts (0 = identical, 1 = disjoint)."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    union = words_a | words_b
    if not union:
        return 0.0
    intersection = words_a & words_b
    return 1.0 - len(intersection) / len(union)


# ── Image Perturbation ───────────────────────────────────────────────────────

def create_perturbed_image(
    image: "Image.Image",
    perturbation_type: str = "center_mask",
    mask_fraction: float = 0.5,
    blur_radius: float = 20.0,
) -> "Image.Image":
    """Create a perturbed version of an image for faithfulness testing.

    Perturbation types:
      - center_mask:       gray-out central region
      - gaussian_blur:     heavy Gaussian blur over entire image
      - random_patch:      randomly placed gray patches
      - shuffle_quadrants: rearrange the four quadrants
    """
    img = image.copy()
    w, h = img.size

    if perturbation_type == "center_mask":
        draw = ImageDraw.Draw(img)
        cx, cy = w // 2, h // 2
        half_w = int(w * mask_fraction / 2)
        half_h = int(h * mask_fraction / 2)
        draw.rectangle(
            [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
            fill=(128, 128, 128),
        )

    elif perturbation_type == "gaussian_blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    elif perturbation_type == "random_patch":
        draw = ImageDraw.Draw(img)
        rng = np.random.RandomState(42)
        num_patches = max(3, int(w * h / 10000))
        for _ in range(num_patches):
            pw, ph = rng.randint(20, max(21, w // 5)), rng.randint(20, max(21, h // 5))
            px, py = rng.randint(0, max(1, w - pw)), rng.randint(0, max(1, h - ph))
            draw.rectangle([px, py, px + pw, py + ph], fill=(128, 128, 128))

    elif perturbation_type == "shuffle_quadrants":
        arr = np.array(img)
        mid_h, mid_w = h // 2, w // 2
        q = [
            arr[:mid_h, :mid_w],
            arr[:mid_h, mid_w:],
            arr[mid_h:, :mid_w],
            arr[mid_h:, mid_w:],
        ]
        # Rotate quadrants: [Q3, Q1, Q4, Q2]
        new_arr = np.vstack([
            np.hstack([q[2], q[0]]),
            np.hstack([q[3], q[1]]),
        ])
        img = Image.fromarray(new_arr)

    return img


# ── Model Wrapper ────────────────────────────────────────────────────────────

class ModelWrapper:
    """Unified wrapper for loading and generating with different VLM families.

    Supports:
      - Qwen2-VL / Qwen2.5-VL (via our load_model_and_processor)
      - HuluMed family (AutoModelForCausalLM with trust_remote_code)

    The investigation needs to compare multiple model families, so we keep
    a wrapper rather than using load_model_and_processor directly.
    """

    FAMILY_QWEN2VL = "qwen2vl"
    FAMILY_QWEN25VL = "qwen25vl"
    FAMILY_HULUMED = "hulumed"

    def __init__(
        self, model_name: str, model_path: str, device: str = "cuda:0",
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._loaded = False
        self._family = self._detect_family()

    def _detect_family(self) -> str:
        path_lower = self.model_path.lower()
        name_lower = self.model_name.lower()
        combined = path_lower + " " + name_lower

        if "hulu" in combined:
            return self.FAMILY_HULUMED
        if "qwen2.5" in combined or "qwen25" in combined:
            return self.FAMILY_QWEN25VL
        # Default to Qwen2-VL for qwen2, medvlm_r1, lingshu, etc.
        return self.FAMILY_QWEN2VL

    def load(self):
        """Load the model into memory."""
        if self._loaded:
            return

        log = logging.getLogger("investigation")
        log.info("Loading model %s from %s (family=%s)", self.model_name, self.model_path, self._family)

        attn_impl = "flash_attention_2" if "cuda" in self.device else "eager"

        if self._family in (self.FAMILY_QWEN2VL, self.FAMILY_QWEN25VL):
            # Use our codebase's model loader for Qwen models
            from model_loader_and_checker import load_model_and_processor
            self.model, self.processor, self.device = load_model_and_processor(
                model_id=self._base_model_id(),
                checkpoint_path=self.model_path if self._is_local_checkpoint() else None,
                device=self.device.split(":")[0] if ":" in self.device else self.device,
            )

        elif self._family == self.FAMILY_HULUMED:
            from transformers import AutoModelForCausalLM, AutoProcessor
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=self.device if ":" in self.device else "auto",
                attn_implementation=attn_impl,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True,
            )
            self.tokenizer = self.processor.tokenizer
            self.model.eval()

        self._loaded = True
        log.info("Model %s loaded successfully", self.model_name)

    def _base_model_id(self) -> str:
        """Get the base HF model ID for Qwen family loading."""
        if os.path.isdir(self.model_path):
            # Local checkpoint — infer base model from name
            name_lower = self.model_name.lower()
            if "qwen2.5" in name_lower or "qwen25" in name_lower:
                return "Qwen/Qwen2.5-VL-3B-Instruct"
            return "Qwen/Qwen2-VL-2B-Instruct"
        return self.model_path

    def _is_local_checkpoint(self) -> bool:
        return os.path.isdir(self.model_path)

    def generate(
        self,
        image: "Image.Image",
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a text response for an image + prompt."""
        if not self._loaded:
            self.load()
        if self._family in (self.FAMILY_QWEN2VL, self.FAMILY_QWEN25VL):
            return self._generate_qwen(image, prompt, max_new_tokens, temperature)
        elif self._family == self.FAMILY_HULUMED:
            return self._generate_hulumed(image, prompt, max_new_tokens, temperature)
        raise ValueError(f"Unknown family: {self._family}")

    def _generate_qwen(
        self, image: "Image.Image", prompt: str,
        max_new_tokens: int, temperature: float,
    ) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Try qwen_vl_utils for proper image processing
        image_inputs = [image]
        video_inputs = None
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
        except ImportError:
            pass

        inputs = self.processor(
            text=[text_input], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        inputs = inputs.to(
            self.model.device if hasattr(self.model, "device") else self.device,
        )

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_len:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()

    def _generate_hulumed(
        self, image: "Image.Image", prompt: str,
        max_new_tokens: int, temperature: float,
    ) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")

        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        inputs = self.processor(
            conversation=conversation, return_tensors="pt", add_generation_prompt=True,
        )
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens, modals=["image"],
            use_cache=True, pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_len:]
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    def unload(self):
        """Free GPU memory."""
        del self.model, self.processor, self.tokenizer
        self.model = self.processor = self.tokenizer = None
        self._loaded = False
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Prompts ──────────────────────────────────────────────────────────────────
# Free-text prompt for H1 (baseline-style, no concept guidance)
PROMPT_FREETEXT = (
    "{question}\n"
    "First output the thinking process in <think> </think> "
    "and final choice (A, B, C, D ...) in <answer> </answer> tags."
)

# Concept-guided prompt for H2 (ours-style, with concept list)
PROMPT_CONCEPT = (
    "{question}\n\n"
    "Respond in the following structured format:\n"
    "<think>\n"
    "<modality>the imaging modality (e.g. X_RAY, FUNDUS, DERMOSCOPY, PATHOLOGY, SKIN)</modality>\n"
    "<concepts>[list of relevant clinical findings from: {concept_list}]</concepts>\n"
    "</think>\n"
    "<answer>the correct option letter (A/B/C/D)</answer>"
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_valid_concepts(modality_tag: str) -> List[str]:
    """Get valid concept list for a modality tag, using our CONCEPT_MAP."""
    # Try direct MODALITY_TAG_MAP lookup first
    mapped = MODALITY_TAG_MAP.get(modality_tag.upper(), None)
    if mapped and mapped in CONCEPT_MAP:
        return CONCEPT_MAP[mapped]
    # Fuzzy match
    for key, concepts in CONCEPT_MAP.items():
        if key.lower() in modality_tag.lower() or modality_tag.lower() in key.lower():
            return concepts
    return []


def _safe_generate(
    model: ModelWrapper, image: "Image.Image", prompt: str, **kwargs,
) -> Optional[str]:
    """Generate with error handling."""
    try:
        return model.generate(image, prompt, **kwargs)
    except Exception as e:
        logging.getLogger("investigation").warning("Generation failed: %s", e)
        return None


class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


def save_json(data: Any, path: str):
    """Save data to JSON with numpy encoding."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NpEncoder, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# H1: Visual Reasoning Collapse — Faithfulness Test
# Ref: Turpin et al., NeurIPS 2023 (extended to medical VLMs)
# ═══════════════════════════════════════════════════════════════════════════

def run_h1(
    models: Dict[str, ModelWrapper],
    samples: List[Dict],
    output_dir: str,
    perturbation_types: List[str],
    logger: logging.Logger,
) -> Dict:
    """H1: Test whether free-text reasoning is faithful to visual evidence.

    For each sample:
      1. Generate response on original image
      2. Generate response on perturbed image (same prompt)
      3. Measure: reasoning change, answer flip, concept change

    If reasoning does NOT change despite major image perturbation,
    the model exhibits Visual Reasoning Collapse (unfaithful reasoning).
    """
    logger.info("=" * 70)
    logger.info("H1: Visual Reasoning Collapse -- Faithfulness Test")
    logger.info("=" * 70)

    h1_dir = os.path.join(output_dir, "h1_faithfulness")
    os.makedirs(h1_dir, exist_ok=True)
    results: Dict[str, Any] = {"per_model": {}, "detailed": []}

    for mname, mwrap in models.items():
        logger.info("Model: %s", mname)
        mwrap.load()
        model_agg: Dict[str, list] = {pt: [] for pt in perturbation_types}

        for idx, sample in enumerate(samples):
            if idx % 25 == 0:
                logger.info("  sample %d / %d", idx + 1, len(samples))

            image = sample.get("image")
            if image is None:
                continue

            # Parse ground truth using our parse_solution
            parsed = parse_solution(sample.get("solution", ""))
            modality_tag = parsed["modality_tag"]
            gt_concepts = parsed["concepts"]
            gt_letter = parsed["answer_letter"]
            valid_concepts = _get_valid_concepts(modality_tag)

            prompt = PROMPT_FREETEXT.format(question=sample.get("problem", ""))

            # Generate on original image
            orig_resp = _safe_generate(mwrap, image, prompt)
            if orig_resp is None:
                continue

            orig_answer = extract_answer_letter(orig_resp)
            orig_think = extract_think_content(orig_resp)
            orig_concepts = extract_concepts_from_text(orig_think, valid_concepts)

            # Generate on each perturbation
            for pt in perturbation_types:
                pert_img = create_perturbed_image(image, perturbation_type=pt)
                pert_resp = _safe_generate(mwrap, pert_img, prompt)
                if pert_resp is None:
                    continue

                pert_answer = extract_answer_letter(pert_resp)
                pert_think = extract_think_content(pert_resp)
                pert_concepts = extract_concepts_from_text(pert_think, valid_concepts)

                # Concept change ratio (symmetric difference / union)
                orig_set = set(c.lower() for c in orig_concepts)
                pert_set = set(c.lower() for c in pert_concepts)
                union = orig_set | pert_set
                concept_change = (
                    len(orig_set.symmetric_difference(pert_set)) / len(union)
                    if union else 0.0
                )

                # Reasoning change (Jaccard distance > threshold)
                reasoning_dist = word_level_jaccard_distance(orig_think, pert_think)
                reasoning_changed = reasoning_dist > 0.15

                rec = {
                    "sample_id": idx, "model": mname, "perturbation": pt,
                    "modality": modality_tag, "gt_answer": gt_letter,
                    "orig_answer": orig_answer, "pert_answer": pert_answer,
                    "answer_flipped": orig_answer != pert_answer,
                    "reasoning_changed": reasoning_changed,
                    "reasoning_distance": float(reasoning_dist),
                    "concept_change_ratio": float(concept_change),
                    "orig_concepts": orig_concepts, "pert_concepts": pert_concepts,
                }
                model_agg[pt].append(rec)
                results["detailed"].append(rec)

        # Aggregate per perturbation type
        summary = {}
        for pt, recs in model_agg.items():
            if not recs:
                continue
            n = len(recs)
            summary[pt] = {
                "n": n,
                "reasoning_change_rate": sum(r["reasoning_changed"] for r in recs) / n,
                "answer_flip_rate": sum(r["answer_flipped"] for r in recs) / n,
                "mean_concept_change": float(np.mean([r["concept_change_ratio"] for r in recs])),
                "mean_reasoning_distance": float(np.mean([r["reasoning_distance"] for r in recs])),
                "faithfulness_score": sum(r["reasoning_changed"] for r in recs) / n,
            }
        results["per_model"][mname] = summary
        mwrap.unload()

    save_json(results, os.path.join(h1_dir, "h1_results.json"))
    if HAS_MPL:
        _plot_h1(results, h1_dir)
    logger.info("H1 complete. Results saved to %s", h1_dir)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# H2: Concept Bottleneck Analysis — Free-text vs Structured Reasoning
# Ref: Koh et al., ICML 2020 (CBM); Oikarinen et al., ICLR 2024
# ═══════════════════════════════════════════════════════════════════════════

def run_h2(
    models: Dict[str, ModelWrapper],
    samples: List[Dict],
    output_dir: str,
    logger: logging.Logger,
) -> Dict:
    """H2: Compare free-text CoT vs concept-guided reasoning.

    For each sample, run the same model with two prompts:
      1. Free-text CoT (PROMPT_FREETEXT) — unrestricted reasoning
      2. Concept-guided (PROMPT_CONCEPT) — structured with concept list

    Metrics: accuracy, concept F1, precision, recall, hallucination rate.
    Hypothesis: concept-guided should have higher F1 and lower hallucination.
    """
    logger.info("=" * 70)
    logger.info("H2: Concept Bottleneck Analysis -- Faithfulness & Robustness")
    logger.info("=" * 70)

    h2_dir = os.path.join(output_dir, "h2_concept_analysis")
    os.makedirs(h2_dir, exist_ok=True)
    results: Dict[str, Any] = {"per_model": {}, "per_modality": {}, "detailed": []}

    for mname, mwrap in models.items():
        logger.info("Model: %s", mname)
        mwrap.load()
        records = []
        modality_buckets: Dict[str, list] = defaultdict(list)

        for idx, sample in enumerate(samples):
            if idx % 25 == 0:
                logger.info("  sample %d / %d", idx + 1, len(samples))

            image = sample.get("image")
            if image is None:
                continue

            parsed = parse_solution(sample.get("solution", ""))
            modality_tag = parsed["modality_tag"]
            gt_concepts = parsed["concepts"]
            gt_letter = parsed["answer_letter"]
            valid_concepts = _get_valid_concepts(modality_tag)
            concept_list_str = ", ".join(valid_concepts[:15])

            # 1) Free-text CoT
            ft_resp = _safe_generate(
                mwrap, image, PROMPT_FREETEXT.format(question=sample["problem"]),
            )
            if ft_resp is None:
                continue
            ft_answer = extract_answer_letter(ft_resp)
            ft_think = extract_think_content(ft_resp)
            ft_concepts = extract_concepts_from_text(ft_think, valid_concepts)

            # 2) Concept-guided
            cg_resp = _safe_generate(
                mwrap, image,
                PROMPT_CONCEPT.format(
                    question=sample["problem"], concept_list=concept_list_str,
                ),
            )
            if cg_resp is None:
                continue
            cg_answer = extract_answer_letter(cg_resp)
            cg_think = extract_think_content(cg_resp)
            cg_concepts = extract_concepts_from_text(cg_think, valid_concepts)

            rec = {
                "sample_id": idx, "model": mname, "modality": modality_tag,
                "gt_answer": gt_letter, "gt_concepts": gt_concepts,
                "ft_answer": ft_answer, "ft_correct": ft_answer == gt_letter,
                "ft_concepts": ft_concepts,
                "ft_f1": compute_concept_f1(ft_concepts, gt_concepts),
                "ft_halluc": compute_hallucination_rate(ft_concepts, gt_concepts),
                "cg_answer": cg_answer, "cg_correct": cg_answer == gt_letter,
                "cg_concepts": cg_concepts,
                "cg_f1": compute_concept_f1(cg_concepts, gt_concepts),
                "cg_halluc": compute_hallucination_rate(cg_concepts, gt_concepts),
            }
            records.append(rec)
            modality_buckets[modality_tag].append(rec)

        if records:
            results["per_model"][mname] = {
                "n": len(records),
                "freetext": _agg_h2(records, "ft"),
                "concept_guided": _agg_h2(records, "cg"),
            }
            for mod, recs in modality_buckets.items():
                results["per_modality"][f"{mname}/{mod}"] = {
                    "n": len(recs),
                    "freetext": _agg_h2(recs, "ft"),
                    "concept_guided": _agg_h2(recs, "cg"),
                }
        results["detailed"].extend(records)
        mwrap.unload()

    save_json(results, os.path.join(h2_dir, "h2_results.json"))
    if HAS_MPL:
        _plot_h2(results, h2_dir)
    logger.info("H2 complete. Results saved to %s", h2_dir)
    return results


def _agg_h2(records: List[Dict], prefix: str) -> Dict[str, float]:
    """Aggregate H2 metrics for a set of records."""
    n = len(records)
    if n == 0:
        return {}
    acc = sum(r[f"{prefix}_correct"] for r in records) / n
    f1s = [r[f"{prefix}_f1"]["f1"] for r in records]
    precs = [r[f"{prefix}_f1"]["precision"] for r in records]
    recs_list = [r[f"{prefix}_f1"]["recall"] for r in records]
    halluc = [r[f"{prefix}_halluc"]["hallucination_rate"] for r in records]
    return {
        "accuracy": float(acc),
        "concept_f1": float(np.mean(f1s)),
        "concept_precision": float(np.mean(precs)),
        "concept_recall": float(np.mean(recs_list)),
        "hallucination_rate": float(np.mean(halluc)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# H3: Reward Decomposition & Vanishing Advantages
# Ref: MO-GRPO (arXiv:2509.22047), VL-Rethinker (arXiv:2504.08837),
#      Lightman et al., ICLR 2024 (Process Reward Models)
# ═══════════════════════════════════════════════════════════════════════════

def run_h3(
    models: Dict[str, ModelWrapper],
    samples: List[Dict],
    output_dir: str,
    num_generations: int,
    logger: logging.Logger,
) -> Dict:
    """H3: Analyze reward decomposition and vanishing advantages.

    For each sample, generate G responses (GRPO group) and compute:
      - r_answer:   binary answer correctness
      - r_format:   binary format compliance
      - r_concept:  continuous concept F1 (process reward)
      - r_modality: binary modality correctness

    Then measure:
      - Per-component reward variance (gradient signal strength)
      - Vanishing advantage rate (all-same-reward groups)
      - MO-GRPO vs single-reward advantage distribution

    Hypothesis: concept-aware multi-objective rewards reduce vanishing
    advantages, providing richer gradient signal for GRPO training.
    """
    logger.info("=" * 70)
    logger.info("H3: Reward Decomposition & Vanishing Advantages Analysis")
    logger.info("=" * 70)

    h3_dir = os.path.join(output_dir, "h3_reward_analysis")
    os.makedirs(h3_dir, exist_ok=True)

    # Reward weights matching our config.py TrainConfig defaults
    ALPHA, BETA, GAMMA, DELTA = 0.4, 0.2, 0.3, 0.1
    results: Dict[str, Any] = {"per_model": {}, "distributions": {}, "detailed": []}

    for mname, mwrap in models.items():
        logger.info("Model: %s", mname)
        mwrap.load()

        group_stds_ans: List[float] = []
        group_stds_concept: List[float] = []
        group_stds_full: List[float] = []
        vanish_ans = 0
        vanish_full = 0
        total_groups = 0
        per_sample: List[Dict] = []

        for idx, sample in enumerate(samples):
            if idx % 25 == 0:
                logger.info("  sample %d / %d", idx + 1, len(samples))

            image = sample.get("image")
            if image is None:
                continue

            parsed = parse_solution(sample.get("solution", ""))
            modality_tag = parsed["modality_tag"]
            gt_concepts = parsed["concepts"]
            gt_letter = parsed["answer_letter"]
            valid_concepts = _get_valid_concepts(modality_tag)
            concept_str = ", ".join(valid_concepts[:15])

            prompt = PROMPT_CONCEPT.format(
                question=sample["problem"], concept_list=concept_str,
            )

            # Generate G rollouts (GRPO group)
            gen_rewards = []
            for g in range(num_generations):
                temp = 0.0 if g == 0 else 0.7
                resp = _safe_generate(
                    mwrap, image, prompt, max_new_tokens=512, temperature=temp,
                )
                if resp is None:
                    continue

                pred_letter = extract_answer_letter(resp)
                pred_concepts = extract_concepts_from_text(resp, valid_concepts)

                # Reward components
                r_answer = 1.0 if pred_letter == gt_letter else 0.0

                has_think = bool(re.search(r"<think>.*?</think>", resp, re.DOTALL))
                has_ans_tag = bool(re.search(r"<answer>.*?</answer>", resp, re.DOTALL))
                r_format = 1.0 if (has_think and has_ans_tag) else 0.0

                # Concept F1 as process reward (continuous 0-1)
                r_concept = compute_concept_f1(pred_concepts, gt_concepts)["f1"]

                # Modality correctness
                mod_match = re.search(
                    r"<modality>\s*(.*?)\s*</modality>", resp, re.DOTALL | re.IGNORECASE,
                )
                pred_mod = mod_match.group(1).strip().upper() if mod_match else ""
                r_modality = 1.0 if pred_mod == modality_tag else 0.0

                gen_rewards.append({
                    "r_answer": r_answer, "r_format": r_format,
                    "r_concept": r_concept, "r_modality": r_modality,
                })

            if len(gen_rewards) < 2:
                continue

            total_groups += 1

            arr_ans = np.array([r["r_answer"] for r in gen_rewards])
            arr_con = np.array([r["r_concept"] for r in gen_rewards])
            arr_full = np.array([
                ALPHA * r["r_answer"] + BETA * r["r_format"]
                + GAMMA * r["r_concept"] + DELTA * r["r_modality"]
                for r in gen_rewards
            ])

            std_ans = float(np.std(arr_ans))
            std_con = float(np.std(arr_con))
            std_full = float(np.std(arr_full))

            # MO-GRPO normalization (arXiv:2509.22047):
            # Per-objective z-normalization before combining
            mo_advantages = np.zeros(len(gen_rewards))
            for key in ("r_answer", "r_format", "r_concept", "r_modality"):
                vals = np.array([r[key] for r in gen_rewards])
                s = np.std(vals)
                if s > 1e-8:
                    mo_advantages += (vals - np.mean(vals)) / s
            std_mo = float(np.std(mo_advantages))

            # Vanishing advantages detection (VL-Rethinker, arXiv:2504.08837)
            if std_ans < 1e-8:
                vanish_ans += 1
            if std_full < 1e-8:
                vanish_full += 1

            group_stds_ans.append(std_ans)
            group_stds_concept.append(std_con)
            group_stds_full.append(std_full)

            per_sample.append({
                "sample_id": idx, "model": mname, "modality": modality_tag,
                "gen_rewards": [{k: float(v) for k, v in r.items()} for r in gen_rewards],
                "std_answer": std_ans, "std_concept": std_con,
                "std_full": std_full, "std_mo_grpo": std_mo,
                "vanish_answer": std_ans < 1e-8, "vanish_full": std_full < 1e-8,
            })

        if total_groups > 0:
            results["per_model"][mname] = {
                "total_groups": total_groups,
                "vanishing_rate_answer_only": vanish_ans / total_groups,
                "vanishing_rate_full_reward": vanish_full / total_groups,
                "mean_std_answer": float(np.mean(group_stds_ans)),
                "mean_std_concept": float(np.mean(group_stds_concept)),
                "mean_std_full": float(np.mean(group_stds_full)),
            }
            results["distributions"][mname] = {
                "answer_stds": group_stds_ans,
                "concept_stds": group_stds_concept,
                "full_stds": group_stds_full,
            }
        results["detailed"].extend(per_sample)
        mwrap.unload()

    save_json(results, os.path.join(h3_dir, "h3_results.json"))
    if HAS_MPL:
        _plot_h3(results, h3_dir)
    logger.info("H3 complete. Results saved to %s", h3_dir)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

PLT_PARAMS = {
    "figure.dpi": 200, "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "figure.titlesize": 13,
}
C_RED = "#D55E00"
C_BLUE = "#0072B2"
C_GREEN = "#009E73"
C_ORANGE = "#E69F00"
C_PURPLE = "#CC79A7"


def _plot_h1(results: Dict, save_dir: str):
    """Plot H1 faithfulness results: reasoning change, answer flip, concept change."""
    plt.rcParams.update(PLT_PARAMS)
    models = list(results["per_model"].keys())
    if not models:
        return

    pt_types = sorted(
        set(pt for m in models for pt in results["per_model"][m].keys()),
    )
    if not pt_types:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        "H1: Visual Reasoning Collapse -- Faithfulness Under Perturbation\n"
        "(Ref: Turpin et al., NeurIPS 2023, extended to medical VLMs)",
        fontweight="bold",
    )

    metrics = [
        ("reasoning_change_rate", "Reasoning Change Rate", "Higher = more faithful"),
        ("answer_flip_rate", "Answer Flip Rate", "Higher = less robust"),
        ("mean_concept_change", "Concept Change Ratio", "How much concepts change"),
    ]
    colors = [C_RED, C_BLUE, C_GREEN, C_ORANGE, C_PURPLE]

    for ax, (key, ylabel, subtitle) in zip(axes, metrics):
        x = np.arange(len(pt_types))
        w = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            vals = [results["per_model"][m].get(pt, {}).get(key, 0) for pt in pt_types]
            ax.bar(x + i * w, vals, w, label=m, color=colors[i % len(colors)], alpha=0.85)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.set_xticks(x + w * (len(models) - 1) / 2)
        ax.set_xticklabels([pt.replace("_", "\n") for pt in pt_types], fontsize=8)
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(save_dir, f"h1_faithfulness.{ext}"), bbox_inches="tight")
    plt.close(fig)


def _plot_h2(results: Dict, save_dir: str):
    """Plot H2 concept analysis: free-text vs concept-guided comparison."""
    plt.rcParams.update(PLT_PARAMS)
    models = list(results["per_model"].keys())
    if not models:
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        "H2: Concept Bottleneck Analysis -- Free-Text CoT vs Structured Reasoning\n"
        "(Ref: Koh et al., ICML 2020; Oikarinen et al., ICLR 2024)",
        fontweight="bold",
    )

    metrics = [
        ("accuracy", "Answer Accuracy"),
        ("concept_f1", "Concept F1"),
        ("concept_precision", "Concept Precision"),
        ("hallucination_rate", "Hallucination Rate"),
    ]

    for ax, (key, title) in zip(axes, metrics):
        ft_vals, cg_vals = [], []
        for m in models:
            d = results["per_model"].get(m, {})
            ft_vals.append(d.get("freetext", {}).get(key, 0))
            cg_vals.append(d.get("concept_guided", {}).get(key, 0))

        x = np.arange(len(models))
        w = 0.35
        ax.bar(x - w / 2, ft_vals, w, label="Free-text CoT", color=C_RED, alpha=0.85)
        ax.bar(x + w / 2, cg_vals, w, label="Concept-guided", color=C_GREEN, alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(save_dir, f"h2_concept_analysis.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # Per-modality breakdown
    if results["per_modality"]:
        _plot_h2_modality(results, save_dir)


def _plot_h2_modality(results: Dict, save_dir: str):
    """Plot H2 per-modality concept F1 breakdown."""
    plt.rcParams.update(PLT_PARAMS)
    items = sorted(results["per_modality"].items())
    if not items:
        return

    labels = [k for k, _ in items]
    ft_f1 = [v.get("freetext", {}).get("concept_f1", 0) for _, v in items]
    cg_f1 = [v.get("concept_guided", {}).get("concept_f1", 0) for _, v in items]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.4)))
    y = np.arange(len(labels))
    h = 0.35
    ax.barh(y - h / 2, ft_f1, h, label="Free-text CoT", color=C_RED, alpha=0.85)
    ax.barh(y + h / 2, cg_f1, h, label="Concept-guided", color=C_GREEN, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Concept F1")
    ax.set_title("H2: Concept F1 by Model / Modality")
    ax.legend()
    ax.set_xlim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(save_dir, f"h2_modality_breakdown.{ext}"), bbox_inches="tight")
    plt.close(fig)


def _plot_h3(results: Dict, save_dir: str):
    """Plot H3 reward decomposition: vanishing rate, signal strength, distribution."""
    plt.rcParams.update(PLT_PARAMS)
    models = list(results["per_model"].keys())
    if not models:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        "H3: Reward Decomposition & Vanishing Advantages\n"
        "(Ref: MO-GRPO arXiv:2509.22047; VL-Rethinker arXiv:2504.08837; "
        "Lightman et al. ICLR 2024 PRM)",
        fontweight="bold",
    )

    # Panel 1: Vanishing advantages rate
    ax = axes[0]
    x = np.arange(len(models))
    w = 0.35
    v_ans = [results["per_model"][m].get("vanishing_rate_answer_only", 0) for m in models]
    v_full = [results["per_model"][m].get("vanishing_rate_full_reward", 0) for m in models]
    ax.bar(x - w / 2, v_ans, w, label="Answer-only (MedVLM-R1)", color=C_RED, alpha=0.85)
    ax.bar(x + w / 2, v_full, w, label="Concept-aware (Ours)", color=C_GREEN, alpha=0.85)
    ax.set_ylabel("Vanishing Rate")
    ax.set_title("Vanishing Advantages Rate\n(lower = better for GRPO)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: Mean reward std (gradient signal strength)
    ax = axes[1]
    s_ans = [results["per_model"][m].get("mean_std_answer", 0) for m in models]
    s_con = [results["per_model"][m].get("mean_std_concept", 0) for m in models]
    s_full = [results["per_model"][m].get("mean_std_full", 0) for m in models]
    w3 = 0.25
    ax.bar(x - w3, s_ans, w3, label="R_answer", color=C_RED, alpha=0.85)
    ax.bar(x, s_con, w3, label="R_concept", color=C_BLUE, alpha=0.85)
    ax.bar(x + w3, s_full, w3, label="R_full (MO-GRPO)", color=C_GREEN, alpha=0.85)
    ax.set_ylabel("Mean Reward Std per Group")
    ax.set_title("Gradient Signal Strength\n(higher = more informative)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 3: Reward std distribution histogram
    ax = axes[2]
    m0 = models[0]
    dist = results.get("distributions", {}).get(m0, {})
    if dist:
        bins = np.linspace(0, 0.55, 25)
        if dist.get("answer_stds"):
            ax.hist(
                dist["answer_stds"], bins=bins, alpha=0.6,
                label="Answer-only", color=C_RED, density=True,
            )
        if dist.get("full_stds"):
            ax.hist(
                dist["full_stds"], bins=bins, alpha=0.6,
                label="Full reward", color=C_GREEN, density=True,
            )
        ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Reward Std per Group")
        ax.set_ylabel("Density")
        ax.set_title(f"Reward Std Distribution ({m0})\n(spike at 0 = vanishing advantages)")
        ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(save_dir, f"h3_reward_analysis.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Summary Report
# ═══════════════════════════════════════════════════════════════════════════

def write_report(
    h1: Optional[Dict], h2: Optional[Dict], h3: Optional[Dict],
    output_dir: str, logger: logging.Logger,
):
    """Write a consolidated markdown report of all investigation results."""
    path = os.path.join(output_dir, "INVESTIGATION_REPORT.md")
    L = []
    L.append("# Investigation Report: Concept-Based Reasoning in Medical VLMs\n")
    L.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    L.append("---\n")

    if h1 and h1.get("per_model"):
        L.append("## H1: Visual Reasoning Collapse\n")
        L.append("**Hypothesis**: Free-text CoT is decoupled from visual evidence.\n")
        L.append("| Model | Perturbation | Reasoning Change | Answer Flip | Concept Change | Faithfulness |")
        L.append("|---|---|---|---|---|---|")
        for m, data in h1["per_model"].items():
            for pt, v in data.items():
                L.append(
                    f"| {m} | {pt} | {v.get('reasoning_change_rate',0):.3f} "
                    f"| {v.get('answer_flip_rate',0):.3f} "
                    f"| {v.get('mean_concept_change',0):.3f} "
                    f"| {v.get('faithfulness_score',0):.3f} |"
                )
        L.append("")

    if h2 and h2.get("per_model"):
        L.append("## H2: Concept Bottleneck Analysis\n")
        L.append("**Hypothesis**: Structured concept reasoning is more faithful.\n")
        L.append("| Model | Prompt | Acc | F1 | Precision | Recall | Halluc |")
        L.append("|---|---|---|---|---|---|---|")
        for m, data in h2["per_model"].items():
            for ptype in ("freetext", "concept_guided"):
                d = data.get(ptype, {})
                L.append(
                    f"| {m} | {ptype} | {d.get('accuracy',0):.3f} "
                    f"| {d.get('concept_f1',0):.3f} | {d.get('concept_precision',0):.3f} "
                    f"| {d.get('concept_recall',0):.3f} | {d.get('hallucination_rate',0):.3f} |"
                )
        L.append("")

    if h3 and h3.get("per_model"):
        L.append("## H3: Reward Decomposition\n")
        L.append("**Hypothesis**: Concept-aware rewards reduce vanishing advantages.\n")
        L.append("| Model | Vanish (Ans) | Vanish (Full) | Std (Ans) | Std (Full) |")
        L.append("|---|---|---|---|---|")
        for m, d in h3["per_model"].items():
            L.append(
                f"| {m} | {d.get('vanishing_rate_answer_only',0):.3f} "
                f"| {d.get('vanishing_rate_full_reward',0):.3f} "
                f"| {d.get('mean_std_answer',0):.4f} "
                f"| {d.get('mean_std_full',0):.4f} |"
            )
        L.append("")

    L.append("---\n")
    L.append("## References\n")
    refs = [
        "Pan et al., MedVLM-R1, MICCAI 2025, arXiv:2502.19634",
        "Turpin et al., NeurIPS 2023 (CoT Unfaithfulness)",
        "Koh et al., ICML 2020 (Concept Bottleneck Models)",
        "Oikarinen et al., ICLR 2024 (Label-free CBM)",
        "Lightman et al., ICLR 2024 (Process Reward Models)",
        "Ichihara et al., MO-GRPO, arXiv:2509.22047",
        "VL-Rethinker, arXiv:2504.08837 (Vanishing Advantages)",
    ]
    for r in refs:
        L.append(f"- {r}")

    with open(path, "w") as f:
        f.write("\n".join(L))
    logger.info("Report written to %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_samples(
    dataset_path: str, max_samples: int, logger: logging.Logger,
) -> List[Dict]:
    """Load dataset samples from HuggingFace or local JSON/JSONL.

    Expected fields per sample: image (PIL or path), problem (str), solution (str).
    """
    samples: List[Dict] = []

    if os.path.isfile(dataset_path):
        logger.info("Loading local file: %s", dataset_path)
        if dataset_path.endswith(".jsonl"):
            with open(dataset_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    samples.append(json.loads(line))
        else:
            with open(dataset_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                samples = data[:max_samples]
            elif isinstance(data, dict) and "data" in data:
                samples = data["data"][:max_samples]

        # Resolve image paths to PIL Images
        base_dir = os.path.dirname(dataset_path)
        for s in samples:
            if "image" in s and isinstance(s["image"], str):
                img_path = s["image"]
                if not os.path.isabs(img_path):
                    img_path = os.path.join(base_dir, img_path)
                if os.path.isfile(img_path):
                    s["image"] = Image.open(img_path).convert("RGB")
                else:
                    s["image"] = None

    else:
        # Assume HuggingFace dataset identifier
        logger.info("Loading HuggingFace dataset: %s", dataset_path)
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_path)
            split_name = "test" if "test" in ds else list(ds.keys())[0]
            for i, item in enumerate(ds[split_name]):
                if i >= max_samples:
                    break
                samples.append(dict(item))
        except Exception as e:
            logger.error("Failed to load HF dataset: %s", e)

    logger.info("Loaded %d samples", len(samples))
    return samples


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Investigation: Concept-Based Reasoning in Medical VLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python investigation_h1_h2_h3.py \\
    --dataset_path ./test_data.json \\
    --models medvlm_r1,qwen25vl \\
    --hypotheses h1,h2 --max_samples 50

  python investigation_h1_h2_h3.py \\
    --dataset_path abhijitdas/medvlm-r1-dataset \\
    --models medvlm_r1,qwen25vl,lingshu \\
    --model_paths JZPeterPan/MedVLM-R1,Qwen/Qwen2.5-VL-3B-Instruct,lingshu-medical-mllm/Lingshu-7B \\
    --hypotheses h1,h2,h3 --max_samples 200 --num_generations 6
        """,
    )
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="HuggingFace dataset name or path to local JSON/JSONL")
    parser.add_argument("--output_dir", type=str, default="./investigation_outputs",
                        help="Directory for all output files")
    parser.add_argument("--models", type=str, default="qwen25vl",
                        help="Comma-separated model keys: qwen2vl,qwen25vl,medvlm_r1,lingshu,hulumed,hulumed_4b")
    parser.add_argument("--model_paths", type=str, default=None,
                        help="Comma-separated HF IDs or local paths (same order as --models)")
    parser.add_argument("--hypotheses", type=str, default="h1,h2,h3",
                        help="Which hypotheses to run: h1,h2,h3")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Max samples to evaluate")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="GRPO group size for H3")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for inference")
    parser.add_argument("--perturbations", type=str,
                        default="center_mask,gaussian_blur,shuffle_quadrants",
                        help="Comma-separated perturbation types for H1")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    logger.info("=" * 70)
    logger.info("Investigation: Concept-Based Reasoning in Medical VLMs")
    logger.info("=" * 70)
    logger.info("Config: %s", vars(args))

    # Resolve model names to paths
    model_names = [m.strip() for m in args.models.split(",")]
    if args.model_paths:
        model_paths = [p.strip() for p in args.model_paths.split(",")]
    else:
        model_paths = [DEFAULT_MODEL_PATHS.get(m, m) for m in model_names]

    if len(model_names) != len(model_paths):
        logger.error("Mismatch: %d names vs %d paths", len(model_names), len(model_paths))
        sys.exit(1)

    models = {}
    for name, path in zip(model_names, model_paths):
        models[name] = ModelWrapper(name, path, device=args.device)
        logger.info("Registered: %s -> %s", name, path)

    # Load data
    samples = load_samples(args.dataset_path, args.max_samples * 2, logger)
    if not samples:
        logger.error("No samples loaded. Check --dataset_path.")
        sys.exit(1)

    eval_samples = samples[:args.max_samples]
    hyps = [h.strip() for h in args.hypotheses.split(",")]
    pts = [p.strip() for p in args.perturbations.split(",")]

    h1_res = h2_res = h3_res = None

    try:
        if "h1" in hyps:
            h1_res = run_h1(models, eval_samples, args.output_dir, pts, logger)
        if "h2" in hyps:
            h2_res = run_h2(models, eval_samples, args.output_dir, logger)
        if "h3" in hyps:
            h3_res = run_h3(
                models, eval_samples, args.output_dir, args.num_generations, logger,
            )
    except Exception as e:
        logger.error("Error during investigation: %s\n%s", e, traceback.format_exc())

    write_report(h1_res, h2_res, h3_res, args.output_dir, logger)

    logger.info("=" * 70)
    logger.info("Done. All outputs in: %s", args.output_dir)
    logger.info("  h1_faithfulness/     - perturbation test results + plots")
    logger.info("  h2_concept_analysis/ - concept F1, hallucination + plots")
    logger.info("  h3_reward_analysis/  - reward decomposition + plots")
    logger.info("  INVESTIGATION_REPORT.md - consolidated markdown report")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
