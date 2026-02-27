"""
train_eval.py - GRPO training with MO-GRPO, curriculum RL, and selective replay.

Supports two training methods via --method flag:
  - "baseline": MedVLM-R1 reproduction (answer + format rewards, standard GRPO)
  - "ours":     Concept-aware MO-GRPO with curriculum RL + selective sample replay

Key innovations (ours method):
  - MO-GRPO:     Per-objective variance normalization (arXiv 2509.22047)
  - Curriculum:  3-stage training: warmup → main → hardmine (VCRL + Curr-ReFT)
  - SSR:         Selective Sample Replay (VL-Rethinker, arXiv 2504.08837)

References:
  - GRPO:          DeepSeekMath (arXiv 2402.03300)
  - MedVLM-R1:     https://github.com/JZPeterPan/MedVLM-R1 (MICCAI 2025)
  - MO-GRPO:       arXiv 2509.22047 (Ichihara et al., 2025)
  - VCRL:          NeurIPS 2025 (Variance-based Curriculum RL)
  - Curr-ReFT:     arXiv 2503.07065 (Curriculum for small VLMs)
  - VL-Rethinker:  arXiv 2504.08837 (Selective Sample Replay)

Usage:
  python train_eval.py --method ours                        # our full method
  python train_eval.py --method baseline                    # MedVLM-R1 reproduction
  python train_eval.py --method ours --resume-from <ckpt>   # resume training
  python train_eval.py --eval-only --checkpoint <ckpt> --method ours
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import DatasetDict
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import (
    CONCEPT_MAP,
    MODALITIES,
    MODALITY_TAG_MAP,
    MODEL_ID,
    OUTPUT_DIR,
    SEED,
    TRAIN_CFG,
    EVAL_CFG,
    Method,
    get_config,
    get_prompts,
)
from data_prep_and_viewer import (
    PREPARED_DATA_DIR,
    augment_question,
    load_and_prepare,
)
from model_loader_and_checker import load_model_and_processor


# ── Reward Functions ─────────────────────────────────────────────────────────

def format_reward(
    completions: list[str], method: Method = Method.OURS,
) -> list[float]:
    """Binary reward: 1.0 if output matches the expected structured format.

    Baseline: <think>...</think><answer>...</answer>
    Ours:     <think><modality>...</modality><concepts>...</concepts>...</think><answer>...</answer>
    """
    if method == Method.BASELINE:
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    else:
        pattern = (
            r"<think>.*?<modality>.*?</modality>.*?<concepts>.*?</concepts>"
            r".*?</think>\s*<answer>.*?</answer>"
        )
    return [
        1.0 if re.fullmatch(pattern, c.strip(), re.DOTALL) else 0.0
        for c in completions
    ]


def accuracy_reward(
    completions: list[str],
    ground_truth_letters: list[str],
) -> list[float]:
    """Grade the answer letter extracted from <answer> tags.

    Scoring: 1.0 exact match, 0.5 correct with extra text, 0.0 otherwise.
    """
    rewards = []
    for completion, gt_letter in zip(completions, ground_truth_letters):
        gt_letter = gt_letter.strip().upper()
        match = re.search(
            r"<answer>(.*?)</answer>", completion, re.DOTALL | re.IGNORECASE,
        )
        if not match:
            rewards.append(0.0)
            continue

        student_answer = match.group(1).strip()
        sa_upper = student_answer.upper()
        if not sa_upper:
            rewards.append(0.0)
            continue

        letters_found = re.findall(r"[A-J]", sa_upper)
        distinct_letters = set(letters_found)

        if len(letters_found) == 1 and len(distinct_letters) == 1:
            found_letter = distinct_letters.pop()
            if found_letter == gt_letter:
                leftover = re.sub(r"(?i)\(?[A-J]\)?", "", student_answer).strip()
                rewards.append(0.5 if leftover else 1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    return rewards


def concept_reward(
    completions: list[str],
    ground_truth_concepts: list[list[str]],
) -> list[float]:
    """F1-based reward for clinical concept identification (ours method only)."""
    rewards = []
    for completion, gt_concepts in zip(completions, ground_truth_concepts):
        match = re.search(
            r"<concepts>\s*\[(.*?)\]\s*</concepts>", completion, re.DOTALL,
        )
        if not match:
            rewards.append(0.0)
            continue

        predicted = {
            c.strip().strip("'\"").lower()
            for c in match.group(1).split(",") if c.strip()
        }
        gt_set = {c.lower() for c in gt_concepts}

        if not gt_set:
            rewards.append(1.0 if not predicted else 0.0)
            continue
        if not predicted:
            rewards.append(0.0)
            continue

        tp = len(predicted & gt_set)
        precision = tp / len(predicted)
        recall = tp / len(gt_set)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )
        rewards.append(f1)

    return rewards


def modality_reward(
    completions: list[str],
    ground_truth_modalities: list[str],
) -> list[float]:
    """Binary reward for correct modality identification (ours method only).

    Normalizes predictions via MODALITY_TAG_MAP before comparison.
    """
    rewards = []
    for completion, gt_mod in zip(completions, ground_truth_modalities):
        match = re.search(r"<modality>(.*?)</modality>", completion, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue

        pred_raw = match.group(1).strip()

        # Direct match
        if pred_raw.lower() == gt_mod.lower():
            rewards.append(1.0)
            continue

        # Normalize via tag map
        pred_key = pred_raw.upper().replace(" ", "_").replace("-", "_")
        pred_normalized = MODALITY_TAG_MAP.get(pred_key, pred_raw)
        rewards.append(1.0 if pred_normalized.lower() == gt_mod.lower() else 0.0)

    return rewards


# ── Reward Composition & MO-GRPO Advantages ─────────────────────────────────

def compute_rewards_decomposed(
    completions: list[str],
    gt_letters: list[str],
    gt_concepts: list[list[str]],
    gt_modalities: list[str],
    method: Method,
) -> dict[str, list[float]]:
    """Compute individual reward components (not yet combined).

    Returns dict mapping reward name -> list of per-completion scores.
    """
    rewards = {
        "format": format_reward(completions, method),
        "accuracy": accuracy_reward(completions, gt_letters),
    }
    if method == Method.OURS:
        rewards["concept"] = concept_reward(completions, gt_concepts)
        rewards["modality"] = modality_reward(completions, gt_modalities)
    return rewards


def get_reward_weights(cfg) -> dict[str, float]:
    """Get the weight dict for each reward component."""
    return {
        "format": cfg.format_reward_weight,
        "accuracy": cfg.accuracy_reward_weight,
        "concept": cfg.concept_reward_weight,
        "modality": cfg.modality_reward_weight,
    }


def compute_advantages(
    reward_components: dict[str, torch.Tensor],
    weights: dict[str, float],
    mo_normalize: bool = True,
) -> torch.Tensor:
    """Compute group-normalized advantages.

    Args:
        reward_components: {name: tensor of shape (B, G)}.
        weights: {name: scalar weight}.
        mo_normalize: If True, use MO-GRPO per-objective normalization
                      (Ref: arXiv 2509.22047). If False, standard GRPO.

    MO-GRPO normalizes each reward component independently before weighted
    summation, preventing high-variance objectives from dominating gradients.
    Standard GRPO sums first, then normalizes the combined reward.

    Returns:
        Advantages tensor of shape (B, G).
    """
    first_tensor = next(iter(reward_components.values()))

    if mo_normalize:
        # MO-GRPO: per-objective normalization, then weighted sum
        advantages = torch.zeros_like(first_tensor)
        for name, rewards in reward_components.items():
            w = weights.get(name, 0.0)
            if w == 0.0:
                continue
            mean_r = rewards.mean(dim=1, keepdim=True)
            std_r = rewards.std(dim=1, keepdim=True)
            normalized = (rewards - mean_r) / (std_r + 1e-4)
            advantages = advantages + w * normalized
        return advantages
    else:
        # Standard GRPO: weighted sum first, then normalize
        combined = torch.zeros_like(first_tensor)
        for name, rewards in reward_components.items():
            w = weights.get(name, 0.0)
            combined = combined + w * rewards
        mean_r = combined.mean(dim=1, keepdim=True)
        std_r = combined.std(dim=1, keepdim=True)
        return (combined - mean_r) / (std_r + 1e-4)


# ── Selective Sample Replay (VL-Rethinker, arXiv 2504.08837) ────────────────

class ReplayBuffer:
    """Stores high-advantage rollouts to prevent wasted training steps.

    When all completions in a GRPO group get the same reward, advantages
    are zero and the gradient vanishes. SSR stores informative rollouts
    and replays them to provide supplementary gradient signal.
    """

    def __init__(self, max_size: int = 500, min_advantage: float = 0.1):
        self.max_size = max_size
        self.min_advantage = min_advantage
        self.entries: list[dict] = []

    def add(self, entry: dict) -> None:
        """Add a rollout if its advantage exceeds the threshold."""
        if abs(entry["advantage"]) >= self.min_advantage:
            self.entries.append(entry)
            if len(self.entries) > self.max_size:
                self.entries.sort(key=lambda e: abs(e["advantage"]), reverse=True)
                self.entries = self.entries[: self.max_size]

    def sample(self, n: int) -> list[dict]:
        """Randomly sample n entries from the buffer."""
        n = min(n, len(self.entries))
        if n == 0:
            return []
        return [self.entries[i] for i in random.sample(range(len(self.entries)), n)]

    def __len__(self) -> int:
        return len(self.entries)


# ── Difficulty Estimator (VCRL, NeurIPS 2025) ──────────────────────────────

class DifficultyEstimator:
    """Track per-sample difficulty for curriculum scheduling.

    Difficulty = 1 - (rolling accuracy over last K attempts).
    Samples at ~50% accuracy are at the "learning frontier" and yield
    the highest-variance advantages (most informative for RL).
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.history: dict[str, list[bool]] = defaultdict(list)

    def update(self, sample_id: str, correct: bool) -> None:
        h = self.history[sample_id]
        h.append(correct)
        if len(h) > self.window_size:
            h.pop(0)

    def get_difficulty(self, sample_id: str) -> float:
        h = self.history.get(sample_id, [])
        if not h:
            return 0.5  # unknown → neutral
        return 1.0 - (sum(h) / len(h))

    def get_sampling_weights(
        self,
        sample_ids: list[str],
        target_difficulty: float = 0.5,
        bandwidth: float = 0.3,
    ) -> list[float]:
        """Gaussian-kernel weights centered on target difficulty."""
        weights = []
        for sid in sample_ids:
            diff = self.get_difficulty(sid)
            w = math.exp(-0.5 * ((diff - target_difficulty) / bandwidth) ** 2)
            weights.append(max(w, 0.01))  # small floor to avoid zero weight
        return weights


# ── Curriculum Scheduler (Curr-ReFT, arXiv 2503.07065) ─────────────────────

class CurriculumScheduler:
    """Three-stage curriculum for GRPO training.

    Stage 1 (warmup):   High G for exploration, build difficulty estimates
    Stage 2 (main):     VCRL difficulty-targeted sampling
    Stage 3 (hardmine): Focus on hardest samples, reduced LR
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.stage1_end = cfg.warmup_steps
        self.stage2_end = cfg.warmup_steps + cfg.main_steps
        self.total_steps = cfg.warmup_steps + cfg.main_steps + cfg.hardmine_steps

    def get_stage(self, step: int) -> str:
        if step < self.stage1_end:
            return "warmup"
        elif step < self.stage2_end:
            return "main"
        return "hardmine"

    def get_num_generations(self, step: int, default_G: int) -> int:
        if self.get_stage(step) == "warmup":
            return self.cfg.warmup_num_generations
        return default_G

    def get_lr_factor(self, step: int) -> float:
        if self.get_stage(step) == "hardmine":
            return self.cfg.hardmine_lr_factor
        return 1.0


# ── Prompt Building ──────────────────────────────────────────────────────────

def build_prompt_messages(
    problem: str,
    method: Method,
    augment: bool = False,
) -> list[dict]:
    """Build chat messages for a single sample.

    Args:
        problem: The question text.
        method: BASELINE or OURS.
        augment: If True, apply prompt augmentation (ours only).
    """
    sys_prompt, q_template = get_prompts(method)

    if method == Method.OURS:
        if augment:
            problem = augment_question(problem)
        # Shuffle modality order to prevent memorization
        modalities_str = ", ".join(random.sample(MODALITIES, len(MODALITIES)))
        user_text = q_template.format(question=problem, modalities=modalities_str)
    else:
        user_text = q_template.format(question=problem)

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


# ── Collation ────────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """Collate dataset samples into a training batch."""
    images, problems, gt_letters, gt_concepts, gt_modalities = (
        [], [], [], [], [],
    )
    for sample in batch:
        img = sample["image"]
        if isinstance(img, str):
            img = Image.open(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

        problems.append(sample["problem"])
        gt_letters.append(sample.get("answer_letter", ""))

        concepts = sample.get("concepts", "[]")
        if isinstance(concepts, str):
            concepts = json.loads(concepts)
        gt_concepts.append(concepts)

        gt_modalities.append(sample.get("modality", "unknown"))

    return {
        "images": images,
        "problems": problems,
        "gt_letters": gt_letters,
        "gt_concepts": gt_concepts,
        "gt_modalities": gt_modalities,
        # Use first 100 chars of problem as stable sample ID for difficulty tracking
        "sample_ids": [p[:100] for p in problems],
    }


# ── GRPO Training Step ──────────────────────────────────────────────────────

def grpo_step(
    model,
    ref_model,
    processor,
    batch: dict,
    cfg,
    device: str,
    method: Method,
    G: int | None = None,
) -> tuple[torch.Tensor, dict, list[list[str]], torch.Tensor, list[bool]]:
    """Execute one GRPO training step.

    For each sample in the batch:
      1. Generate G completions via sampling.
      2. Score with reward functions.
      3. Compute group-normalized advantages (MO-GRPO for ours, standard for baseline).
      4. Compute GRPO loss with KL penalty against reference model.

    Returns:
        loss: Scalar tensor with gradient.
        metrics: Dict of logged metrics.
        completions_by_sample: Nested list [B][G] of completion strings.
        advantages: Tensor (B, G).
        sample_correct: List[bool] per sample (any generation correct?).
    """
    model.train()
    G = G or cfg.num_generations

    images = batch["images"]
    problems = batch["problems"]
    gt_letters = batch["gt_letters"]
    gt_concepts = batch["gt_concepts"]
    gt_modalities = batch["gt_modalities"]
    B = len(images)

    augment = method == Method.OURS and cfg.prompt_augmentation

    # ── 1. Build prompts ──
    all_prompt_texts = []
    all_images = []
    for img, problem in zip(images, problems):
        messages = build_prompt_messages(problem, method, augment=augment)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        all_prompt_texts.append(text)
        all_images.append(img)

    # ── 2. Tokenize prompts ──
    prompt_inputs = processor(
        text=all_prompt_texts,
        images=all_images,
        return_tensors="pt",
        padding=True,
    )
    prompt_inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in prompt_inputs.items()
    }
    prompt_length = prompt_inputs["input_ids"].shape[1]

    # ── 3. Generate G completions per sample ──
    all_completions_text = []  # length B*G (flattened across generations)
    all_completion_ids = []    # G tensors, each (B, comp_len_g)

    with torch.no_grad():
        for _g in range(G):
            generated_ids = model.generate(
                **prompt_inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                do_sample=True,
                top_p=0.95,
            )
            completion_ids = generated_ids[:, prompt_length:]
            all_completion_ids.append(completion_ids)
            texts = processor.batch_decode(completion_ids, skip_special_tokens=True)
            all_completions_text.extend(texts)

    # Reshape to [B][G]
    completions_by_sample = [
        [all_completions_text[g * B + b] for g in range(G)]
        for b in range(B)
    ]

    # ── 4. Compute decomposed rewards ──
    reward_components: dict[str, list] = defaultdict(list)
    for b in range(B):
        decomposed = compute_rewards_decomposed(
            completions_by_sample[b],
            [gt_letters[b]] * G,
            [gt_concepts[b]] * G,
            [gt_modalities[b]] * G,
            method,
        )
        for name, values in decomposed.items():
            reward_components[name].append(values)

    reward_tensors = {
        name: torch.tensor(values, dtype=torch.float32, device=device)
        for name, values in reward_components.items()
    }

    # ── 5. Compute advantages ──
    weights = get_reward_weights(cfg)
    mo_normalize = method == Method.OURS and cfg.mo_grpo_normalize
    advantages = compute_advantages(reward_tensors, weights, mo_normalize)

    # ── 6. Compute GRPO loss ──
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for g in range(G):
        comp_ids = all_completion_ids[g]
        comp_len = comp_ids.shape[1]

        full_ids = torch.cat([prompt_inputs["input_ids"], comp_ids], dim=1)
        full_mask = torch.cat([
            prompt_inputs["attention_mask"],
            torch.ones_like(comp_ids),
        ], dim=1)

        # Completion mask (tokens up to and including first EOS)
        eos_id = processor.tokenizer.eos_token_id
        is_eos = comp_ids == eos_id
        eos_positions = torch.full((B,), comp_len, device=device, dtype=torch.long)
        for b_idx in range(B):
            eos_indices = is_eos[b_idx].nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                eos_positions[b_idx] = eos_indices[0]

        seq_indices = torch.arange(comp_len, device=device).unsqueeze(0).expand(B, -1)
        completion_mask = (seq_indices <= eos_positions.unsqueeze(1)).float()

        # Forward kwargs (shared by policy and reference)
        fwd_kwargs = {"input_ids": full_ids, "attention_mask": full_mask}
        if "pixel_values" in prompt_inputs:
            fwd_kwargs["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            fwd_kwargs["image_grid_thw"] = prompt_inputs["image_grid_thw"]

        # Policy log probs (with gradients)
        outputs = model(**fwd_kwargs)
        shift_logits = outputs.logits[
            :, prompt_length - 1 : prompt_length - 1 + comp_len, :
        ]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = log_probs.gather(
            dim=-1, index=comp_ids.unsqueeze(-1),
        ).squeeze(-1)

        # Reference log probs (frozen, no gradients)
        with torch.no_grad():
            ref_outputs = ref_model(**fwd_kwargs)
            ref_shift = ref_outputs.logits[
                :, prompt_length - 1 : prompt_length - 1 + comp_len, :
            ]
            ref_log_probs = F.log_softmax(ref_shift, dim=-1)
            ref_per_token_logps = ref_log_probs.gather(
                dim=-1, index=comp_ids.unsqueeze(-1),
            ).squeeze(-1)

        # KL divergence (Schulman unbiased estimator)
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )

        # GRPO loss: -exp(logp - logp.detach()) * advantage + beta * KL
        adv = advantages[:, g].unsqueeze(1)
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * adv
        per_token_loss = -(per_token_loss - cfg.beta * per_token_kl)

        masked_loss = (
            (per_token_loss * completion_mask).sum(dim=1)
            / completion_mask.sum(dim=1).clamp(min=1)
        )
        total_loss = total_loss + masked_loss.mean()

    total_loss = total_loss / G

    # ── 7. Metrics ──
    combined_reward = torch.zeros(B, G, device=device)
    for name, rt in reward_tensors.items():
        combined_reward += weights.get(name, 0.0) * rt

    metrics = {
        "loss": total_loss.item(),
        "mean_reward": combined_reward.mean().item(),
        "mean_advantage": advantages.mean().item(),
        "std_advantage": advantages.std().item(),
    }
    for name, rt in reward_tensors.items():
        metrics[f"mean_{name}_reward"] = rt.mean().item()

    # Per-sample accuracy for difficulty tracking
    sample_correct = []
    for b in range(B):
        any_correct = any(
            accuracy_reward([c], [gt_letters[b]])[0] >= 1.0
            for c in completions_by_sample[b]
        )
        sample_correct.append(any_correct)

    return total_loss, metrics, completions_by_sample, advantages, sample_correct


# ── Replay Loss ──────────────────────────────────────────────────────────────

def compute_replay_loss(
    model,
    ref_model,
    processor,
    replay_items: list[dict],
    cfg,
    device: str,
    method: Method,
) -> torch.Tensor | None:
    """Compute GRPO loss for replay buffer entries.

    Uses stored completions and advantages, but recomputes log probabilities
    with the current policy for correct gradient flow.
    """
    if not replay_items:
        return None

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    valid_count = 0

    for entry in replay_items:
        messages = build_prompt_messages(entry["problem"], method)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Tokenize prompt
        prompt_inputs = processor(
            text=[text], images=[entry["image"]],
            return_tensors="pt", padding=True,
        )
        prompt_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in prompt_inputs.items()
        }
        prompt_length = prompt_inputs["input_ids"].shape[1]

        # Tokenize stored completion
        comp_token_ids = processor.tokenizer.encode(
            entry["completion"], add_special_tokens=False,
        )
        if not comp_token_ids:
            continue
        comp_ids = torch.tensor(
            [comp_token_ids], dtype=torch.long, device=device,
        )
        comp_len = comp_ids.shape[1]

        # Build full sequence
        full_ids = torch.cat([prompt_inputs["input_ids"], comp_ids], dim=1)
        full_mask = torch.cat([
            prompt_inputs["attention_mask"],
            torch.ones_like(comp_ids),
        ], dim=1)

        fwd_kwargs = {"input_ids": full_ids, "attention_mask": full_mask}
        if "pixel_values" in prompt_inputs:
            fwd_kwargs["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            fwd_kwargs["image_grid_thw"] = prompt_inputs["image_grid_thw"]

        # Policy log probs
        outputs = model(**fwd_kwargs)
        shift_logits = outputs.logits[
            :, prompt_length - 1 : prompt_length - 1 + comp_len, :
        ]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = log_probs.gather(
            dim=-1, index=comp_ids.unsqueeze(-1),
        ).squeeze(-1)

        # Reference log probs
        with torch.no_grad():
            ref_out = ref_model(**fwd_kwargs)
            ref_shift = ref_out.logits[
                :, prompt_length - 1 : prompt_length - 1 + comp_len, :
            ]
            ref_logps = F.log_softmax(ref_shift, dim=-1)
            ref_per_token = ref_logps.gather(
                dim=-1, index=comp_ids.unsqueeze(-1),
            ).squeeze(-1)

        # KL
        per_token_kl = (
            torch.exp(ref_per_token - per_token_logps)
            - (ref_per_token - per_token_logps)
            - 1
        )

        # GRPO loss with stored advantage
        advantage = torch.tensor(entry["advantage"], device=device)
        per_token_loss = torch.exp(
            per_token_logps - per_token_logps.detach()
        ) * advantage
        per_token_loss = -(per_token_loss - cfg.beta * per_token_kl)

        total_loss = total_loss + per_token_loss.mean()
        valid_count += 1

    if valid_count == 0:
        return None
    return total_loss / valid_count


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model,
    processor,
    eval_dataset,
    device: str,
    method: Method = Method.OURS,
    cfg=EVAL_CFG,
    max_samples: int = 100,
) -> dict[str, float]:
    """Evaluate the model on a dataset split."""
    model.eval()
    n = min(max_samples, len(eval_dataset))

    all_completions = []
    all_gt_letters = []
    all_gt_concepts = []
    all_gt_modalities = []

    for i in tqdm(range(n), desc="Evaluating"):
        sample = eval_dataset[i]

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

        all_completions.append(completion)
        all_gt_letters.append(sample.get("answer_letter", ""))

        concepts = sample.get("concepts", "[]")
        if isinstance(concepts, str):
            concepts = json.loads(concepts)
        all_gt_concepts.append(concepts)
        all_gt_modalities.append(sample.get("modality", "unknown"))

    # Compute metrics
    fmt_rewards = format_reward(all_completions, method)
    acc_rewards = accuracy_reward(all_completions, all_gt_letters)

    metrics = {
        "eval_samples": n,
        "format_compliance": sum(fmt_rewards) / n,
        "accuracy": sum(1 for r in acc_rewards if r >= 1.0) / n,
        "accuracy_partial": sum(acc_rewards) / n,
    }

    w = get_reward_weights(TRAIN_CFG)
    total_r = sum(
        w["format"] * f + w["accuracy"] * a
        for f, a in zip(fmt_rewards, acc_rewards)
    )

    if method == Method.OURS:
        con_rewards = concept_reward(all_completions, all_gt_concepts)
        mod_rewards = modality_reward(all_completions, all_gt_modalities)
        metrics["concept_f1"] = sum(con_rewards) / n
        metrics["modality_accuracy"] = sum(mod_rewards) / n
        total_r += sum(
            w["concept"] * c + w["modality"] * m
            for c, m in zip(con_rewards, mod_rewards)
        )

    metrics["mean_reward"] = total_r / n
    return metrics


# ── Training Loop ────────────────────────────────────────────────────────────

def train(
    model_id: str | None = None,
    checkpoint_path: str | None = None,
    resume_from: str | None = None,
    cfg=None,
    device: str | None = None,
):
    """Main training loop with curriculum RL and selective sample replay."""
    from config import MODEL_ID as DEFAULT_MODEL

    cfg = cfg or TRAIN_CFG
    model_id = model_id or DEFAULT_MODEL
    method = Method(cfg.method)
    torch.manual_seed(SEED)
    random.seed(SEED)

    # ── Load data ──
    if PREPARED_DATA_DIR.exists():
        splits = DatasetDict.load_from_disk(str(PREPARED_DATA_DIR))
    else:
        splits = load_and_prepare()

    train_dataset = splits["train"]
    val_dataset = splits.get("validation", splits.get("test"))

    print(f"Method:         {method.value}")
    print(f"Train samples:  {len(train_dataset)}")
    print(f"Val samples:    {len(val_dataset)}")

    # ── Load model ──
    load_path = resume_from or checkpoint_path
    model, processor, device = load_model_and_processor(
        model_id=model_id,
        checkpoint_path=load_path,
        device=device,
        max_pixels=cfg.max_pixels,
        min_pixels=cfg.min_pixels,
    )

    # ── Frozen reference model ──
    print("Creating frozen reference model...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # ── Curriculum & replay setup ──
    curriculum_enabled = method == Method.OURS and cfg.curriculum.enabled
    replay_enabled = method == Method.OURS and cfg.replay.enabled

    if curriculum_enabled:
        cur_scheduler = CurriculumScheduler(cfg.curriculum)
        difficulty_est = DifficultyEstimator(cfg.curriculum.difficulty_window)
        total_steps = cur_scheduler.total_steps
        # Precompute sample IDs for difficulty-weighted sampling
        all_sample_ids = [
            train_dataset[i]["problem"][:100]
            for i in range(len(train_dataset))
        ]
    else:
        cur_scheduler = None
        difficulty_est = None
        total_steps = (
            len(train_dataset) * cfg.num_epochs
            // (cfg.per_device_batch_size * cfg.gradient_accumulation_steps)
        )

    replay_buf = (
        ReplayBuffer(cfg.replay.buffer_size, cfg.replay.min_advantage)
        if replay_enabled else None
    )

    # ── LR Scheduler ──
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=cfg.learning_rate * 0.1,
    )

    # ── DataLoader ──
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )

    # ── Print training plan ──
    print(f"\nStarting {'MO-GRPO' if method == Method.OURS else 'GRPO'} training")
    print(f"  Total steps: {total_steps}")
    print(f"  G={cfg.num_generations}, beta={cfg.beta}, lr={cfg.learning_rate}")
    if curriculum_enabled:
        cc = cfg.curriculum
        print(f"  Curriculum: warmup({cc.warmup_steps}) → "
              f"main({cc.main_steps}) → hardmine({cc.hardmine_steps})")
    if replay_enabled:
        print(f"  Replay: buffer={cfg.replay.buffer_size}, "
              f"ratio={cfg.replay.replay_ratio}")

    # ── Training state ──
    global_step = 0
    batch_idx = 0
    best_eval_reward = -float("inf")
    log_history = []
    base_lr = cfg.learning_rate
    prev_stage = None
    train_iter = iter(train_loader)
    replay_interval = (
        max(1, int(1.0 / cfg.replay.replay_ratio)) if replay_enabled else 0
    )

    pbar = tqdm(total=total_steps, desc="Training")

    while global_step < total_steps:
        # ── Curriculum stage management ──
        if curriculum_enabled:
            stage = cur_scheduler.get_stage(global_step)
            G = cur_scheduler.get_num_generations(
                global_step, cfg.num_generations,
            )

            if stage != prev_stage:
                print(f"\n  [Stage] → {stage} (step {global_step})")
                prev_stage = stage

                if stage == "hardmine" and difficulty_est:
                    # Filter to hardest samples
                    n_hard = max(
                        1,
                        int(len(train_dataset) * cfg.curriculum.hardmine_top_fraction),
                    )
                    difficulties = [
                        (i, difficulty_est.get_difficulty(all_sample_ids[i]))
                        for i in range(len(train_dataset))
                    ]
                    difficulties.sort(key=lambda x: x[1], reverse=True)
                    hard_indices = [d[0] for d in difficulties[:n_hard]]
                    hard_dataset = train_dataset.select(hard_indices)
                    train_loader = DataLoader(
                        hard_dataset,
                        batch_size=cfg.per_device_batch_size,
                        shuffle=True, collate_fn=collate_fn,
                        num_workers=0, drop_last=True,
                    )
                    train_iter = iter(train_loader)
                    print(f"    Filtered to {len(hard_dataset)} hardest samples")

                elif stage == "main" and difficulty_est:
                    _rebuild_weighted_loader(
                        difficulty_est, all_sample_ids, train_dataset, cfg,
                    )

                # Adjust LR for hardmine
                lr_factor = cur_scheduler.get_lr_factor(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = base_lr * lr_factor

            # Periodically update sampling weights during main stage
            if (stage == "main" and difficulty_est
                    and global_step > 0 and global_step % 100 == 0):
                weights = difficulty_est.get_sampling_weights(
                    all_sample_ids,
                    cfg.curriculum.target_difficulty,
                    cfg.curriculum.difficulty_bandwidth,
                )
                sampler = WeightedRandomSampler(weights, len(train_dataset))
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=cfg.per_device_batch_size,
                    sampler=sampler, collate_fn=collate_fn,
                    num_workers=0, drop_last=True,
                )
                train_iter = iter(train_loader)
        else:
            G = cfg.num_generations
            stage = "main"

        # ── Get batch ──
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # ── GRPO step ──
        loss, metrics, completions_by_sample, advantages, sample_correct = (
            grpo_step(
                model=model, ref_model=ref_model, processor=processor,
                batch=batch, cfg=cfg, device=device, method=method, G=G,
            )
        )

        # ── Update difficulty estimator ──
        if difficulty_est:
            for sid, correct in zip(batch["sample_ids"], sample_correct):
                difficulty_est.update(sid, correct)

        # ── Add to replay buffer ──
        if replay_buf:
            B = len(batch["images"])
            for b in range(B):
                for g in range(G):
                    adv_val = advantages[b, g].item()
                    if abs(adv_val) >= cfg.replay.min_advantage:
                        replay_buf.add({
                            "image": batch["images"][b],
                            "problem": batch["problems"][b],
                            "gt_letter": batch["gt_letters"][b],
                            "gt_concepts": batch["gt_concepts"][b],
                            "gt_modality": batch["gt_modalities"][b],
                            "completion": completions_by_sample[b][g],
                            "advantage": adv_val,
                        })

        # ── Backward ──
        scaled_loss = loss / cfg.gradient_accumulation_steps
        scaled_loss.backward()

        # ── Replay loss (supplementary gradient signal) ──
        if (replay_buf and len(replay_buf) > 0
                and replay_interval > 0
                and batch_idx % replay_interval == 0):
            replay_items = replay_buf.sample(cfg.per_device_batch_size)
            replay_loss = compute_replay_loss(
                model, ref_model, processor, replay_items, cfg, device, method,
            )
            if replay_loss is not None:
                (replay_loss / cfg.gradient_accumulation_steps).backward()

        batch_idx += 1

        # ── Gradient update ──
        if batch_idx % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.max_grad_norm,
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            pbar.update(1)

            # ── Logging ──
            if global_step % cfg.logging_steps == 0:
                metrics["step"] = global_step
                metrics["stage"] = stage
                metrics["lr"] = optimizer.param_groups[0]["lr"]
                if replay_buf:
                    metrics["replay_buffer_size"] = len(replay_buf)
                log_history.append(metrics)

                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "rwd": f"{metrics['mean_reward']:.3f}",
                    "stg": stage[:4],
                    "lr": f"{metrics['lr']:.2e}",
                })

            # ── Evaluation ──
            if global_step % cfg.eval_steps == 0 and val_dataset is not None:
                eval_metrics = evaluate(
                    model, processor, val_dataset, device, method,
                    max_samples=50,
                )
                eval_str = (
                    f"acc={eval_metrics['accuracy']:.3f} "
                    f"fmt={eval_metrics['format_compliance']:.3f} "
                    f"rwd={eval_metrics['mean_reward']:.3f}"
                )
                print(f"\n  [Eval @ step {global_step}] {eval_str}")

                eval_metrics["step"] = global_step
                log_history.append({"eval": eval_metrics})

                if eval_metrics["mean_reward"] > best_eval_reward:
                    best_eval_reward = eval_metrics["mean_reward"]
                    best_dir = Path(cfg.output_dir) / "best"
                    best_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(best_dir))
                    processor.save_pretrained(str(best_dir))
                    print(f"  Saved best (reward={best_eval_reward:.3f})")

                model.train()

            # ── Checkpoint ──
            if global_step % cfg.save_steps == 0:
                ckpt_dir = Path(cfg.output_dir) / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(ckpt_dir))
                processor.save_pretrained(str(ckpt_dir))

    pbar.close()

    # ── Save final model ──
    final_dir = Path(cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"\nSaved final model to {final_dir}")

    # ── Save training log ──
    log_path = Path(cfg.logging_dir) / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"Saved training log to {log_path}")

    # ── Final evaluation ──
    if val_dataset is not None:
        print("\nRunning final evaluation...")
        final_metrics = evaluate(
            model, processor, val_dataset, device, method, max_samples=100,
        )
        print(f"Final eval: {json.dumps(final_metrics, indent=2)}")

    return model, processor


def _rebuild_weighted_loader(difficulty_est, sample_ids, dataset, cfg):
    """Helper: rebuild DataLoader with VCRL difficulty-weighted sampling."""
    weights = difficulty_est.get_sampling_weights(
        sample_ids,
        cfg.curriculum.target_difficulty,
        cfg.curriculum.difficulty_bandwidth,
    )
    sampler = WeightedRandomSampler(weights, len(dataset))
    return DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GRPO training for MedVLM-R1 (baseline and ours)",
    )
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "mps", "cpu"])
    parser.add_argument("--method", type=str, default="ours",
                        choices=["baseline", "ours"])
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    # Get method-specific config
    cfg = get_config(args.method)

    # Override with CLI args
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.per_device_batch_size = args.batch_size
    if args.num_generations is not None:
        cfg.num_generations = args.num_generations
    if args.beta is not None:
        cfg.beta = args.beta

    method = Method(args.method)

    if args.eval_only:
        model, processor, device = load_model_and_processor(
            model_id=args.model_id,
            checkpoint_path=args.checkpoint or args.resume_from,
            device=args.device,
        )
        splits = DatasetDict.load_from_disk(str(PREPARED_DATA_DIR))
        val_ds = splits.get("validation", splits.get("test"))
        metrics = evaluate(model, processor, val_ds, device, method)
        print(f"\nEvaluation Results:\n{json.dumps(metrics, indent=2)}")
    else:
        train(
            model_id=args.model_id,
            checkpoint_path=args.checkpoint,
            resume_from=args.resume_from,
            cfg=cfg,
            device=args.device,
        )


if __name__ == "__main__":
    main()
