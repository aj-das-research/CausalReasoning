"""
train_eval.py - GRPO (Group Relative Policy Optimization) training for MedVLM-R1.

Implements R1-style reinforcement learning:
  1. For each prompt, generate G completions via sampling.
  2. Score each with reward functions (format + accuracy + concept).
  3. Compute group-normalized advantages.
  4. Update policy with clipped gradient + KL penalty against reference model.

Reward functions:
  - format_reward:   1.0 if output matches <think>...<modality>...<concepts>...</think><answer>...</answer>
  - accuracy_reward:  1.0 for correct letter, 0.5 for correct + extra text, 0.0 otherwise
  - concept_reward:   partial credit for matching ground-truth concepts

Usage:
  python train_eval.py                              # train with defaults
  python train_eval.py --num-epochs 3 --lr 2e-6     # custom hyperparams
  python train_eval.py --resume-from <checkpoint>    # resume training
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    CONCEPT_MAP,
    MODALITIES,
    OUTPUT_DIR,
    SEED,
    SYSTEM_PROMPT,
    TRAIN_CFG,
    EVAL_CFG,
)
from data_prep_and_viewer import PREPARED_DATA_DIR, load_and_prepare
from model_loader_and_checker import load_model_and_processor


# ── Reward Functions ─────────────────────────────────────────────────────────

def format_reward(completions: list[str]) -> list[float]:
    """Binary reward: 1.0 if output matches the R1 structured format, else 0.0.

    Expected format:
      <think>
      <modality>...</modality>
      <concepts>[...]</concepts>
      ...reasoning...
      </think>
      <answer>...</answer>
    """
    pattern = r"<think>.*?<modality>.*?</modality>.*?<concepts>.*?</concepts>.*?</think>\s*<answer>.*?</answer>"
    return [
        1.0 if re.fullmatch(pattern, c.strip(), re.DOTALL) else 0.0
        for c in completions
    ]


def accuracy_reward(
    completions: list[str],
    ground_truth_letters: list[str],
) -> list[float]:
    """Grade the answer letter extracted from <answer> tags.

    Scoring:
      1.0 - Correct single letter, no extra text
      0.5 - Correct letter but extra text present
      0.0 - Wrong letter, no letter, or multiple letters
    """
    rewards = []
    for completion, gt_letter in zip(completions, ground_truth_letters):
        reward = 0.0
        gt_letter = gt_letter.strip().upper()

        # Extract answer from tags
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL | re.IGNORECASE)
        if not match:
            rewards.append(0.0)
            continue

        student_answer = match.group(1).strip()
        sa_upper = student_answer.upper()

        if not sa_upper:
            rewards.append(0.0)
            continue

        # Find letter-like characters (A-J)
        letters_found = re.findall(r"[A-J]", sa_upper)
        distinct_letters = set(letters_found)

        if len(letters_found) == 1 and len(distinct_letters) == 1:
            found_letter = distinct_letters.pop()
            if found_letter == gt_letter:
                # Check if there's extra text beyond the letter
                leftover = re.sub(r"(?i)\(?[A-J]\)?", "", student_answer).strip()
                reward = 0.5 if leftover else 1.0
            else:
                reward = 0.0
        else:
            reward = 0.0

        rewards.append(reward)

    return rewards


def concept_reward(
    completions: list[str],
    ground_truth_concepts: list[list[str]],
) -> list[float]:
    """Reward for correctly identifying clinical concepts.

    Computes F1-like score between predicted and ground-truth concepts.
    """
    rewards = []
    for completion, gt_concepts in zip(completions, ground_truth_concepts):
        # Extract concepts from <concepts>[...]</concepts>
        match = re.search(r"<concepts>\s*\[(.*?)\]\s*</concepts>", completion, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue

        # Parse predicted concepts
        raw = match.group(1)
        predicted = {c.strip().strip("'\"").lower() for c in raw.split(",") if c.strip()}
        gt_set = {c.lower() for c in gt_concepts}

        if not gt_set:
            rewards.append(1.0 if not predicted else 0.0)
            continue

        if not predicted:
            rewards.append(0.0)
            continue

        # F1 score
        tp = len(predicted & gt_set)
        precision = tp / len(predicted) if predicted else 0
        recall = tp / len(gt_set) if gt_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        rewards.append(f1)

    return rewards


def compute_rewards(
    completions: list[str],
    gt_letters: list[str],
    gt_concepts: list[list[str]],
    cfg: Any = TRAIN_CFG,
) -> torch.Tensor:
    """Compute combined rewards for a batch of completions.

    Returns tensor of shape (batch_size,) with weighted sum of rewards.
    """
    fmt_rewards = format_reward(completions)
    acc_rewards = accuracy_reward(completions, gt_letters)
    con_rewards = concept_reward(completions, gt_concepts)

    combined = []
    for f, a, c in zip(fmt_rewards, acc_rewards, con_rewards):
        total = (
            cfg.format_reward_weight * f
            + cfg.accuracy_reward_weight * a
            + cfg.concept_reward_weight * c
        )
        combined.append(total)

    return torch.tensor(combined, dtype=torch.float32)


# ── Log Probability Computation ──────────────────────────────────────────────

@torch.no_grad()
def get_per_token_logps(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    prompt_length: int = 0,
) -> torch.Tensor:
    """Compute per-token log probabilities for the completion portion.

    Args:
        model: The VLM model.
        input_ids: Full sequence (prompt + completion), shape (B, T).
        attention_mask: Shape (B, T).
        pixel_values: Image pixel values.
        image_grid_thw: Image grid dimensions for Qwen2-VL.
        prompt_length: Number of prompt tokens to skip in the output.

    Returns:
        Per-token log probabilities for the completion tokens, shape (B, T-prompt_length).
    """
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        kwargs["image_grid_thw"] = image_grid_thw

    outputs = model(**kwargs)
    logits = outputs.logits  # (B, T, V)

    # Shift: logits[t] predicts token[t+1]
    # We want log p(token[t]) for t in [prompt_length, T)
    # So we use logits[prompt_length-1 : T-1] to predict tokens[prompt_length : T]
    shift_logits = logits[:, prompt_length - 1 : -1, :]  # (B, completion_len, V)
    shift_labels = input_ids[:, prompt_length:]           # (B, completion_len)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, completion_len)

    return per_token_logps


# ── GRPO Training Step ───────────────────────────────────────────────────────

def grpo_step(
    model,
    ref_model,
    processor,
    batch: dict,
    cfg: Any = TRAIN_CFG,
    device: str = "cuda",
) -> dict[str, float]:
    """Execute one GRPO training step.

    For each sample in the batch:
      1. Generate G completions.
      2. Score with reward functions.
      3. Compute group-normalized advantages.
      4. Compute GRPO loss with KL penalty.

    Returns dict of metrics (loss, rewards, etc.).
    """
    model.train()
    G = cfg.num_generations

    images = batch["images"]          # list of PIL Images
    problems = batch["problems"]      # list of str
    gt_letters = batch["gt_letters"]  # list of str
    gt_concepts = batch["gt_concepts"]  # list of list[str]
    B = len(images)

    # ── 1. Build prompts ──
    all_prompt_texts = []
    all_images_flat = []
    modalities_str = ", ".join(MODALITIES)

    for img, problem in zip(images, problems):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            f"{problem}\n\n"
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
        all_prompt_texts.append(text)
        all_images_flat.append(img)

    # ── 2. Tokenize prompts ──
    prompt_inputs = processor(
        text=all_prompt_texts,
        images=all_images_flat,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    prompt_inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in prompt_inputs.items()
    }
    prompt_length = prompt_inputs["input_ids"].shape[1]

    # ── 3. Generate G completions per sample ──
    all_completions_text = []  # B * G strings
    all_completion_ids = []     # B * G tensors

    generation_config = {
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "do_sample": True,
        "top_p": 0.95,
    }

    with torch.no_grad():
        for g in range(G):
            generated_ids = model.generate(
                **prompt_inputs,
                **generation_config,
            )
            # Extract completion portion
            completion_ids = generated_ids[:, prompt_length:]
            all_completion_ids.append(completion_ids)

            # Decode
            texts = processor.batch_decode(completion_ids, skip_special_tokens=True)
            all_completions_text.extend(texts)

    # Reshape: all_completions_text is [sample0_gen0, sample1_gen0, ..., sample0_gen1, ...]
    # Rearrange to [sample0_gen0, sample0_gen1, ..., sample1_gen0, ...]
    completions_by_sample = []
    for b in range(B):
        sample_completions = [all_completions_text[g * B + b] for g in range(G)]
        completions_by_sample.append(sample_completions)

    # ── 4. Compute rewards ──
    all_rewards = []
    for b in range(B):
        sample_gt_letters = [gt_letters[b]] * G
        sample_gt_concepts = [gt_concepts[b]] * G
        rewards = compute_rewards(
            completions_by_sample[b],
            sample_gt_letters,
            sample_gt_concepts,
            cfg,
        )
        all_rewards.append(rewards)

    rewards_tensor = torch.stack(all_rewards).to(device)  # (B, G)

    # ── 5. Group-normalized advantages ──
    mean_rewards = rewards_tensor.mean(dim=1, keepdim=True)
    std_rewards = rewards_tensor.std(dim=1, keepdim=True)
    advantages = (rewards_tensor - mean_rewards) / (std_rewards + 1e-4)  # (B, G)

    # ── 6. Compute loss over completions ──
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for g in range(G):
        comp_ids = all_completion_ids[g]  # (B, comp_len)
        comp_len = comp_ids.shape[1]

        # Build full sequence for log-prob computation
        full_ids = torch.cat([prompt_inputs["input_ids"], comp_ids], dim=1)
        full_mask = torch.cat([
            prompt_inputs["attention_mask"],
            torch.ones_like(comp_ids),
        ], dim=1)

        # Create completion mask (mask out padding / post-EOS tokens)
        eos_id = processor.tokenizer.eos_token_id
        is_eos = comp_ids == eos_id
        # Find first EOS position per sequence
        eos_positions = torch.full((B,), comp_len, device=device, dtype=torch.long)
        for b_idx in range(B):
            eos_indices = (is_eos[b_idx]).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                eos_positions[b_idx] = eos_indices[0]

        # Mask: 1 for tokens up to and including first EOS, 0 after
        seq_indices = torch.arange(comp_len, device=device).unsqueeze(0).expand(B, -1)
        completion_mask = (seq_indices <= eos_positions.unsqueeze(1)).float()

        # Policy log probs (with gradients)
        kwargs = {
            "input_ids": full_ids,
            "attention_mask": full_mask,
        }
        if "pixel_values" in prompt_inputs:
            kwargs["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            kwargs["image_grid_thw"] = prompt_inputs["image_grid_thw"]

        outputs = model(**kwargs)
        logits = outputs.logits
        shift_logits = logits[:, prompt_length - 1 : prompt_length - 1 + comp_len, :]
        shift_labels = comp_ids

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, comp_len)

        # Reference model log probs (no gradients)
        with torch.no_grad():
            ref_outputs = ref_model(**kwargs)
            ref_logits = ref_outputs.logits
            ref_shift_logits = ref_logits[:, prompt_length - 1 : prompt_length - 1 + comp_len, :]
            ref_log_probs = F.log_softmax(ref_shift_logits, dim=-1)
            ref_per_token_logps = ref_log_probs.gather(
                dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

        # KL divergence (unbiased estimator)
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )

        # GRPO loss per token
        adv = advantages[:, g].unsqueeze(1)  # (B, 1)
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * adv
        per_token_loss = -(per_token_loss - cfg.beta * per_token_kl)

        # Mean over valid tokens per sequence, then mean over batch
        masked_loss = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
        total_loss = total_loss + masked_loss.mean()

    total_loss = total_loss / G  # Average over generations

    # ── 7. Metrics ──
    metrics = {
        "loss": total_loss.item(),
        "mean_reward": rewards_tensor.mean().item(),
        "mean_format_reward": sum(
            format_reward(completions_by_sample[b])[0] for b in range(B)
        ) / B,
        "mean_accuracy_reward": sum(
            accuracy_reward(completions_by_sample[b], [gt_letters[b]] * G)[0] for b in range(B)
        ) / B,
        "mean_advantage": advantages.mean().item(),
        "std_advantage": advantages.std().item(),
    }

    return total_loss, metrics


# ── Collate Function ─────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """Collate dataset samples into a training batch."""
    images = []
    problems = []
    gt_letters = []
    gt_concepts = []

    for sample in batch:
        # Handle image
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

    return {
        "images": images,
        "problems": problems,
        "gt_letters": gt_letters,
        "gt_concepts": gt_concepts,
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model,
    processor,
    eval_dataset,
    device: str,
    cfg: Any = EVAL_CFG,
    max_samples: int = 100,
) -> dict[str, float]:
    """Evaluate the model on a dataset split.

    Generates one completion per sample (greedy/low-temperature) and
    computes accuracy, format compliance, and concept F1.
    """
    model.eval()
    n = min(max_samples, len(eval_dataset))
    modalities_str = ", ".join(MODALITIES)

    all_completions = []
    all_gt_letters = []
    all_gt_concepts = []

    for i in tqdm(range(n), desc="Evaluating"):
        sample = eval_dataset[i]

        img = sample["image"]
        if isinstance(img, str):
            img = Image.open(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

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

        all_completions.append(completion)
        all_gt_letters.append(sample.get("answer_letter", ""))

        concepts = sample.get("concepts", "[]")
        if isinstance(concepts, str):
            concepts = json.loads(concepts)
        all_gt_concepts.append(concepts)

    # Compute metrics
    fmt_rewards = format_reward(all_completions)
    acc_rewards = accuracy_reward(all_completions, all_gt_letters)
    con_rewards = concept_reward(all_completions, all_gt_concepts)

    metrics = {
        "eval_samples": n,
        "format_compliance": sum(fmt_rewards) / n,
        "accuracy": sum(1 for r in acc_rewards if r >= 1.0) / n,
        "accuracy_partial": sum(acc_rewards) / n,
        "concept_f1": sum(con_rewards) / n,
        "mean_reward": sum(
            TRAIN_CFG.format_reward_weight * f + TRAIN_CFG.accuracy_reward_weight * a + TRAIN_CFG.concept_reward_weight * c
            for f, a, c in zip(fmt_rewards, acc_rewards, con_rewards)
        ) / n,
    }

    return metrics


# ── Training Loop ────────────────────────────────────────────────────────────

def train(
    model_id: str = None,
    checkpoint_path: str | None = None,
    resume_from: str | None = None,
    cfg: Any = None,
    device: str | None = None,
):
    """Main training loop implementing GRPO.

    Args:
        model_id: HuggingFace model identifier.
        checkpoint_path: Path to a pre-trained checkpoint.
        resume_from: Path to resume training from.
        cfg: TrainConfig instance.
        device: Target device.
    """
    from config import TRAIN_CFG as DEFAULT_CFG, MODEL_ID as DEFAULT_MODEL

    cfg = cfg or DEFAULT_CFG
    model_id = model_id or DEFAULT_MODEL
    torch.manual_seed(SEED)

    # ── Load data ──
    if PREPARED_DATA_DIR.exists():
        splits = DatasetDict.load_from_disk(str(PREPARED_DATA_DIR))
    else:
        splits = load_and_prepare()

    train_dataset = splits["train"]
    val_dataset = splits.get("validation", splits.get("test"))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # ── Load model ──
    load_path = resume_from or checkpoint_path
    model, processor, device = load_model_and_processor(
        model_id=model_id,
        checkpoint_path=load_path,
        device=device,
        max_pixels=cfg.max_pixels,
        min_pixels=cfg.min_pixels,
    )

    # ── Create reference model (frozen copy) ──
    print("Creating frozen reference model...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # ── Enable gradient checkpointing ──
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # ── LR Scheduler ──
    total_steps = (
        len(train_dataset) * cfg.num_epochs
        // (cfg.per_device_batch_size * cfg.gradient_accumulation_steps)
    )
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=cfg.learning_rate * 0.1
    )

    # ── DataLoader ──
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # PIL images don't pickle well
        drop_last=True,
    )

    # ── Training ──
    print(f"\nStarting GRPO training for {cfg.num_epochs} epochs ({total_steps} steps)")
    print(f"  G={cfg.num_generations}, beta={cfg.beta}, lr={cfg.learning_rate}")

    global_step = 0
    best_eval_reward = -float("inf")
    log_history = []

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_reward = 0.0
        step_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            loss, metrics = grpo_step(
                model=model,
                ref_model=ref_model,
                processor=processor,
                batch=batch,
                cfg=cfg,
                device=device,
            )

            # Scale loss for gradient accumulation
            scaled_loss = loss / cfg.gradient_accumulation_steps
            scaled_loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % cfg.logging_steps == 0:
                    log_entry = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "lr": scheduler.get_last_lr()[0],
                        **metrics,
                    }
                    log_history.append(log_entry)

                    pbar.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "reward": f"{metrics['mean_reward']:.3f}",
                        "fmt": f"{metrics['mean_format_reward']:.2f}",
                        "acc": f"{metrics['mean_accuracy_reward']:.2f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    })

                # Evaluation
                if global_step % cfg.eval_steps == 0 and val_dataset is not None:
                    eval_metrics = evaluate(
                        model, processor, val_dataset, device, max_samples=50
                    )
                    print(f"\n  [Eval @ step {global_step}] "
                          f"acc={eval_metrics['accuracy']:.3f} "
                          f"fmt={eval_metrics['format_compliance']:.3f} "
                          f"concept_f1={eval_metrics['concept_f1']:.3f} "
                          f"reward={eval_metrics['mean_reward']:.3f}")

                    eval_metrics["step"] = global_step
                    log_history.append({"eval": eval_metrics})

                    # Save best
                    if eval_metrics["mean_reward"] > best_eval_reward:
                        best_eval_reward = eval_metrics["mean_reward"]
                        best_dir = Path(cfg.output_dir) / "best"
                        best_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(str(best_dir))
                        processor.save_pretrained(str(best_dir))
                        print(f"  Saved best model (reward={best_eval_reward:.3f})")

                    model.train()

                # Checkpoint
                if global_step % cfg.save_steps == 0:
                    ckpt_dir = Path(cfg.output_dir) / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(ckpt_dir))
                    processor.save_pretrained(str(ckpt_dir))
                    print(f"\n  Saved checkpoint at step {global_step}")

            epoch_loss += metrics["loss"]
            epoch_reward += metrics["mean_reward"]
            step_count += 1

        # End of epoch summary
        avg_loss = epoch_loss / max(step_count, 1)
        avg_reward = epoch_reward / max(step_count, 1)
        print(f"\nEpoch {epoch+1} complete: avg_loss={avg_loss:.4f}, avg_reward={avg_reward:.3f}")

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
        final_metrics = evaluate(model, processor, val_dataset, device, max_samples=100)
        print(f"Final eval: {json.dumps(final_metrics, indent=2)}")

    return model, processor


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO training for MedVLM-R1")
    parser.add_argument("--model-id", type=str, default=None, help="HuggingFace model ID")
    parser.add_argument("--checkpoint", type=str, default=None, help="Pre-trained checkpoint path")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    args = parser.parse_args()

    # Override config with CLI args
    cfg = TRAIN_CFG
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

    if args.eval_only:
        model, processor, device = load_model_and_processor(
            model_id=args.model_id,
            checkpoint_path=args.checkpoint or args.resume_from,
            device=args.device,
        )
        splits = DatasetDict.load_from_disk(str(PREPARED_DATA_DIR))
        val_ds = splits.get("validation", splits.get("test"))
        metrics = evaluate(model, processor, val_ds, device)
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
