"""
config.py - Central configuration for MedVLM-R1 training pipeline.

All constants, concept maps, hyperparameters, paths, and prompt templates
live here so the other 5 scripts stay clean and DRY.

Supports two methods via --method flag:
  - "baseline": MedVLM-R1 reproduction (answer-only + format reward, free-text <think>)
  - "ours":     Concept-aware MO-GRPO with curriculum RL + selective sample replay

References:
  - MedVLM-R1:       https://github.com/JZPeterPan/MedVLM-R1  (MICCAI 2025)
  - MO-GRPO:         arXiv 2509.22047 (Ichihara et al., 2025)
  - VCRL:            NeurIPS 2025 (Variance-based Curriculum RL)
  - VL-Rethinker:    arXiv 2504.08837 (Selective Sample Replay)
  - Curr-ReFT:       arXiv 2503.07065 (Curriculum for small VLMs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ── Method Selection ─────────────────────────────────────────────────────────

class Method(str, Enum):
    """Training method selector.

    BASELINE: Reproduces MedVLM-R1 — binary answer reward + format reward,
              free-text reasoning in <think>, standard GRPO advantages.
    OURS:     Concept-aware MO-GRPO — multi-objective rewards with per-objective
              variance normalization, structured reasoning with <modality> and
              <concepts> tags, curriculum RL stages, selective sample replay.
    """

    BASELINE = "baseline"
    OURS = "ours"


# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
VIS_DIR = OUTPUT_DIR / "visualizations"

for _d in (OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, VIS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Model ────────────────────────────────────────────────────────────────────

# Qwen2.5-VL-3B is the recommended default (Med-R1, VLM-R1 standard).
# Fall back to 2B if GPU-constrained.
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


# ── Dataset ──────────────────────────────────────────────────────────────────

DATASET_ID = "abhijitdas/medvlm-r1-dataset"  # placeholder - set to your HF dataset

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42


# ── Concept Map ──────────────────────────────────────────────────────────────

CONCEPT_MAP: dict[str, list[str]] = {
    "Derm7pt": [
        "pigment network", "streaks", "dots and globules",
        "blue-whitish veil", "regression structures", "No positive finding",
    ],
    "SkinCon": [
        "Papule", "Plaque", "Pustule", "Bulla", "Patch", "Nodule", "Ulcer",
        "Crust", "Erosion", "Atrophy", "Exudate", "Telangiectasia", "Scale",
        "Scar", "Friable", "Dome-shaped", "Brown(Hyperpigmentation)",
        "White(Hypopigmentation)", "Purple", "Yellow", "Black", "Erythema",
    ],
    "corda": [
        "Enlarged cardiomediastinum", "Cardiomegaly", "Lung opacity",
        "Lung lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
        "Pneumothorax", "Pleural effusion", "Pleural other", "No positive finding",
    ],
    "ddr": [
        "hard exudates", "haemorrhages", "microaneurysms",
        "soft exudates", "No positive finding",
    ],
    "pedicxr": [
        "Atelectasis", "Boot-shaped heart", "Bronchial thickening",
        "Cardiomegaly", "Consolidation", "Dextro cardia",
        "Diffuse aveolar opacity", "Egg on string sign", "Enlarged PA",
        "Infiltration", "Interstitial lung disease - ILD", "Lung cyst",
        "Lung hyperinflation", "No finding", "Other lesion",
        "Other nodule/mass", "Other opacity",
        "Peribronchovascular interstitial opacity", "Pleural effusion",
        "Reticulonodular opacity",
    ],
    "quilt-1m": [
        "acanthosis", "atypical cells", "benign", "bladder", "bone", "breast",
        "Calcification", "cartilage", "CD20", "CD3", "CD31", "CD34",
        "chromogranin", "colon", "cords", "Cytokeratin", "dermis", "Desmin",
        "EMA", "epidermis", "epithelium", "erythrocytes", "esophagus",
        "fibroma", "gallbladder", "granulomas", "granulomatous inflammation",
        "hair follicle", "hair follicles", "hemosiderin", "HMB45",
        "Hyperkeratosis", "in situ", "inflammatory cells", "invasion",
        "islands", "Keratinization", "kidney", "lamina propria", "leukemia",
        "lipoma", "liver", "lung", "lymph node", "lymphocytic infiltrate",
        "lymphoid", "malignancy", "malignant", "Melanocytes", "metastasis",
        "muscle", "muscularis mucosa", "nerve", "osteoclasts", "ovary", "P53",
        "pancreas", "parakeratosis", "PAS", "prostate", "S100", "skin",
        "smooth muscle", "spongiosis", "Suppurative", "synaptophysin",
    ],
}

CONCEPT_TO_MODALITY: dict[str, str] = {}
for _mod, _concepts in CONCEPT_MAP.items():
    for _c in _concepts:
        CONCEPT_TO_MODALITY[_c] = _mod

MODALITIES = list(CONCEPT_MAP.keys())


# ── Prompt Templates ─────────────────────────────────────────────────────────
# Two sets: one for "ours" (structured), one for "baseline" (free-text).

# -- Baseline: matches MedVLM-R1 exactly --
SYSTEM_PROMPT_BASELINE = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

QUESTION_TEMPLATE_BASELINE = (
    "{question}\n\n"
    "Your task:\n"
    "1. Think through the question step by step, enclose your reasoning "
    "process in <think>...</think> tags.\n"
    "2. Then provide the correct single-letter choice (A, B, C, D,...) "
    "inside <answer>...</answer> tags.\n"
    "3. No extra information or text outside of these tags."
)

# -- Ours: structured with modality + concepts --
SYSTEM_PROMPT_OURS = (
    "You are a medical image analysis assistant. "
    "When given a medical image and a question, first reason step-by-step "
    "about what you observe, identifying the imaging modality and relevant "
    "clinical concepts. Then provide the answer.\n\n"
    "Format your response EXACTLY as:\n"
    "<think>\n"
    "<modality>MODALITY_NAME</modality>\n"
    "<concepts>[concept1, concept2, ...]</concepts>\n"
    "Your reasoning here.\n"
    "</think>\n"
    "<answer>LETTER</answer>"
)

QUESTION_TEMPLATE_OURS = (
    "{question}\n\n"
    "Your task:\n"
    "1. Identify the imaging modality from: {modalities}\n"
    "2. Identify relevant clinical concepts from the concept list for that modality.\n"
    "3. Reason step-by-step inside <think>...</think> tags, including "
    "<modality> and <concepts> sub-tags.\n"
    "4. Provide the single correct letter choice inside <answer>...</answer> tags.\n"
    "5. No extra text outside these tags."
)

# Convenience aliases (selected at runtime based on method)
SYSTEM_PROMPT = SYSTEM_PROMPT_OURS  # default
QUESTION_TEMPLATE = QUESTION_TEMPLATE_OURS


def get_prompts(method: Method) -> tuple[str, str]:
    """Return (system_prompt, question_template) for the given method."""
    if method == Method.BASELINE:
        return SYSTEM_PROMPT_BASELINE, QUESTION_TEMPLATE_BASELINE
    return SYSTEM_PROMPT_OURS, QUESTION_TEMPLATE_OURS


# ── Prompt Augmentation Templates ────────────────────────────────────────────
# Used in "ours" method to prevent question-pattern overfitting.
# Ref: Strategy doc, Section 2.5.

QUESTION_REPHRASINGS = [
    "Based on the image, {question_core}",
    "Looking at this medical image, {question_core}",
    "Given the findings in the image, {question_core}",
    "Considering the visual evidence, {question_core}",
    "From the image provided, {question_core}",
]


# ── Solution Parsing ─────────────────────────────────────────────────────────

MODALITY_TAG_MAP: dict[str, str] = {
    "X_RAY": "corda",
    "XRAY": "corda",
    "CHEST_XRAY": "corda",
    "DERM": "Derm7pt",
    "SKIN": "SkinCon",
    "SKINCON": "SkinCon",
    "FUNDUS": "ddr",
    "DDR": "ddr",
    "PEDI": "pedicxr",
    "PEDICXR": "pedicxr",
    "HISTO": "quilt-1m",
    "PATHOLOGY": "quilt-1m",
    "QUILT": "quilt-1m",
}


# ── Training Hyperparameters ─────────────────────────────────────────────────

@dataclass
class CurriculumConfig:
    """Curriculum RL stage configuration.

    Ref: VCRL (NeurIPS 2025), Curr-ReFT (arXiv 2503.07065).

    Three stages:
      1. warmup   — easiest subset, high G for exploration
      2. main     — full dataset, VCRL difficulty targeting
      3. hardmine — hardest 20%, lower LR
    """

    enabled: bool = True

    # Stage 1: Warmup
    warmup_steps: int = 200
    warmup_num_generations: int = 8   # more exploration early on

    # Stage 2: Main (VCRL difficulty targeting)
    main_steps: int = 600
    target_difficulty: float = 0.5    # prioritize samples where model is ~50% correct
    difficulty_window: int = 50       # rolling window for difficulty estimation
    difficulty_bandwidth: float = 0.3 # samples within [0.5 - 0.3, 0.5 + 0.3] preferred

    # Stage 3: Hard mining
    hardmine_steps: int = 200
    hardmine_lr_factor: float = 0.5   # reduce LR by this factor
    hardmine_top_fraction: float = 0.2  # focus on hardest 20%


@dataclass
class ReplayConfig:
    """Selective Sample Replay (SSR) configuration.

    Ref: VL-Rethinker (arXiv 2504.08837).
    Stores high-advantage rollouts and mixes them into future batches
    to prevent wasted training steps from zero-variance groups.
    """

    enabled: bool = True
    buffer_size: int = 500            # max stored (prompt, rollout, advantage) tuples
    replay_ratio: float = 0.3         # fraction of each batch from replay buffer
    min_advantage: float = 0.1        # minimum |advantage| to store in buffer


@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""

    # Method
    method: str = Method.OURS.value

    # GRPO
    num_generations: int = 4
    temperature: float = 1.0
    max_new_tokens: int = 512
    max_prompt_length: int = 512
    beta: float = 0.04                # KL penalty coefficient

    # Optimizer
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Schedule
    num_epochs: int = 2
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    save_steps: int = 50
    eval_steps: int = 50
    logging_steps: int = 10

    # Image processing
    max_pixels: int = 401_408
    min_pixels: int = 3_136

    # Precision
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Paths
    output_dir: str = str(CHECKPOINT_DIR)
    logging_dir: str = str(LOG_DIR)

    # Reward weights — used by BOTH methods (baseline ignores concept/modality)
    format_reward_weight: float = 1.0
    accuracy_reward_weight: float = 1.0
    concept_reward_weight: float = 0.5  # "ours" only
    modality_reward_weight: float = 0.3 # "ours" only

    # MO-GRPO: per-objective variance normalization (Ref: arXiv 2509.22047)
    # When True, each reward component is independently standardized before
    # summation, so no single reward dominates the gradient signal.
    mo_grpo_normalize: bool = True

    # Prompt augmentation (ours only)
    prompt_augmentation: bool = True
    concept_perturbation: bool = True   # shuffle concept list order during training

    # Curriculum
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    # Replay
    replay: ReplayConfig = field(default_factory=ReplayConfig)

    # Device
    device: Optional[str] = None


@dataclass
class EvalConfig:
    """Evaluation / test configuration."""

    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    batch_size: int = 4
    max_pixels: int = 401_408
    min_pixels: int = 3_136
    device: Optional[str] = None


# ── Preset Configurations ────────────────────────────────────────────────────

def make_baseline_config() -> TrainConfig:
    """MedVLM-R1 reproduction config: answer + format reward only."""
    cfg = TrainConfig(method=Method.BASELINE.value)
    cfg.concept_reward_weight = 0.0
    cfg.modality_reward_weight = 0.0
    cfg.mo_grpo_normalize = False
    cfg.prompt_augmentation = False
    cfg.concept_perturbation = False
    cfg.curriculum = CurriculumConfig(enabled=False)
    cfg.replay = ReplayConfig(enabled=False)
    return cfg


def make_ours_config() -> TrainConfig:
    """Our method: concept-aware MO-GRPO + curriculum + replay."""
    return TrainConfig(method=Method.OURS.value)


def get_config(method: str | Method) -> TrainConfig:
    """Get the TrainConfig for a given method string."""
    if isinstance(method, str):
        method = Method(method)
    if method == Method.BASELINE:
        return make_baseline_config()
    return make_ours_config()


# ── Convenience Instances ────────────────────────────────────────────────────

TRAIN_CFG = TrainConfig()
EVAL_CFG = EvalConfig()
