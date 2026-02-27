"""
config.py - Central configuration for MedVLM-R1 training pipeline.

All constants, concept maps, hyperparameters, paths, and prompt templates
live here so the other 5 scripts stay clean and DRY.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
VIS_DIR = OUTPUT_DIR / "visualizations"

for _d in (OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, VIS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Model ────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


# ── Dataset ──────────────────────────────────────────────────────────────────

# The HF dataset that ships images + MCQ problems + solutions with concept tags.
# Users should replace this with their own dataset identifier.
DATASET_ID = "abhijitdas/medvlm-r1-dataset"  # placeholder - set to your HF dataset

# Train / val / test split ratios (applied when the dataset has no predefined split)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42


# ── Concept Map ──────────────────────────────────────────────────────────────
# Fixed list of concepts per modality. The model must pick from these.

CONCEPT_MAP: dict[str, list[str]] = {
    "Derm7pt": [
        "pigment network",
        "streaks",
        "dots and globules",
        "blue-whitish veil",
        "regression structures",
        "No positive finding",
    ],
    "SkinCon": [
        "Papule",
        "Plaque",
        "Pustule",
        "Bulla",
        "Patch",
        "Nodule",
        "Ulcer",
        "Crust",
        "Erosion",
        "Atrophy",
        "Exudate",
        "Telangiectasia",
        "Scale",
        "Scar",
        "Friable",
        "Dome-shaped",
        "Brown(Hyperpigmentation)",
        "White(Hypopigmentation)",
        "Purple",
        "Yellow",
        "Black",
        "Erythema",
    ],
    "corda": [
        "Enlarged cardiomediastinum",
        "Cardiomegaly",
        "Lung opacity",
        "Lung lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural effusion",
        "Pleural other",
        "No positive finding",
    ],
    "ddr": [
        "hard exudates",
        "haemorrhages",
        "microaneurysms",
        "soft exudates",
        "No positive finding",
    ],
    "pedicxr": [
        "Atelectasis",
        "Boot-shaped heart",
        "Bronchial thickening",
        "Cardiomegaly",
        "Consolidation",
        "Dextro cardia",
        "Diffuse aveolar opacity",
        "Egg on string sign",
        "Enlarged PA",
        "Infiltration",
        "Interstitial lung disease - ILD",
        "Lung cyst",
        "Lung hyperinflation",
        "No finding",
        "Other lesion",
        "Other nodule/mass",
        "Other opacity",
        "Peribronchovascular interstitial opacity",
        "Pleural effusion",
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

# Reverse lookup: concept -> modality
CONCEPT_TO_MODALITY: dict[str, str] = {}
for _mod, _concepts in CONCEPT_MAP.items():
    for _c in _concepts:
        CONCEPT_TO_MODALITY[_c] = _mod

# All modalities
MODALITIES = list(CONCEPT_MAP.keys())


# ── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
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

QUESTION_TEMPLATE = (
    "{question}\n\n"
    "Your task:\n"
    "1. Identify the imaging modality from: {modalities}\n"
    "2. Identify relevant clinical concepts from the concept list for that modality.\n"
    "3. Reason step-by-step inside <think>...</think> tags, including "
    "<modality> and <concepts> sub-tags.\n"
    "4. Provide the single correct letter choice inside <answer>...</answer> tags.\n"
    "5. No extra text outside these tags."
)


# ── Solution Parsing ─────────────────────────────────────────────────────────
# Maps modality tag names found in solution strings to our canonical names.

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
class TrainConfig:
    """All training hyperparameters in one place."""

    # GRPO
    num_generations: int = 4           # G completions per prompt
    temperature: float = 1.0           # sampling temperature for generation
    max_new_tokens: int = 512          # max tokens per completion
    max_prompt_length: int = 512       # max prompt token length
    beta: float = 0.04                 # KL penalty coefficient

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
    max_pixels: int = 401_408          # ~632x632
    min_pixels: int = 3_136            # ~56x56

    # Precision
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Paths
    output_dir: str = str(CHECKPOINT_DIR)
    logging_dir: str = str(LOG_DIR)

    # Reward weights
    format_reward_weight: float = 1.0
    accuracy_reward_weight: float = 1.0
    concept_reward_weight: float = 0.5  # bonus for correct concepts

    # Device
    device: Optional[str] = None       # auto-detected if None


@dataclass
class EvalConfig:
    """Evaluation / test configuration."""

    max_new_tokens: int = 512
    temperature: float = 0.1           # lower temp for deterministic eval
    top_p: float = 0.9
    batch_size: int = 4
    max_pixels: int = 401_408
    min_pixels: int = 3_136
    device: Optional[str] = None


# ── Convenience Instances ────────────────────────────────────────────────────

TRAIN_CFG = TrainConfig()
EVAL_CFG = EvalConfig()
