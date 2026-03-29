# ============================================================================
# Cell 1: Install Dependencies + GPU Check  (Cluster version)
# ============================================================================
# On the cluster, install deps via requirements.txt BEFORE launching Jupyter,
# or uncomment the pip install line below to install inside the notebook.

# Uncomment if deps are not pre-installed in your conda/venv:
# !pip install -q -r requirements.txt

# ── Force single GPU to avoid DataParallel + LoRA + NCCL conflicts ───────────
# Must be set BEFORE torch is imported. DataParallel with LoRA +
# gradient_checkpointing triggers NCCL Error 5 on multi-GPU nodes.
# SLURM --gres=gpu:1 requests 1 GPU but may expose multiple logical devices;
# this guarantees only GPU 0 is used, keeping Trainer in single-device mode.
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# Allow HuggingFace downloads during run (unset any offline flags)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

import torch
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {name}  ({mem:.1f} GB)")
print(f"\nCUDA available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ============================================================================
# Cell 2: WandB Login  (optional on cluster)
# ============================================================================
try:
    import wandb
    wandb.login()
    print("WandB logged in successfully.")
except Exception as e:
    print(f"WandB login skipped ({e}). Set use_wandb=False in Cell 3.")


# ============================================================================
# Cell 3: Configuration -- ALL configs from config.py inlined
# ============================================================================
# Poly-Detox Configuration
# ========================
# All hyperparameters and experiment configurations as specified in the paper:
#   - mT5-base (580M params) as foundation model
#   - LoRA: r=32, alpha=64, Q+V modules (baseline)
#   - Ablation: r in {8,16,32,64}, alpha in {16,32,64,128},
#               modules in {Q+V, All Attention, All Linear}
#   - Few-shot: 50, 100, 200 examples
#   - Languages: English (source), Spanish + Hindi (targets)
#
# Dataset Sources (HuggingFace):
#   1. textdetox/multilingual_paradetox       - parallel toxic/neutral pairs (train)
#      Schema: split (lang code), toxic_sentence, neutral_sentence
#      EN=400 pairs, ES=400 pairs, HI=400 pairs
#   2. textdetox/multilingual_paradetox_test  - toxic-only test inputs
#      Schema: split (lang code), text
#      EN=600, ES=600, HI=600
#   3. ScaDSAI/ParaDeHate                     - English hate-speech pairs (train)
#      Schema: Original_Text, Converted_Text
#      8,276 pairs
#   4. s-nlp/synthdetoxm                      - PENDING ACCESS (commented out)
#      Schema: toxic_sentence, neutral_sentence, lang
#      ES=5,826 pairs

import os
import logging
from dataclasses import dataclass, field
from typing import List

# -- Paths (local / cluster storage) ------------------------------------------
# All checkpoints, results, and cached datasets are stored locally.
# Change BASE_DIR if you want outputs elsewhere (e.g. /disk/scratch/yourUUN/).
#
#
#
#
# To resume, re-run Cells 1-4 (setup) then jump to the target cell.

BASE_DIR       = os.path.join(os.getcwd(), "output", "poly_detox")
# (DRIVE_ROOT removed -- not needed on cluster)
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

for d in [RESULTS_DIR, CHECKPOINTS_DIR, DATA_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Output directory: {BASE_DIR}")
print(f"  Checkpoints: {CHECKPOINTS_DIR}")
print(f"  Results:     {RESULTS_DIR}")
print(f"  Data cache:  {DATA_CACHE_DIR}")

# ── Check for existing checkpoints (useful when resuming a session) ──────────
for exp in ["english_lora", "english_full_ft"]:
    best = os.path.join(CHECKPOINTS_DIR, exp, "best")
    status = "FOUND" if os.path.exists(best) else "not found"
    print(f"  [{status}] {exp}/best")


# ── Dataset configuration ────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """
    Dataset sources from HuggingFace (Section 3.1).
    Schemas verified against actual HuggingFace dataset pages.
    """
    # ── Source 1: Multilingual ParaDetox (train) ──
    # Columns: split, toxic_sentence, neutral_sentence
    # 400 rows per language (en, es, hi, ru, uk, de, zh, ar, am)
    paradetox_name: str = "textdetox/multilingual_paradetox"

    # ── Source 2: Multilingual ParaDetox Test (test-only, toxic inputs) ──
    # Columns: split, text (toxic only, NO neutral reference)
    # 600 rows per language
    paradetox_test_name: str = "textdetox/multilingual_paradetox_test"

    # ── Source 3: ParaDeHate (English hate speech pairs) ──
    # Columns: Original_Text, Converted_Text
    # 8,276 English pairs
    paradehate_name: str = "ScaDSAI/ParaDeHate"

    # ── Source 4: SynthDetoxM (PENDING ACCESS) ──
    # Columns: toxic_sentence, neutral_sentence, lang
    # ES=5,826 pairs -- uncomment when access is granted
    # synthdetoxm_name: str = "s-nlp/synthdetoxm"

    # ── Source 5: es_paradetox (additional Spanish pairs) ──
    # Columns: toxic_sentence, neutral_sentence
    # 565 Spanish pairs — augments ES training from ~300 to ~720 pairs
    es_paradetox_name: str = "textdetox/es_paradetox"

    # ── Source 6: Multilingual Toxic Lexicon (Delete baseline) ──
    # 176k toxic terms across 15 languages
    toxic_lexicon_name: str = "textdetox/multilingual_toxic_lexicon"

    # ── Augmentation flags ──
    use_es_paradetox: bool = True       # augment ES training with es_paradetox
    use_hf_toxic_lexicon: bool = True   # enhance Delete baseline with HF lexicon

    # ── Languages ──
    source_lang: str = "en"
    target_langs: List[str] = field(default_factory=lambda: ["es", "hi"])

    # ── Preprocessing (Section 3.2) ──
    max_input_length: int = 128
    max_target_length: int = 128
    length_ratio_threshold: float = 3.0  # remove pairs where ratio > 3:1

    # ── Few-shot subset sizes (Section 2.2 / RQ2) ──
    few_shot_sizes: List[int] = field(default_factory=lambda: [50, 100, 200])

    # ── Train/val split ratios ──
    en_val_ratio: float = 0.12       # ~1000 pairs for validation
    en_test_ratio: float = 0.08      # ~700 pairs for ref-based test
    es_val_ratio: float = 0.25       # ~100 pairs for validation
    hi_val_ratio: float = 0.25       # ~100 pairs for validation


# ── LoRA configuration (Section 4.1) ─────────────────────────────────────────

@dataclass
class LoRAConfig:
    """LoRA hyperparameters. Baseline: r=32, alpha=64, Q+V modules."""
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: ["q", "v"]  # Q+V projection matrices
    )
    # Ablation search space (Section 2.2)
    ablation_ranks: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    ablation_alphas: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    ablation_modules: dict = field(default_factory=lambda: {
        "qv": ["q", "v"],
        "all_attn": ["q", "k", "v", "o"],
        "all_linear": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    })


# ── Training configuration (Section 4.2) ─────────────────────────────────────
# NOTE: batch_size=8 for T4 GPU to avoid OOM (paper uses 16 for larger GPUs)

@dataclass
class TrainingConfig:
    """Training hyperparameters from Section 4.2."""
    model_name: str = "google/mt5-base"
    learning_rate: float = 3e-4
    batch_size: int = 4               # MIG-16GB optimized (LoRA-only)
    eval_batch_size: int = 8           # MIG-16GB optimized
    max_epochs: int = 5                # sufficient for 6762 samples
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 5  # increased from 3: teacher-forced eval_loss is noisy signal
                                       # for seq2seq detox; patience=3 stopped training too early
    gradient_accumulation_steps: int = 4  # effective batch = 4*4 = 16
    fp16: bool = False
    gradient_checkpointing: bool = True  # save ~40% GPU memory
    seed: int = 42

    # Generation parameters
    max_gen_length: int = 128
    num_beams: int = 4
    no_repeat_ngram_size: int = 3

    # Few-shot specific
    few_shot_epochs: int = 20       # more epochs for small data
    few_shot_lr: float = 1e-4       # lower LR for few-shot

    # WandB (Weights & Biases) configuration
    use_wandb: bool = True
    wandb_project: str = "poly-detox"
    ablation_wandb_project: str = "poly-detox-ablation"  # separate project for ablation runs
    wandb_entity: str = ""           # leave empty for default entity


# ── Evaluation configuration (Section 4.3) ───────────────────────────────────

@dataclass
class EvalConfig:
    """Evaluation thresholds and model names from Section 4.3."""
    # Toxicity scoring - Unitary Multilingual Toxic Comment Classifier
    toxicity_model: str = "textdetox/xlmr-large-toxicity-classifier-v2"
    toxicity_threshold: float = 0.5   # T < 0.5 for successful detoxification

    # Semantic similarity - SentenceBERT / LaBSE
    similarity_model: str = "sentence-transformers/LaBSE"
    similarity_threshold: float = 0.75  # S > 0.75

    # Fluency - causal LM perplexity
    # EN: English-only GPT-2 (stronger than multilingual XGLM for EN)
    # ES/HI: facebook/xglm-564M -- multilingual causal LM (30 langs),
    #         significantly more reliable than small language-specific GPT-2s
    #         (surajp/gpt2-hindi gave constant PPL~17; datificate/gpt2-small-spanish gave PPL~2500 for normal text)
    fluency_models: dict = field(default_factory=lambda: {
        "en": "gpt2",
        "es": "facebook/xglm-564M",
        "hi": "facebook/xglm-564M",
    })

    # Bootstrap resampling (Section 2.2)
    bootstrap_n_samples: int = 1000
    bootstrap_ci: float = 0.95

    # Error analysis sample size
    error_analysis_n: int = 100


# ── Experiment presets ────────────────────────────────────────────────────────

EXPERIMENT_PRESETS = {
    "baseline_english": {
        "description": "Baseline experiments on English (Table 2)",
        "methods": ["delete", "identity", "full_ft", "lora"],
        "lang": "en",
    },
    "zero_shot": {
        "description": "Zero-shot cross-lingual transfer (Table 3 / RQ1)",
        "methods": ["lora"],
        "langs": ["es", "hi"],
        "shots": 0,
    },
    "few_shot": {
        "description": "Few-shot adaptation (RQ2/RQ3)",
        "methods": ["lora", "full_ft"],
        "langs": ["es", "hi"],
        "shots": [50, 100, 200],
    },
    "ablation": {
        "description": "LoRA hyperparameter ablation (RQ4)",
        "langs": ["es", "hi"],
        "shots": 100,
        "ranks": [8, 16, 32, 64],
        "alphas": [16, 32, 64, 128],
        "modules": ["qv", "all_attn", "all_linear"],
    },
}


def get_default_configs():
    """Return default configuration objects."""
    return DataConfig(), LoRAConfig(), TrainingConfig(), EvalConfig()


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("poly_detox")

print("\nConfiguration loaded successfully.")
print(f"  Model: {TrainingConfig().model_name}")
print(f"  LoRA: r={LoRAConfig().rank}, alpha={LoRAConfig().alpha}, modules={LoRAConfig().target_modules}")
print(f"  Training: batch_size={TrainingConfig().batch_size}, grad_accum={TrainingConfig().gradient_accumulation_steps} (effective={TrainingConfig().batch_size * TrainingConfig().gradient_accumulation_steps})")
print(f"  Target languages: {DataConfig().target_langs}")
print(f"  WandB: project={TrainingConfig().wandb_project}, enabled={TrainingConfig().use_wandb}")
print(f"  Datasets: {DataConfig().paradetox_name}, {DataConfig().paradehate_name}")
print(f"  Augmentation: es_paradetox={DataConfig().use_es_paradetox}, hf_lexicon={DataConfig().use_hf_toxic_lexicon}")

# ============================================================================
# Cell 4: Data Loading & Preprocessing -- inline ALL of data_utils.py
# ============================================================================
# Dataset Sources & Schemas (verified):
#   1. textdetox/multilingual_paradetox
#      Columns: split (lang code), toxic_sentence, neutral_sentence
#      400 rows per lang: en, es, hi, ru, uk, de, zh, ar, am  (3,600 total)
#
#   2. textdetox/multilingual_paradetox_test
#      Columns: split (lang code), text  (toxic only, NO neutral reference)
#      600 rows per lang: en, es, hi, ...  (9,000 total)
#
#   3. ScaDSAI/ParaDeHate
#      Columns: Original_Text, Converted_Text
#      8,276 English hate-speech pairs
#
#   4. s-nlp/synthdetoxm  (PENDING ACCESS)
#      Columns: toxic_sentence, neutral_sentence, lang
#      ES=5,826 pairs
#
# Data Flow:
#   English:  ParaDeHate(8276) + paradetox EN(400) -> train/val/test_ref
#   Spanish:  paradetox ES(400) -> train/val + 50/100/200-shot subsets
#   Hindi:    paradetox HI(400) -> train/val + 50/100/200-shot subsets
#   Test:     paradetox_test (600 toxic-only per lang) for all languages

import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING -- exact schemas from HuggingFace
# ══════════════════════════════════════════════════════════════════════════════

def load_paradetox(cfg: DataConfig) -> Dict[str, List[dict]]:
    """
    Load textdetox/multilingual_paradetox (Section 3.1, Source 1).
    Schema: split (lang code), toxic_sentence, neutral_sentence
    Returns dict: lang_code -> list of {"toxic": str, "neutral": str}
    EN=400, ES=400, HI=400 pairs.
    """
    logger.info(f"Loading {cfg.paradetox_name} ...")
    pairs_by_lang = defaultdict(list)
    wanted_langs = {cfg.source_lang} | set(cfg.target_langs)

    for lang in wanted_langs:
        try:
            ds = load_dataset(cfg.paradetox_name, split=lang, cache_dir=DATA_CACHE_DIR)
        except ValueError:
            logger.warning(f"  paradetox: split '{lang}' not found, skipping.")
            continue
        for row in ds:
            toxic = (row.get("toxic_sentence") or "").strip()
            neutral = (row.get("neutral_sentence") or "").strip()
            if toxic and neutral:
                pairs_by_lang[lang].append({"toxic": toxic, "neutral": neutral})

    for lang in sorted(pairs_by_lang):
        logger.info(f"  paradetox {lang}: {len(pairs_by_lang[lang])} pairs")
    return dict(pairs_by_lang)


def load_paradetox_test(cfg: DataConfig) -> Dict[str, List[str]]:
    """
    Load textdetox/multilingual_paradetox_test (Section 3.1, Source 2).
    Schema: split (lang code), text (toxic only -- NO neutral reference)
    Returns dict: lang_code -> list of toxic strings
    EN=600, ES=600, HI=600 inputs.
    """
    logger.info(f"Loading {cfg.paradetox_test_name} ...")
    texts_by_lang = defaultdict(list)
    wanted_langs = {cfg.source_lang} | set(cfg.target_langs)

    for lang in wanted_langs:
        try:
            ds = load_dataset(cfg.paradetox_test_name, split=lang, cache_dir=DATA_CACHE_DIR)
        except ValueError:
            logger.warning(f"  paradetox_test: split '{lang}' not found, skipping.")
            continue
        for row in ds:
            text = (row.get("text") or "").strip()
            if text:
                texts_by_lang[lang].append(text)

    for lang in sorted(texts_by_lang):
        logger.info(f"  paradetox_test {lang}: {len(texts_by_lang[lang])} toxic inputs")
    return dict(texts_by_lang)


def load_paradehate(cfg: DataConfig) -> List[dict]:
    """
    Load ScaDSAI/ParaDeHate (Section 3.1, Source 3).
    Schema: Original_Text, Converted_Text
    8,276 English hate-speech detoxification pairs.
    Returns list of {"toxic": str, "neutral": str}
    """
    logger.info(f"Loading {cfg.paradehate_name} ...")
    ds = load_dataset(cfg.paradehate_name, split="train", cache_dir=DATA_CACHE_DIR)

    pairs = []
    for row in ds:
        toxic = (row.get("Original Text") or row.get("Original_Text") or "").strip()
        neutral = (row.get("Converted Text") or row.get("Converted_Text") or "").strip()
        if toxic and neutral:
            pairs.append({"toxic": toxic, "neutral": neutral})

    logger.info(f"  ParaDeHate en: {len(pairs)} pairs")
    return pairs


def load_es_paradetox(cfg: DataConfig) -> List[dict]:
    """
    Load textdetox/es_paradetox (Source 5) — additional Spanish parallel pairs.
    Schema: toxic_sentence, neutral_sentence (same as multilingual_paradetox).
    565 pairs — augments ES training from ~300 to ~720 post-split.
    """
    logger.info(f"Loading {cfg.es_paradetox_name} ...")
    try:
        ds = load_dataset(cfg.es_paradetox_name, split="train", cache_dir=DATA_CACHE_DIR)
        pairs = []
        for row in ds:
            toxic   = (row.get("toxic_sentence")  or row.get("toxic")   or row.get("original")   or "").strip()
            neutral = (row.get("neutral_sentence") or row.get("neutral") or row.get("detoxified") or "").strip()
            if toxic and neutral:
                pairs.append({"toxic": toxic, "neutral": neutral})
        logger.info(f"  es_paradetox: {len(pairs)} pairs")
        return pairs
    except Exception as e:
        logger.warning(f"  Could not load es_paradetox ({e}); skipping augmentation.")
        return []


# ── PENDING ACCESS -- uncomment when s-nlp/synthdetoxm access is granted ──
#
# def load_synthdetoxm(cfg: DataConfig) -> Dict[str, List[dict]]:
#     """Load s-nlp/synthdetoxm: toxic_sentence, neutral_sentence, lang"""
#     ds = load_dataset("s-nlp/synthdetoxm", split="train", cache_dir=DATA_CACHE_DIR)
#     pairs_by_lang = defaultdict(list)
#     wanted_langs = set(cfg.target_langs)
#     for row in ds:
#         lang = (row.get("lang") or "").strip().lower()[:2]
#         if lang not in wanted_langs: continue
#         toxic = (row.get("toxic_sentence") or "").strip()
#         neutral = (row.get("neutral_sentence") or "").strip()
#         if toxic and neutral:
#             pairs_by_lang[lang].append({"toxic": toxic, "neutral": neutral})
#     return dict(pairs_by_lang)


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING (Section 3.2)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_pairs(pairs: List[dict], cfg: DataConfig) -> List[dict]:
    """
    Preprocessing pipeline from Section 3.2:
    1. Remove empty / whitespace-only entries
    2. Length filtering: remove pairs where word-count ratio > 3:1
    3. Deduplication: hash-based exact matching
    """
    if not pairs:
        return pairs

    n_orig = len(pairs)
    filtered = [p for p in pairs if p["toxic"].strip() and p["neutral"].strip()]
    n_empty = n_orig - len(filtered)

    length_ok = []
    for p in filtered:
        len_t = len(p["toxic"].split())
        len_n = len(p["neutral"].split())
        if len_t == 0 or len_n == 0:
            continue
        ratio = max(len_t, len_n) / max(min(len_t, len_n), 1)
        if ratio <= cfg.length_ratio_threshold:
            length_ok.append(p)
    n_length = len(filtered) - len(length_ok)

    seen = set()
    deduped = []
    for p in length_ok:
        h = hashlib.md5((p["toxic"] + "|||" + p["neutral"]).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(p)
    n_dup = len(length_ok) - len(deduped)

    logger.info(f"  Preprocess: {n_orig} -> {len(deduped)} (empty={n_empty}, length={n_length}, dup={n_dup})")
    return deduped


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / VAL / TEST SPLITTING
# ══════════════════════════════════════════════════════════════════════════════

def split_pairs(pairs, val_ratio, test_ratio=0.0, seed=42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(pairs))
    shuffled = [pairs[i] for i in indices]
    n_val = max(1, int(len(pairs) * val_ratio))
    n_test = int(len(pairs) * test_ratio)
    n_train = len(pairs) - n_val - n_test
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


# ══════════════════════════════════════════════════════════════════════════════
# FEW-SHOT DIVERSITY SAMPLING (Section 3.2)
# ══════════════════════════════════════════════════════════════════════════════

def diversity_sample(pairs, n_samples, seed=42):
    """
    Diversity-based selection via k-means in LaBSE embedding space.
    Selects examples nearest to cluster centroids to maximize coverage.
    Falls back to random sampling if LaBSE unavailable.
    """
    if len(pairs) <= n_samples:
        return list(pairs)

    logger.info(f"  Diversity sampling {n_samples} from {len(pairs)} pairs ...")
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans

        model = SentenceTransformer("sentence-transformers/LaBSE")
        texts = [p["toxic"] for p in pairs]
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)

        kmeans = KMeans(n_clusters=n_samples, random_state=seed, n_init=10)
        kmeans.fit(embeddings)

        selected = set()
        final = []
        for centroid in kmeans.cluster_centers_:
            dists = np.linalg.norm(embeddings - centroid, axis=1)
            for idx in np.argsort(dists):
                if idx not in selected:
                    selected.add(idx)
                    final.append(pairs[idx])
                    break

        if len(final) < n_samples:
            rng = np.random.RandomState(seed)
            extras = [i for i in range(len(pairs)) if i not in selected]
            rng.shuffle(extras)
            for idx in extras[:n_samples - len(final)]:
                final.append(pairs[idx])

        return final[:n_samples]

    except Exception as e:
        logger.warning(f"LaBSE sampling failed ({e}); random fallback")
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(pairs), size=n_samples, replace=False)
        return [pairs[i] for i in idx]


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZATION -- mT5 SentencePiece (Section 3.2)
# ══════════════════════════════════════════════════════════════════════════════

def pairs_to_hf_dataset(pairs, tokenizer, max_input_length=128, max_target_length=128, prefix="detoxify: "):
    inputs  = [prefix + p["toxic"]   for p in pairs]
    targets = [p["neutral"]          for p in pairs]

    model_inputs = tokenizer(inputs,  max_length=max_input_length,  truncation=True, padding="max_length")
    labels       = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = [
        [t if t != tokenizer.pad_token_id else -100 for t in ids]
        for ids in labels["input_ids"]
    ]
    return Dataset.from_dict(model_inputs)


def toxic_texts_to_hf_dataset(texts, tokenizer, max_input_length=128, prefix="detoxify: "):
    inputs = [prefix + t for t in texts]
    return Dataset.from_dict(tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length"))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DATA PREPARATION  (English + Spanish + Hindi)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_all_data(cfg: DataConfig = None, tokenizer=None) -> dict:
    """
    Full data pipeline: load -> preprocess -> split -> tokenize.

    Produces:
      English:  en_train/val/test_ref (parallel) + en_test_noref (toxic-only)
      Spanish:  es_train/val (parallel) + es_test_noref + es_{50,100,200}shot
      Hindi:    hi_train/val (parallel) + hi_test_noref + hi_{50,100,200}shot
    """
    if cfg is None:      cfg = DataConfig()
    if tokenizer is None: tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

    data = {}

    # ── 1. Load all HuggingFace datasets ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Loading datasets from HuggingFace (EN + ES + HI)")
    logger.info("=" * 60)

    paradetox      = load_paradetox(cfg)       # {lang -> [{toxic, neutral}]}
    paradetox_test = load_paradetox_test(cfg)  # {lang -> [toxic_str]}
    paradehate     = load_paradehate(cfg)       # [{toxic, neutral}]  (EN only)
    es_extra = load_es_paradetox(cfg) if cfg.use_es_paradetox else []  # Source 5
    # synthdetox = load_synthdetoxm(cfg)        # PENDING ACCESS

    # ── 2. ENGLISH data ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Preparing ENGLISH data")
    logger.info("=" * 60)

    en_pairs = paradehate + paradetox.get("en", [])
    en_pairs = preprocess_pairs(en_pairs, cfg)
    logger.info(f"  Combined EN pairs (ParaDeHate + paradetox): {len(en_pairs)}")

    en_train, en_val, en_test_ref = split_pairs(
        en_pairs, val_ratio=cfg.en_val_ratio, test_ratio=cfg.en_test_ratio, seed=42,
    )
    logger.info(f"  EN split -> train={len(en_train)}, val={len(en_val)}, test_ref={len(en_test_ref)}")

    data["en_train_raw"]    = en_train
    data["en_val_raw"]      = en_val
    data["en_test_ref_raw"] = en_test_ref
    data["en_test_noref_raw"] = paradetox_test.get("en", [])
    logger.info(f"  EN test (no-ref): {len(data['en_test_noref_raw'])} toxic inputs")

    data["en_train"]    = pairs_to_hf_dataset(en_train,    tokenizer, cfg.max_input_length, cfg.max_target_length)
    data["en_val"]      = pairs_to_hf_dataset(en_val,      tokenizer, cfg.max_input_length, cfg.max_target_length)
    data["en_test_ref"] = pairs_to_hf_dataset(en_test_ref, tokenizer, cfg.max_input_length, cfg.max_target_length)

    # ── 3. SPANISH data ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Preparing SPANISH data")
    logger.info("=" * 60)

    es_pairs = paradetox.get("es", [])
    if es_extra:
        n_before = len(es_pairs)
        es_pairs = es_pairs + es_extra
        logger.info(f"  ES augmented: paradetox({n_before}) + es_paradetox({len(es_extra)}) = {len(es_pairs)}")
    # es_pairs += synthdetox.get("es", [])  # PENDING
    es_pairs = preprocess_pairs(es_pairs, cfg)
    logger.info(f"  ES pairs after preprocessing: {len(es_pairs)}")

    es_train, es_val, _ = split_pairs(es_pairs, val_ratio=cfg.es_val_ratio, seed=42)
    logger.info(f"  ES split -> train={len(es_train)}, val={len(es_val)}")

    data["es_train_raw"]      = es_train
    data["es_val_raw"]        = es_val
    data["es_test_noref_raw"] = paradetox_test.get("es", [])
    logger.info(f"  ES test (no-ref): {len(data['es_test_noref_raw'])} toxic inputs")

    data["es_val"] = pairs_to_hf_dataset(es_val, tokenizer, cfg.max_input_length, cfg.max_target_length)

    # ── 4. HINDI data ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Preparing HINDI data")
    logger.info("=" * 60)

    hi_pairs = preprocess_pairs(paradetox.get("hi", []), cfg)
    logger.info(f"  HI pairs (paradetox): {len(hi_pairs)}")

    hi_train, hi_val, _ = split_pairs(hi_pairs, val_ratio=cfg.hi_val_ratio, seed=42)
    logger.info(f"  HI split -> train={len(hi_train)}, val={len(hi_val)}")

    data["hi_train_raw"]      = hi_train
    data["hi_val_raw"]        = hi_val
    data["hi_test_noref_raw"] = paradetox_test.get("hi", [])
    logger.info(f"  HI test (no-ref): {len(data['hi_test_noref_raw'])} toxic inputs")

    data["hi_val"] = pairs_to_hf_dataset(hi_val, tokenizer, cfg.max_input_length, cfg.max_target_length)

    # ── 5. Few-shot subsets for ALL target languages (ES + HI) ───────────────
    logger.info("\n" + "-" * 40)
    logger.info("Creating few-shot subsets (diversity sampling) for ES + HI")
    logger.info("-" * 40)

    target_train_map = {"es": es_train, "hi": hi_train}

    for lang in cfg.target_langs:
        lang_train = target_train_map.get(lang, [])
        for n_shot in cfg.few_shot_sizes:
            subset = diversity_sample(lang_train, n_shot, seed=42)
            data[f"{lang}_{n_shot}shot_raw"] = subset
            data[f"{lang}_{n_shot}shot"]     = pairs_to_hf_dataset(
                subset, tokenizer, cfg.max_input_length, cfg.max_target_length,
            )
            logger.info(f"  {lang}_{n_shot}shot: {len(subset)} pairs")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("DATA SUMMARY")
    logger.info("=" * 60)
    for key in sorted(data.keys()):
        val = data[key]
        if key.endswith("_raw"):
            n = len(val)
            kind = "pairs" if val and isinstance(val[0], dict) else "texts"
            logger.info(f"  {key:<30} {n:>6} {kind}")
        elif hasattr(val, "__len__"):
            logger.info(f"  {key:<30} {len(val):>6} tokenized examples")

    return data


# ── Run data preparation ──────────────────────────────────────────────────────
data_cfg = DataConfig()
train_cfg = TrainingConfig()
lora_cfg  = LoRAConfig()
eval_cfg  = EvalConfig()

tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name)
data = prepare_all_data(data_cfg, tokenizer)

# ── Show sample pairs from each language ─────────────────────────────────────
print("\n--- Sample English training pair ---")
p = data["en_train_raw"][0]
print(f"  TOXIC:   {p['toxic'][:100]}")
print(f"  NEUTRAL: {p['neutral'][:100]}")

print("\n--- Sample Spanish training pair ---")
if data["es_train_raw"]:
    p = data["es_train_raw"][0]
    print(f"  TOXIC:   {p['toxic'][:100]}")
    print(f"  NEUTRAL: {p['neutral'][:100]}")

print("\n--- Sample Spanish test (no-ref) ---")
if data["es_test_noref_raw"]:
    print(f"  TOXIC:   {data['es_test_noref_raw'][0][:100]}")

print("\n--- Sample Hindi training pair ---")
if data.get("hi_train_raw"):
    p = data["hi_train_raw"][0]
    print(f"  TOXIC:   {p['toxic'][:100]}")
    print(f"  NEUTRAL: {p['neutral'][:100]}")

print("\n--- Sample Hindi test (no-ref) ---")
if data.get("hi_test_noref_raw"):
    print(f"  TOXIC:   {data['hi_test_noref_raw'][0][:100]}")

print(f"\n--- Dataset sizes ---")
print(f"  EN train: {len(data['en_train_raw'])} | val: {len(data['en_val_raw'])} | test_ref: {len(data['en_test_ref_raw'])} | test_noref: {len(data['en_test_noref_raw'])}")
print(f"  ES train: {len(data['es_train_raw'])} | val: {len(data['es_val_raw'])} | test_noref: {len(data['es_test_noref_raw'])}")
print(f"  HI train: {len(data['hi_train_raw'])} | val: {len(data['hi_val_raw'])} | test_noref: {len(data['hi_test_noref_raw'])}")
for lang in data_cfg.target_langs:
    for n in data_cfg.few_shot_sizes:
        print(f"  {lang}_{n}shot: {len(data[f'{lang}_{n}shot_raw'])} pairs")

# ============================================================================
# Cell 5: Baselines -- inline baselines.py
# ============================================================================
# Section 5.1: Delete Baseline and Identity Baseline
# 1. Delete Baseline - rule-based profanity removal (EN + ES + HI lexicons)
# 2. Identity Baseline - returns input unchanged

import re

# ── Multilingual profanity lexicon (EN + ES + HI) ───────────────────────────
PROFANITY_LEXICON = {
    "en": {
        "fuck", "shit", "damn", "ass", "bitch", "bastard", "dick", "crap",
        "piss", "hell", "slut", "whore", "idiot", "stupid", "moron", "dumb",
        "retard", "nigger", "faggot", "cunt", "kill", "hate", "ugly",
        "loser", "suck", "pathetic", "trash", "worthless", "disgusting",
        "horrible", "terrible", "awful", "scum", "pig", "die", "rape",
        "fucking", "shitty", "bullshit", "asshole", "motherfucker", "dumbass",
        "jackass", "dickhead", "goddamn", "crappy", "bitchy", "slutty",
    },
    "es": {
        "mierda", "puta", "joder", "cono", "gilipollas", "cabron", "imbecil",
        "estupido", "idiota", "pendejo", "culero", "chingar", "verga",
        "marica", "zorra", "basura", "cerdo", "asqueroso", "maldito",
        "pinche", "hijueputa", "carajo", "tonto", "baboso", "mamon",
    },
    "hi": {
        # Hindi profanity (transliterated/Devanagari common terms)
        "madarchod", "bhenchod", "chutiya", "gandu", "randi", "haramzada",
        "harami", "kutta", "kamine", "saala", "bhadwa", "kutte", "kamina",
        "gadha", "ullu", "bakwaas", "bewakoof", "pagal", "nalayak",
        # Devanagari script terms
        "मादरचोद", "भेनचोद", "चूतिया", "गांडू", "रंडी", "हरामजादा",
        "हरामी", "कुत्ता", "कमीना", "साला", "भड़वा", "गधा", "उल्लू",
        "बकवास", "बेवकूफ", "पागल", "नालायक",
    },
}

# Flatten for quick lookup
ALL_PROFANITY = set()
for words in PROFANITY_LEXICON.values():
    ALL_PROFANITY.update(words)


def load_hf_toxic_lexicon(langs, cache_dir=None):
    """
    Load textdetox/multilingual_toxic_lexicon to augment the Delete baseline.
    176k toxic terms across 15 languages. Falls back gracefully if unavailable.
    """
    try:
        if cache_dir is None:
            cache_dir = DATA_CACHE_DIR
        logger.info("Loading textdetox/multilingual_toxic_lexicon ...")
        hf_lex = {lang: set() for lang in langs}
        for lang in langs:
            try:
                ds = load_dataset("textdetox/multilingual_toxic_lexicon", split=lang, cache_dir=cache_dir)
            except ValueError:
                logger.warning(f"  toxic_lexicon: split '{lang}' not found, skipping.")
                continue
            for row in ds:
                word = (row.get("word") or row.get("term") or row.get("text") or "").strip().lower()
                if word:
                    hf_lex[lang].add(word)
        totals = {k: len(v) for k, v in hf_lex.items() if v}
        logger.info(f"  HF toxic lexicon loaded: {totals}")
        return hf_lex
    except Exception as e:
        logger.warning(f"  Could not load HF toxic lexicon ({e}); using built-in lexicon only.")
        return {}


class DeleteBaseline:
    """
    Section 5.1 - Baseline 1: Delete Baseline.
    Identifies and removes all tokens flagged by the multilingual profanity lexicon.
    Augmented with HF toxic lexicon (176k terms) when available.
    """
    def __init__(self, lang: str = None, use_hf_lexicon: bool = True):
        if lang and lang in PROFANITY_LEXICON:
            self.lexicon = set(PROFANITY_LEXICON[lang])
        else:
            self.lexicon = set(ALL_PROFANITY)

        # Augment with HF toxic lexicon (176k terms across 15 languages)
        if use_hf_lexicon:
            target_langs = [lang] if lang else list(PROFANITY_LEXICON.keys())
            hf_lex = load_hf_toxic_lexicon(langs=target_langs)
            before = len(self.lexicon)
            for lang_words in hf_lex.values():
                self.lexicon.update(lang_words)
            if len(self.lexicon) > before:
                logger.info(f"  DeleteBaseline: {before} -> {len(self.lexicon)} terms (+{len(self.lexicon)-before} from HF)")

        try:
            from better_profanity import profanity
            profanity.load_censor_words()
            self.use_library = True
        except ImportError:
            self.use_library = False

    def detoxify(self, text: str) -> str:
        tokens = text.split()
        cleaned = []
        for token in tokens:
            word_lower = re.sub(r'[^\w]', '', token.lower())
            # Exact match first; then prefix match to handle morphological
            # inflections (e.g. lexicon has कुत्ता, text has कुत्ते/कुत्तों)
            is_toxic = word_lower in self.lexicon or any(
                word_lower.startswith(w) or w.startswith(word_lower)
                for w in self.lexicon if len(w) >= 4 and len(word_lower) >= 4
            )
            if not is_toxic:
                cleaned.append(token)
        result = " ".join(cleaned).strip()

        if self.use_library:
            try:
                from better_profanity import profanity
                result = profanity.censor(result, censor_char="")
                result = re.sub(r'\s+', ' ', result).strip()
            except Exception:
                pass

        return result if result else text

    def detoxify_batch(self, texts):
        return [self.detoxify(t) for t in texts]


class IdentityBaseline:
    """Section 5.1 - Baseline 2: Identity Baseline (no-op)."""
    def detoxify(self, text: str) -> str:
        return text

    def detoxify_batch(self, texts):
        return list(texts)


def run_baseline(baseline_name: str, test_pairs: list, lang: str = "en") -> list:
    if baseline_name == "delete":
        model = DeleteBaseline(lang=lang)
    elif baseline_name == "identity":
        model = IdentityBaseline()
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    return [
        {"toxic": p["toxic"], "neutral": p["neutral"], "prediction": model.detoxify(p["toxic"])}
        for p in test_pairs
    ]


# ── Quick test across all three languages ────────────────────────────────────
print("Baselines loaded. Quick test (EN / ES / HI):")

en_test = "You are a stupid idiot and should go to hell"
es_test = "Eres un idiota estupido y un gilipollas"
hi_test = "तुम एक बेवकूफ और नालायक इंसान हो"

for lang, text in [("en", en_test), ("es", es_test), ("hi", hi_test)]:
    d = DeleteBaseline(lang=lang)
    print(f"\n  [{lang.upper()}] Input:  {text}")
    print(f"  [{lang.upper()}] Delete: {d.detoxify(text)}")

# ============================================================================
# Cell 6: Model Utilities -- inline model_utils.py
# ============================================================================
# Load mT5-base with:
#   - LoRA adapters (PEFT): r=32, alpha=64, Q+V modules (baseline config)
#   - Full fine-tuning: all parameters trainable
#   - Ablation configs: r in {8,16,32,64}, alpha in {16,32,64,128},
#     modules in {Q+V, All Attention, All Linear}
#
# Equation (1): W' = W + BA where B in R^{d x r}, A in R^{r x d}

from peft import (
    LoraConfig as PeftLoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
from transformers import (
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
)


def get_tokenizer(model_name: str = "google/mt5-base"):
    """Load mT5 SentencePiece tokenizer (250K vocabulary)."""
    return AutoTokenizer.from_pretrained(model_name)


def load_mt5_base(
    model_name: str = "google/mt5-base",
    device: str = None,
) -> T5ForConditionalGeneration:
    """
    Load the base mT5-base model (580M parameters).
    Section 4: "We utilize the mT5-base model with 580M parameters."
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading {model_name} on {device}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    return model


def apply_lora(
    model,
    rank: int = 32, 
    alpha: int = 64,
    target_modules: List[str] = None,
    dropout: float = 0.1,
) -> PeftModel:
    """
    Apply LoRA adapters to mT5 model (Section 4.1).

    Equation (1): W' = W + BA
    - Freezes all base model parameters
    - Injects trainable low-rank matrices into specified attention layers
    - Default targets Q and V projection matrices

    Parameter reduction (Equation 2):
      Reduction = 1 - 2r/d ~ 93.75% per layer (r=32, d=1024)
      Overall: 580M -> ~3M trainable (99.5% reduction)
    """
    if target_modules is None:
        target_modules = ["q", "v"]

    lora_config = PeftLoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)

    # Log parameter statistics
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        f"  LoRA applied: r={rank}, alpha={alpha}, modules={target_modules}"
    )
    logger.info(
        f"  Trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    return peft_model


def load_model_for_training(
    method: str = "lora",
    lora_cfg_obj: LoRAConfig = None,
    train_cfg_obj: TrainingConfig = None,
    device: str = None,
):
    """
    Load model configured for training.

    Args:
        method: 'lora' or 'full_ft'
        lora_cfg_obj: LoRA configuration (used only for method='lora')
        train_cfg_obj: Training configuration

    Returns:
        (model, tokenizer)
    """
    if lora_cfg_obj is None:
        lora_cfg_obj = LoRAConfig()
    if train_cfg_obj is None:
        train_cfg_obj = TrainingConfig()

    tok = get_tokenizer(train_cfg_obj.model_name)
    model = load_mt5_base(train_cfg_obj.model_name, device)

    if method == "lora":
        model = apply_lora(
            model,
            rank=lora_cfg_obj.rank,
            alpha=lora_cfg_obj.alpha,
            target_modules=lora_cfg_obj.target_modules,
            dropout=lora_cfg_obj.dropout,
        )
    elif method == "full_ft":
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Full fine-tuning: {trainable:,} trainable parameters")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'lora' or 'full_ft'.")

    return model, tok


def save_model(model, tok, save_dir: str, method: str = "lora"):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)

    if method == "lora" and isinstance(model, PeftModel):
        # Save only LoRA adapter weights (~8MB vs ~2.3GB for full model)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        adapter_size = sum(
            os.path.getsize(os.path.join(save_dir, f))
            for f in os.listdir(save_dir)
            if f.endswith(('.bin', '.safetensors'))
        )
        logger.info(f"  LoRA adapter saved to {save_dir} ({adapter_size / 1e6:.1f}MB)")
    else:
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        model_size = sum(
            os.path.getsize(os.path.join(save_dir, f))
            for f in os.listdir(save_dir)
            if f.endswith(('.bin', '.safetensors'))
        )
        logger.info(f"  Full model saved to {save_dir} ({model_size / 1e6:.1f}MB)")


def load_trained_model(
    checkpoint_dir: str,
    method: str = "lora",
    base_model_name: str = "google/mt5-base",
    device: str = None,
    is_trainable: bool = False,   # set True when loading for continued fine-tuning
):
    """Load a trained model from checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(checkpoint_dir)

    if method == "lora":
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        # is_trainable=True keeps LoRA requires_grad=True for few-shot fine-tuning.
        # Without it, PeftModel defaults to inference mode (frozen LoRA weights)
        # and no gradient updates flow through the adapters during training.
        model = PeftModel.from_pretrained(base_model, checkpoint_dir,
                                          is_trainable=is_trainable)
        model = model.to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
        model = model.to(device)

    if not is_trainable:
        model.eval()
    return model, tok


def generate_detoxified(
    model,
    tok,
    texts: List[str],
    max_length: int = 128,
    num_beams: int = 4,
    batch_size: int = 16,
    prefix: str = "detoxify: ",
    device: str = None,
) -> List[str]:
    """
    Generate detoxified text for a batch of toxic inputs.
    Uses beam search with num_beams=4 (Section 4.2 training config).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_outputs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = [prefix + t for t in texts[i:i + batch_size]]

        inputs = tok(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        decoded = tok.batch_decode(outputs, skip_special_tokens=True)
        all_outputs.extend(decoded)

    return all_outputs


def get_param_stats(model) -> dict:
    """Get parameter statistics for efficiency reporting (RQ3)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0,
        "reduction_pct": 100 * (1 - trainable / total) if total > 0 else 0,
    }


print("Model utilities loaded successfully.")

# ============================================================================
# Cell 7: Training Pipeline with Checkpointing + WandB
# ============================================================================
# T4 GPU safety features:
#   1. gradient_checkpointing=True  -- cuts GPU memory ~40% (slower but safe)
#   2. Auto-resume from mid-training checkpoint  -- resume if session dies
#   3. Skip-if-complete  -- re-running a cell won't restart finished experiments
#   4. All checkpoints on Google Drive  -- survive runtime resets
#
# Checkpoint layout (all on Drive):
#   checkpoints/{experiment}/checkpoint-{step}/  <- mid-training (last 2 kept)
#   checkpoints/{experiment}/best/               <- best model (final)
# ============================================================================

import json, time
import psutil

try:
    import wandb
    WANDB_AVAILABLE = True
    print("WandB available:", wandb.__version__)
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not installed; training will run without logging.")

from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)


class EfficiencyTracker:
    """Track wall-clock time, peak GPU memory, and CPU memory."""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_gpu_memory = 0

    def start(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def stop(self):
        self.end_time = time.time()
        if torch.cuda.is_available():
            self.peak_gpu_memory = torch.cuda.max_memory_allocated() / 1e9

    @property
    def elapsed_hours(self):
        return (self.end_time - self.start_time) / 3600 if (self.start_time and self.end_time) else 0.0

    def summary(self, model=None, checkpoint_dir=None):
        result = {
            "training_time_hours": round(self.elapsed_hours, 3),
            "peak_gpu_memory_gb":  round(self.peak_gpu_memory, 2),
            "peak_cpu_memory_gb":  round(psutil.Process().memory_info().rss / 1e9, 2),
        }
        if model is not None:
            result.update(get_param_stats(model))
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            size = sum(
                os.path.getsize(os.path.join(checkpoint_dir, f))
                for f in os.listdir(checkpoint_dir)
                if os.path.isfile(os.path.join(checkpoint_dir, f))
            )
            result["checkpoint_size_mb"] = round(size / 1e6, 1)
        return result


def _find_latest_checkpoint(output_dir: str):
    """Return path to latest mid-training checkpoint, or None."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")
         and os.path.isdir(os.path.join(output_dir, d))],
        key=lambda x: int(x.split("-")[-1])
    )
    return os.path.join(output_dir, ckpts[-1]) if ckpts else None


def _is_already_complete(best_dir: str, method: str) -> bool:
    """True if best/ already has a saved model (skip re-training)."""
    if not os.path.exists(best_dir):
        return False
    # LoRA saves adapter_config.json; full FT saves config.json
    marker = "adapter_config.json" if method == "lora" else "config.json"
    return os.path.exists(os.path.join(best_dir, marker))


def train_model(
    method: str,
    train_dataset,
    val_dataset,
    experiment_name: str,
    train_cfg: TrainingConfig = None,
    lora_cfg: LoRAConfig = None,
    resume_from: str = None,   # for loading a COMPLETED checkpoint (e.g. english->few-shot)
) -> dict:
    """
    Train a detoxification model (Section 4.2).
    method: 'lora' or 'full_ft'

    T4 GPU features:
    - gradient_checkpointing reduces peak VRAM ~40%
    - auto-resumes from the latest mid-training checkpoint if session restarted
    - skips entirely if best/ already exists on Drive
    """
    if train_cfg is None: train_cfg = TrainingConfig()
    if lora_cfg  is None: lora_cfg  = LoRAConfig()

    output_dir = os.path.join(CHECKPOINTS_DIR, experiment_name)
    best_dir   = os.path.join(output_dir, "best")
    os.makedirs(output_dir, exist_ok=True)

    # ── Skip if already finished ───────────────────────────────────────────────
    if _is_already_complete(best_dir, method):
        print(f"  [SKIP] {experiment_name}: best checkpoint already on Drive at {best_dir}")
        results_path = os.path.join(RESULTS_DIR, f"{experiment_name}_train.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                return json.load(f)
        return {"experiment": experiment_name, "method": method,
                "checkpoint_dir": best_dir, "skipped": True,
                "training_time_hours": 0, "peak_gpu_memory_gb": 0}

    # ── WandB init ──────────────────────────────────────────────────────────────
    use_wandb = train_cfg.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb_kw = {
            "project": train_cfg.wandb_project,
            "name": experiment_name,
            "config": {
                "method": method,
                "lora_rank": lora_cfg.rank,
                "lora_alpha": lora_cfg.alpha,
                "lora_modules": lora_cfg.target_modules,
                "learning_rate": train_cfg.learning_rate,
                "batch_size": train_cfg.batch_size,
                "grad_accum": train_cfg.gradient_accumulation_steps,
                "max_epochs": train_cfg.max_epochs,
                "n_train": len(train_dataset) if train_dataset else 0,
                "gradient_checkpointing": True,
            },
            "reinit": True,
        }
        if train_cfg.wandb_entity:
            wandb_kw["entity"] = train_cfg.wandb_entity
        wandb.init(**wandb_kw)
        print(f"  WandB run: {experiment_name}")

    tracker = EfficiencyTracker()

    # ── Load model ──────────────────────────────────────────────────────────────
    if resume_from:
        # Loading a fully-trained checkpoint to continue fine-tuning (e.g. EN->ES)
        # is_trainable=True is critical: without it LoRA weights are frozen (requires_grad=False)
        # and few-shot training produces zero gradient updates on the adapters.
        model, tokenizer_m = load_trained_model(resume_from, method, is_trainable=True)
    else:
        model, tokenizer_m = load_model_for_training(method, lora_cfg, train_cfg)

    # ── Enable input gradients for gradient checkpointing + LoRA ─────────────
    # Required when using gradient_checkpointing with PEFT/LoRA; harmless for full FT
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer_m, model=model, padding=True, label_pad_token_id=-100,
    )

    n_train    = len(train_dataset)
    is_few     = n_train <= 200
    lr         = train_cfg.few_shot_lr    if is_few else train_cfg.learning_rate
    epochs     = train_cfg.few_shot_epochs if is_few else train_cfg.max_epochs
    eff_batch  = train_cfg.batch_size
    grad_accum = train_cfg.gradient_accumulation_steps
    if is_few and n_train < train_cfg.batch_size:
        eff_batch  = max(1, n_train // 2)
        grad_accum = max(1, train_cfg.batch_size // eff_batch)
    # Evaluate once per epoch for few-shot: prevents premature early stopping.
    # (Old formula gave eval_steps=1 for 50-shot, triggering patience=3 after 3 steps.)
    # For full training, evaluate ~twice per epoch for responsiveness.
    steps_per_epoch = max(1, n_train // eff_batch)
    eval_steps = steps_per_epoch if is_few else max(1, steps_per_epoch // 2)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=eff_batch,
        per_device_eval_batch_size=train_cfg.eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,
        fp16=train_cfg.fp16 and torch.cuda.is_available(),
        bf16=getattr(train_cfg, "bf16", False) and torch.cuda.is_available(),
        # ── T4 GPU memory savings ──────────────────────────────────────────────
        gradient_checkpointing=True,           # ~40% less VRAM, ~20% slower
        gradient_checkpointing_kwargs={"use_reentrant": False},  # avoids LoRA warnings
        dataloader_num_workers=2,
        # ── Checkpoint strategy (on Drive) ────────────────────────────────────
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,                    # keep last 3 mid-training checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        logging_steps=max(1, eval_steps // 2),
        report_to="wandb" if use_wandb else "none",
        seed=train_cfg.seed,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=train_cfg.early_stopping_patience)],
    )

    # ── Auto-resume from mid-training checkpoint (handles Colab disconnects) ──
    mid_ckpt = _find_latest_checkpoint(output_dir)
    if mid_ckpt:
        print(f"  [RESUME] Found mid-training checkpoint: {mid_ckpt}")
    else:
        print(f"  Starting fresh: {experiment_name} | method={method} | n={n_train} | epochs={epochs} | lr={lr}")

    print(f"  gradient_checkpointing=ON  |  batch={eff_batch}x{grad_accum}={eff_batch*grad_accum} effective")

    tracker.start()
    train_result = trainer.train(resume_from_checkpoint=mid_ckpt)
    tracker.stop()

    # ── Save best model to Drive ──────────────────────────────────────────────
    save_model(model, tokenizer_m, best_dir, method)

    efficiency = tracker.summary(model, best_dir)
    eval_result = trainer.evaluate()

    results = {
        "experiment": experiment_name,
        "method": method,
        "n_train": n_train,
        "train_loss": train_result.training_loss,
        "val_loss": eval_result.get("eval_loss"),
        "checkpoint_dir": best_dir,
        "resumed_from": mid_ckpt,
        **efficiency,
    }

    results_path = os.path.join(RESULTS_DIR, f"{experiment_name}_train.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if use_wandb:
        wandb.log({
            "final_train_loss": results["train_loss"],
            "final_val_loss":   results.get("val_loss"),
            "training_time_h":  efficiency.get("training_time_hours", 0),
            "peak_gpu_gb":      efficiency.get("peak_gpu_memory_gb", 0),
        })
        wandb.finish()

    print(f"  Done. train_loss={results['train_loss']:.4f} | "
          f"time={efficiency['training_time_hours']:.2f}h | "
          f"GPU={efficiency['peak_gpu_memory_gb']:.1f}GB")
    return results


def train_english_baseline(method, train_dataset, val_dataset, train_cfg=None, lora_cfg=None):
    """Stage 1: Train on English data (~8K pairs)."""
    return train_model(method=method, train_dataset=train_dataset, val_dataset=val_dataset,
                       experiment_name=f"english_{method}", train_cfg=train_cfg, lora_cfg=lora_cfg)


def train_few_shot(method, lang, n_shots, train_dataset, val_dataset, english_checkpoint,
                   train_cfg=None, lora_cfg=None):
    """Stage 2: Few-shot adaptation for a target language (ES or HI)."""
    return train_model(method=method, train_dataset=train_dataset, val_dataset=val_dataset,
                       experiment_name=f"fewshot_{lang}_{n_shots}_{method}",
                       train_cfg=train_cfg, lora_cfg=lora_cfg, resume_from=english_checkpoint)


def train_ablation(lang, n_shots, rank, alpha, target_modules, module_name,
                   train_dataset, val_dataset, train_cfg=None):
    """LoRA ablation: vary r, alpha, target modules (100-shot)."""
    if train_cfg is None:
        train_cfg = TrainingConfig()
    # Ablation runs go to a separate WandB project to keep main project clean
    import dataclasses
    ablation_cfg = dataclasses.replace(train_cfg, wandb_project=train_cfg.ablation_wandb_project)
    _lora_cfg = LoRAConfig(rank=rank, alpha=alpha, target_modules=target_modules)
    return train_model(method="lora", train_dataset=train_dataset, val_dataset=val_dataset,
                       experiment_name=f"ablation_{lang}_{n_shots}shot_r{rank}_a{alpha}_{module_name}",
                       train_cfg=ablation_cfg, lora_cfg=_lora_cfg)


# ── Convenience: show all checkpoints on Drive ─────────────────────────────────
def show_checkpoint_status():
    """Print status of all experiments on Drive."""
    print(f"\nCheckpoint status: {CHECKPOINTS_DIR}")
    if not os.path.exists(CHECKPOINTS_DIR):
        print("  (none yet)")
        return
    for exp in sorted(os.listdir(CHECKPOINTS_DIR)):
        exp_dir = os.path.join(CHECKPOINTS_DIR, exp)
        best    = os.path.join(exp_dir, "best")
        mid     = _find_latest_checkpoint(exp_dir)
        if os.path.exists(best):
            size_mb = sum(os.path.getsize(os.path.join(best, f)) for f in os.listdir(best)
                         if os.path.isfile(os.path.join(best, f))) / 1e6
            print(f"  [COMPLETE] {exp:<45} best/ ({size_mb:.0f}MB)")
        elif mid:
            step = mid.split("-")[-1]
            print(f"  [PARTIAL ] {exp:<45} checkpoint-{step} (resume available)")
        else:
            print(f"  [EMPTY   ] {exp}")

show_checkpoint_status()
print("\nTraining pipeline loaded. gradient_checkpointing=ON for all runs.")

# ============================================================================
# Cell 8: Evaluator -- inline evaluator.py
# ============================================================================
# Primary metrics (Section 4.3):
#   T  -- Toxicity score via textdetox/xlmr-large-toxicity-classifier-v2 (lower=better)
#   S  -- Semantic Similarity via LaBSE cosine                       (higher=better)
#   FL -- Fluency via language-specific GPT-2 perplexity             (lower PPL = better)
#   BLEU -- sacrebleu (only when neutral references available)
#   chrF -- character n-gram F-score (sacrebleu, better for Hindi/morphologically rich langs)
#
# Composite metric:
#   J2 -- Joint Score (2-component):  mean(STA_i × SIM_i)
#   J3 -- Joint Score (3-component):  mean(STA_i × SIM_i × FL_norm_i)
#          STA_i = 1 if tox_i < 0.5, else 0  (per-sample detox success)
#          FL_norm_i = max(0, 1 - log(PPL_i)/log(1000))  (PPL -> [0,1])
#   Reference: TextDetox 2024 shared task evaluation protocol
#
# evaluate()       -- requires {toxic, neutral, prediction}  (ref-based: BLEU, chrF)
# evaluate_noref() -- requires toxic + predictions only      (no BLEU/chrF)

import math
import random
from typing import Optional
import pandas as pd


# ── Fluency normalization & J-Score ──────────────────────────────────────────

def normalize_fluency(ppl_values: list, base_ppl: float = 1000.0) -> list:
    """
    Convert perplexity scores to normalised fluency in [0, 1].
    FL_norm = max(0, 1 - log(PPL) / log(base_ppl))
    Intuition:
      PPL ~  10 (very fluent)   -> FL_norm ~ 0.90
      PPL ~  50 (natural text)  -> FL_norm ~ 0.78
      PPL ~ 200 (disfluent)     -> FL_norm ~ 0.62
      PPL >= 1000               -> FL_norm =  0.00
    """
    result = []
    for ppl in ppl_values:
        if ppl is None or ppl <= 0:
            result.append(None)
        else:
            result.append(max(0.0, 1.0 - math.log(ppl) / math.log(base_ppl)))
    return result


def compute_j_score(
    tox_scores: list,
    sim_scores: list,
    fl_norm_scores: list = None,
    threshold: float = 0.5,
) -> dict:
    """
    Joint Score (J-Score) -- composite detoxification metric.

    J2 = mean_i( STA_i × SIM_i )
    J3 = mean_i( STA_i × SIM_i × FL_norm_i )   (requires fluency)

    where STA_i = 1 if tox_scores[i] < threshold else 0.

    J2 and J3 both lie in [0, 1]; higher = better overall detoxification.
    A perfect run (all detoxified, all semantically identical, all fluent) = 1.0.
    """
    n = len(tox_scores)
    sta = [1.0 if t < threshold else 0.0 for t in tox_scores]
    j2_vals = [sta[i] * sim_scores[i] for i in range(n)]
    metrics = {"j2_score": round(float(np.mean(j2_vals)), 4)}

    if fl_norm_scores is not None:
        valid_fl = [fl_norm_scores[i] for i in range(n) if fl_norm_scores[i] is not None]
        if len(valid_fl) == n:
            j3_vals = [sta[i] * sim_scores[i] * fl_norm_scores[i] for i in range(n)]
            metrics["j3_score"] = round(float(np.mean(j3_vals)), 4)

    return metrics


# ── Scorer classes ─────────────────────────────────────────────────────────────

class ToxicityScorer:
    def __init__(self, model_name="textdetox/xlmr-large-toxicity-classifier-v2"):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=model_name, top_k=None,
                             device=0 if torch.cuda.is_available() else -1)

    def score(self, texts: List[str]) -> List[float]:
        """
        Robust label extraction for multiple label conventions:
          - textdetox/xlmr-large-toxicity-classifier-v2: id2label=None -> "LABEL_0"/"LABEL_1"
            (TextDetox convention: LABEL_0=neutral, LABEL_1=toxic)
          - unitary/multilingual-toxic-xlm-roberta:       labels "toxic" / "non_toxic"
          - Other models: explicit "toxic" / "neutral" labels
        """
        scores = []
        for out in self.pipe(texts, batch_size=32, truncation=True, max_length=512):
            label_map = {x["label"].lower(): x["score"] for x in out}
            if "toxic" in label_map:
                # Explicit toxic label (most direct)
                tox = label_map["toxic"]
            elif "neutral" in label_map:
                # Explicit neutral label: infer toxic score
                tox = 1.0 - label_map["neutral"]
            elif "non_toxic" in label_map:
                # unitary model convention
                tox = 1.0 - label_map["non_toxic"]
            elif "label_1" in label_map:
                # Generic LABEL_0/LABEL_1 (model has no id2label in config).
                # TextDetox convention: LABEL_0=neutral, LABEL_1=toxic.
                tox = label_map["label_1"]
            elif "label_0" in label_map:
                # Fallback if only LABEL_0 present
                tox = label_map["label_0"]
            else:
                tox = 0.5  # truly unknown format
            scores.append(float(tox))
        return scores


class SemanticSimilarity:
    def __init__(self, model_name="sentence-transformers/LaBSE"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def score(self, src_texts: List[str], tgt_texts: List[str]) -> List[float]:
        src_emb = self.model.encode(src_texts, batch_size=64, show_progress_bar=False)
        tgt_emb = self.model.encode(tgt_texts, batch_size=64, show_progress_bar=False)
        sims = []
        for s, t in zip(src_emb, tgt_emb):
            denom = (np.linalg.norm(s) * np.linalg.norm(t))
            sims.append(float(np.dot(s, t) / denom) if denom > 0 else 0.0)
        return sims


class FluencyScorer:
    def __init__(self, lang="en", fluency_models=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer as AT
        if fluency_models is None:
            fluency_models = {"en": "gpt2", "es": "datificate/gpt2-small-spanish",
                              "hi": "surajp/gpt2-hindi"}
        model_name = fluency_models.get(lang, "gpt2")
        try:
            self.tok   = AT.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.ok = True
        except Exception as e:
            print(f"  FluencyScorer ({lang}): could not load {model_name}: {e}")
            self.ok = False

    def perplexity(self, texts: List[str]) -> List[Optional[float]]:
        if not self.ok:
            return [None] * len(texts)
        results = []
        for text in texts:
            try:
                enc = self.tok(text, return_tensors="pt", truncation=True, max_length=256)
                if torch.cuda.is_available():
                    enc = {k: v.cuda() for k, v in enc.items()}
                with torch.no_grad():
                    loss = self.model(**enc, labels=enc["input_ids"]).loss.item()
                results.append(math.exp(loss))
            except Exception:
                results.append(None)
        return results


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(values: List[float], n=1000, ci=0.95) -> dict:
    if not values:
        return {"mean": 0, "lower": 0, "upper": 0}
    rng = random.Random(42)
    boot_means = [np.mean(rng.choices(values, k=len(values))) for _ in range(n)]
    alpha = 1 - ci
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return {"mean": round(float(np.mean(values)), 4),
            "lower": round(float(lo), 4), "upper": round(float(hi), 4)}


def save_evaluation(metrics: dict, name: str):
    path = os.path.join(RESULTS_DIR, f"{name}_eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)


# ── Main Evaluator ────────────────────────────────────────────────────────────

class DetoxEvaluator:
    def __init__(self, lang="en"):
        self.lang = lang
        self.tox_scorer  = ToxicityScorer()
        self.sim_scorer  = SemanticSimilarity()
        self._flu_scorer = None
        self.eval_cfg    = EvalConfig()

    def _fluency(self):
        if self._flu_scorer is None:
            self._flu_scorer = FluencyScorer(self.lang, self.eval_cfg.fluency_models)
        return self._flu_scorer

    def evaluate(self, results: List[dict], compute_fluency=True,
                 compute_bootstrap=True, compute_errors=True) -> dict:
        """
        Evaluate WITH references.
        Computes: T, STA, S, FL/PPL, BLEU, chrF, J2, J3, Bootstrap CI.
        """
        preds  = [r["prediction"] for r in results]
        inputs = [r["toxic"]      for r in results]
        refs   = [r["neutral"]    for r in results]

        tox = self.tox_scorer.score(preds)
        sim = self.sim_scorer.score(inputs, preds)
        ppl = self._fluency().perplexity(preds) if compute_fluency else [None] * len(preds)

        # BLEU
        from sacrebleu.metrics import BLEU, CHRF
        bleu_score = BLEU(effective_order=True).corpus_score(preds, [refs]).score if refs else 0.0

        # chrF -- character n-gram F-score (better for Hindi & morphologically rich langs)
        chrf_score = CHRF().corpus_score(preds, [refs]).score if refs else 0.0

        # Fluency normalisation & J-Score
        fl_norm = normalize_fluency(ppl) if compute_fluency else None
        j_scores = compute_j_score(tox, sim, fl_norm, self.eval_cfg.toxicity_threshold)

        metrics = {
            "language":   self.lang,
            "n_samples":  len(preds),
            # Toxicity
            "toxicity_mean":  round(float(np.mean(tox)), 4),
            "sta":            round(sum(t < self.eval_cfg.toxicity_threshold for t in tox) / len(tox), 4),
            "toxicity_pct_below_threshold": round(100 * sum(t < self.eval_cfg.toxicity_threshold for t in tox) / len(tox), 2),
            # Similarity
            "similarity_mean": round(float(np.mean(sim)), 4),
            "similarity_pct_above_threshold": round(100 * sum(s > self.eval_cfg.similarity_threshold for s in sim) / len(sim), 2),
            # Fluency
            "fluency_ppl_mean": round(float(np.mean([p for p in ppl if p is not None])), 2) if any(p for p in ppl) else None,
            "fluency_norm_mean": round(float(np.mean([f for f in fl_norm if f is not None])), 4) if fl_norm and any(f is not None for f in fl_norm) else None,
            # Reference-based
            "bleu":  round(bleu_score, 2),
            "chrf":  round(chrf_score, 2),
            # Joint Score (composite)
            **j_scores,
        }

        if compute_bootstrap:
            metrics["toxicity_ci"]   = bootstrap_ci(tox)
            metrics["similarity_ci"] = bootstrap_ci(sim)
            j2_vals = [(1.0 if t < self.eval_cfg.toxicity_threshold else 0.0) * s for t, s in zip(tox, sim)]
            metrics["j2_ci"] = bootstrap_ci(j2_vals)

        return metrics

    def evaluate_noref(self, toxic_texts: List[str], predictions: List[str],
                       compute_fluency=True, compute_bootstrap=True) -> dict:
        """
        Evaluate WITHOUT references.
        Computes: T, STA, S, FL/PPL, J2, J3. No BLEU/chrF (no refs).
        """
        tox = self.tox_scorer.score(predictions)
        sim = self.sim_scorer.score(toxic_texts, predictions)
        ppl = self._fluency().perplexity(predictions) if compute_fluency else [None] * len(predictions)

        fl_norm = normalize_fluency(ppl) if compute_fluency else None
        j_scores = compute_j_score(tox, sim, fl_norm, self.eval_cfg.toxicity_threshold)

        metrics = {
            "language":   self.lang,
            "n_samples":  len(predictions),
            # Toxicity
            "toxicity_mean":  round(float(np.mean(tox)), 4),
            "sta":            round(sum(t < self.eval_cfg.toxicity_threshold for t in tox) / len(tox), 4),
            "toxicity_pct_below_threshold": round(100 * sum(t < self.eval_cfg.toxicity_threshold for t in tox) / len(tox), 2),
            # Similarity
            "similarity_mean": round(float(np.mean(sim)), 4),
            "similarity_pct_above_threshold": round(100 * sum(s > self.eval_cfg.similarity_threshold for s in sim) / len(sim), 2),
            # Fluency
            "fluency_ppl_mean": round(float(np.mean([p for p in ppl if p is not None])), 2) if any(p for p in ppl) else None,
            "fluency_norm_mean": round(float(np.mean([f for f in fl_norm if f is not None])), 4) if fl_norm and any(f is not None for f in fl_norm) else None,
            # No references
            "bleu": None,
            "chrf": None,
            # Joint Score (composite)
            **j_scores,
        }

        if compute_bootstrap:
            metrics["toxicity_ci"]   = bootstrap_ci(tox)
            metrics["similarity_ci"] = bootstrap_ci(sim)
            j2_vals = [(1.0 if t < self.eval_cfg.toxicity_threshold else 0.0) * s for t, s in zip(tox, sim)]
            metrics["j2_ci"] = bootstrap_ci(j2_vals)

        return metrics


# ── Results table ─────────────────────────────────────────────────────────────

def format_results_table(metrics_list: List[dict], title: str = "") -> str:
    if not metrics_list:
        return "No results."
    rows = []
    for m in metrics_list:
        rows.append({
            "Experiment":  m.get("experiment", m.get("method", "?")),
            "Lang":        m.get("language", m.get("lang", "?")),
            "STA%":        m.get("toxicity_pct_below_threshold", "-"),
            "Sim":         m.get("similarity_mean", "-"),
            "PPL":         m.get("fluency_ppl_mean", "-"),
            "BLEU":        m.get("bleu", "-"),
            "chrF":        m.get("chrf", "-"),
            "J2":          m.get("j2_score", "-"),
            "J3":          m.get("j3_score", "-"),
        })
    df = pd.DataFrame(rows)
    header = f"\n{'='*80}\n{title}\n{'='*80}\n" if title else ""
    return header + df.to_string(index=False)


print("Evaluator loaded.")
print("Metrics: Toxicity (T), STA, Similarity (S), Fluency (PPL), BLEU, chrF, J2, J3")
print("  J2 = mean(STA_i x SIM_i)            [always available]")
print("  J3 = mean(STA_i x SIM_i x FL_norm_i) [when fluency enabled]")

# ============================================================================
# Cell 9: Experiment A -- Baseline Experiments (EN + ES + HI)
# ============================================================================
# Runs Delete and Identity baselines on:
#   - English test set (with neutral refs -> BLEU computed)
#   - Spanish test set (no refs -> evaluate_noref)
#   - Hindi test set   (no refs -> evaluate_noref)
# Results saved to RESULTS_DIR/baseline_*.json

print("=" * 60)
print("EXPERIMENT A: Baseline Experiments")
print("=" * 60)

baseline_results = {}

# ── English baselines (has neutral refs -> use evaluate()) ────────────────────
for bname in ["delete", "identity"]:
    print(f"\n  Running {bname} baseline on EN ...")
    preds = run_baseline(bname, data["en_test_ref_raw"], lang="en")
    evaluator = DetoxEvaluator(lang="en")
    metrics = evaluator.evaluate(preds, compute_fluency=True, compute_bootstrap=True)
    metrics["experiment"] = f"baseline_{bname}"
    metrics["lang"] = "en"
    save_evaluation(metrics, f"baseline_{bname}_en")
    baseline_results[f"{bname}_en"] = metrics
    print(f"    Tox={metrics['toxicity_mean']:.3f}  Sim={metrics['similarity_mean']:.3f}  BLEU={metrics['bleu']:.1f}")

# ── Spanish baselines (no refs -> evaluate_noref()) ───────────────────────────
for bname in ["delete", "identity"]:
    print(f"\n  Running {bname} baseline on ES ...")
    if bname == "delete":
        _del_es = DeleteBaseline(lang="es")
        es_preds = [_del_es.detoxify(t) for t in data["es_test_noref_raw"]]
    else:
        es_preds = list(data["es_test_noref_raw"])
    evaluator = DetoxEvaluator(lang="es")
    metrics = evaluator.evaluate_noref(data["es_test_noref_raw"], es_preds,
                                       compute_fluency=True, compute_bootstrap=True)
    metrics["experiment"] = f"baseline_{bname}"
    metrics["lang"] = "es"
    save_evaluation(metrics, f"baseline_{bname}_es")
    baseline_results[f"{bname}_es"] = metrics
    print(f"    Tox={metrics['toxicity_mean']:.3f}  Sim={metrics['similarity_mean']:.3f}")

# ── Hindi baselines (no refs -> evaluate_noref()) ─────────────────────────────
for bname in ["delete", "identity"]:
    print(f"\n  Running {bname} baseline on HI ...")
    if bname == "delete":
        _del_hi = DeleteBaseline(lang="hi")
        hi_preds = [_del_hi.detoxify(t) for t in data["hi_test_noref_raw"]]
    else:
        hi_preds = list(data["hi_test_noref_raw"])
    evaluator = DetoxEvaluator(lang="hi")
    metrics = evaluator.evaluate_noref(data["hi_test_noref_raw"], hi_preds,
                                       compute_fluency=True, compute_bootstrap=True)
    metrics["experiment"] = f"baseline_{bname}"
    metrics["lang"] = "hi"
    save_evaluation(metrics, f"baseline_{bname}_hi")
    baseline_results[f"{bname}_hi"] = metrics
    print(f"    Tox={metrics['toxicity_mean']:.3f}  Sim={metrics['similarity_mean']:.3f}")

# ── Summary table ─────────────────────────────────────────────────────────────
print(format_results_table(list(baseline_results.values()), title="Baseline Results (Table 2)"))
print("\nBaseline experiments done. Results saved to:", RESULTS_DIR)

# ============================================================================
# Cell 10: Experiment B -- English Training (LoRA + Full FT)
# ============================================================================
# Stage 1 of the multi-stage training pipeline (Section 4.2):
#   - Trains both LoRA (r=32, alpha=64, Q+V) and Full FT on English data
#   - English dataset: ParaDeHate(8276) + paradetox EN(400) -> ~7K train pairs
#   - Saves best checkpoint to CHECKPOINTS_DIR/english_{method}/best/
#   - Evaluates on EN val set (with refs -> BLEU) and EN test set
#
# Expected runtime: LoRA ~1.5h, Full FT ~3h on T4 GPU

print("=" * 60)
print("EXPERIMENT B: English Training (LoRA + Full FT)")
print("=" * 60)

english_train_results = {}
english_checkpoints   = {}

for method in ["lora"]:  # full_ft skipped (OOM on 16GB MIG)
    print(f"\n  Training {method.upper()} on English data ...")
    print(f"  Train: {len(data['en_train'])} | Val: {len(data['en_val'])}")

    result = train_english_baseline(
        method=method,
        train_dataset=data["en_train"],
        val_dataset=data["en_val"],
        train_cfg=train_cfg,
        lora_cfg=lora_cfg,
    )
    english_train_results[method] = result
    english_checkpoints[method]   = result["checkpoint_dir"]
    print(f"  [{method}] train_loss={result['train_loss']:.4f} | val_loss={result['val_loss']:.4f}")
    print(f"  [{method}] time={result['training_time_hours']:.2f}h | GPU={result['peak_gpu_memory_gb']:.1f}GB")
    if method == "lora":
        print(f"  [{method}] trainable={result.get('trainable_params', 0):,} params ({result.get('trainable_pct', 0):.2f}%)")

# ── Evaluate on EN test set (has refs) ────────────────────────────────────────
print("\n  Evaluating on English test set ...")
en_eval_results = {}
for method, ckpt in english_checkpoints.items():
    model_m, tok_m = load_trained_model(ckpt, method)
    preds = generate_detoxified(model_m, tok_m, [p["toxic"] for p in data["en_test_ref_raw"]])
    result_pairs = [{"toxic": p["toxic"], "neutral": p["neutral"], "prediction": pred}
                    for p, pred in zip(data["en_test_ref_raw"], preds)]
    evaluator = DetoxEvaluator(lang="en")
    metrics = evaluator.evaluate(result_pairs, compute_fluency=True, compute_bootstrap=True)
    metrics["experiment"] = f"english_{method}"
    metrics["lang"] = "en"
    save_evaluation(metrics, f"english_{method}_en")
    en_eval_results[method] = metrics
    del model_m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  [{method}] Tox={metrics['toxicity_mean']:.3f} Sim={metrics['similarity_mean']:.3f} BLEU={metrics['bleu']:.1f}")

print(format_results_table(list(en_eval_results.values()), title="English Training Results (Table 2)"))
print(f"\nCheckpoints: {english_checkpoints}")
print("English training done.")

# ============================================================================
# Cell 11: Experiment C -- Zero-Shot Transfer (EN -> ES + EN -> HI)
# ============================================================================
# RQ1: Can English-trained LoRA/FullFT transfer zero-shot to ES and HI?
#   - Evaluates BOTH LoRA and Full FT checkpoints (zero additional training)
#   - Run generate_detoxified on ES + HI test (600 toxic inputs each, no refs)
#   - Evaluate with evaluate_noref() (Tox, Sim, Fluency, J2; no BLEU)
#
# Produces the "0-shot" columns in Table 3 for both methods.
# NOTE: Full FT is evaluated zero-shot here to complete the LoRA vs Full FT
#       comparison for RQ1 (otherwise the comparison matrix is incomplete).

print("=" * 60)
print("EXPERIMENT C: Zero-Shot Transfer EN -> ES / HI (RQ1)")
print("=" * 60)

# Requires english_checkpoints from Cell 10.
# If re-running standalone, set paths manually:
# english_checkpoints = {
#     "lora":    "/content/drive/MyDrive/poly_detox/checkpoints/english_lora/best",
#     "full_ft": "/content/drive/MyDrive/poly_detox/checkpoints/english_full_ft/best",
# }

zeroshot_results = {}

# Evaluate both LoRA and Full FT zero-shot (RQ1 comparison)
for method in ["lora"]:  # full_ft skipped (OOM on 16GB MIG)
    ckpt = english_checkpoints.get(method)
    if ckpt is None or not os.path.exists(ckpt):
        print(f"\n  [SKIP] {method}: checkpoint not found at {ckpt}. "
              "Train with Cell 10 first.")
        continue

    print(f"\n{'─'*50}")
    print(f"  Method: {method} (zero-shot, no target-language training)")
    print(f"{'─'*50}")

    model_z, tok_z = load_trained_model(ckpt, method)

    for lang in data_cfg.target_langs:
        lang_test_noref = data[f"{lang}_test_noref_raw"]
        print(f"\n  Zero-shot {method} -> {lang.upper()} "
              f"({len(lang_test_noref)} toxic inputs) ...")

        preds = generate_detoxified(model_z, tok_z, lang_test_noref,
                                    max_length=train_cfg.max_gen_length,
                                    num_beams=train_cfg.num_beams)

        evaluator = DetoxEvaluator(lang=lang)
        metrics = evaluator.evaluate_noref(lang_test_noref, preds,
                                           compute_fluency=True,
                                           compute_bootstrap=True)
        metrics["experiment"] = f"zeroshot_{lang}_{method}"
        metrics["method"] = f"{method} (0-shot)"
        metrics["lang"] = lang
        metrics["n_shots"] = 0
        save_evaluation(metrics, f"zeroshot_{lang}_{method}")
        zeroshot_results[f"{lang}_{method}"] = metrics

        print(f"  [{lang}] Tox={metrics['toxicity_mean']:.3f}  "
              f"Sim={metrics['similarity_mean']:.3f}  "
              f"PPL={metrics['fluency_ppl_mean']}")
        print(f"  [{lang}] Tox<0.5: {metrics['toxicity_pct_below_threshold']:.1f}%  "
              f"Sim>0.75: {metrics['similarity_pct_above_threshold']:.1f}%  "
              f"J2={metrics.get('j2_score', 'n/a')}")

    del model_z
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(format_results_table(list(zeroshot_results.values()),
                           title="Zero-Shot Transfer Results (RQ1 -- Table 3)"))
print("\nZero-shot experiments done.")


# ============================================================================
# Cell 12: Experiment D -- Few-Shot Adaptation (RQ2 + RQ3)
# ============================================================================
# RQ2: How does sample efficiency scale with 50/100/200 shots?
# RQ3: LoRA vs Full FT at each shot count?
#
# For EACH (lang, n_shots, method) combination:
#   - Load English checkpoint as starting point
#   - Continue training on {lang}_{n_shots} diversity-sampled subset
#   - Evaluate on {lang}_test_noref (600 toxic inputs)
#   - Save metrics to RESULTS_DIR/fewshot_{lang}_{n_shots}_{method}_eval.json
#
# Languages: ES + HI
# Shot sizes: 50, 100, 200
# Methods: lora, full_ft
# Total runs: 2 x 3 x 2 = 12 experiments
# Expected runtime: ~4-6h total on T4 GPU

print("=" * 60)
print("EXPERIMENT D: Few-Shot Adaptation (RQ2 / RQ3)")
print("=" * 60)
print("12 runs: 2 langs x 3 shot sizes x 2 methods\n")

fewshot_results = {}

for lang in data_cfg.target_langs:
    lang_test_noref = data[f"{lang}_test_noref_raw"]
    lang_val        = data[f"{lang}_val"]
    evaluator       = DetoxEvaluator(lang=lang)

    for n_shots in data_cfg.few_shot_sizes:
        shot_key    = f"{lang}_{n_shots}shot"
        train_split = data[shot_key]
        print(f"\n  --- {lang.upper()} {n_shots}-shot ({len(train_split)} pairs) ---")

        for method in ["lora"]:  # full_ft skipped (OOM on 16GB MIG)
            en_ckpt = english_checkpoints.get(method,
                      f"{CHECKPOINTS_DIR}/english_{method}/best")
            print(f"  Training {method.upper()} from {en_ckpt} ...")

            train_res = train_few_shot(
                method=method,
                lang=lang,
                n_shots=n_shots,
                train_dataset=train_split,
                val_dataset=lang_val,
                english_checkpoint=en_ckpt,
                train_cfg=train_cfg,
                lora_cfg=lora_cfg,
            )
            ckpt = train_res["checkpoint_dir"]

            # Evaluate on no-ref test set
            model_f, tok_f = load_trained_model(ckpt, method)
            preds = generate_detoxified(model_f, tok_f, lang_test_noref,
                                        max_length=train_cfg.max_gen_length,
                                        num_beams=train_cfg.num_beams)

            metrics = evaluator.evaluate_noref(lang_test_noref, preds,
                                               compute_fluency=True, compute_bootstrap=True)
            metrics["experiment"] = f"fewshot_{lang}_{n_shots}_{method}"
            metrics["lang"]       = lang
            metrics["n_shots"]    = n_shots
            metrics["method"]     = method
            metrics["train_loss"] = train_res.get("train_loss")
            metrics["val_loss"]   = train_res.get("val_loss")
            metrics["training_time_hours"]  = train_res.get("training_time_hours")
            metrics["peak_gpu_memory_gb"]   = train_res.get("peak_gpu_memory_gb")
            metrics["trainable_params"]     = train_res.get("trainable_params")
            metrics["checkpoint_size_mb"]   = train_res.get("checkpoint_size_mb")

            save_evaluation(metrics, f"fewshot_{lang}_{n_shots}_{method}")
            fewshot_results[f"{lang}_{n_shots}_{method}"] = metrics

            del model_f
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"  [{lang} {n_shots}s {method}] "
                  f"Tox={metrics['toxicity_mean']:.3f}  "
                  f"Sim={metrics['similarity_mean']:.3f}  "
                  f"PPL={metrics['fluency_ppl_mean']}  "
                  f"t={metrics.get('training_time_hours', 0):.2f}h  "
                  f"GPU={metrics.get('peak_gpu_memory_gb', 0):.1f}GB")

print(format_results_table(list(fewshot_results.values()),
                           title="Few-Shot Adaptation Results (RQ2 / RQ3 -- Table 3)"))
print("\nFew-shot experiments done.")

# ============================================================================
# Cell 13: Experiment E -- LoRA Ablation Study (RQ4)
# ============================================================================
# RQ4: How do LoRA hyperparameters affect detoxification performance?
#
# Fixed: 100-shot, both ES and HI languages
# Vary one dimension at a time (others fixed to baseline r=32, a=64, qv):
#   - Rank:    r in {8, 16, 32*, 64}          (*=baseline)
#   - Alpha:   alpha in {16, 32, 64*, 128}     (*=baseline)
#   - Modules: qv* / all_attn / all_linear     (*=baseline)
#
# Total: (4 ranks + 4 alphas + 3 modules - 3 duplicates of baseline) * 2 langs
#      = 8 unique configs x 2 langs = 16 runs
# Expected runtime: ~2-3h on T4 GPU

print("=" * 60)
print("EXPERIMENT E: LoRA Ablation Study (RQ4)")
print("=" * 60)

ablation_results = {}

# Ablation configurations (one dimension varied at a time)
ABLATION_CONFIGS = []

# Vary rank (alpha=64, modules=qv fixed)
for r in lora_cfg.ablation_ranks:
    ABLATION_CONFIGS.append({
        "rank": r, "alpha": 64, "modules": ["q", "v"],
        "module_name": "qv", "vary": "rank",
    })

# Vary alpha (rank=32, modules=qv fixed)  -- skip alpha=64 already in rank sweep
for a in lora_cfg.ablation_alphas:
    if a == 64:
        continue   # already covered by baseline in rank sweep
    ABLATION_CONFIGS.append({
        "rank": 32, "alpha": a, "modules": ["q", "v"],
        "module_name": "qv", "vary": "alpha",
    })

# Vary target modules (rank=32, alpha=64 fixed) -- skip qv already covered
for mname, mods in lora_cfg.ablation_modules.items():
    if mname == "qv":
        continue   # already covered by baseline in rank sweep
    ABLATION_CONFIGS.append({
        "rank": 32, "alpha": 64, "modules": mods,
        "module_name": mname, "vary": "modules",
    })

print(f"  Running {len(ABLATION_CONFIGS)} configs x {len(data_cfg.target_langs)} langs "
      f"= {len(ABLATION_CONFIGS) * len(data_cfg.target_langs)} experiments\n")

for lang in data_cfg.target_langs:
    lang_test_noref = data[f"{lang}_test_noref_raw"]
    lang_val        = data[f"{lang}_val"]
    train_split     = data[f"{lang}_100shot"]
    evaluator       = DetoxEvaluator(lang=lang)

    print(f"\n  --- Ablation on {lang.upper()} (100-shot) ---")
    for cfg_ab in ABLATION_CONFIGS:
        r     = cfg_ab["rank"]
        a     = cfg_ab["alpha"]
        mods  = cfg_ab["modules"]
        mname = cfg_ab["module_name"]
        label = f"r{r}_a{a}_{mname}"

        print(f"  [{lang}] r={r}, alpha={a}, modules={mname} ...")

        train_res = train_ablation(
            lang=lang,
            n_shots=100,
            rank=r,
            alpha=a,
            target_modules=mods,
            module_name=mname,
            train_dataset=train_split,
            val_dataset=lang_val,
            train_cfg=train_cfg,
        )
        ckpt = train_res["checkpoint_dir"]

        model_ab, tok_ab = load_trained_model(ckpt, "lora")
        preds = generate_detoxified(model_ab, tok_ab, lang_test_noref,
                                    max_length=train_cfg.max_gen_length,
                                    num_beams=train_cfg.num_beams)

        metrics = evaluator.evaluate_noref(lang_test_noref, preds,
                                           compute_fluency=False, compute_bootstrap=True)
        metrics["experiment"]  = f"ablation_{lang}_100shot_{label}"
        metrics["lang"]        = lang
        metrics["n_shots"]     = 100
        metrics["lora_rank"]   = r
        metrics["lora_alpha"]  = a
        metrics["lora_modules"]= mname
        metrics["vary"]        = cfg_ab["vary"]
        metrics["train_loss"]  = train_res.get("train_loss")
        metrics["trainable_params"]   = train_res.get("trainable_params")
        metrics["checkpoint_size_mb"] = train_res.get("checkpoint_size_mb")

        save_evaluation(metrics, f"ablation_{lang}_100shot_{label}")
        ablation_results[f"{lang}_{label}"] = metrics

        del model_ab
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"    Tox={metrics['toxicity_mean']:.3f}  "
              f"Sim={metrics['similarity_mean']:.3f}  "
              f"params={metrics.get('trainable_params', '?'):,}")

print(format_results_table(list(ablation_results.values()),
                           title="LoRA Ablation Results (RQ4)"))
print("\nAblation experiments done.")

# ============================================================================
# Cell 14: Visualization & Final Results Summary
# ============================================================================
# Generates all result plots and prints the final summary tables:
#   Fig 1 -- Few-shot sample efficiency (Toxicity vs n_shots, ES + HI)
#   Fig 2 -- LoRA vs Full FT comparison (Toxicity + Similarity, bar chart)
#   Fig 3 -- LoRA ablation: rank vs Toxicity (ES + HI)
#   Fig 4 -- LoRA ablation: alpha vs Toxicity (ES + HI)
#   Fig 5 -- LoRA ablation: module vs Toxicity (bar chart)
#   Fig 6 -- Efficiency: Training time and GPU memory (LoRA vs Full FT)
#   Table summary -- all experiments side by side

import matplotlib.pyplot as plt
import seaborn as sns
import glob as _glob
import os

sns.set_theme(style="whitegrid", palette="colorblind")
FIGS_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Load all saved eval JSONs ─────────────────────────────────────────────────
def load_all_results(results_dir):
    records = []
    for path in sorted(_glob.glob(os.path.join(results_dir, "*_eval.json"))):
        with open(path) as f:
            r = json.load(f)
        r["_file"] = os.path.basename(path)
        records.append(r)
    return records

all_results = load_all_results(RESULTS_DIR)
df_all = pd.DataFrame(all_results)
print(f"Loaded {len(df_all)} result files from {RESULTS_DIR}")
print(df_all[["_file", "lang", "toxicity_mean", "similarity_mean", "bleu"]].to_string(index=False))

# ── Fig 1: Few-shot sample efficiency ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric, ylabel, threshold in zip(
    axes,
    ["toxicity_mean", "similarity_mean"],
    ["Toxicity Score (lower=better)", "Semantic Similarity (higher=better)"],
    [0.5, 0.75],
):
    for lang, linestyle in [("es", "-"), ("hi", "--")]:
        for method, marker in [("lora", "o"), ("full_ft", "s")]:
            pts = []
            for n_shot in [0] + data_cfg.few_shot_sizes:
                if n_shot == 0:
                    key = f"zeroshot_{lang}_{method}"
                else:
                    key = f"fewshot_{lang}_{n_shot}_{method}"
                r = fewshot_results.get(key) or zeroshot_results.get(f"{lang}_{method}")
                if r:
                    pts.append((n_shot, r[metric]))
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker=marker, linestyle=linestyle,
                        label=f"{lang.upper()} {method}")
    ax.axhline(threshold, color="red", linestyle=":", alpha=0.5, label="threshold")
    ax.set_xlabel("Number of shots")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel.split('(')[0].strip()} vs Shot Count")
    ax.legend(fontsize=8)
    ax.set_xticks([0] + data_cfg.few_shot_sizes)

fig.suptitle("Fig 1: Few-Shot Sample Efficiency (EN→ES/HI)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "fig1_sample_efficiency.png"), dpi=150)
plt.show()

# ── Fig 2: LoRA vs Full FT (200-shot, ES + HI) ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric, title in zip(
    axes,
    ["toxicity_mean", "similarity_mean"],
    ["Toxicity (lower=better)", "Similarity (higher=better)"],
):
    groups, vals, colors = [], [], []
    for lang in data_cfg.target_langs:
        for method, color in [("lora", "#4878CF"), ("full_ft", "#6ACC65")]:
            r = fewshot_results.get(f"{lang}_200_{method}")
            if r:
                groups.append(f"{lang.upper()}\n{method}")
                vals.append(r[metric])
                colors.append(color)
    ax.bar(groups, vals, color=colors)
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " "))

fig.suptitle("Fig 2: LoRA vs Full FT at 200-shot (RQ3)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "fig2_lora_vs_fullft.png"), dpi=150)
plt.show()

# ── Fig 3 + 4: Ablation -- rank and alpha ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for row, (vary_key, x_vals, x_label) in enumerate([
    ("rank",  lora_cfg.ablation_ranks,  "LoRA Rank (r)"),
    ("alpha", lora_cfg.ablation_alphas, "LoRA Alpha"),
]):
    for col, lang in enumerate(data_cfg.target_langs):
        ax = axes[row][col]
        tox_vals, sim_vals = [], []
        for xv in x_vals:
            if vary_key == "rank":
                label = f"r{xv}_a64_qv"
            else:
                label = f"r32_a{xv}_qv"
            r = ablation_results.get(f"{lang}_{label}")
            tox_vals.append(r["toxicity_mean"] if r else None)
            sim_vals.append(r["similarity_mean"] if r else None)
        ax.plot(x_vals, tox_vals, "o-", label="Toxicity", color="#D65F5F")
        ax.plot(x_vals, sim_vals, "s--", label="Similarity", color="#4878CF")
        ax.axhline(0.5, color="red",  linestyle=":", alpha=0.4)
        ax.axhline(0.75, color="blue", linestyle=":", alpha=0.4)
        ax.set_xlabel(x_label)
        ax.set_title(f"{x_label} ablation -- {lang.upper()}")
        ax.legend(fontsize=8)
        ax.set_xticks(x_vals)

fig.suptitle("Figs 3-4: LoRA Rank & Alpha Ablation (RQ4)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "fig34_ablation_rank_alpha.png"), dpi=150)
plt.show()

# ── Fig 5: Module ablation ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(data_cfg.target_langs), figsize=(10, 4))
if len(data_cfg.target_langs) == 1:
    axes = [axes]
module_names = list(lora_cfg.ablation_modules.keys())
for ax, lang in zip(axes, data_cfg.target_langs):
    tox_vals = []
    for mname in module_names:
        label = f"r32_a64_{mname}"
        r = ablation_results.get(f"{lang}_{label}")
        tox_vals.append(r["toxicity_mean"] if r else 0)
    ax.bar(module_names, tox_vals, color=["#4878CF", "#6ACC65", "#D65F5F"])
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.5, label="threshold 0.5")
    ax.set_title(f"Module ablation -- {lang.upper()}")
    ax.set_ylabel("Toxicity (lower=better)")
    ax.legend(fontsize=8)

fig.suptitle("Fig 5: Target Module Ablation (RQ4)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "fig5_ablation_modules.png"), dpi=150)
plt.show()

# ── Fig 6: Efficiency (LoRA vs Full FT) ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
methods = ["lora"]  # full_ft skipped (OOM on 16GB MIG)
times  = [english_train_results.get(m, {}).get("training_time_hours", 0) for m in methods]
gpus   = [english_train_results.get(m, {}).get("peak_gpu_memory_gb",  0) for m in methods]
axes[0].bar(methods, times, color=["#4878CF", "#D65F5F"])
axes[0].set_ylabel("Training Time (hours)")
axes[0].set_title("Training Time: LoRA vs Full FT (EN)")
axes[1].bar(methods, gpus, color=["#4878CF", "#D65F5F"])
axes[1].set_ylabel("Peak GPU Memory (GB)")
axes[1].set_title("GPU Memory: LoRA vs Full FT (EN)")

fig.suptitle("Fig 6: Efficiency Comparison (RQ3)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "fig6_efficiency.png"), dpi=150)
plt.show()

# ── Final summary table ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

summary_rows = []

# Baselines
for key, r in baseline_results.items():
    summary_rows.append({
        "Experiment": r.get("experiment", key), "Lang": r.get("lang", "?"),
        "Method": "baseline", "Shots": "-",
        "Toxicity": r.get("toxicity_mean"), "Tox<0.5%": r.get("toxicity_pct_below_threshold"),
        "Similarity": r.get("similarity_mean"), "BLEU": r.get("bleu"),
    })

# Zero-shot
for _zkey, r in zeroshot_results.items():
    summary_rows.append({
        "Experiment": r.get("experiment"), "Lang": r.get("lang"),
        "Method": r.get("method", "lora"), "Shots": 0,
        "Toxicity": r.get("toxicity_mean"), "Tox<0.5%": r.get("toxicity_pct_below_threshold"),
        "Similarity": r.get("similarity_mean"), "BLEU": "-",
    })

# Few-shot
for key, r in fewshot_results.items():
    summary_rows.append({
        "Experiment": r.get("experiment", key), "Lang": r.get("lang"),
        "Method": r.get("method"), "Shots": r.get("n_shots"),
        "Toxicity": r.get("toxicity_mean"), "Tox<0.5%": r.get("toxicity_pct_below_threshold"),
        "Similarity": r.get("similarity_mean"), "BLEU": "-",
    })

df_summary = pd.DataFrame(summary_rows).sort_values(["Lang", "Method", "Shots"])
print(df_summary.to_string(index=False))

# Save summary CSV
summary_csv = os.path.join(RESULTS_DIR, "all_results_summary.csv")
df_summary.to_csv(summary_csv, index=False)
print(f"\nSummary saved to: {summary_csv}")
print(f"Figures saved to: {FIGS_DIR}")
print("\nAll experiments complete!")

