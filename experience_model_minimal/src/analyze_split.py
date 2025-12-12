"""Comprehensive Train/Val Split Leakage and Overlap Analysis.

A scientifically rigorous tool for evaluating overfitting risk by detecting
various forms of data leakage between training and validation sets.

Theoretical Framework:
======================

1. EXACT DUPLICATES
   - Complete sample duplicates (critical leakage)
   - Prompt-only duplicates (same input, possibly different output)
   - Completion-only duplicates (same output pattern)

2. TASK/INSTRUCTION LEAKAGE  
   - Same task appearing in both splits means model can memorize
     task → trajectory mappings rather than learning dynamics

3. ENTITY LEAKAGE (Domain-Specific)
   - Same product IDs (ASINs) in both splits
   - Model can memorize product-specific transitions

4. N-GRAM OVERLAP ANALYSIS
   - Jaccard similarity on character/word n-grams
   - Measures surface-level pattern overlap
   - High overlap → model can rely on memorization

5. SEMANTIC SIMILARITY (Optional)
   - Embedding-based nearest neighbor analysis
   - Detects paraphrases and semantically similar samples

6. DISTRIBUTION METRICS
   - Token distribution divergence (KL, Jensen-Shannon)
   - Length distribution comparison
   - Vocabulary overlap

7. COMPRESSION-BASED SIMILARITY
   - Normalized Compression Distance (NCD)
   - Information-theoretic measure of redundancy

Usage:
    python src/analyze_split.py --train data/train.jsonl --val data/val.jsonl

    # With semantic analysis (requires sentence-transformers)
    python src/analyze_split.py --train data/train.jsonl --val data/val.jsonl --semantic

Output:
    - Detailed metrics for each leakage category
    - Overall "Overfitting Risk Score" (0-100)
    - Actionable recommendations
"""

import argparse
import gzip
import hashlib
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Sample:
    """A single data sample with extracted features."""
    raw: dict                           # Original JSON
    prompt: str                         # User message content
    completion: str                     # Assistant message content
    full_text: str                      # Concatenated for hashing
    instruction: str = ""               # Extracted task instruction
    entities: Set[str] = field(default_factory=set)  # ASINs, product IDs
    prompt_hash: str = ""               # MD5 hash for fast comparison
    completion_hash: str = ""
    full_hash: str = ""
    
    def __post_init__(self):
        self.prompt_hash = hashlib.md5(self.prompt.encode()).hexdigest()
        self.completion_hash = hashlib.md5(self.completion.encode()).hexdigest()
        self.full_hash = hashlib.md5(self.full_text.encode()).hexdigest()


@dataclass
class OverlapMetrics:
    """Container for all computed overlap metrics."""
    # Exact duplicates
    exact_duplicate_count: int = 0
    exact_duplicate_rate: float = 0.0
    prompt_duplicate_count: int = 0
    prompt_duplicate_rate: float = 0.0
    completion_duplicate_count: int = 0
    completion_duplicate_rate: float = 0.0
    
    # Instruction leakage
    instruction_overlap_count: int = 0
    instruction_overlap_rate: float = 0.0
    unique_train_instructions: int = 0
    unique_val_instructions: int = 0
    
    # Entity leakage
    entity_overlap_count: int = 0
    entity_overlap_rate: float = 0.0
    unique_train_entities: int = 0
    unique_val_entities: int = 0
    leaked_entities: List[str] = field(default_factory=list)
    
    # N-gram overlap
    ngram_metrics: Dict[str, Dict] = field(default_factory=dict)
    
    # Semantic similarity (optional)
    semantic_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Distribution metrics
    vocab_overlap_rate: float = 0.0
    kl_divergence: float = 0.0
    js_divergence: float = 0.0
    length_ks_statistic: float = 0.0
    
    # Compression-based
    ncd_score: float = 0.0
    
    # Overall risk
    risk_score: float = 0.0
    risk_level: str = "Unknown"
    risk_breakdown: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Data Loading
# =============================================================================

def load_samples(path: Path) -> List[Sample]:
    """Load and parse JSONL samples."""
    samples = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                
                # Handle chat format
                if "messages" in data:
                    prompt = data["messages"][0]["content"]
                    completion = data["messages"][1]["content"]
                else:
                    prompt = data.get("prompt", "")
                    completion = data.get("completion", "")
                
                full_text = prompt + "\n" + completion
                
                # Extract instruction
                instruction = extract_instruction(prompt)
                
                # Extract entities (ASINs)
                entities = extract_entities(full_text)
                
                sample = Sample(
                    raw=data,
                    prompt=prompt,
                    completion=completion,
                    full_text=full_text,
                    instruction=instruction,
                    entities=entities,
                )
                samples.append(sample)
                
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.warning(f"Error parsing line {line_num}: {e}")
    
    return samples


def extract_instruction(prompt: str) -> str:
    """Extract the task instruction from a prompt."""
    # Pattern: "Task: {instruction}\n\n"
    match = re.search(r"Task:\s*(.+?)(?:\n\n|$)", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Pattern: "Instruction:\n{instruction}\n\n"
    match = re.search(r"Instruction:\s*\n(.+?)(?:\n\n|\[)", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return ""


def extract_entities(text: str) -> Set[str]:
    """Extract entity identifiers from text (ASINs, product IDs)."""
    entities = set()
    
    # ASIN pattern (Amazon Standard Identification Number)
    asin_pattern = r"\b[B][0-9A-Z]{9}\b"
    entities.update(re.findall(asin_pattern, text))
    
    # Generic product ID pattern
    product_pattern = r"Product[:\s]+([A-Z0-9]+)"
    for match in re.finditer(product_pattern, text):
        entities.add(match.group(1))
    
    return entities


# =============================================================================
# Exact Duplicate Analysis
# =============================================================================

def compute_exact_duplicates(
    train_samples: List[Sample],
    val_samples: List[Sample],
) -> Tuple[int, int, int, List[int]]:
    """Compute exact duplicate counts.
    
    Returns:
        (full_duplicates, prompt_duplicates, completion_duplicates, duplicate_indices)
    """
    train_full_hashes = {s.full_hash for s in train_samples}
    train_prompt_hashes = {s.prompt_hash for s in train_samples}
    train_completion_hashes = {s.completion_hash for s in train_samples}
    
    full_dups = 0
    prompt_dups = 0
    completion_dups = 0
    duplicate_indices = []
    
    for i, sample in enumerate(val_samples):
        if sample.full_hash in train_full_hashes:
            full_dups += 1
            duplicate_indices.append(i)
        if sample.prompt_hash in train_prompt_hashes:
            prompt_dups += 1
        if sample.completion_hash in train_completion_hashes:
            completion_dups += 1
    
    return full_dups, prompt_dups, completion_dups, duplicate_indices


# =============================================================================
# Instruction Leakage Analysis
# =============================================================================

def compute_instruction_overlap(
    train_samples: List[Sample],
    val_samples: List[Sample],
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Compute instruction overlap between splits.
    
    Returns:
        (train_instructions, val_instructions, overlapping_instructions)
    """
    train_instructions = {s.instruction for s in train_samples if s.instruction}
    val_instructions = {s.instruction for s in val_samples if s.instruction}
    overlap = train_instructions & val_instructions
    
    return train_instructions, val_instructions, overlap


# =============================================================================
# Entity Leakage Analysis
# =============================================================================

def compute_entity_overlap(
    train_samples: List[Sample],
    val_samples: List[Sample],
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Compute entity (ASIN) overlap between splits.
    
    Returns:
        (train_entities, val_entities, overlapping_entities)
    """
    train_entities = set()
    val_entities = set()
    
    for s in train_samples:
        train_entities.update(s.entities)
    for s in val_samples:
        val_entities.update(s.entities)
    
    overlap = train_entities & val_entities
    
    return train_entities, val_entities, overlap


# =============================================================================
# N-gram Overlap Analysis
# =============================================================================

def get_ngrams(text: str, n: int, char_level: bool = False) -> Set[Tuple]:
    """Extract n-grams from text."""
    if char_level:
        tokens = list(text.lower())
    else:
        tokens = text.lower().split()
    
    if len(tokens) < n:
        return set()
    
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def compute_ngram_overlap(
    train_samples: List[Sample],
    val_samples: List[Sample],
    ns: List[int] = [1, 2, 3, 5],
    char_level: bool = False,
) -> Dict[int, Dict]:
    """Compute n-gram overlap statistics.
    
    For each n, computes:
    - Jaccard similarity of n-gram sets
    - % of val n-grams seen in training
    - % of unique val n-grams (novelty)
    
    Returns:
        Dict mapping n to metrics dict
    """
    # Combine all text
    train_text = " ".join(s.full_text for s in train_samples)
    val_text = " ".join(s.full_text for s in val_samples)
    
    results = {}
    
    for n in ns:
        train_ngrams = get_ngrams(train_text, n, char_level)
        val_ngrams = get_ngrams(val_text, n, char_level)
        
        if not val_ngrams:
            results[n] = {
                "jaccard": 0.0,
                "val_coverage": 0.0,
                "val_novelty": 1.0,
                "train_count": len(train_ngrams),
                "val_count": 0,
                "overlap_count": 0,
            }
            continue
        
        overlap = train_ngrams & val_ngrams
        union = train_ngrams | val_ngrams
        
        jaccard = len(overlap) / len(union) if union else 0.0
        val_coverage = len(overlap) / len(val_ngrams)  # How much of val is in train
        val_novelty = 1.0 - val_coverage  # How much of val is new
        
        results[n] = {
            "jaccard": jaccard,
            "val_coverage": val_coverage,
            "val_novelty": val_novelty,
            "train_count": len(train_ngrams),
            "val_count": len(val_ngrams),
            "overlap_count": len(overlap),
        }
    
    return results


# =============================================================================
# Distribution Analysis
# =============================================================================

def compute_token_distributions(
    train_samples: List[Sample],
    val_samples: List[Sample],
) -> Tuple[Counter, Counter, float, float, float]:
    """Compute token distribution metrics.
    
    Returns:
        (train_dist, val_dist, vocab_overlap, kl_div, js_div)
    """
    train_tokens = Counter()
    val_tokens = Counter()
    
    for s in train_samples:
        train_tokens.update(s.full_text.lower().split())
    for s in val_samples:
        val_tokens.update(s.full_text.lower().split())
    
    train_vocab = set(train_tokens.keys())
    val_vocab = set(val_tokens.keys())
    vocab_overlap = len(train_vocab & val_vocab) / len(val_vocab) if val_vocab else 0.0
    
    # Normalize to probability distributions
    train_total = sum(train_tokens.values())
    val_total = sum(val_tokens.values())
    
    all_tokens = train_vocab | val_vocab
    
    # Compute KL and JS divergence
    epsilon = 1e-10  # Smoothing for zero probabilities
    kl_div = 0.0
    js_div = 0.0
    
    for token in all_tokens:
        p = (train_tokens[token] / train_total) if train_total > 0 else epsilon
        q = (val_tokens[token] / val_total) if val_total > 0 else epsilon
        
        # Smooth
        p = max(p, epsilon)
        q = max(q, epsilon)
        
        # KL(Q || P) - how much val diverges from train
        kl_div += q * math.log(q / p)
        
        # JS divergence (symmetric)
        m = (p + q) / 2
        js_div += 0.5 * p * math.log(p / m) + 0.5 * q * math.log(q / m)
    
    return train_tokens, val_tokens, vocab_overlap, kl_div, js_div


def compute_length_statistics(
    train_samples: List[Sample],
    val_samples: List[Sample],
) -> Tuple[Dict, Dict, float]:
    """Compute length distribution statistics.
    
    Returns:
        (train_stats, val_stats, ks_statistic)
    """
    train_lengths = [len(s.full_text) for s in train_samples]
    val_lengths = [len(s.full_text) for s in val_samples]
    
    def stats(lengths):
        if not lengths:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        mean = sum(lengths) / len(lengths)
        variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        return {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": min(lengths),
            "max": max(lengths),
        }
    
    train_stats = stats(train_lengths)
    val_stats = stats(val_lengths)
    
    # Kolmogorov-Smirnov statistic (simplified)
    # Measures max difference between cumulative distributions
    if train_lengths and val_lengths:
        train_sorted = sorted(train_lengths)
        val_sorted = sorted(val_lengths)
        
        # Compute empirical CDFs at each point
        all_points = sorted(set(train_lengths + val_lengths))
        max_diff = 0.0
        
        for point in all_points:
            train_cdf = sum(1 for x in train_lengths if x <= point) / len(train_lengths)
            val_cdf = sum(1 for x in val_lengths if x <= point) / len(val_lengths)
            max_diff = max(max_diff, abs(train_cdf - val_cdf))
        
        ks_stat = max_diff
    else:
        ks_stat = 0.0
    
    return train_stats, val_stats, ks_stat


# =============================================================================
# Compression-Based Similarity
# =============================================================================

def normalized_compression_distance(text1: str, text2: str) -> float:
    """Compute Normalized Compression Distance (NCD).
    
    NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
    
    Where C(x) is the compressed size of x.
    
    NCD ranges from 0 (identical) to 1+ (completely different).
    Lower values indicate higher similarity/redundancy.
    """
    def compressed_size(text: str) -> int:
        return len(gzip.compress(text.encode('utf-8')))
    
    c_x = compressed_size(text1)
    c_y = compressed_size(text2)
    c_xy = compressed_size(text1 + text2)
    
    ncd = (c_xy - min(c_x, c_y)) / max(c_x, c_y)
    return ncd


# =============================================================================
# Semantic Similarity (Optional)
# =============================================================================

def compute_semantic_similarity(
    train_samples: List[Sample],
    val_samples: List[Sample],
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Compute embedding-based semantic similarity metrics.
    
    For each val sample, finds nearest neighbor in train set.
    Reports distribution of similarity scores.
    
    Requires: pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        logger.warning("sentence-transformers not installed. Skipping semantic analysis.")
        return {}
    
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Embed prompts (truncate for efficiency)
    max_length = 512
    train_texts = [s.prompt[:max_length] for s in train_samples]
    val_texts = [s.prompt[:max_length] for s in val_samples]
    
    logger.info("Computing embeddings...")
    train_embeddings = model.encode(train_texts, batch_size=batch_size, show_progress_bar=True)
    val_embeddings = model.encode(val_texts, batch_size=batch_size, show_progress_bar=True)
    
    # Compute cosine similarity matrix (val x train)
    # Normalize embeddings
    train_embeddings = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    val_embeddings = val_embeddings / np.linalg.norm(val_embeddings, axis=1, keepdims=True)
    
    # For each val sample, find max similarity to any train sample
    logger.info("Computing similarity scores...")
    max_similarities = []
    
    for i, val_emb in enumerate(val_embeddings):
        similarities = np.dot(train_embeddings, val_emb)
        max_sim = np.max(similarities)
        max_similarities.append(max_sim)
    
    max_similarities = np.array(max_similarities)
    
    # Compute statistics
    metrics = {
        "mean_max_similarity": float(np.mean(max_similarities)),
        "median_max_similarity": float(np.median(max_similarities)),
        "std_max_similarity": float(np.std(max_similarities)),
        "min_max_similarity": float(np.min(max_similarities)),
        "max_max_similarity": float(np.max(max_similarities)),
        "pct_above_0.9": float(np.mean(max_similarities > 0.9) * 100),
        "pct_above_0.95": float(np.mean(max_similarities > 0.95) * 100),
        "pct_above_0.99": float(np.mean(max_similarities > 0.99) * 100),
    }
    
    return metrics


# =============================================================================
# Risk Score Computation
# =============================================================================

def compute_risk_score(metrics: OverlapMetrics) -> Tuple[float, str, Dict[str, float]]:
    """Compute overall overfitting risk score (0-100).
    
    Combines multiple signals with domain-appropriate weights.
    
    Returns:
        (risk_score, risk_level, breakdown)
    """
    breakdown = {}
    
    # Exact duplicates (critical, up to 30 points)
    dup_risk = min(metrics.exact_duplicate_rate * 100 * 3, 30)
    breakdown["exact_duplicates"] = dup_risk
    
    # Prompt duplicates (high risk, up to 20 points)
    prompt_dup_risk = min(metrics.prompt_duplicate_rate * 100 * 2, 20)
    breakdown["prompt_duplicates"] = prompt_dup_risk
    
    # Instruction leakage (high risk, up to 20 points)
    instruction_risk = min(metrics.instruction_overlap_rate * 100 * 0.5, 20)
    breakdown["instruction_leakage"] = instruction_risk
    
    # Entity leakage (medium risk, up to 15 points)
    entity_risk = min(metrics.entity_overlap_rate * 100 * 0.2, 15)
    breakdown["entity_leakage"] = entity_risk
    
    # N-gram overlap (medium risk, up to 10 points)
    # Use bigram coverage as primary signal
    bigram_coverage = metrics.ngram_metrics.get(2, {}).get("val_coverage", 0)
    ngram_risk = min(bigram_coverage * 100 * 0.15, 10)
    breakdown["ngram_overlap"] = ngram_risk
    
    # Semantic similarity (if available, up to 5 points)
    if metrics.semantic_metrics:
        sem_risk = metrics.semantic_metrics.get("pct_above_0.95", 0) * 0.1
        breakdown["semantic_similarity"] = min(sem_risk, 5)
    
    # Total risk
    risk_score = sum(breakdown.values())
    risk_score = min(risk_score, 100)
    
    # Risk level
    if risk_score >= 50:
        risk_level = "CRITICAL"
    elif risk_score >= 30:
        risk_level = "HIGH"
    elif risk_score >= 15:
        risk_level = "MODERATE"
    elif risk_score >= 5:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"
    
    return risk_score, risk_level, breakdown


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_split(
    train_path: Path,
    val_path: Path,
    semantic: bool = False,
    output_path: Optional[Path] = None,
) -> OverlapMetrics:
    """Run comprehensive overlap analysis.
    
    Args:
        train_path: Path to training JSONL
        val_path: Path to validation JSONL
        semantic: Whether to run semantic similarity analysis
        output_path: Optional path to save detailed results
    
    Returns:
        OverlapMetrics with all computed metrics
    """
    logger.info(f"Loading training data: {train_path}")
    train_samples = load_samples(train_path)
    logger.info(f"Loaded {len(train_samples)} training samples")
    
    logger.info(f"Loading validation data: {val_path}")
    val_samples = load_samples(val_path)
    logger.info(f"Loaded {len(val_samples)} validation samples")
    
    metrics = OverlapMetrics()
    
    # === Exact Duplicates ===
    logger.info("Computing exact duplicates...")
    full_dups, prompt_dups, comp_dups, dup_indices = compute_exact_duplicates(
        train_samples, val_samples
    )
    metrics.exact_duplicate_count = full_dups
    metrics.exact_duplicate_rate = full_dups / len(val_samples) if val_samples else 0
    metrics.prompt_duplicate_count = prompt_dups
    metrics.prompt_duplicate_rate = prompt_dups / len(val_samples) if val_samples else 0
    metrics.completion_duplicate_count = comp_dups
    metrics.completion_duplicate_rate = comp_dups / len(val_samples) if val_samples else 0
    
    # === Instruction Leakage ===
    logger.info("Computing instruction overlap...")
    train_instr, val_instr, overlap_instr = compute_instruction_overlap(
        train_samples, val_samples
    )
    metrics.unique_train_instructions = len(train_instr)
    metrics.unique_val_instructions = len(val_instr)
    metrics.instruction_overlap_count = len(overlap_instr)
    metrics.instruction_overlap_rate = len(overlap_instr) / len(val_instr) if val_instr else 0
    
    # === Entity Leakage ===
    logger.info("Computing entity overlap...")
    train_ent, val_ent, overlap_ent = compute_entity_overlap(train_samples, val_samples)
    metrics.unique_train_entities = len(train_ent)
    metrics.unique_val_entities = len(val_ent)
    metrics.entity_overlap_count = len(overlap_ent)
    metrics.entity_overlap_rate = len(overlap_ent) / len(val_ent) if val_ent else 0
    metrics.leaked_entities = list(overlap_ent)[:20]  # Sample
    
    # === N-gram Overlap ===
    logger.info("Computing n-gram overlap...")
    metrics.ngram_metrics = compute_ngram_overlap(
        train_samples, val_samples, ns=[1, 2, 3, 5]
    )
    
    # === Distribution Analysis ===
    logger.info("Computing distribution metrics...")
    _, _, vocab_overlap, kl_div, js_div = compute_token_distributions(
        train_samples, val_samples
    )
    metrics.vocab_overlap_rate = vocab_overlap
    metrics.kl_divergence = kl_div
    metrics.js_divergence = js_div
    
    _, _, ks_stat = compute_length_statistics(train_samples, val_samples)
    metrics.length_ks_statistic = ks_stat
    
    # === Compression-Based ===
    logger.info("Computing compression-based similarity...")
    train_text = " ".join(s.full_text[:500] for s in train_samples[:100])  # Sample for efficiency
    val_text = " ".join(s.full_text[:500] for s in val_samples[:100])
    metrics.ncd_score = normalized_compression_distance(train_text, val_text)
    
    # === Semantic Similarity (Optional) ===
    if semantic:
        logger.info("Computing semantic similarity...")
        metrics.semantic_metrics = compute_semantic_similarity(train_samples, val_samples)
    
    # === Risk Score ===
    logger.info("Computing risk score...")
    risk_score, risk_level, breakdown = compute_risk_score(metrics)
    metrics.risk_score = risk_score
    metrics.risk_level = risk_level
    metrics.risk_breakdown = breakdown
    
    # === Save Results ===
    if output_path:
        results = {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "exact_duplicates": {
                "full": metrics.exact_duplicate_count,
                "full_rate": metrics.exact_duplicate_rate,
                "prompt": metrics.prompt_duplicate_count,
                "prompt_rate": metrics.prompt_duplicate_rate,
                "completion": metrics.completion_duplicate_count,
                "completion_rate": metrics.completion_duplicate_rate,
            },
            "instruction_leakage": {
                "train_unique": metrics.unique_train_instructions,
                "val_unique": metrics.unique_val_instructions,
                "overlap": metrics.instruction_overlap_count,
                "overlap_rate": metrics.instruction_overlap_rate,
            },
            "entity_leakage": {
                "train_unique": metrics.unique_train_entities,
                "val_unique": metrics.unique_val_entities,
                "overlap": metrics.entity_overlap_count,
                "overlap_rate": metrics.entity_overlap_rate,
                "sample_leaked": metrics.leaked_entities,
            },
            "ngram_overlap": metrics.ngram_metrics,
            "distribution": {
                "vocab_overlap_rate": metrics.vocab_overlap_rate,
                "kl_divergence": metrics.kl_divergence,
                "js_divergence": metrics.js_divergence,
                "length_ks_statistic": metrics.length_ks_statistic,
            },
            "compression": {
                "ncd_score": metrics.ncd_score,
            },
            "semantic": metrics.semantic_metrics,
            "risk": {
                "score": metrics.risk_score,
                "level": metrics.risk_level,
                "breakdown": metrics.risk_breakdown,
            },
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved detailed results to {output_path}")
    
    return metrics


# =============================================================================
# Report Generation
# =============================================================================

def print_report(metrics: OverlapMetrics, train_count: int, val_count: int) -> None:
    """Print a formatted analysis report."""
    
    print("\n" + "=" * 80)
    print("TRAIN/VAL SPLIT OVERLAP ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\nDataset: {train_count} train / {val_count} val samples")
    
    # Risk Score Summary
    print("\n" + "-" * 80)
    print("OVERALL RISK ASSESSMENT")
    print("-" * 80)
    
    risk_colors = {
        "MINIMAL": "✓",
        "LOW": "○",
        "MODERATE": "⚠",
        "HIGH": "✗",
        "CRITICAL": "✗✗",
    }
    
    print(f"\n  Risk Score: {metrics.risk_score:.1f}/100")
    print(f"  Risk Level: {risk_colors.get(metrics.risk_level, '?')} {metrics.risk_level}")
    
    print("\n  Risk Breakdown:")
    for component, score in sorted(metrics.risk_breakdown.items(), key=lambda x: -x[1]):
        bar = "█" * int(score / 2) + "░" * (50 - int(score / 2))
        print(f"    {component:25s} {bar} {score:.1f}")
    
    # Exact Duplicates
    print("\n" + "-" * 80)
    print("1. EXACT DUPLICATES")
    print("-" * 80)
    print(f"  Full duplicates:       {metrics.exact_duplicate_count:5d} ({metrics.exact_duplicate_rate*100:.2f}%)")
    print(f"  Prompt duplicates:     {metrics.prompt_duplicate_count:5d} ({metrics.prompt_duplicate_rate*100:.2f}%)")
    print(f"  Completion duplicates: {metrics.completion_duplicate_count:5d} ({metrics.completion_duplicate_rate*100:.2f}%)")
    
    if metrics.exact_duplicate_rate > 0:
        print("\n  ⚠ WARNING: Exact duplicates found! These should be removed.")
    
    # Instruction Leakage
    print("\n" + "-" * 80)
    print("2. INSTRUCTION/TASK LEAKAGE")
    print("-" * 80)
    print(f"  Unique train instructions: {metrics.unique_train_instructions}")
    print(f"  Unique val instructions:   {metrics.unique_val_instructions}")
    print(f"  Overlapping instructions:  {metrics.instruction_overlap_count} ({metrics.instruction_overlap_rate*100:.2f}%)")
    
    if metrics.instruction_overlap_rate > 0.5:
        print("\n  ⚠ WARNING: High instruction overlap. Model may memorize task→response mappings.")
    
    # Entity Leakage
    print("\n" + "-" * 80)
    print("3. ENTITY (ASIN/PRODUCT) LEAKAGE")
    print("-" * 80)
    print(f"  Unique train entities: {metrics.unique_train_entities}")
    print(f"  Unique val entities:   {metrics.unique_val_entities}")
    print(f"  Overlapping entities:  {metrics.entity_overlap_count} ({metrics.entity_overlap_rate*100:.2f}%)")
    
    if metrics.leaked_entities:
        print(f"  Sample leaked: {', '.join(metrics.leaked_entities[:5])}...")
    
    # N-gram Overlap
    print("\n" + "-" * 80)
    print("4. N-GRAM OVERLAP")
    print("-" * 80)
    print("  N  │ Jaccard │ Val Coverage │ Val Novelty │  Train   │   Val   ")
    print("  ───┼─────────┼──────────────┼─────────────┼──────────┼─────────")
    
    for n, data in sorted(metrics.ngram_metrics.items()):
        print(f"  {n:2d} │ {data['jaccard']:.4f}  │   {data['val_coverage']:.4f}     │   {data['val_novelty']:.4f}    │ {data['train_count']:8d} │ {data['val_count']:7d}")
    
    print("\n  Interpretation:")
    print("  - Jaccard: Overall set similarity (0=disjoint, 1=identical)")
    print("  - Val Coverage: % of val n-grams seen in train (lower=better)")
    print("  - Val Novelty: % of val n-grams NOT in train (higher=better)")
    
    bigram_cov = metrics.ngram_metrics.get(2, {}).get("val_coverage", 0)
    if bigram_cov > 0.8:
        print("\n  ⚠ WARNING: High bigram coverage suggests surface-level memorization risk.")
    
    # Distribution Analysis
    print("\n" + "-" * 80)
    print("5. DISTRIBUTION ANALYSIS")
    print("-" * 80)
    print(f"  Vocabulary overlap:     {metrics.vocab_overlap_rate*100:.2f}%")
    print(f"  KL divergence (val||train): {metrics.kl_divergence:.4f}")
    print(f"  Jensen-Shannon divergence:  {metrics.js_divergence:.4f}")
    print(f"  Length KS statistic:        {metrics.length_ks_statistic:.4f}")
    
    print("\n  Interpretation:")
    print("  - KL/JS divergence: Higher = more distribution shift (not necessarily bad)")
    print("  - KS statistic: Max diff in length CDFs (higher = different distributions)")
    
    # Compression-Based
    print("\n" + "-" * 80)
    print("6. COMPRESSION-BASED SIMILARITY")
    print("-" * 80)
    print(f"  Normalized Compression Distance: {metrics.ncd_score:.4f}")
    print("\n  Interpretation:")
    print("  - NCD ranges from 0 (identical) to 1+ (very different)")
    print("  - Lower values indicate higher redundancy/leakage risk")
    
    # Semantic Similarity
    if metrics.semantic_metrics:
        print("\n" + "-" * 80)
        print("7. SEMANTIC SIMILARITY (Embedding-Based)")
        print("-" * 80)
        sm = metrics.semantic_metrics
        print(f"  Mean max similarity:   {sm.get('mean_max_similarity', 0):.4f}")
        print(f"  Median max similarity: {sm.get('median_max_similarity', 0):.4f}")
        print(f"  % samples > 0.90 sim:  {sm.get('pct_above_0.9', 0):.2f}%")
        print(f"  % samples > 0.95 sim:  {sm.get('pct_above_0.95', 0):.2f}%")
        print(f"  % samples > 0.99 sim:  {sm.get('pct_above_0.99', 0):.2f}%")
        
        if sm.get('pct_above_0.95', 0) > 10:
            print("\n  ⚠ WARNING: Many val samples highly similar to train samples.")
    
    # Recommendations
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS")
    print("-" * 80)
    
    recommendations = []
    
    if metrics.exact_duplicate_rate > 0:
        recommendations.append("Remove exact duplicates from validation set")
    
    if metrics.prompt_duplicate_rate > 0.1:
        recommendations.append("Consider re-splitting data to avoid prompt leakage")
    
    if metrics.instruction_overlap_rate > 0.5:
        recommendations.append("Split data by instruction/task, not by sample")
    
    if metrics.entity_overlap_rate > 0.8:
        recommendations.append("Consider entity-aware splitting (by product ID)")
    
    bigram_cov = metrics.ngram_metrics.get(2, {}).get("val_coverage", 0)
    if bigram_cov > 0.9:
        recommendations.append("High surface overlap - consider more diverse val set")
    
    if not recommendations:
        recommendations.append("✓ No critical issues detected. Split appears reasonable.")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "=" * 80)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze train/val split for data leakage and overfitting risk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/analyze_split.py --train data/train.jsonl --val data/val.jsonl
  python src/analyze_split.py --train data/train.jsonl --val data/val.jsonl --semantic
  python src/analyze_split.py --train data/train.jsonl --val data/val.jsonl --output report.json
        """
    )
    
    parser.add_argument(
        "--train", type=Path, required=True,
        help="Path to training data (JSONL)"
    )
    parser.add_argument(
        "--val", type=Path, required=True,
        help="Path to validation data (JSONL)"
    )
    parser.add_argument(
        "--semantic", action="store_true",
        help="Run semantic similarity analysis (requires sentence-transformers)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to save detailed results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.train.exists():
        logger.error(f"Training file not found: {args.train}")
        return 1
    if not args.val.exists():
        logger.error(f"Validation file not found: {args.val}")
        return 1
    
    # Run analysis
    metrics = analyze_split(
        args.train, args.val,
        semantic=args.semantic,
        output_path=args.output,
    )
    
    # Count samples for report
    with open(args.train) as f:
        train_count = sum(1 for line in f if line.strip())
    with open(args.val) as f:
        val_count = sum(1 for line in f if line.strip())
    
    # Print report
    print_report(metrics, train_count, val_count)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
