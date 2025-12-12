# Theoretical Foundations of Train/Val Leakage Analysis

This document provides rigorous theoretical background for the data leakage detection methods implemented in `analyze_split.py`.

## 1. The Problem of Data Leakage

### 1.1 Definition

**Data leakage** occurs when information from the validation (or test) set is inadvertently available during training. This leads to:
- Overly optimistic performance estimates
- Poor generalization to truly unseen data
- Models that memorize rather than learn

### 1.2 Types of Leakage in Sequential Decision Tasks

For experience model training (state, action → next_state, reward), leakage can occur at multiple levels:

| Level | Description | Detection Method |
|-------|-------------|------------------|
| **Sample-level** | Identical samples in both splits | Hash-based matching |
| **Input-level** | Same prompts with different outputs | Prompt hash matching |
| **Task-level** | Same instructions in both splits | Instruction extraction + matching |
| **Entity-level** | Same entities (products, IDs) in both | Regex extraction + set overlap |
| **Pattern-level** | Overlapping surface patterns | N-gram analysis |
| **Semantic-level** | Similar meaning despite different text | Embedding similarity |

## 2. Exact Duplicate Detection

### 2.1 Method
We use MD5 hashing for O(1) lookup:

```
H(sample) = MD5(prompt + completion)
leakage = H(val_sample) ∈ {H(train_sample) | train_sample ∈ Train}
```

### 2.2 Interpretation
- **Any exact duplicates is critical leakage**
- Model can achieve perfect accuracy by memorization
- Should be 0% in a proper split

## 3. N-gram Overlap Analysis

### 3.1 Jaccard Similarity

For n-gram sets A (train) and B (val):

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

- Range: [0, 1]
- 0 = completely disjoint
- 1 = identical sets

### 3.2 Validation Coverage

$$\text{ValCoverage}_n = \frac{|B \cap A|}{|B|}$$

This measures **what fraction of validation n-grams appear in training**. High coverage means:
- Model can rely on memorized patterns
- Less need to generalize

### 3.3 Validation Novelty

$$\text{ValNovelty}_n = 1 - \text{ValCoverage}_n = \frac{|B \setminus A|}{|B|}$$

This measures **what fraction of validation n-grams are truly new**. Higher is better for assessing generalization.

### 3.4 Interpretation by N

| N | What it captures |
|---|------------------|
| 1 (unigrams) | Vocabulary overlap |
| 2 (bigrams) | Local patterns, common phrases |
| 3 (trigrams) | Longer patterns, template fragments |
| 5+ | Near-exact sequence matches |

**Rule of thumb**: Bigram coverage > 80% is concerning.

## 4. Instruction/Task Leakage

### 4.1 Why It Matters

For experience models, the **instruction** defines the task goal. If the same instruction appears in train and val:

```
Train: "Find red wallet" → trajectory_1
Val:   "Find red wallet" → trajectory_2
```

The model can memorize:
- Common action sequences for this goal
- Typical products that satisfy this goal
- Expected reward patterns

This doesn't test whether the model learned **environment dynamics**, just **task-specific shortcuts**.

### 4.2 Metric

$$\text{InstructionLeakage} = \frac{|\text{Instructions}_{val} \cap \text{Instructions}_{train}|}{|\text{Instructions}_{val}|}$$

**Target**: < 10% for rigorous evaluation

### 4.3 Mitigation: Instruction-based Splitting

Instead of random sample splitting:
```python
# Bad: random split
train, val = random_split(all_samples, 0.9)

# Good: instruction-based split  
instructions = set(s.instruction for s in all_samples)
train_instructions, val_instructions = random_split(instructions, 0.9)
train = [s for s in all_samples if s.instruction in train_instructions]
val = [s for s in all_samples if s.instruction in val_instructions]
```

## 5. Entity Leakage

### 5.1 Domain-Specific Entities

For WebShop/e-commerce:
- **ASINs** (Amazon product IDs): `B08WHJLGWT`
- **Product names**: "Sony WH-1000XM4"
- **Categories**: "Electronics > Headphones"

### 5.2 Why It Matters

If the same product appears in train and val:
- Model memorizes product-specific transitions
- "B08XYZ → Buy Now → reward=0.95" becomes a lookup table
- Doesn't test generalization to new products

### 5.3 Metric

$$\text{EntityLeakage} = \frac{|\text{Entities}_{val} \cap \text{Entities}_{train}|}{|\text{Entities}_{val}|}$$

**Target**: < 50% is acceptable, < 20% is ideal

## 6. Distribution Divergence

### 6.1 KL Divergence

Kullback-Leibler divergence measures how much Q (val distribution) diverges from P (train distribution):

$$D_{KL}(Q || P) = \sum_x Q(x) \log \frac{Q(x)}{P(x)}$$

- Not symmetric
- Undefined if P(x) = 0 where Q(x) > 0
- We apply smoothing: ε = 10⁻¹⁰

### 6.2 Jensen-Shannon Divergence

Symmetric version:

$$D_{JS}(P || Q) = \frac{1}{2} D_{KL}(P || M) + \frac{1}{2} D_{KL}(Q || M)$$

where M = (P + Q) / 2

- Range: [0, ln(2)] ≈ [0, 0.693] for base-e
- More stable than KL divergence

### 6.3 Interpretation

| JS Divergence | Interpretation |
|---------------|----------------|
| < 0.05 | Very similar distributions (potential leakage) |
| 0.05 - 0.20 | Normal variation |
| > 0.20 | Significant distribution shift (may be problematic) |

**Note**: High divergence isn't necessarily bad—it may indicate good diversity. Low divergence can indicate leakage OR that the data is uniformly sampled.

## 7. Normalized Compression Distance (NCD)

### 7.1 Information-Theoretic Foundation

Based on Kolmogorov complexity. The intuition:
- If X and Y share structure, compressing X+Y together should be nearly as efficient as compressing them separately
- If they're independent, no compression benefit

### 7.2 Formula

$$\text{NCD}(X, Y) = \frac{C(XY) - \min(C(X), C(Y))}{\max(C(X), C(Y))}$$

Where C(·) is compressed size (we use gzip).

### 7.3 Interpretation

| NCD | Interpretation |
|-----|----------------|
| 0.0 | Identical (X = Y) |
| < 0.5 | High similarity/redundancy |
| 0.5 - 0.8 | Moderate similarity |
| > 0.8 | Low similarity (good for val) |

**Advantage**: Language-agnostic, captures any redundancy.

## 8. Semantic Similarity (Embedding-Based)

### 8.1 Method

1. Encode train and val samples with sentence transformer
2. For each val sample, find max cosine similarity to any train sample
3. Report distribution of max similarities

$$\text{MaxSim}(v) = \max_{t \in \text{Train}} \frac{\mathbf{e}_v \cdot \mathbf{e}_t}{||\mathbf{e}_v|| \cdot ||\mathbf{e}_t||}$$

### 8.2 Interpretation

| Max Similarity | Interpretation |
|----------------|----------------|
| > 0.99 | Near-duplicate (paraphrase) |
| > 0.95 | Very similar (likely leaked) |
| > 0.90 | Similar topic/structure |
| < 0.80 | Reasonably distinct |

### 8.3 Aggregate Metrics

- **% above 0.95**: Fraction of val set that has a near-duplicate in train
- Target: < 5%

## 9. Composite Risk Score

### 9.1 Weighted Combination

We combine signals with domain-appropriate weights:

```
Risk = w₁·ExactDup + w₂·PromptDup + w₃·InstrLeak + w₄·EntityLeak + w₅·NGramOverlap
```

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Exact duplicates | 3× | Critical, direct leakage |
| Prompt duplicates | 2× | High risk, same input |
| Instruction leakage | 0.5× | Task-level, indirect |
| Entity leakage | 0.2× | Moderate, entity-specific |
| N-gram overlap | 0.15× | Surface patterns |

### 9.2 Risk Levels

| Score | Level | Action |
|-------|-------|--------|
| 0-5 | MINIMAL | Proceed with training |
| 5-15 | LOW | Monitor for overfitting |
| 15-30 | MODERATE | Consider re-splitting |
| 30-50 | HIGH | Re-split recommended |
| 50+ | CRITICAL | Re-split required |

## 10. Best Practices for Splitting

### 10.1 Hierarchy of Splitting Strategies

From most rigorous to least:

1. **Temporal split**: Train on older data, validate on newer
2. **Entity-based split**: No product overlap
3. **Instruction-based split**: No task overlap
4. **Stratified random**: Preserve class distributions
5. **Random**: Simple but prone to leakage

### 10.2 For Experience Models Specifically

Recommended approach:
1. Group samples by (instruction, primary_entity)
2. Split groups, not individual samples
3. Verify with `analyze_split.py`
4. Target: instruction overlap < 10%, entity overlap < 30%

## 11. References

1. Kapoor, S., & Narayanan, A. (2022). Leakage and the Reproducibility Crisis in ML-based Science. arXiv:2207.07048

2. Kaufman, S., et al. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. ACM TKDD.

3. Li, M., et al. (2004). The Similarity Metric. IEEE Transactions on Information Theory.

4. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.

5. Jaccard, P. (1912). The Distribution of the Flora in the Alpine Zone. New Phytologist.
