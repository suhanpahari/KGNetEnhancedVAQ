# Dataset-Specific Evaluation Guide

## Overview

This guide describes the complete evaluation pipeline for all four supported datasets. Each dataset has its own specialized evaluator with dataset-specific metrics.

**Evaluation Modules**: `update_kgnet/evaluation/`

| Dataset | Evaluator | Primary Metric | Additional Metrics |
|---------|-----------|----------------|-------------------|
| VQA v2.0 | [vqa_v2_eval.py](evaluation/vqa_v2_eval.py) | VQA Accuracy (soft) | Per-answer-type accuracy |
| GQA | [gqa_eval.py](evaluation/gqa_eval.py) | Exact Match Accuracy | Per-question-type accuracy |
| OK-VQA | [okvqa_eval.py](evaluation/okvqa_eval.py) | Soft Accuracy | BLEU, ROUGE-1/2/L |
| ReasonVQA | [reasonvqa_eval.py](evaluation/reasonvqa_eval.py) | Exact + Soft Match | Answer BLEU/ROUGE, Reasoning BLEU/ROUGE |

---

## 1. VQA v2.0 Evaluation

### Metrics

**VQA Accuracy (Soft Matching)**:
```python
# Count how many annotators provided the predicted answer
count = gt_answers.count(predicted_answer)
# VQA accuracy: min(count/3, 1)
score = min(count / 3.0, 1.0)
```

**Per-Answer-Type Breakdown**:
- `yes/no`: Binary questions
- `number`: Counting questions
- `other`: Open-ended questions

### Usage

```bash
cd evaluation/
python vqa_v2_eval.py \
    --checkpoint ../checkpoints/vqa_v2_best.pth \
    --config ../configs/vqa_v2_config.yaml \
    --data_root ../../data/coco/images/ \
    --imdb_val ../../data/imdb/imdb_minival2014.npy \
    --answer_vocab ../../data/answers_vqa.txt \
    --batch_size 16 \
    --output ../results/vqa_v2_results.json
```

### Output Format

```json
{
  "predictions": {
    "question_id": "predicted_answer",
    ...
  },
  "metrics": {
    "overall_accuracy": 72.45,
    "total_questions": 10000,
    "yes/no_accuracy": 85.2,
    "number_accuracy": 55.3,
    "other_accuracy": 68.9
  }
}
```

### Expected Performance

| Model | VQA Accuracy | Target |
|-------|--------------|--------|
| VisualBERT (baseline) | ~70% | - |
| **KG-VQA (ours)** | **>72%** | ✅ |

---

## 2. GQA Evaluation

### Metrics

**Exact Match Accuracy**:
```python
# Binary match (case-insensitive)
score = 1.0 if predicted.lower() == ground_truth.lower() else 0.0
```

**Per-Question-Type Breakdown**:
- `global`: Scene-level questions
- `object`: Object-centric questions
- `relation`: Relationship questions
- `attribute`: Property questions
- `category`: Classification questions
- `query`: Information retrieval questions

### Usage

```bash
cd evaluation/
python gqa_eval.py \
    --checkpoint ../checkpoints/gqa_best.pth \
    --config ../configs/gqa_config.yaml \
    --data_root ../../data/gqa/images/ \
    --questions ../../data/gqa/val_balanced_questions.json \
    --scene_graphs ../../data/gqa/val_sceneGraphs.json \
    --batch_size 16 \
    --output ../results/gqa_results.json
```

### Output Format

```json
{
  "predictions": {
    "question_id": "predicted_answer",
    ...
  },
  "metrics": {
    "overall_accuracy": 60.5,
    "total_questions": 12578,
    "global_accuracy": 72.3,
    "object_accuracy": 65.8,
    "relation_accuracy": 58.4,
    "attribute_accuracy": 63.1,
    "category_accuracy": 70.2,
    "query_accuracy": 55.9
  }
}
```

### Expected Performance

| Model | GQA Accuracy | Target |
|-------|--------------|--------|
| Bottom-Up (baseline) | ~57% | - |
| **KG-VQA (ours)** | **>60%** | ✅ |

---

## 3. OK-VQA Evaluation

### Metrics

**Soft Accuracy** (similar to VQA v2.0):
```python
count = gt_answers.count(predicted_answer)
score = min(count / 3.0, 1.0)
```

**Generation Metrics** (for free-form generation mode):
- **BLEU**: Measures n-gram overlap with reference answers
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence

### Usage

```bash
cd evaluation/
python okvqa_eval.py \
    --checkpoint ../checkpoints/okvqa_best.pth \
    --config ../configs/okvqa_config.yaml \
    --data_root ../../data/coco/images/ \
    --questions ../../data/okvqa/OpenEnded_mscoco_val2014_questions.json \
    --annotations ../../data/okvqa/mscoco_val2014_annotations.json \
    --batch_size 8 \
    --output ../results/okvqa_results.json
```

### Output Format

```json
{
  "predictions": {
    "question_id": "predicted_answer",
    ...
  },
  "metrics": {
    "overall_accuracy": 50.8,
    "total_questions": 5046,
    "bleu": 0.3245,
    "rouge1": 0.4567,
    "rouge2": 0.2891,
    "rougeL": 0.4123
  }
}
```

### Expected Performance

| Model | OK-VQA Accuracy | Target |
|-------|-----------------|--------|
| VisualBERT (baseline) | ~45% | - |
| **KG-VQA (ours)** | **>50%** | ✅ |

---

## 4. ReasonVQA Evaluation

### Metrics

**Answer Accuracy**:
- **Exact Match**: Binary match with any reference answer
- **Soft Match**: Partial string matching (substring containment)

**Answer Generation Metrics**:
- **BLEU**: Answer quality
- **ROUGE-1/2/L**: Answer overlap with references

**Reasoning Evaluation Metrics**:
- **Reasoning BLEU**: Quality of generated reasoning chain
- **Reasoning ROUGE-1/2/L**: Overlap with ground-truth rationale

### Usage

```bash
cd evaluation/
python reasonvqa_eval.py \
    --checkpoint ../checkpoints/reasonvqa_best.pth \
    --config ../configs/reasonvqa_config.yaml \
    --data_root ../../data/reasonvqa/images/ \
    --questions ../../data/reasonvqa/val_questions.json \
    --annotations ../../data/reasonvqa/val_annotations.json \
    --batch_size 8 \
    --output ../results/reasonvqa_results.json
```

### Output Format

```json
{
  "predictions": {
    "question_id": {
      "answer": "predicted_answer",
      "reasoning": "step-by-step reasoning chain"
    },
    ...
  },
  "metrics": {
    "exact_match_accuracy": 35.2,
    "soft_match_accuracy": 48.7,
    "total_questions": 5000,
    "answer_bleu": 0.2156,
    "answer_rouge1": 0.3478,
    "answer_rouge2": 0.1892,
    "answer_rougeL": 0.3123,
    "reasoning_bleu": 0.1845,
    "reasoning_rouge1": 0.4012,
    "reasoning_rouge2": 0.2456,
    "reasoning_rougeL": 0.3689
  }
}
```

### Expected Performance

| Model | Answer Exact Match | Answer BLEU | Reasoning BLEU | Target |
|-------|-------------------|-------------|----------------|--------|
| **KG-VQA (ours)** | **>35%** | **>20** | **>18** | ✅ |

---

## Running All Evaluations

### Option 1: Single Script (Recommended)

```bash
cd scripts/
bash run_evaluation.sh
```

This will:
1. Evaluate on all 4 datasets sequentially
2. Save results to `../results/`
3. Print summary statistics

### Option 2: Individual Datasets

```bash
# Evaluate only VQA v2.0
cd evaluation/
python vqa_v2_eval.py --checkpoint ../checkpoints/vqa_v2_best.pth ...

# Evaluate only GQA
python gqa_eval.py --checkpoint ../checkpoints/gqa_best.pth ...

# Evaluate only OK-VQA
python okvqa_eval.py --checkpoint ../checkpoints/okvqa_best.pth ...

# Evaluate only ReasonVQA
python reasonvqa_eval.py --checkpoint ../checkpoints/reasonvqa_best.pth ...
```

---

## Evaluation Pipeline Architecture

### Common Components

All evaluators inherit common functionality:

```python
class DatasetEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def evaluate(self, dataloader):
        """Run inference on dataset."""
        predictions = {}
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model.inference(
                    batch['images'],
                    batch['questions'],
                    mode='classify' or 'generate'
                )
                # Store predictions
        return predictions

    def compute_metrics(self, predictions, annotations):
        """Dataset-specific metric computation."""
        # Implemented by each evaluator
        pass

    def evaluate_with_metrics(self, dataloader, annotations_file):
        """Full evaluation with all metrics."""
        predictions = self.evaluate(dataloader)
        # Load annotations
        # Compute metrics
        return {'predictions': ..., 'metrics': ...}
```

### Inference Modes

**Classification Mode** (VQA v2.0, GQA):
```python
outputs = model.inference(images, questions, mode='classify')
# Returns: Tensor of shape (batch_size, num_classes)
predicted_indices = outputs.argmax(dim=1)
predicted_answers = [answer_vocab[idx] for idx in predicted_indices]
```

**Generation Mode** (OK-VQA, ReasonVQA):
```python
outputs = model.inference(images, questions, mode='generate')
# Returns: List[str] or List[Tuple[str, str]] (answer, reasoning)
predicted_answers = outputs  # Already decoded strings
```

---

## Dataset-Specific Considerations

### VQA v2.0
- **Format**: IMDB (`.npy`) files with COCO images
- **Answer Vocabulary**: 3,129 classes from `answers_vqa.txt`
- **Evaluation**: Soft matching with multiple annotators
- **Note**: Questions have multiple acceptable answers

### GQA
- **Format**: JSON files with scene graphs
- **Answer Vocabulary**: 1,878 classes
- **Evaluation**: Exact match only
- **Note**: Scene graphs provide additional context

### OK-VQA
- **Format**: JSON files (similar to VQA v2.0) with COCO images
- **Mode**: Generation preferred (free-form answers)
- **Evaluation**: Soft accuracy + BLEU/ROUGE
- **Note**: Requires external knowledge

### ReasonVQA
- **Format**: JSON files with reasoning rationale
- **Mode**: Generation with chain-of-thought
- **Evaluation**: Answer + Reasoning metrics
- **Note**: Two outputs per question (answer + reasoning)

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Solution**: Reduce batch size
```bash
python vqa_v2_eval.py --batch_size 4 ...  # Instead of 16
```

### Issue 2: Missing NLTK/ROUGE

**Solution**: Install required packages
```bash
pip install nltk rouge-score
python -c "import nltk; nltk.download('punkt')"
```

### Issue 3: Incorrect Answer Format

**Check**: Ensure model mode matches dataset
- VQA v2.0 / GQA: Use `mode='classify'`
- OK-VQA / ReasonVQA: Use `mode='generate'`

### Issue 4: Dimension Mismatch

**Verify**: Check that models were trained with correct configs
```python
# Ensure vision/text/knowledge features are all 768-dim
# Ensure fusion output is 1024-dim
# Ensure LLM input projection is correct
```

---

## Advanced Evaluation

### Per-Question-Type Analysis

All evaluators support breaking down results by question type:

```python
# In evaluator classes
def compute_per_type_accuracy(self, predictions, annotations):
    type_scores = defaultdict(list)
    for qid, pred in predictions.items():
        q_type = annotations[qid]['question_type']
        score = self.compute_score(pred, annotations[qid])
        type_scores[q_type].append(score)

    return {
        f"{qtype}_accuracy": 100 * np.mean(scores)
        for qtype, scores in type_scores.items()
    }
```

### Confidence Calibration

Evaluators can compute confidence scores (if model provides them):

```python
# If model returns (answer, confidence)
def compute_calibration(self, predictions):
    # ECE (Expected Calibration Error)
    # Reliability diagrams
    pass
```

### Error Analysis

```python
# Save incorrect predictions for manual review
errors = []
for qid, pred in predictions.items():
    if pred != ground_truth[qid]:
        errors.append({
            'question_id': qid,
            'question': questions[qid],
            'predicted': pred,
            'ground_truth': ground_truth[qid]
        })

with open('errors.json', 'w') as f:
    json.dump(errors, f, indent=2)
```

---

## Summary

**Complete Evaluation Suite**: 4 dataset-specific evaluators with proper metrics

| Component | Status | Lines |
|-----------|--------|-------|
| VQA v2.0 Evaluator | ✅ | 178 |
| GQA Evaluator | ✅ | 210 |
| OK-VQA Evaluator | ✅ | 268 |
| ReasonVQA Evaluator | ✅ | 290 |
| Evaluation Script | ✅ | 90 |

**Total**: ~1,036 lines of evaluation code

**All metrics properly implemented**:
- VQA soft accuracy
- GQA exact match
- OK-VQA generation metrics
- ReasonVQA reasoning evaluation

**Ready to benchmark on all 4 datasets!** 🎯
