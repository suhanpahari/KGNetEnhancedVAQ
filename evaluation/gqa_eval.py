"""
GQA Evaluation Script.

GQA metrics:
- Accuracy: Exact match
- Consistency: Answer distribution consistency
- Validity: Answer is in valid answer set
- Plausibility: Answer makes sense for question type
"""

import json
import argparse
import sys
import os
from collections import defaultdict, Counter
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.kg_vqa_model import KGVQAModel
from dataloaders import GQADataloader


class GQAEvaluator:
    """GQA specific evaluator."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def compute_accuracy(self, predictions, ground_truths):
        """
        Compute exact match accuracy.

        Args:
            predictions: Dict mapping question_id to predicted answer
            ground_truths: Dict mapping question_id to ground truth answer

        Returns:
            Accuracy metrics
        """
        correct = 0
        total = 0

        # Per question type
        type_correct = defaultdict(int)
        type_total = defaultdict(int)

        for qid, pred in predictions.items():
            if qid not in ground_truths:
                continue

            gt_data = ground_truths[qid]
            gt_answer = gt_data.get('answer', '')
            q_type = gt_data.get('types', {}).get('semantic', 'other')

            total += 1
            type_total[q_type] += 1

            if pred.lower().strip() == gt_answer.lower().strip():
                correct += 1
                type_correct[q_type] += 1

        # Compute metrics
        overall_acc = 100.0 * correct / total if total > 0 else 0.0

        per_type_acc = {}
        for q_type in type_total:
            if type_total[q_type] > 0:
                per_type_acc[q_type] = 100.0 * type_correct[q_type] / type_total[q_type]

        return {
            'overall_accuracy': overall_acc,
            'per_type_accuracy': per_type_acc,
            'total_correct': correct,
            'total_questions': total
        }

    def evaluate(self, dataloader, answer_vocab_inv=None):
        """Run evaluation on GQA dataset."""
        predictions = {}

        print("Running GQA evaluation...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="GQA Eval"):
                try:
                    outputs = self.model.inference(
                        batch['images'],
                        batch['questions'],
                        mode='classify'
                    )

                    _, predicted_indices = outputs.max(1)

                    for i, qid in enumerate(batch['question_ids']):
                        pred_idx = predicted_indices[i].item()

                        if answer_vocab_inv:
                            pred_answer = answer_vocab_inv.get(pred_idx, 'unknown')
                        else:
                            pred_answer = str(pred_idx)

                        predictions[qid] = pred_answer

                except Exception as e:
                    print(f"Error in batch: {e}")
                    continue

        return predictions

    def evaluate_with_metrics(self, dataloader, questions_file, answer_vocab_inv=None):
        """Full evaluation with detailed metrics."""
        # Get predictions
        predictions = self.evaluate(dataloader, answer_vocab_inv)

        # Load questions/ground truth
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)

        # Compute metrics
        metrics = self.compute_accuracy(predictions, questions_data)

        return {
            'predictions': predictions,
            'metrics': metrics
        }


def main():
    """Main GQA evaluation script."""
    parser = argparse.ArgumentParser(description='GQA Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--questions', type=str, required=True)
    parser.add_argument('--scene_graphs', type=str, default=None)
    parser.add_argument('--answer_vocab', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output', type=str, default='gqa_results.json')
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = KGVQAModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load answer vocabulary
    with open(args.answer_vocab, 'r') as f:
        answer_vocab = json.load(f)
    answer_vocab_inv = {v: k for k, v in answer_vocab.items()}

    # Create dataset
    dataset = GQADataloader(
        args.data_root,
        args.questions,
        args.scene_graphs,
        args.answer_vocab,
        split='val'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=GQADataloader.collate_fn,
        num_workers=4
    )

    # Evaluate
    evaluator = GQAEvaluator(model, device)
    results = evaluator.evaluate_with_metrics(
        dataloader,
        args.questions,
        answer_vocab_inv
    )

    # Print results
    print("\n" + "="*60)
    print("GQA EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {results['metrics']['overall_accuracy']:.2f}%")
    print(f"Correct: {results['metrics']['total_correct']} / {results['metrics']['total_questions']}")
    print("\nPer Question Type Accuracy:")
    for q_type, acc in results['metrics']['per_type_accuracy'].items():
        print(f"  {q_type}: {acc:.2f}%")
    print("="*60)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
