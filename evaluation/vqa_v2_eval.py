"""
VQA v2.0 Evaluation Script.

Official VQA v2.0 metric:
- Accuracy = min(# humans that said answer / 3, 1)
- Handles multiple acceptable answers per question
"""

import json
import argparse
import sys
import os
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.kg_vqa_model import KGVQAModel
from dataloaders import VQAv2Dataloader


class VQAv2Evaluator:
    """VQA v2.0 specific evaluator."""

    def __init__(self, model, device='cuda'):
        """
        Initialize evaluator.

        Args:
            model: KGVQAModel instance
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.eval()

    def compute_vqa_accuracy(self, predictions, annotations):
        """
        Compute VQA v2.0 accuracy.

        Args:
            predictions: Dict mapping question_id to predicted answer
            annotations: Dict mapping question_id to answer data

        Returns:
            Overall accuracy and per-answer-type accuracy
        """
        total_score = 0.0
        total_count = 0

        # Per answer type
        answer_type_scores = defaultdict(lambda: {'score': 0.0, 'count': 0})

        for qid, pred_answer in predictions.items():
            if qid not in annotations:
                continue

            anno = annotations[qid]
            gt_answers = anno.get('answers', [])
            answer_type = anno.get('answer_type', 'other')

            # Count how many humans gave this answer
            count = sum(1 for a in gt_answers if a['answer'] == pred_answer)

            # VQA accuracy: min(count/3, 1)
            score = min(count / 3.0, 1.0)

            total_score += score
            total_count += 1

            answer_type_scores[answer_type]['score'] += score
            answer_type_scores[answer_type]['count'] += 1

        # Compute overall accuracy
        overall_acc = 100.0 * total_score / total_count if total_count > 0 else 0.0

        # Compute per-type accuracy
        per_type_acc = {}
        for ans_type, data in answer_type_scores.items():
            if data['count'] > 0:
                per_type_acc[ans_type] = 100.0 * data['score'] / data['count']

        return {
            'overall_accuracy': overall_acc,
            'per_type_accuracy': per_type_acc,
            'total_questions': total_count
        }

    def evaluate(self, dataloader, answer_vocab_inv=None):
        """
        Run evaluation on VQA v2.0 dataset.

        Args:
            dataloader: VQA v2.0 DataLoader
            answer_vocab_inv: Inverse answer vocabulary (idx -> answer)

        Returns:
            Evaluation results dictionary
        """
        predictions = {}

        print("Running VQA v2.0 evaluation...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="VQA v2.0 Eval"):
                try:
                    # Get predictions
                    outputs = self.model.inference(
                        batch['images'],
                        batch['questions'],
                        mode='classify'
                    )

                    # Get predicted answers
                    _, predicted_indices = outputs.max(1)

                    # Map to answers
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

    def evaluate_with_metrics(self, dataloader, annotations_file, answer_vocab_inv=None):
        """
        Full evaluation with detailed metrics.

        Args:
            dataloader: VQA v2.0 DataLoader
            annotations_file: Path to annotations JSON
            answer_vocab_inv: Inverse answer vocabulary

        Returns:
            Complete evaluation results
        """
        # Get predictions
        predictions = self.evaluate(dataloader, answer_vocab_inv)

        # Load annotations
        with open(annotations_file, 'r') as f:
            anno_data = json.load(f)

        # Build annotation dict
        annotations = {}
        for anno in anno_data.get('annotations', []):
            qid = anno['question_id']
            annotations[qid] = anno

        # Compute metrics
        metrics = self.compute_vqa_accuracy(predictions, annotations)

        return {
            'predictions': predictions,
            'metrics': metrics
        }


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='VQA v2.0 Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--imdb_file', type=str, required=True)
    parser.add_argument('--annotations', type=str, required=True)
    parser.add_argument('--answer_vocab', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output', type=str, default='vqa_v2_results.json')
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = KGVQAModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load answer vocabulary
    with open(args.answer_vocab, 'r') as f:
        answer_list = [line.strip() for line in f]
    answer_vocab_inv = {i: ans for i, ans in enumerate(answer_list)}

    # Create dataset
    dataset = VQAv2Dataloader(
        args.data_root,
        args.imdb_file,
        args.answer_vocab,
        split='val'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=VQAv2Dataloader.collate_fn,
        num_workers=4
    )

    # Evaluate
    evaluator = VQAv2Evaluator(model, device)
    results = evaluator.evaluate_with_metrics(
        dataloader,
        args.annotations,
        answer_vocab_inv
    )

    # Print results
    print("\n" + "="*60)
    print("VQA v2.0 EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {results['metrics']['overall_accuracy']:.2f}%")
    print(f"Total Questions: {results['metrics']['total_questions']}")
    print("\nPer Answer Type Accuracy:")
    for ans_type, acc in results['metrics']['per_type_accuracy'].items():
        print(f"  {ans_type}: {acc:.2f}%")
    print("="*60)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
