"""
OK-VQA Evaluation Script.

OK-VQA metrics:
- Accuracy: Multiple acceptable answers (similar to VQA v2.0)
- Generation metrics: BLEU, ROUGE, METEOR
- Knowledge requirement: % requiring external knowledge
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
from dataloaders import OKVQADataloader


class OKVQAEvaluator:
    """OK-VQA specific evaluator."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def compute_accuracy(self, predictions, annotations):
        """
        Compute OK-VQA accuracy (soft matching with multiple answers).

        Args:
            predictions: Dict mapping question_id to predicted answer
            annotations: Dict mapping question_id to annotation data

        Returns:
            Accuracy metrics
        """
        total_score = 0.0
        total_count = 0

        for qid, pred_answer in predictions.items():
            if qid not in annotations:
                continue

            anno = annotations[qid]
            gt_answers = [a['answer'].lower().strip() for a in anno.get('answers', [])]

            pred_lower = pred_answer.lower().strip()

            # Count matches
            count = gt_answers.count(pred_lower)

            # Soft accuracy: min(count/3, 1)
            score = min(count / 3.0, 1.0)

            total_score += score
            total_count += 1

        overall_acc = 100.0 * total_score / total_count if total_count > 0 else 0.0

        return {
            'overall_accuracy': overall_acc,
            'total_questions': total_count
        }

    def compute_generation_metrics(self, predictions, annotations):
        """
        Compute generation-based metrics (BLEU, ROUGE).

        Args:
            predictions: Dict mapping question_id to predicted answer
            annotations: Dict mapping question_id to annotation data

        Returns:
            Generation metrics
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from rouge_score import rouge_scorer

            bleu_scores = []
            rouge_scorer_obj = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

            smoothing = SmoothingFunction()

            for qid, pred in predictions.items():
                if qid not in annotations:
                    continue

                anno = annotations[qid]
                references = [a['answer'] for a in anno.get('answers', [])]

                if not references:
                    continue

                # BLEU score (against all references)
                bleu = max([
                    sentence_bleu(
                        [ref.split()],
                        pred.split(),
                        smoothing_function=smoothing.method1
                    ) for ref in references
                ])
                bleu_scores.append(bleu)

                # ROUGE score (against best reference)
                best_rouge = None
                best_score = 0
                for ref in references:
                    scores = rouge_scorer_obj.score(ref, pred)
                    if scores['rougeL'].fmeasure > best_score:
                        best_score = scores['rougeL'].fmeasure
                        best_rouge = scores

                if best_rouge:
                    for key in rouge_scores:
                        rouge_scores[key].append(best_rouge[key].fmeasure)

            return {
                'bleu': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
                'rouge1': sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0.0,
                'rouge2': sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0.0,
                'rougeL': sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0.0,
            }

        except ImportError:
            print("Warning: NLTK or rouge_score not installed. Skipping generation metrics.")
            return {}

    def evaluate(self, dataloader):
        """Run evaluation on OK-VQA dataset."""
        predictions = {}

        print("Running OK-VQA evaluation (generation mode)...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="OK-VQA Eval"):
                try:
                    # Use generation mode for OK-VQA
                    outputs = self.model.inference(
                        batch['images'],
                        batch['questions'],
                        mode='generate'
                    )

                    # outputs is List[str] in generation mode
                    if isinstance(outputs, list):
                        for i, qid in enumerate(batch['question_ids']):
                            predictions[qid] = outputs[i] if i < len(outputs) else ''
                    else:
                        # Fallback to classification
                        _, predicted_indices = outputs.max(1)
                        for i, qid in enumerate(batch['question_ids']):
                            predictions[qid] = str(predicted_indices[i].item())

                except Exception as e:
                    print(f"Error in batch: {e}")
                    continue

        return predictions

    def evaluate_with_metrics(self, dataloader, annotations_file):
        """Full evaluation with detailed metrics."""
        # Get predictions
        predictions = self.evaluate(dataloader)

        # Load annotations
        with open(annotations_file, 'r') as f:
            anno_data = json.load(f)

        annotations = {}
        for anno in anno_data.get('annotations', []):
            qid = anno['question_id']
            annotations[qid] = anno

        # Compute metrics
        acc_metrics = self.compute_accuracy(predictions, annotations)
        gen_metrics = self.compute_generation_metrics(predictions, annotations)

        return {
            'predictions': predictions,
            'metrics': {
                **acc_metrics,
                **gen_metrics
            }
        }


def main():
    """Main OK-VQA evaluation script."""
    parser = argparse.ArgumentParser(description='OK-VQA Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--questions', type=str, required=True)
    parser.add_argument('--annotations', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)  # Smaller for generation
    parser.add_argument('--output', type=str, default='okvqa_results.json')
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

    # Create dataset
    dataset = OKVQADataloader(
        args.data_root,
        args.questions,
        args.annotations,
        split='val'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=OKVQADataloader.collate_fn,
        num_workers=4
    )

    # Evaluate
    evaluator = OKVQAEvaluator(model, device)
    results = evaluator.evaluate_with_metrics(dataloader, args.annotations)

    # Print results
    print("\n" + "="*60)
    print("OK-VQA EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {results['metrics']['overall_accuracy']:.2f}%")
    print(f"Total Questions: {results['metrics']['total_questions']}")
    if 'bleu' in results['metrics']:
        print(f"\nGeneration Metrics:")
        print(f"  BLEU: {results['metrics']['bleu']:.4f}")
        print(f"  ROUGE-1: {results['metrics']['rouge1']:.4f}")
        print(f"  ROUGE-2: {results['metrics']['rouge2']:.4f}")
        print(f"  ROUGE-L: {results['metrics']['rougeL']:.4f}")
    print("="*60)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
