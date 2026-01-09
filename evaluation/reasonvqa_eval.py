"""
ReasonVQA Evaluation Script.

ReasonVQA metrics:
- Answer accuracy: Exact match and soft matching
- Generation metrics: BLEU, ROUGE, METEOR
- Reasoning evaluation: BLEU/ROUGE on reasoning chains
- Chain-of-thought quality assessment
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
from dataloaders import ReasonVQADataloader


class ReasonVQAEvaluator:
    """ReasonVQA specific evaluator with reasoning assessment."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def compute_answer_accuracy(self, predictions, annotations):
        """
        Compute answer accuracy (exact match and soft match).

        Args:
            predictions: Dict mapping question_id to {'answer': str, 'reasoning': str}
            annotations: Dict mapping question_id to annotation data

        Returns:
            Answer accuracy metrics
        """
        exact_matches = 0
        soft_matches = 0
        total_count = 0

        for qid, pred in predictions.items():
            if qid not in annotations:
                continue

            anno = annotations[qid]
            gt_answers = [a['answer'].lower().strip() for a in anno.get('answers', [])]
            pred_answer = pred['answer'].lower().strip()

            # Exact match
            if pred_answer in gt_answers:
                exact_matches += 1

            # Soft match (partial string matching)
            for gt in gt_answers:
                if pred_answer in gt or gt in pred_answer:
                    soft_matches += 1
                    break

            total_count += 1

        exact_acc = 100.0 * exact_matches / total_count if total_count > 0 else 0.0
        soft_acc = 100.0 * soft_matches / total_count if total_count > 0 else 0.0

        return {
            'exact_match_accuracy': exact_acc,
            'soft_match_accuracy': soft_acc,
            'total_questions': total_count
        }

    def compute_generation_metrics(self, predictions, annotations):
        """
        Compute generation metrics for answers (BLEU, ROUGE).

        Args:
            predictions: Dict mapping question_id to {'answer': str, 'reasoning': str}
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

                pred_answer = pred['answer']

                # BLEU score (best match among references)
                bleu = max([
                    sentence_bleu(
                        [ref.split()],
                        pred_answer.split(),
                        smoothing_function=smoothing.method1
                    ) for ref in references
                ])
                bleu_scores.append(bleu)

                # ROUGE score (best match among references)
                best_rouge = None
                best_score = 0
                for ref in references:
                    scores = rouge_scorer_obj.score(ref, pred_answer)
                    if scores['rougeL'].fmeasure > best_score:
                        best_score = scores['rougeL'].fmeasure
                        best_rouge = scores

                if best_rouge:
                    for key in rouge_scores:
                        rouge_scores[key].append(best_rouge[key].fmeasure)

            return {
                'answer_bleu': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
                'answer_rouge1': sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0.0,
                'answer_rouge2': sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0.0,
                'answer_rougeL': sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0.0,
            }

        except ImportError:
            print("Warning: NLTK or rouge_score not installed. Skipping generation metrics.")
            return {}

    def compute_reasoning_metrics(self, predictions, annotations):
        """
        Evaluate reasoning chain quality.

        Args:
            predictions: Dict mapping question_id to {'answer': str, 'reasoning': str}
            annotations: Dict mapping question_id to annotation data

        Returns:
            Reasoning evaluation metrics
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
                gt_reasoning = anno.get('rationale', '') or anno.get('reasoning', '')

                if not gt_reasoning or 'reasoning' not in pred:
                    continue

                pred_reasoning = pred['reasoning']

                # BLEU score for reasoning
                bleu = sentence_bleu(
                    [gt_reasoning.split()],
                    pred_reasoning.split(),
                    smoothing_function=smoothing.method1
                )
                bleu_scores.append(bleu)

                # ROUGE score for reasoning
                scores = rouge_scorer_obj.score(gt_reasoning, pred_reasoning)
                for key in rouge_scores:
                    rouge_scores[key].append(scores[key].fmeasure)

            return {
                'reasoning_bleu': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
                'reasoning_rouge1': sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0.0,
                'reasoning_rouge2': sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0.0,
                'reasoning_rougeL': sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0.0,
            }

        except ImportError:
            print("Warning: NLTK or rouge_score not installed. Skipping reasoning metrics.")
            return {}

    def evaluate(self, dataloader):
        """Run evaluation on ReasonVQA dataset."""
        predictions = {}

        print("Running ReasonVQA evaluation (generation mode with reasoning)...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="ReasonVQA Eval"):
                try:
                    # Use generation mode for ReasonVQA
                    outputs = self.model.inference(
                        batch['images'],
                        batch['questions'],
                        mode='generate'
                    )

                    # If model supports chain-of-thought, extract reasoning
                    # Otherwise, outputs is just List[str] of answers
                    if isinstance(outputs, list):
                        # Check if outputs are tuples (answer, reasoning)
                        for i, qid in enumerate(batch['question_ids']):
                            if i >= len(outputs):
                                predictions[qid] = {'answer': '', 'reasoning': ''}
                            elif isinstance(outputs[i], tuple):
                                predictions[qid] = {
                                    'answer': outputs[i][0],
                                    'reasoning': outputs[i][1]
                                }
                            else:
                                predictions[qid] = {
                                    'answer': outputs[i] if i < len(outputs) else '',
                                    'reasoning': ''
                                }
                    else:
                        # Fallback to classification
                        _, predicted_indices = outputs.max(1)
                        for i, qid in enumerate(batch['question_ids']):
                            predictions[qid] = {
                                'answer': str(predicted_indices[i].item()),
                                'reasoning': ''
                            }

                except Exception as e:
                    print(f"Error in batch: {e}")
                    continue

        return predictions

    def evaluate_with_metrics(self, dataloader, annotations_file):
        """Full evaluation with all metrics."""
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
        answer_acc_metrics = self.compute_answer_accuracy(predictions, annotations)
        answer_gen_metrics = self.compute_generation_metrics(predictions, annotations)
        reasoning_metrics = self.compute_reasoning_metrics(predictions, annotations)

        return {
            'predictions': predictions,
            'metrics': {
                **answer_acc_metrics,
                **answer_gen_metrics,
                **reasoning_metrics
            }
        }


def main():
    """Main ReasonVQA evaluation script."""
    parser = argparse.ArgumentParser(description='ReasonVQA Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--questions', type=str, required=True)
    parser.add_argument('--annotations', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)  # Smaller for generation
    parser.add_argument('--output', type=str, default='reasonvqa_results.json')
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
    dataset = ReasonVQADataloader(
        args.data_root,
        args.questions,
        args.annotations,
        split='val'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ReasonVQADataloader.collate_fn,
        num_workers=4
    )

    # Evaluate
    evaluator = ReasonVQAEvaluator(model, device)
    results = evaluator.evaluate_with_metrics(dataloader, args.annotations)

    # Print results
    print("\n" + "="*60)
    print("REASONVQA EVALUATION RESULTS")
    print("="*60)
    print(f"Exact Match Accuracy: {results['metrics']['exact_match_accuracy']:.2f}%")
    print(f"Soft Match Accuracy: {results['metrics']['soft_match_accuracy']:.2f}%")
    print(f"Total Questions: {results['metrics']['total_questions']}")

    if 'answer_bleu' in results['metrics']:
        print(f"\nAnswer Generation Metrics:")
        print(f"  BLEU: {results['metrics']['answer_bleu']:.4f}")
        print(f"  ROUGE-1: {results['metrics']['answer_rouge1']:.4f}")
        print(f"  ROUGE-2: {results['metrics']['answer_rouge2']:.4f}")
        print(f"  ROUGE-L: {results['metrics']['answer_rougeL']:.4f}")

    if 'reasoning_bleu' in results['metrics']:
        print(f"\nReasoning Chain Metrics:")
        print(f"  BLEU: {results['metrics']['reasoning_bleu']:.4f}")
        print(f"  ROUGE-1: {results['metrics']['reasoning_rouge1']:.4f}")
        print(f"  ROUGE-2: {results['metrics']['reasoning_rouge2']:.4f}")
        print(f"  ROUGE-L: {results['metrics']['reasoning_rougeL']:.4f}")

    print("="*60)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
