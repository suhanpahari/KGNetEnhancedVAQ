"""Evaluation pipeline for all VQA datasets."""

import torch
import sys
import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.kg_vqa_model import KGVQAModel


def compute_vqa_accuracy(predictions, ground_truths):
    """
    VQA v2.0 accuracy: min(# humans said answer / 3, 1).
    """
    correct = 0
    total = len(predictions)
    
    for pred, gt_list in zip(predictions, ground_truths):
        if isinstance(gt_list, str):
            gt_list = [gt_list]
        
        # Count how many humans said this answer
        count = gt_list.count(pred) if pred in gt_list else 0
        score = min(count / 3.0, 1.0)
        correct += score
    
    return 100.0 * correct / total if total > 0 else 0.0


def compute_exact_match_accuracy(predictions, ground_truths):
    """
    Exact match accuracy for GQA.
    """
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    return 100.0 * correct / len(predictions) if predictions else 0.0


def compute_generation_metrics(predictions, references):
    """
    Compute BLEU and ROUGE for generation tasks.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from rouge_score import rouge_scorer
        
        bleu_scores = []
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            # BLEU
            bleu = sentence_bleu([ref.split()], pred.split())
            bleu_scores.append(bleu)
            
            # ROUGE
            scores = rouge_scorer_obj.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
        
        return {
            'bleu': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
            'rouge1': sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0.0,
            'rouge2': sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0.0,
            'rougeL': sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0.0,
        }
    except ImportError:
        print("Warning: NLTK or rouge_score not installed. Skipping generation metrics.")
        return {}


def evaluate_model(model, dataloader, dataset_type, device, output_file=None):
    """
    Evaluate model on a dataset.
    
    Args:
        model: KGVQAModel
        dataloader: DataLoader
        dataset_type: 'vqa_v2', 'gqa', 'okvqa', 'reasonvqa'
        device: torch device
        output_file: Path to save predictions (optional)
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    predictions = []
    ground_truths = []
    question_ids = []
    
    print(f"\nEvaluating on {dataset_type}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            try:
                # Get predictions
                if batch['mode'] == 'classify':
                    outputs = model.inference(batch['images'], batch['questions'], mode='classify')
                    _, predicted = outputs.max(1)
                    pred_list = predicted.cpu().tolist()
                else:
                    outputs = model.inference(batch['images'], batch['questions'], mode='generate')
                    pred_list = outputs if isinstance(outputs, list) else [outputs]
                
                predictions.extend(pred_list)
                ground_truths.extend(batch['answers'] if isinstance(batch['answers'], list) else batch['answers'].cpu().tolist())
                question_ids.extend(batch['question_ids'])
                
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
    
    # Compute metrics based on dataset type
    metrics = {}
    
    if dataset_type == 'vqa_v2':
        metrics['accuracy'] = compute_vqa_accuracy(predictions, ground_truths)
    
    elif dataset_type == 'gqa':
        metrics['accuracy'] = compute_exact_match_accuracy(predictions, ground_truths)
    
    elif dataset_type in ['okvqa', 'reasonvqa']:
        # Generation metrics
        if predictions and isinstance(predictions[0], str):
            gen_metrics = compute_generation_metrics(predictions, ground_truths)
            metrics.update(gen_metrics)
        metrics['accuracy'] = compute_exact_match_accuracy(predictions, ground_truths)
    
    # Save predictions
    if output_file:
        results = {
            'predictions': predictions,
            'ground_truths': ground_truths,
            'question_ids': question_ids,
            'metrics': metrics
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved predictions to {output_file}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True, choices=['vqa_v2', 'gqa', 'okvqa', 'reasonvqa'])
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='../results/')
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    model = KGVQAModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dataloader
    from torch.utils.data import DataLoader
    
    if args.dataset == 'vqa_v2':
        from dataloaders import VQAv2Dataloader
        dataset = VQAv2Dataloader(args.data_root, config['imdb_val'], config['answer_vocab'], split='val')
        collate_fn = VQAv2Dataloader.collate_fn
    
    elif args.dataset == 'gqa':
        from dataloaders import GQADataloader
        dataset = GQADataloader(args.data_root, config['questions_val'], config.get('scene_graphs'), config.get('answer_vocab'), split='val')
        collate_fn = GQADataloader.collate_fn
    
    elif args.dataset == 'okvqa':
        from dataloaders import OKVQADataloader
        dataset = OKVQADataloader(args.data_root, config['questions_val'], config.get('annotations_val'), split='val')
        collate_fn = OKVQADataloader.collate_fn
    
    elif args.dataset == 'reasonvqa':
        from dataloaders import ReasonVQADataloader
        dataset = ReasonVQADataloader(args.data_root, config['data_val'], split='val')
        collate_fn = ReasonVQADataloader.collate_fn
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Evaluate
    output_file = os.path.join(args.output_dir, f'{args.dataset}_predictions.json')
    metrics = evaluate_model(model, dataloader, args.dataset, device, output_file)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results for {args.dataset.upper()}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    print(f"{'='*50}\n")
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, f'{args.dataset}_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")


if __name__ == '__main__':
    main()
