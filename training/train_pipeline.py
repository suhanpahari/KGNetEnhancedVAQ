"""Training pipeline for KG-VQA model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
import os
import sys
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.kg_vqa_model import KGVQAModel
from dataloaders.vqa_v2_dataloader import VQAv2Dataloader


def train_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    processed_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        # Move to device
        batch['answers'] = batch['answers'].to(device)

        # Forward pass with mixed precision
        with autocast():
            try:
                loss, outputs = model.train_step(batch, criterion)
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch['answers'].size(0)
        correct += predicted.eq(batch['answers']).sum().item()
        processed_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / processed_batches,
            'acc': 100. * correct / max(total, 1)
        })

    return total_loss / max(processed_batches, 1), 100. * correct / max(total, 1)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    processed_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch['answers'] = batch['answers'].to(device)

            try:
                outputs = model.inference(batch['images'], batch['questions'])
                loss = criterion(outputs, batch['answers'])

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch['answers'].size(0)
                correct += predicted.eq(batch['answers']).sum().item()
                processed_batches += 1
            except Exception as e:
                print(f"Validation error: {e}")
                continue

    return total_loss / max(processed_batches, 1), 100. * correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--imdb_train', type=str, required=True)
    parser.add_argument('--imdb_val', type=str, required=True)
    parser.add_argument('--answer_vocab', type=str, required=True)
    parser.add_argument('--kg_index_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with args
    config['kg_index_path'] = args.kg_index_path
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = VQAv2Dataloader(args.data_root, args.imdb_train, args.answer_vocab, split='train')
    val_dataset = VQAv2Dataloader(args.data_root, args.imdb_val, args.answer_vocab, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=VQAv2Dataloader.collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=VQAv2Dataloader.collate_fn,
        num_workers=4
    )
    
    # Create model
    print("Initializing model...")
    model = KGVQAModel(config).to(device)
    
    # Optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # Training loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_acc = 0
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch+1
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"Saved best model with acc: {val_acc:.2f}%")


if __name__ == '__main__':
    main()
