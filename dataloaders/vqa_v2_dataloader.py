"""VQA v2.0 Dataloader."""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import json


class VQAv2Dataloader(Dataset):
    """VQA v2.0 dataset loader."""

    def __init__(self, data_root, imdb_file, answer_vocab_file, split='train'):
        """
        Initialize dataloader.
        
        Args:
            data_root: Root directory for data
            imdb_file: Path to IMDB .npy file
            answer_vocab_file: Path to answer vocabulary
            split: 'train', 'val', or 'test'
        """
        self.data_root = data_root
        self.split = split
        
        # Load IMDB data
        self.imdb = np.load(imdb_file, allow_pickle=True)
        if len(self.imdb) > 0 and isinstance(self.imdb[0], (int, np.integer)):
            self.imdb = self.imdb[1:]  # Skip metadata
        
        # Load answer vocabulary
        self.answer_dict = self._load_answer_vocab(answer_vocab_file)
        
        print(f"Loaded {len(self.imdb)} samples for {split}")

    def _load_answer_vocab(self, vocab_file):
        """Load answer vocabulary."""
        if vocab_file.endswith('.txt'):
            with open(vocab_file, 'r') as f:
                answers = [line.strip() for line in f]
            return {ans: idx for idx, ans in enumerate(answers)}
        elif vocab_file.endswith('.json'):
            with open(vocab_file, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported vocab file format: {vocab_file}")

    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):
        item = self.imdb[idx]
        
        # Get image path
        image_id = item.get('image_id', item.get('image_name', ''))
        if isinstance(image_id, int):
            image_path = os.path.join(
                self.data_root,
                f'COCO_{self.split}2014_{image_id:012d}.jpg'
            )
        else:
            image_path = os.path.join(self.data_root, image_id)
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, IOError, OSError) as e:
            # Fallback to dummy image
            print(f"Warning: Could not load image {image_path}: {e}")
            image = Image.new('RGB', (224, 224))
        
        # Get question
        question = ' '.join(item['question_tokens']) if 'question_tokens' in item else item.get('question', '')
        
        # Get answer
        answers_list = item.get('answers', [''])
        answer = item.get('answer', (answers_list[0] if answers_list else ''))
        answer_idx = self.answer_dict.get(answer, 0)
        
        # Get question ID
        question_id = item.get('question_id', idx)
        
        return {
            'image': image,
            'image_path': image_path,
            'question': question,
            'answer': answer,
            'answer_idx': answer_idx,
            'question_id': question_id,
            'mode': 'classify'
        }

    @staticmethod
    def collate_fn(batch):
        """Custom collate function."""
        images = [item['image'] for item in batch]
        questions = [item['question'] for item in batch]
        answer_indices = torch.tensor([item['answer_idx'] for item in batch])
        question_ids = [item['question_id'] for item in batch]
        
        return {
            'images': images,
            'questions': questions,
            'answers': answer_indices,
            'question_ids': question_ids,
            'mode': 'classify'
        }
