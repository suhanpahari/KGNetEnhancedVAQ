"""Base VQA Dataloader - Common functionality for all datasets."""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from retrieval.entity_extractor import EntityExtractor


class BaseVQADataloader(Dataset):
    """Base class for all VQA dataset loaders."""

    def __init__(self, data_root, split='train'):
        """
        Initialize base dataloader.
        
        Args:
            data_root: Root directory for data
            split: 'train', 'val', or 'test'
        """
        self.data_root = data_root
        self.split = split
        self.entity_extractor = EntityExtractor()
        
        print(f"Initialized {self.__class__.__name__} for {split}")

    def load_image(self, image_path):
        """Load image with fallback to dummy image."""
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Fallback to dummy image
            image = Image.new('RGB', (224, 224), color='gray')
        return image

    def extract_entities(self, question):
        """Extract entities from question text."""
        return self.entity_extractor.extract_from_text(question)

    def __len__(self):
        raise NotImplementedError("Subclass must implement __len__")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclass must implement __getitem__")

    @staticmethod
    def collate_fn(batch):
        """Default collate function."""
        images = [item['image'] for item in batch]
        questions = [item['question'] for item in batch]
        answer_indices = torch.tensor([item['answer_idx'] for item in batch])
        question_ids = [item['question_id'] for item in batch]
        modes = [item['mode'] for item in batch]
        
        return {
            'images': images,
            'questions': questions,
            'answers': answer_indices,
            'question_ids': question_ids,
            'mode': modes[0] if len(set(modes)) == 1 else 'classify'
        }
