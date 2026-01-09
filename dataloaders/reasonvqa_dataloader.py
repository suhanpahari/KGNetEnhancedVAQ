"""ReasonVQA Dataset Loader."""

import json
import os
import torch
from base_vqa_dataloader import BaseVQADataloader


class ReasonVQADataloader(BaseVQADataloader):
    """
    ReasonVQA dataset loader.
    
    Dataset Info:
    - Requires multi-step reasoning
    - Includes reasoning rationale/explanation
    - Free-form answers (generation mode)
    - Chain-of-thought reasoning beneficial
    """

    def __init__(self, data_root, data_file, split='train'):
        """
        Initialize ReasonVQA dataloader.
        
        Args:
            data_root: Root directory for images
            data_file: Path to ReasonVQA data JSON file
            split: 'train', 'val', or 'test'
        """
        super().__init__(data_root, split)
        
        # Load data
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        # ReasonVQA format may vary, adjust as needed
        if isinstance(self.data, dict):
            if 'questions' in self.data:
                self.samples = self.data['questions']
            elif 'data' in self.data:
                self.samples = self.data['data']
            else:
                self.samples = list(self.data.values())
        else:
            self.samples = self.data
        
        print(f"Loaded {len(self.samples)} ReasonVQA samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get image path (adjust field names based on actual format)
        image_id = sample.get('image_id', sample.get('img_id', ''))
        image_filename = sample.get('image', sample.get('img_file', f"{image_id}.jpg"))
        image_path = os.path.join(self.data_root, image_filename)
        
        # Load image
        image = self.load_image(image_path)
        
        # Get question
        question = sample.get('question', sample.get('query', ''))
        
        # Get answer
        answer = sample.get('answer', sample.get('response', ''))
        
        # Get reasoning/rationale (if available)
        rationale = sample.get('rationale', sample.get('reasoning', sample.get('explanation', '')))
        
        # Get question ID
        question_id = sample.get('question_id', sample.get('qid', idx))
        
        return {
            'image': image,
            'image_path': image_path,
            'question': question,
            'answer': answer,
            'rationale': rationale,  # Reasoning steps/explanation
            'answer_idx': 0,  # Not used for generation mode
            'question_id': question_id,
            'mode': 'generate'  # ReasonVQA uses generation
        }

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for ReasonVQA."""
        images = [item['image'] for item in batch]
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        rationales = [item['rationale'] for item in batch]
        question_ids = [item['question_id'] for item in batch]
        
        return {
            'images': images,
            'questions': questions,
            'answers': answers,  # Text answers for generation
            'rationales': rationales,  # For training with reasoning
            'question_ids': question_ids,
            'mode': 'generate'
        }
