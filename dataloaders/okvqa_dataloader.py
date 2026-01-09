"""OK-VQA Dataset Loader."""

import json
import os
import torch
from base_vqa_dataloader import BaseVQADataloader


class OKVQADataloader(BaseVQADataloader):
    """
    OK-VQA (Outside Knowledge VQA) dataset loader.
    
    Dataset Info:
    - Requires external commonsense knowledge
    - Free-form answers (can use generation mode)
    - Multiple acceptable answers per question
    - Built on COCO images
    """

    def __init__(self, data_root, questions_file, annotations_file=None, split='train', answer_vocab=None):
        """
        Initialize OK-VQA dataloader.

        Args:
            data_root: Root directory for COCO images
            questions_file: Path to questions JSON file
            annotations_file: Path to annotations JSON file (optional for test)
            split: 'train', 'val', or 'test'
            answer_vocab: Optional dict mapping answers to indices for classify mode
        """
        super().__init__(data_root, split)

        self.answer_vocab = answer_vocab or {}

        # Load questions
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)
            self.questions = questions_data['questions']

        # Load annotations (if available)
        self.annotations = {}
        if annotations_file and os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                anno_data = json.load(f)
                for anno in anno_data['annotations']:
                    self.annotations[anno['question_id']] = anno
            print(f"Loaded {len(self.annotations)} annotations")

        print(f"Loaded {len(self.questions)} OK-VQA questions")

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q_data = self.questions[idx]
        question_id = q_data['question_id']
        
        # Get image path
        image_id = q_data['image_id']
        image_filename = f"COCO_{self.split}2014_{image_id:012d}.jpg"
        image_path = os.path.join(self.data_root, image_filename)
        
        # Load image
        image = self.load_image(image_path)
        
        # Get question
        question = q_data['question']
        
        # Get answers (multiple acceptable answers)
        answers = []
        answer_text = ""
        if question_id in self.annotations:
            anno = self.annotations[question_id]
            answers = [ans['answer'] for ans in anno.get('answers', [])]
            # Most common answer
            if answers:
                answer_text = max(set(answers), key=answers.count)
        
        # For OK-VQA, we can use generation mode
        # But also support classification with answer vocabulary
        answer_idx = self.answer_vocab.get(answer_text, 0) if self.answer_vocab else 0
        
        return {
            'image': image,
            'image_path': image_path,
            'question': question,
            'answer': answer_text,
            'answers_list': answers,  # Multiple acceptable answers
            'answer_idx': answer_idx,
            'question_id': question_id,
            'mode': 'generate'  # OK-VQA typically uses generation
        }

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for OK-VQA."""
        images = [item['image'] for item in batch]
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        answers_lists = [item['answers_list'] for item in batch]
        question_ids = [item['question_id'] for item in batch]
        
        return {
            'images': images,
            'questions': questions,
            'answers': answers,  # For generation, use text answers
            'answers_lists': answers_lists,
            'question_ids': question_ids,
            'mode': 'generate'
        }
