"""GQA Dataset Loader."""

import json
import os
import torch
from base_vqa_dataloader import BaseVQADataloader


class GQADataloader(BaseVQADataloader):
    """
    GQA (Visual Genome Question Answering) dataset loader.
    
    Dataset Info:
    - 1,878 answer classes
    - Scene graph annotations available
    - Balanced evaluation split
    """

    def __init__(self, data_root, questions_file, scene_graph_file=None, 
                 answer_vocab_file=None, split='train'):
        """
        Initialize GQA dataloader.
        
        Args:
            data_root: Root directory for images (Visual Genome images)
            questions_file: Path to GQA questions JSON file
            scene_graph_file: Path to scene graph JSON (optional)
            answer_vocab_file: Path to answer vocabulary JSON
            split: 'train', 'val', or 'test'
        """
        super().__init__(data_root, split)
        
        # Load questions
        with open(questions_file, 'r') as f:
            self.questions = json.load(f)
        
        # Convert to list format
        self.question_ids = list(self.questions.keys())
        
        # Load scene graphs (optional)
        self.scene_graphs = {}
        if scene_graph_file and os.path.exists(scene_graph_file):
            with open(scene_graph_file, 'r') as f:
                self.scene_graphs = json.load(f)
            print(f"Loaded {len(self.scene_graphs)} scene graphs")
        
        # Load answer vocabulary
        if answer_vocab_file and os.path.exists(answer_vocab_file):
            with open(answer_vocab_file, 'r') as f:
                self.answer_dict = json.load(f)
        else:
            # Build answer vocab from training data
            self.answer_dict = self._build_answer_vocab()
        
        print(f"Loaded {len(self.question_ids)} GQA questions")
        print(f"Answer vocabulary size: {len(self.answer_dict)}")

    def _build_answer_vocab(self):
        """Build answer vocabulary from questions."""
        answers = set()
        for qid, q_data in self.questions.items():
            if 'answer' in q_data:
                answers.add(q_data['answer'])
        
        answer_dict = {ans: idx for idx, ans in enumerate(sorted(answers))}
        return answer_dict

    def __len__(self):
        return len(self.question_ids)

    def __getitem__(self, idx):
        qid = self.question_ids[idx]
        q_data = self.questions[qid]
        
        # Get image ID and path
        image_id = q_data['imageId']
        image_path = os.path.join(self.data_root, f"{image_id}.jpg")
        
        # Load image
        image = self.load_image(image_path)
        
        # Get question
        question = q_data['question']
        
        # Get answer
        answer = q_data.get('answer', '')
        answer_idx = self.answer_dict.get(answer, 0)
        
        # Get scene graph info (if available)
        scene_graph = self.scene_graphs.get(image_id, {})
        
        return {
            'image': image,
            'image_path': image_path,
            'question': question,
            'answer': answer,
            'answer_idx': answer_idx,
            'question_id': qid,
            'scene_graph': scene_graph,
            'mode': 'classify'
        }

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for GQA."""
        images = [item['image'] for item in batch]
        questions = [item['question'] for item in batch]
        answer_indices = torch.tensor([item['answer_idx'] for item in batch])
        question_ids = [item['question_id'] for item in batch]
        scene_graphs = [item['scene_graph'] for item in batch]
        
        return {
            'images': images,
            'questions': questions,
            'answers': answer_indices,
            'question_ids': question_ids,
            'scene_graphs': scene_graphs,
            'mode': 'classify'
        }
