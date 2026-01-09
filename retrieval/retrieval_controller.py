"""Adaptive Retrieval Controller for question-type aware retrieval."""

import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveRetrievalController:
    """Dynamically adjusts retrieval based on question type."""

    def __init__(self):
        """Initialize controller."""
        # Question type patterns
        self.question_patterns = {
            'counting': ['how many', 'count', 'number of'],
            'color': ['what color', 'color of'],
            'position': ['where', 'location', 'position'],
            'reasoning': ['why', 'because', 'reason'],
            'knowledge': ['what is', 'who is', 'when', 'which'],
            'yes_no': ['is', 'are', 'does', 'can', 'will']
        }

    def determine_retrieval_strategy(self,
                                     question: str,
                                     dataset_type: str = 'vqa_v2') -> Dict:
        """
        Determine retrieval strategy based on question.

        Args:
            question: Question text
            dataset_type: Dataset type

        Returns:
            Dict with top_k, sources, and retrieval_mode
        """
        question_lower = question.lower()

        # Classify question type
        question_type = self._classify_question(question_lower)

        # Adjust parameters based on type
        if question_type == 'knowledge':
            # Knowledge-intensive questions need more retrieval
            top_k = 20
            sources = ['conceptnet', 'wikipedia', 'visual_genome', 'custom']
            mode = 'dense'
        elif question_type == 'reasoning':
            # Reasoning questions need diverse knowledge
            top_k = 15
            sources = ['conceptnet', 'visual_genome', 'custom']
            mode = 'dense'
        elif question_type in ['counting', 'position']:
            # Visual questions need less external knowledge
            top_k = 5
            sources = ['visual_genome', 'custom']
            mode = 'dense'
        else:
            # Default strategy
            top_k = 10
            sources = ['conceptnet', 'visual_genome', 'custom']
            mode = 'dense'

        # Dataset-specific adjustments
        if dataset_type == 'okvqa':
            top_k = max(top_k, 20)  # OK-VQA always needs more knowledge
        elif dataset_type == 'reasonvqa':
            top_k = max(top_k, 15)

        return {
            'top_k': top_k,
            'sources': sources,
            'retrieval_mode': mode,
            'question_type': question_type
        }

    def _classify_question(self, question: str) -> str:
        """Classify question type."""
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if pattern in question:
                    return q_type
        return 'general'
