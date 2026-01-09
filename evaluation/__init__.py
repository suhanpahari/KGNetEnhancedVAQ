"""Evaluation module for VQA datasets."""

from .vqa_v2_eval import VQAv2Evaluator
from .gqa_eval import GQAEvaluator
from .okvqa_eval import OKVQAEvaluator
from .reasonvqa_eval import ReasonVQAEvaluator

__all__ = [
    'VQAv2Evaluator',
    'GQAEvaluator',
    'OKVQAEvaluator',
    'ReasonVQAEvaluator',
]
