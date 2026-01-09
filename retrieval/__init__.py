"""
Retrieval module for RAG-style knowledge retrieval.

This module handles:
- Dense retrieval from vector index
- LLM-based knowledge summarization
- Entity extraction from questions
- Adaptive retrieval strategies
"""

from .rag_retriever import RAGKnowledgeRetriever
from .knowledge_summarizer import LLMKnowledgeSummarizer
from .entity_extractor import EntityExtractor
from .retrieval_controller import AdaptiveRetrievalController

__all__ = [
    'RAGKnowledgeRetriever',
    'LLMKnowledgeSummarizer',
    'EntityExtractor',
    'AdaptiveRetrievalController',
]
