"""
RAG-style Knowledge Retriever.

Implements dense retrieval from vector index for knowledge injection.
Replaces simple CEL (Cross-modal Entity Linking) with advanced RAG.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
import sys
import os

# Add preprocessing to path for importing indexer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
from kg_indexer import KGVectorIndexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGKnowledgeRetriever:
    """RAG-based knowledge retrieval from vector index."""

    def __init__(self,
                 index_path: str,
                 embedding_model='sentence-transformers/all-mpnet-base-v2',
                 rerank_model=None,
                 top_k=10,
                 use_reranking=False,
                 device='cuda'):
        """
        Initialize RAG retriever.

        Args:
            index_path: Path to directory containing FAISS index
            embedding_model: Sentence transformer model
            rerank_model: Cross-encoder model for reranking (optional)
            top_k: Number of results to retrieve
            use_reranking: Whether to use cross-encoder reranking
            device: Device for models
        """
        logger.info("Initializing RAG Knowledge Retriever...")

        self.top_k = top_k
        self.use_reranking = use_reranking
        self.device = device

        # Load indexer
        self.indexer = KGVectorIndexer(
            embedding_model=embedding_model,
            device=device
        )
        self.indexer.load(index_path)

        logger.info(f"Loaded index with {self.indexer.get_stats()['num_triples']} triples")

        # Load reranker if needed
        if use_reranking and rerank_model:
            logger.info(f"Loading reranker: {rerank_model}")
            self.reranker = CrossEncoder(rerank_model, device=device)
        else:
            self.reranker = None

    def retrieve_for_question(self,
                              question: str,
                              entities: Optional[List[str]] = None,
                              image_context: Optional[str] = None,
                              top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve relevant knowledge for a question.

        Args:
            question: Question text
            entities: Extracted entities from question (optional)
            image_context: Visual context description (optional)
            top_k: Override default top_k

        Returns:
            List of knowledge dictionaries with keys:
                - text: Knowledge triple text
                - subject: Subject entity
                - relation: Relation type
                - object: Object entity/value
                - source: Knowledge source
                - score: Relevance score
        """
        k = top_k if top_k is not None else self.top_k

        # Create enhanced query
        query = self._create_query_embedding(question, entities, image_context)

        # Search index
        results = self.indexer.search(query, top_k=k * 2 if self.use_reranking else k)

        knowledge_list = []
        for triple_meta, score in results:
            knowledge_list.append({
                'text': triple_meta['text'],
                'subject': triple_meta['subject'],
                'relation': triple_meta['relation'],
                'object': triple_meta['object'],
                'source': triple_meta['source'],
                'score': score,
                'kg_score': triple_meta.get('score', 1.0)  # Original KG score
            })

        # Rerank if enabled
        if self.use_reranking and self.reranker is not None:
            knowledge_list = self._rerank_knowledge(question, knowledge_list, top_k=k)

        return knowledge_list[:k]

    def _create_query_embedding(self,
                                question: str,
                                entities: Optional[List[str]] = None,
                                image_context: Optional[str] = None) -> str:
        """
        Create enhanced query for retrieval.

        Args:
            question: Question text
            entities: Entities in question
            image_context: Visual context

        Returns:
            Query string
        """
        query_parts = [question]

        # Add entities for focus
        if entities:
            query_parts.append("Entities: " + ", ".join(entities))

        # Add image context if available
        if image_context:
            query_parts.append("Context: " + image_context)

        return " ".join(query_parts)

    def _rerank_knowledge(self,
                         question: str,
                         knowledge_list: List[Dict],
                         top_k: int) -> List[Dict]:
        """
        Rerank retrieved knowledge using cross-encoder.

        Args:
            question: Question text
            knowledge_list: List of retrieved knowledge
            top_k: Number to keep after reranking

        Returns:
            Reranked knowledge list
        """
        if not knowledge_list:
            return knowledge_list

        # Create question-knowledge pairs
        pairs = [[question, k['text']] for k in knowledge_list]

        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)

        # Update scores
        for i, score in enumerate(rerank_scores):
            knowledge_list[i]['rerank_score'] = float(score)
            # Combine original and rerank scores
            knowledge_list[i]['final_score'] = (
                0.3 * knowledge_list[i]['score'] +
                0.7 * float(score)
            )

        # Sort by final score
        knowledge_list.sort(key=lambda x: x.get('final_score', x['score']), reverse=True)

        return knowledge_list[:top_k]

    def multi_source_retrieval(self,
                               question: str,
                               entities: List[str],
                               source_weights: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        Retrieve from multiple sources with source-specific weighting.

        Args:
            question: Question text
            entities: Entities in question
            source_weights: Dictionary mapping source names to weights

        Returns:
            Merged and reweighted knowledge list
        """
        # Default weights
        if source_weights is None:
            source_weights = {
                'conceptnet': 1.0,
                'wikipedia': 0.9,
                'visual_genome': 1.1,  # Higher weight for visual knowledge
                'custom': 0.8,
                'llm_completion': 0.7
            }

        # Retrieve knowledge
        knowledge_list = self.retrieve_for_question(
            question,
            entities,
            top_k=self.top_k * 2  # Retrieve more for diversity
        )

        # Reweight by source
        for k in knowledge_list:
            source = k.get('source', 'unknown')
            weight = source_weights.get(source, 1.0)
            k['score'] = k['score'] * weight

        # Re-sort by weighted scores
        knowledge_list.sort(key=lambda x: x['score'], reverse=True)

        # Diversify results (avoid too many from same source)
        diverse_knowledge = self._diversify_sources(knowledge_list, self.top_k)

        return diverse_knowledge

    def _diversify_sources(self,
                          knowledge_list: List[Dict],
                          target_count: int,
                          max_per_source: int = 5) -> List[Dict]:
        """
        Ensure diversity in knowledge sources.

        Args:
            knowledge_list: List of knowledge
            target_count: Target number of knowledge items
            max_per_source: Maximum items per source

        Returns:
            Diversified knowledge list
        """
        source_counts = {}
        diverse_list = []

        for k in knowledge_list:
            source = k.get('source', 'unknown')
            count = source_counts.get(source, 0)

            if count < max_per_source:
                diverse_list.append(k)
                source_counts[source] = count + 1

            if len(diverse_list) >= target_count:
                break

        return diverse_list

    def batch_retrieve(self,
                       questions: List[str],
                       entities_list: Optional[List[List[str]]] = None,
                       top_k: Optional[int] = None) -> List[List[Dict]]:
        """
        Batch retrieval for multiple questions.

        Args:
            questions: List of questions
            entities_list: List of entity lists for each question
            top_k: Number of results per question

        Returns:
            List of knowledge lists
        """
        if entities_list is None:
            entities_list = [None] * len(questions)

        results = []
        for q, ents in zip(questions, entities_list):
            knowledge = self.retrieve_for_question(q, ents, top_k=top_k)
            results.append(knowledge)

        return results

    def get_entity_knowledge(self, entity: str, top_k: int = 5) -> List[Dict]:
        """
        Get direct knowledge about a specific entity.

        Args:
            entity: Entity name
            top_k: Number of results

        Returns:
            List of knowledge about the entity
        """
        query = f"What is {entity}? Tell me about {entity}."
        return self.retrieve_for_question(query, entities=[entity], top_k=top_k)

    def get_relation_knowledge(self,
                               entity1: str,
                               entity2: str,
                               top_k: int = 3) -> List[Dict]:
        """
        Get knowledge about relation between two entities.

        Args:
            entity1: First entity
            entity2: Second entity
            top_k: Number of results

        Returns:
            List of relational knowledge
        """
        query = f"How are {entity1} and {entity2} related?"
        return self.retrieve_for_question(
            query,
            entities=[entity1, entity2],
            top_k=top_k
        )
