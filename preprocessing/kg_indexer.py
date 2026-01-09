"""
Vector Database Indexer for Knowledge Graph.

Creates FAISS index for efficient dense retrieval of knowledge.
"""

import numpy as np
import pickle
import json
import logging
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KGVectorIndexer:
    """Creates and manages vector index for KG retrieval."""

    def __init__(self,
                 embedding_model='sentence-transformers/all-mpnet-base-v2',
                 vector_db='faiss',
                 device='cuda'):
        """
        Initialize vector indexer.

        Args:
            embedding_model: Sentence transformer model name
            vector_db: Vector database type ('faiss' or 'chromadb')
            device: Device for embedding model
        """
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model, device=device)
        self.vector_db_type = vector_db
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        self.index = None
        self.triple_texts = []  # List of (subject, relation, object) text
        self.triple_metadata = []  # List of metadata dicts
        self.entity2id = {}
        self.relation2id = {}

        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def _create_triple_text(self, subject: str, relation: str, obj: str, source: str = '') -> str:
        """
        Convert triple to text for embedding.

        Args:
            subject: Subject entity
            relation: Relation type
            obj: Object entity/value
            source: Knowledge source

        Returns:
            Text representation
        """
        # Format: "subject relation object" with optional source context
        text = f"{subject} {relation} {obj}"
        if source:
            text += f" (from {source})"
        return text

    def index_knowledge_triples(self, kg_data: Dict, batch_size=128):
        """
        Index all knowledge triples from KG.

        Args:
            kg_data: KG data from kg_builder
            batch_size: Batch size for embedding
        """
        logger.info("Indexing knowledge triples...")

        kg = kg_data['kg']
        all_triples = []

        # Extract all triples
        for subject, relations in kg.items():
            for rel_data in relations:
                triple_text = self._create_triple_text(
                    subject,
                    rel_data['relation'],
                    rel_data['object'],
                    rel_data.get('source', '')
                )

                all_triples.append({
                    'text': triple_text,
                    'subject': subject,
                    'relation': rel_data['relation'],
                    'object': rel_data['object'],
                    'source': rel_data.get('source', ''),
                    'score': rel_data.get('score', 1.0)
                })

        logger.info(f"Total triples to index: {len(all_triples)}")

        # Generate embeddings in batches
        self.triple_texts = [t['text'] for t in all_triples]
        self.triple_metadata = all_triples

        logger.info("Generating embeddings...")
        embeddings = []

        for i in tqdm(range(0, len(self.triple_texts), batch_size), desc="Embedding"):
            batch_texts = self.triple_texts[i:i+batch_size]
            batch_embeddings = self.encoder.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings).astype('float32')
        logger.info(f"Generated embeddings: {embeddings.shape}")

        # Create FAISS index
        self._create_faiss_index(embeddings)

        logger.info("Indexing complete!")

    def _create_faiss_index(self, embeddings: np.ndarray):
        """
        Create FAISS index from embeddings.

        Args:
            embeddings: Numpy array of embeddings
        """
        try:
            import faiss
        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-gpu")
            raise

        logger.info("Creating FAISS index...")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create index (using L2 distance on normalized vectors = cosine similarity)
        if embeddings.shape[0] < 1000000:
            # Use flat index for smaller datasets (exact search)
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
        else:
            # Use IVF index for larger datasets (approximate search)
            nlist = min(4096, embeddings.shape[0] // 100)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.train(embeddings)

        # Add embeddings to index
        self.index.add(embeddings)

        logger.info(f"FAISS index created with {self.index.ntotal} vectors")

    def create_entity_index(self, entities: List[str]):
        """
        Create entity-to-ID mappings.

        Args:
            entities: List of entity names
        """
        logger.info("Creating entity index...")

        self.entity2id = {entity: idx for idx, entity in enumerate(sorted(set(entities)))}

        logger.info(f"Indexed {len(self.entity2id)} entities")

    def create_relation_index(self, relations: List[str]):
        """
        Create relation-to-ID mappings.

        Args:
            relations: List of relation types
        """
        logger.info("Creating relation index...")

        self.relation2id = {relation: idx for idx, relation in enumerate(sorted(set(relations)))}

        logger.info(f"Indexed {len(self.relation2id)} relation types")

    def search(self, query: str, top_k=10):
        """
        Search for relevant knowledge triples.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (triple_metadata, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not created. Call index_knowledge_triples first.")

        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True).astype('float32')

        # Normalize
        import faiss
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Return results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.triple_metadata):
                results.append((self.triple_metadata[idx], float(score)))

        return results

    def save(self, output_dir: str):
        """
        Save index and metadata.

        Args:
            output_dir: Directory to save files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save FAISS index
        import faiss
        index_path = os.path.join(output_dir, 'kg_embeddings.index')
        faiss.write_index(self.index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")

        # Save metadata
        metadata_path = os.path.join(output_dir, 'triple_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'triple_texts': self.triple_texts,
                'triple_metadata': self.triple_metadata,
                'embedding_dim': self.embedding_dim
            }, f)
        logger.info(f"Saved metadata to {metadata_path}")

        # Save entity mappings
        entity_path = os.path.join(output_dir, 'entity2id.json')
        with open(entity_path, 'w') as f:
            json.dump(self.entity2id, f, indent=2)
        logger.info(f"Saved entity mappings to {entity_path}")

        # Save relation mappings
        relation_path = os.path.join(output_dir, 'relation2id.json')
        with open(relation_path, 'w') as f:
            json.dump(self.relation2id, f, indent=2)
        logger.info(f"Saved relation mappings to {relation_path}")

    def load(self, index_dir: str):
        """
        Load pre-built index.

        Args:
            index_dir: Directory containing index files
        """
        import os
        import faiss

        # Load FAISS index
        index_path = os.path.join(index_dir, 'kg_embeddings.index')
        self.index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index from {index_path}")

        # Load metadata
        metadata_path = os.path.join(index_dir, 'triple_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.triple_texts = metadata['triple_texts']
            self.triple_metadata = metadata['triple_metadata']
            self.embedding_dim = metadata['embedding_dim']
        logger.info(f"Loaded metadata from {metadata_path}")

        # Load entity mappings
        entity_path = os.path.join(index_dir, 'entity2id.json')
        with open(entity_path, 'r') as f:
            self.entity2id = json.load(f)
        logger.info(f"Loaded {len(self.entity2id)} entity mappings")

        # Load relation mappings
        relation_path = os.path.join(index_dir, 'relation2id.json')
        with open(relation_path, 'r') as f:
            self.relation2id = json.load(f)
        logger.info(f"Loaded {len(self.relation2id)} relation mappings")

    def get_stats(self) -> Dict:
        """Get indexer statistics."""
        return {
            'num_triples': len(self.triple_metadata),
            'num_entities': len(self.entity2id),
            'num_relations': len(self.relation2id),
            'embedding_dim': self.embedding_dim,
            'index_size': self.index.ntotal if self.index else 0
        }
