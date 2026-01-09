"""
Main preprocessing script for KG construction, completion, and indexing.

Usage:
    python run_preprocessing.py \
        --sources conceptnet wikipedia visual_genome custom \
        --llm_model meta-llama/Llama-3-8B-Instruct \
        --output_dir ../kg_data/
"""

import argparse
import os
import logging
from kg_builder import MultiSourceKGBuilder
from kg_completion import LLMKGCompletion
from kg_indexer import KGVectorIndexer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='KG Preprocessing Pipeline')

    # Input/Output
    parser.add_argument('--output_dir', type=str, default='../kg_data',
                        help='Output directory for KG files')
    parser.add_argument('--vqa_train_path', type=str,
                        default='../../data/imdb/imdb_mirror_train2014.npy',
                        help='Path to VQA training data')
    parser.add_argument('--scene_graph_path', type=str,
                        default='../../data/visual_genome/scene_graphs.json',
                        help='Path to Visual Genome scene graphs')

    # KG Construction
    parser.add_argument('--sources', nargs='+',
                        default=['conceptnet', 'wikipedia', 'visual_genome', 'custom'],
                        help='Knowledge sources to use')
    parser.add_argument('--entities_file', type=str,
                        default='../../visualbert/kg/entities.json',
                        help='File containing entity list')

    # LLM Completion
    parser.add_argument('--use_llm_completion', action='store_true',
                        help='Use LLM for KG completion')
    parser.add_argument('--llm_model', type=str,
                        default='meta-llama/Llama-3-8B-Instruct',
                        help='LLM model for completion')
    parser.add_argument('--max_entities_llm', type=int, default=1000,
                        help='Max entities to process with LLM')
    parser.add_argument('--completion_types', nargs='+',
                        default=['properties'],
                        help='Types of LLM completion: properties, context, relations')

    # Indexing
    parser.add_argument('--embedding_model', type=str,
                        default='sentence-transformers/all-mpnet-base-v2',
                        help='Embedding model for vector index')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for embedding')

    # Processing options
    parser.add_argument('--skip_build', action='store_true',
                        help='Skip KG building (load existing)')
    parser.add_argument('--skip_completion', action='store_true',
                        help='Skip LLM completion')
    parser.add_argument('--skip_indexing', action='store_true',
                        help='Skip vector indexing')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== Step 1: Build Multi-Source KG ==========
    kg_path = os.path.join(args.output_dir, 'unified_kg.pkl')

    if not args.skip_build:
        logger.info("=" * 60)
        logger.info("STEP 1: Building Multi-Source Knowledge Graph")
        logger.info("=" * 60)

        builder = MultiSourceKGBuilder(sources=args.sources)

        # Load entities
        entities = []
        if os.path.exists(args.entities_file):
            import json
            with open(args.entities_file, 'r') as f:
                entities = json.load(f)
            logger.info(f"Loaded {len(entities)} entities from {args.entities_file}")
        else:
            logger.warning(f"Entities file not found: {args.entities_file}")
            # Use default entities
            entities = ['person', 'car', 'dog', 'cat', 'tree', 'house', 'sky', 'water']

        # Build from each source
        if 'conceptnet' in args.sources:
            builder.build_from_conceptnet(entities, max_relations_per_entity=20)

        if 'wikipedia' in args.sources:
            builder.build_from_wikipedia(entities, max_sentences=3)

        if 'visual_genome' in args.sources and os.path.exists(args.scene_graph_path):
            builder.build_from_visual_genome(args.scene_graph_path)

        if 'custom' in args.sources and os.path.exists(args.vqa_train_path):
            builder.build_custom_kg_from_training_data(args.vqa_train_path)

        # Merge sources
        builder.merge_sources()

        # Save
        builder.save(kg_path)

        # Print statistics
        stats = builder.get_statistics()
        logger.info(f"\nKG Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    else:
        logger.info(f"Skipping KG building. Will load from {kg_path}")

    # ========== Step 2: LLM-based KG Completion (Optional) ==========
    enriched_kg_path = os.path.join(args.output_dir, 'enriched_kg.pkl')

    if args.use_llm_completion and not args.skip_completion:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: LLM-based KG Completion")
        logger.info("=" * 60)

        import pickle
        with open(kg_path, 'rb') as f:
            kg_data = pickle.load(f)

        try:
            llm_completer = LLMKGCompletion(
                llm_model=args.llm_model,
                device='cuda',
                load_in_8bit=True
            )

            # Enrich KG
            enriched_kg_data = llm_completer.enrich_kg(
                kg_data,
                max_entities=args.max_entities_llm,
                completion_types=args.completion_types
            )

            # Save enriched KG
            with open(enriched_kg_path, 'wb') as f:
                pickle.dump(enriched_kg_data, f)

            logger.info(f"Saved enriched KG to {enriched_kg_path}")

            # Cleanup
            llm_completer.cleanup()

            # Use enriched KG for indexing
            kg_path = enriched_kg_path

        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            logger.info("Proceeding with non-enriched KG")

    else:
        logger.info("\nSkipping LLM completion")

    # ========== Step 3: Vector Indexing ==========
    if not args.skip_indexing:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Vector Indexing")
        logger.info("=" * 60)

        import pickle
        with open(kg_path, 'rb') as f:
            kg_data = pickle.load(f)

        indexer = KGVectorIndexer(
            embedding_model=args.embedding_model,
            vector_db='faiss',
            device='cuda'
        )

        # Index triples
        indexer.index_knowledge_triples(kg_data, batch_size=args.batch_size)

        # Create entity and relation indices
        indexer.create_entity_index(kg_data['entities'])
        indexer.create_relation_index(kg_data['relations'])

        # Save index
        indexer.save(args.output_dir)

        # Print statistics
        stats = indexer.get_stats()
        logger.info(f"\nIndex Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Files created:")
    logger.info(f"  - unified_kg.pkl (or enriched_kg.pkl)")
    logger.info(f"  - kg_embeddings.index")
    logger.info(f"  - triple_metadata.pkl")
    logger.info(f"  - entity2id.json")
    logger.info(f"  - relation2id.json")


if __name__ == '__main__':
    main()
