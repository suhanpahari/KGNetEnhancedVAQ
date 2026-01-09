"""
Multi-source Knowledge Graph Builder.

Constructs unified knowledge graph from multiple sources:
- ConceptNet: Commonsense knowledge
- Wikipedia: Encyclopedic knowledge
- Visual Genome: Visual scene graphs
- Custom: VQA dataset-specific knowledge
"""

import json
import pickle
import requests
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiSourceKGBuilder:
    """Constructs unified knowledge graph from multiple sources."""

    def __init__(self, sources=None):
        """
        Initialize KG builder.

        Args:
            sources: List of knowledge sources to use
        """
        self.sources = sources or ['conceptnet', 'wikipedia', 'visual_genome', 'custom']
        self.unified_kg = defaultdict(list)  # entity -> [(relation, object, source, score)]
        self.entity_set = set()
        self.relation_types = set()

    def build_from_conceptnet(self, entities: List[str], language='en', max_relations_per_entity=20):
        """
        Build KG from ConceptNet API.

        Args:
            entities: List of entities to query
            language: Language code (default: 'en')
            max_relations_per_entity: Maximum relations to fetch per entity
        """
        logger.info(f"Building KG from ConceptNet for {len(entities)} entities...")

        base_url = 'http://api.conceptnet.io'

        for entity in tqdm(entities, desc="ConceptNet"):
            entity_clean = entity.lower().strip().replace(' ', '_')
            url = f'{base_url}/c/{language}/{entity_clean}'

            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    edges = data.get('edges', [])

                    relations_added = 0
                    for edge in edges:
                        if relations_added >= max_relations_per_entity:
                            break

                        # Extract relation information
                        relation = edge.get('rel', {}).get('label', '')
                        start = edge.get('start', {}).get('label', '')
                        end = edge.get('end', {}).get('label', '')
                        weight = edge.get('weight', 1.0)

                        # Add relation if entity is the subject
                        if start.lower().replace(' ', '_') == entity_clean and relation and end:
                            self.unified_kg[entity].append({
                                'relation': relation,
                                'object': end,
                                'source': 'conceptnet',
                                'score': weight
                            })
                            self.entity_set.add(entity)
                            self.entity_set.add(end)
                            self.relation_types.add(relation)
                            relations_added += 1

            except Exception as e:
                logger.warning(f"Error fetching ConceptNet data for {entity}: {e}")

        logger.info(f"ConceptNet: Added {len(self.unified_kg)} entities with relations")

    def build_from_wikipedia(self, entities: List[str], max_sentences=3):
        """
        Build KG from Wikipedia API.

        Args:
            entities: List of entities to query
            max_sentences: Number of summary sentences to extract
        """
        logger.info(f"Building KG from Wikipedia for {len(entities)} entities...")

        try:
            import wikipediaapi
            wiki = wikipediaapi.Wikipedia('en')
        except ImportError:
            logger.warning("wikipedia-api not installed. Skipping Wikipedia source.")
            return

        for entity in tqdm(entities, desc="Wikipedia"):
            try:
                page = wiki.page(entity)

                if page.exists():
                    # Extract summary as entity description
                    summary = page.summary[:500]  # First 500 chars

                    self.unified_kg[entity].append({
                        'relation': 'DefinedAs',
                        'object': summary,
                        'source': 'wikipedia',
                        'score': 1.0
                    })

                    self.entity_set.add(entity)
                    self.relation_types.add('DefinedAs')

            except Exception as e:
                logger.warning(f"Error fetching Wikipedia data for {entity}: {e}")

        logger.info(f"Wikipedia: Added definitions for entities")

    def build_from_visual_genome(self, scene_graph_path: str):
        """
        Build KG from Visual Genome scene graphs.

        Args:
            scene_graph_path: Path to Visual Genome scene graph JSON file
        """
        logger.info(f"Building KG from Visual Genome scene graphs...")

        try:
            with open(scene_graph_path, 'r') as f:
                scene_graphs = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Scene graph file not found: {scene_graph_path}")
            return

        for sg in tqdm(scene_graphs, desc="Visual Genome"):
            # Extract object relationships
            relationships = sg.get('relationships', [])

            for rel in relationships:
                subject = rel.get('subject', {}).get('name', '').lower()
                predicate = rel.get('predicate', '').lower()
                obj = rel.get('object', {}).get('name', '').lower()

                if subject and predicate and obj:
                    self.unified_kg[subject].append({
                        'relation': predicate,
                        'object': obj,
                        'source': 'visual_genome',
                        'score': 1.0
                    })

                    self.entity_set.add(subject)
                    self.entity_set.add(obj)
                    self.relation_types.add(predicate)

            # Extract object attributes
            objects = sg.get('objects', [])
            for obj_data in objects:
                obj_name = obj_data.get('name', '').lower()
                attributes = obj_data.get('attributes', [])

                for attr in attributes:
                    if obj_name and attr:
                        self.unified_kg[obj_name].append({
                            'relation': 'HasAttribute',
                            'object': attr.lower(),
                            'source': 'visual_genome',
                            'score': 1.0
                        })

                        self.entity_set.add(obj_name)
                        self.relation_types.add('HasAttribute')

        logger.info(f"Visual Genome: Added scene graph relations")

    def build_custom_kg_from_training_data(self,
                                           vqa_train_path: str,
                                           min_cooccurrence=5):
        """
        Build custom KG from VQA training data.

        Extracts co-occurrence patterns and answer distributions.

        Args:
            vqa_train_path: Path to VQA training data (IMDB format)
            min_cooccurrence: Minimum co-occurrence count
        """
        logger.info(f"Building custom KG from VQA training data...")

        try:
            import numpy as np
            import spacy

            # Load spaCy for entity extraction
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                return

            # Load VQA training data
            try:
                imdb = np.load(vqa_train_path, allow_pickle=True)
                if len(imdb) > 0 and isinstance(imdb[0], (int, np.integer)):
                    imdb = imdb[1:]  # Skip first element if it's metadata
            except FileNotFoundError:
                logger.warning(f"VQA training file not found: {vqa_train_path}")
                return

            # Track entity co-occurrences
            entity_pairs = defaultdict(int)
            question_answer_pairs = defaultdict(lambda: defaultdict(int))

            for item in tqdm(imdb[:10000], desc="Custom KG"):  # Limit for efficiency
                question = ' '.join(item['question_tokens']) if 'question_tokens' in item else ''
                answer = item.get('answer', '')

                # Extract entities from question
                doc = nlp(question)
                entities = [ent.text.lower() for ent in doc.ents]

                # Track answer patterns
                for entity in entities:
                    question_answer_pairs[entity][answer] += 1

                # Track entity co-occurrences
                for i, e1 in enumerate(entities):
                    for e2 in entities[i+1:]:
                        pair = tuple(sorted([e1, e2]))
                        entity_pairs[pair] += 1

            # Add co-occurrence relations
            for (e1, e2), count in entity_pairs.items():
                if count >= min_cooccurrence:
                    self.unified_kg[e1].append({
                        'relation': 'CoOccursWith',
                        'object': e2,
                        'source': 'custom',
                        'score': min(count / 100.0, 1.0)  # Normalize
                    })
                    self.entity_set.add(e1)
                    self.entity_set.add(e2)
                    self.relation_types.add('CoOccursWith')

            # Add frequent answer associations
            for entity, answers in question_answer_pairs.items():
                # Get top 3 most frequent answers
                top_answers = sorted(answers.items(), key=lambda x: x[1], reverse=True)[:3]
                for ans, count in top_answers:
                    if count >= min_cooccurrence:
                        self.unified_kg[entity].append({
                            'relation': 'FrequentlyAnsweredBy',
                            'object': ans,
                            'source': 'custom',
                            'score': min(count / 50.0, 1.0)
                        })
                        self.entity_set.add(entity)
                        self.relation_types.add('FrequentlyAnsweredBy')

        except Exception as e:
            logger.error(f"Error building custom KG: {e}")

        logger.info(f"Custom KG: Added co-occurrence and answer patterns")

    def merge_sources(self):
        """Merge and deduplicate relations from all sources."""
        logger.info("Merging knowledge from all sources...")

        # Deduplicate relations (keep highest score)
        for entity, relations in self.unified_kg.items():
            # Group by (relation, object)
            relation_map = {}
            for rel in relations:
                key = (rel['relation'], rel['object'])
                if key not in relation_map or rel['score'] > relation_map[key]['score']:
                    relation_map[key] = rel

            # Update with deduplicated relations
            self.unified_kg[entity] = list(relation_map.values())

        logger.info(f"Total entities: {len(self.entity_set)}")
        logger.info(f"Total relation types: {len(self.relation_types)}")
        total_triples = sum(len(rels) for rels in self.unified_kg.values())
        logger.info(f"Total knowledge triples: {total_triples}")

    def save(self, output_path: str):
        """
        Save unified KG to file.

        Args:
            output_path: Path to save the KG
        """
        kg_data = {
            'kg': dict(self.unified_kg),
            'entities': list(self.entity_set),
            'relations': list(self.relation_types),
            'metadata': {
                'sources': self.sources,
                'num_entities': len(self.entity_set),
                'num_relations': len(self.relation_types),
                'num_triples': sum(len(rels) for rels in self.unified_kg.values())
            }
        }

        with open(output_path, 'wb') as f:
            pickle.dump(kg_data, f)

        logger.info(f"Saved unified KG to {output_path}")

    def load(self, kg_path: str):
        """
        Load pre-built KG from file.

        Args:
            kg_path: Path to the KG file
        """
        with open(kg_path, 'rb') as f:
            kg_data = pickle.load(f)

        self.unified_kg = defaultdict(list, kg_data['kg'])
        self.entity_set = set(kg_data['entities'])
        self.relation_types = set(kg_data['relations'])

        logger.info(f"Loaded KG from {kg_path}")
        logger.info(f"Entities: {len(self.entity_set)}, Relations: {len(self.relation_types)}")

    def get_entity_relations(self, entity: str) -> List[Dict]:
        """
        Get all relations for an entity.

        Args:
            entity: Entity name

        Returns:
            List of relation dictionaries
        """
        return self.unified_kg.get(entity, [])

    def get_statistics(self) -> Dict:
        """Get KG statistics."""
        return {
            'num_entities': len(self.entity_set),
            'num_relation_types': len(self.relation_types),
            'num_triples': sum(len(rels) for rels in self.unified_kg.values()),
            'avg_relations_per_entity': sum(len(rels) for rels in self.unified_kg.values()) / max(len(self.unified_kg), 1),
            'sources': self.sources
        }
