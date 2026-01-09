"""
LLM-based Knowledge Graph Completion.

Uses Llama-3-8B-Instruct to:
- Complete missing relations between entities
- Infer entity properties
- Expand entity context
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMKGCompletion:
    """Uses LLM to complete and enrich knowledge graphs."""

    def __init__(self,
                 llm_model='meta-llama/Llama-3-8B-Instruct',
                 device='cuda',
                 load_in_8bit=True):
        """
        Initialize LLM for KG completion.

        Args:
            llm_model: HuggingFace model name
            device: Device to load model on
            load_in_8bit: Use 8-bit quantization for memory efficiency
        """
        logger.info(f"Loading LLM: {llm_model}")

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)

        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with 8-bit quantization
        if load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    llm_model,
                    quantization_config=quantization_config,
                    device_map='auto',
                    torch_dtype=torch.float16
                )
            except Exception as e:
                logger.warning(f"8-bit loading failed: {e}. Loading in float16.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    llm_model,
                    device_map='auto',
                    torch_dtype=torch.float16
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                device_map='auto',
                torch_dtype=torch.float16
            )

        self.model.eval()
        logger.info("LLM loaded successfully")

    def _generate_response(self, prompt: str, max_new_tokens=100) -> str:
        """
        Generate LLM response.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        # Format prompt for instruction-following
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        inputs = self.tokenizer(formatted_prompt, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the new tokens
        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated.strip()

    def complete_missing_relations(self, entity1: str, entity2: str) -> List[str]:
        """
        Infer semantic relations between two entities.

        Args:
            entity1: First entity
            entity2: Second entity

        Returns:
            List of inferred relations
        """
        prompt = f"""Given two entities: "{entity1}" and "{entity2}", what are the semantic relations between them?

Provide 2-3 concise factual relations in the format:
- {entity1} [relation] {entity2}

Be specific and factual. Only include relations that are commonly known."""

        try:
            response = self._generate_response(prompt, max_new_tokens=150)

            # Parse relations from response
            relations = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    # Extract relation between brackets
                    if '[' in line and ']' in line:
                        rel = line[line.find('[')+1:line.find(']')].strip()
                        if rel:
                            relations.append(rel)

            return relations[:3]  # Limit to top 3

        except Exception as e:
            logger.warning(f"Error completing relations for {entity1}-{entity2}: {e}")
            return []

    def infer_entity_properties(self, entity: str) -> List[Dict[str, str]]:
        """
        Infer properties and attributes of an entity.

        Args:
            entity: Entity name

        Returns:
            List of property dictionaries with 'relation' and 'value'
        """
        prompt = f"""What are the key properties and attributes of "{entity}"?

Provide 3-5 factual properties in the format:
- [Property]: [Value]

For example:
- Type: animal
- Color: typically brown or black
- Habitat: forests

Be concise and factual."""

        try:
            response = self._generate_response(prompt, max_new_tokens=200)

            # Parse properties
            properties = []
            for line in response.split('\n'):
                line = line.strip()
                if ':' in line and (line.startswith('-') or line.startswith('•')):
                    # Remove bullet point
                    line = line.lstrip('- •').strip()
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        prop_name = parts[0].strip()
                        prop_value = parts[1].strip()
                        properties.append({
                            'relation': f'Has{prop_name}',
                            'value': prop_value
                        })

            return properties[:5]  # Limit to top 5

        except Exception as e:
            logger.warning(f"Error inferring properties for {entity}: {e}")
            return []

    def expand_entity_context(self, entity: str, existing_relations: List[str]) -> List[Dict[str, str]]:
        """
        Expand entity knowledge beyond existing relations.

        Args:
            entity: Entity name
            existing_relations: List of existing relation types

        Returns:
            List of additional relation dictionaries
        """
        relations_str = ', '.join(existing_relations) if existing_relations else 'none'

        prompt = f"""Given the entity "{entity}" with known relations: {relations_str}, suggest 3-5 additional relevant facts or relations that would be useful for visual question answering.

Format:
- {entity} [relation] [object]

Focus on visually relevant or commonly queried information."""

        try:
            response = self._generate_response(prompt, max_new_tokens=200)

            # Parse additional relations
            additional_rels = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    # Parse relation and object
                    if '[' in line and ']' in line:
                        parts = line.split('[')
                        if len(parts) >= 3:
                            rel = parts[1].split(']')[0].strip()
                            obj = parts[2].split(']')[0].strip()
                            if rel and obj:
                                additional_rels.append({
                                    'relation': rel,
                                    'object': obj
                                })

            return additional_rels[:5]

        except Exception as e:
            logger.warning(f"Error expanding context for {entity}: {e}")
            return []

    def enrich_kg(self, kg_data: Dict, max_entities=1000, completion_types=['relations', 'properties']):
        """
        Enrich entire knowledge graph with LLM completions.

        Args:
            kg_data: KG data dictionary from kg_builder
            max_entities: Maximum entities to process (for efficiency)
            completion_types: Types of completion to perform

        Returns:
            Enriched KG data
        """
        logger.info(f"Enriching KG with LLM completions (max {max_entities} entities)...")

        kg = kg_data['kg']
        entities = list(kg_data['entities'])[:max_entities]

        enriched_count = 0

        for entity in tqdm(entities, desc="LLM Enrichment"):
            # Get existing relations
            existing_rels = kg.get(entity, [])
            existing_rel_types = list(set(r['relation'] for r in existing_rels))

            # Add inferred properties
            if 'properties' in completion_types:
                properties = self.infer_entity_properties(entity)
                for prop in properties:
                    kg[entity].append({
                        'relation': prop['relation'],
                        'object': prop['value'],
                        'source': 'llm_completion',
                        'score': 0.8  # Lower confidence for LLM-generated
                    })
                    enriched_count += len(properties)

            # Expand context
            if 'context' in completion_types:
                additional_rels = self.expand_entity_context(entity, existing_rel_types)
                for rel in additional_rels:
                    kg[entity].append({
                        'relation': rel['relation'],
                        'object': rel['object'],
                        'source': 'llm_completion',
                        'score': 0.7
                    })
                    enriched_count += len(additional_rels)

        # Update metadata
        kg_data['kg'] = kg
        kg_data['metadata']['llm_enriched'] = True
        kg_data['metadata']['enriched_triples'] = enriched_count

        logger.info(f"Added {enriched_count} LLM-generated triples")

        return kg_data

    def complete_entity_pairs(self, entity_pairs: List[Tuple[str, str]]) -> Dict:
        """
        Complete relations for specific entity pairs.

        Args:
            entity_pairs: List of (entity1, entity2) tuples

        Returns:
            Dictionary mapping entity pairs to inferred relations
        """
        logger.info(f"Completing relations for {len(entity_pairs)} entity pairs...")

        completions = {}

        for e1, e2 in tqdm(entity_pairs, desc="Pair Completion"):
            relations = self.complete_missing_relations(e1, e2)
            if relations:
                completions[(e1, e2)] = relations

        logger.info(f"Completed {len(completions)} entity pairs")

        return completions

    def save_completions(self, completions: Dict, output_path: str):
        """Save LLM completions to file."""
        with open(output_path, 'wb') as f:
            pickle.dump(completions, f)
        logger.info(f"Saved completions to {output_path}")

    def cleanup(self):
        """Free up GPU memory."""
        del self.model
        torch.cuda.empty_cache()
        logger.info("LLM resources freed")
