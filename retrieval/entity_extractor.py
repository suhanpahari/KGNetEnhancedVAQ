"""Entity extraction from questions and images using spaCy."""

import logging
from typing import List
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract entities from text using NER."""

    def __init__(self, spacy_model='en_core_web_sm'):
        """Initialize entity extractor."""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"{spacy_model} not found. Run: python -m spacy download {spacy_model}")
            self.nlp = None

    def extract_from_text(self, text: str) -> List[str]:
        """Extract entities from text."""
        if self.nlp is None:
            return []

        doc = self.nlp(text)
        entities = [ent.text.lower() for ent in doc.ents]

        # Also extract nouns as potential entities
        nouns = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN']]

        # Combine and deduplicate
        all_entities = list(set(entities + nouns))
        return all_entities

    def extract_visual_entities(self, image_objects: List[str]) -> List[str]:
        """Extract entities from detected image objects."""
        return [obj.lower().strip() for obj in image_objects if obj.strip()]

    def batch_extract(self, texts: List[str]) -> List[List[str]]:
        """Batch entity extraction."""
        return [self.extract_from_text(text) for text in texts]
