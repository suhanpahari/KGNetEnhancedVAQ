"""
LLM-based Knowledge Summarizer.

Uses Flan-T5-Large to summarize retrieved knowledge into compact representations.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMKnowledgeSummarizer:
    """LLM-based summarization of retrieved knowledge."""

    def __init__(self,
                 llm_model='google/flan-t5-large',
                 device='cuda',
                 max_input_length=512,
                 max_output_length=128):
        """
        Initialize knowledge summarizer.

        Args:
            llm_model: Seq2Seq model for summarization
            device: Device to load model
            max_input_length: Max input tokens
            max_output_length: Max summary tokens
        """
        logger.info(f"Loading summarization model: {llm_model}")

        self.device = device
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        ).to(device)

        # Add projection layer to ensure 768-dim output
        import torch.nn as nn
        encoder_hidden = self.model.config.d_model  # Flan-T5: d_model attribute
        if encoder_hidden != 768:
            self.projection = nn.Linear(encoder_hidden, 768).to(device)
            logger.info(f"Added encoder projection: {encoder_hidden} → 768")
        else:
            self.projection = None

        self.model.eval()
        logger.info("Summarization model loaded")

    def summarize_knowledge(self,
                           question: str,
                           retrieved_triples: List[Dict],
                           max_triples=10) -> str:
        """
        Summarize retrieved knowledge.

        Args:
            question: Question text
            retrieved_triples: List of knowledge dictionaries
            max_triples: Maximum triples to include

        Returns:
            Summary text
        """
        if not retrieved_triples:
            return ""

        # Format knowledge triples
        knowledge_texts = []
        for triple in retrieved_triples[:max_triples]:
            text = triple.get('text', '')
            if not text:
                text = f"{triple['subject']} {triple['relation']} {triple['object']}"
            knowledge_texts.append(text)

        knowledge_str = "; ".join(knowledge_texts)

        # Create prompt
        prompt = f"""Question: {question}
Retrieved Knowledge: {knowledge_str}

Summarize the most relevant facts for answering this question in 2-3 concise sentences."""

        # Generate summary
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                max_length=self.max_input_length,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_length=self.max_output_length,
                num_beams=4,
                early_stopping=True,
                temperature=0.7
            )

            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return summary.strip()

    def generate_knowledge_features(self, summary: str) -> torch.Tensor:
        """
        Generate feature embedding from summary.

        Args:
            summary: Summary text

        Returns:
            Feature tensor (768-dim)
        """
        if not summary:
            # Return zero vector for empty summary
            return torch.zeros(768, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            inputs = self.tokenizer(
                summary,
                max_length=512,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            # Get encoder hidden states
            encoder_outputs = self.model.encoder(**inputs)
            hidden_states = encoder_outputs.last_hidden_state  # (1, seq_len, hidden_dim)

            # Mean pooling
            features = hidden_states.mean(dim=1).squeeze(0)  # (hidden_dim,)

            # Project to 768 if necessary
            if self.projection is not None:
                features = self.projection(features)

        return features  # (768,)

    def batch_summarize(self,
                       questions: List[str],
                       retrieved_knowledge_list: List[List[Dict]]) -> List[str]:
        """
        Batch summarization.

        Args:
            questions: List of questions
            retrieved_knowledge_list: List of knowledge lists

        Returns:
            List of summaries
        """
        summaries = []
        for q, knowledge in zip(questions, retrieved_knowledge_list):
            summary = self.summarize_knowledge(q, knowledge)
            summaries.append(summary)

        return summaries

    def batch_generate_features(self, summaries: List[str]) -> torch.Tensor:
        """
        Generate features for batch of summaries.

        Args:
            summaries: List of summary texts

        Returns:
            Feature tensor (batch_size, 768)
        """
        features_list = []
        for summary in summaries:
            feats = self.generate_knowledge_features(summary)
            features_list.append(feats)

        return torch.stack(features_list)
