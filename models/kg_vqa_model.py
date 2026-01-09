"""Unified KG-VQA Model integrating all components."""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vision_language.blip2_encoder import BLIP2VisionLanguageEncoder
from retrieval.rag_retriever import RAGKnowledgeRetriever
from retrieval.knowledge_summarizer import LLMKnowledgeSummarizer
from retrieval.entity_extractor import EntityExtractor
from reasoning.fusion_layer import MultiModalFusionLayer
from reasoning.llm_reasoning_head import LLMReasoningHead


class KGVQAModel(nn.Module):
    """Complete KG-aware VQA model."""

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Vision-Language Encoder
        self.vision_encoder = BLIP2VisionLanguageEncoder(
            model_name=config.get('blip_model', 'Salesforce/blip2-opt-2.7b'),
            freeze_vision=config.get('freeze_vision', True)
        )
        
        # RAG Retrieval
        self.rag_retriever = RAGKnowledgeRetriever(
            index_path=config['kg_index_path'],
            top_k=config.get('retrieval_top_k', 10)
        )
        
        # Knowledge Summarizer
        self.knowledge_summarizer = LLMKnowledgeSummarizer(
            llm_model=config.get('summarizer_model', 'google/flan-t5-large')
        )
        
        # Entity Extractor
        self.entity_extractor = EntityExtractor()
        
        # Multi-Modal Fusion
        self.fusion_layer = MultiModalFusionLayer(
            vision_dim=768,
            text_dim=768,
            kg_dim=768,
            hidden_dim=config.get('hidden_dim', 1024)
        )
        
        # LLM Reasoning Head
        self.reasoning_head = LLMReasoningHead(
            llm_model=config.get('llm_model', 'meta-llama/Llama-3-8B-Instruct'),
            num_answers=config.get('num_answers', 3129),
            use_lora=config.get('use_lora', True),
            load_in_8bit=config.get('load_in_8bit', True)
        )

    def forward(self, images, questions, mode='classify', entities=None):
        """
        Forward pass.
        
        Args:
            images: List of image paths or PIL images
            questions: List of question strings
            mode: 'classify' or 'generate'
            entities: Pre-extracted entities (optional)
        
        Returns:
            logits or generated text
        """
        batch_size = len(questions)
        
        # 1. Extract entities if not provided
        if entities is None:
            entities = [self.entity_extractor.extract_from_text(q) for q in questions]
        
        # 2. Retrieve knowledge
        retrieved_knowledge = []
        for q, ents in zip(questions, entities):
            knowledge = self.rag_retriever.retrieve_for_question(q, ents)
            retrieved_knowledge.append(knowledge)
        
        # 3. Summarize knowledge
        kg_summaries = []
        for q, knowledge in zip(questions, retrieved_knowledge):
            summary = self.knowledge_summarizer.summarize_knowledge(q, knowledge)
            kg_summaries.append(summary)
        
        # Generate knowledge features
        kg_feats = self.knowledge_summarizer.batch_generate_features(kg_summaries)
        
        # 4. Extract vision-language features
        vl_outputs = self.vision_encoder(images, questions)
        # Note: qformer_features contain vision-text aligned representations from Q-Former
        vision_feats = vl_outputs['qformer_features']  # Vision features aligned via Q-Former
        text_feats = vl_outputs['text_features']
        
        # 5. Multi-modal fusion
        fused_feats = self.fusion_layer(vision_feats, text_feats, kg_feats)
        
        # 6. LLM reasoning
        outputs = self.reasoning_head(fused_feats, questions, mode=mode)
        
        return outputs

    def train_step(self, batch, criterion):
        """Single training step."""
        images = batch['images']
        questions = batch['questions']
        answers = batch['answers']
        mode = batch.get('mode', 'classify')
        
        outputs = self.forward(images, questions, mode=mode)
        
        if mode == 'classify':
            loss = criterion(outputs, answers)
        else:
            # Generation loss (to be implemented)
            loss = torch.tensor(0.0)
        
        return loss, outputs

    def inference(self, images, questions, mode='classify'):
        """Inference without gradients."""
        with torch.no_grad():
            outputs = self.forward(images, questions, mode=mode)
        return outputs
