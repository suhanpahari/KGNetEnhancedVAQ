"""BLIP-2 Vision-Language Encoder."""

import torch
import torch.nn as nn
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BLIP2VisionLanguageEncoder(nn.Module):
    """BLIP-2 for vision-language encoding."""

    def __init__(self, model_name='Salesforce/blip2-opt-2.7b', freeze_vision=True, device='cuda'):
        super().__init__()
        logger.info(f"Loading BLIP-2 model: {model_name}")

        self.device = device
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        ).to(device)

        # Add projection layers to ensure 768-dim output
        vision_hidden = self.model.vision_model.config.hidden_size
        qformer_hidden = self.model.qformer.config.hidden_size

        if vision_hidden != 768:
            self.vision_projection = nn.Linear(vision_hidden, 768).to(device)
            logger.info(f"Added vision projection: {vision_hidden} → 768")
        else:
            self.vision_projection = None

        if qformer_hidden != 768:
            self.qformer_projection = nn.Linear(qformer_hidden, 768).to(device)
            logger.info(f"Added Q-Former projection: {qformer_hidden} → 768")
        else:
            self.qformer_projection = None

        if freeze_vision:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            logger.info("Vision encoder frozen")

        self.model.eval()

    def extract_vision_features(self, images):
        """Extract vision features from images."""
        if isinstance(images[0], str):
            images = [Image.open(img).convert('RGB') for img in images]

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=inputs.pixel_values,
                return_dict=True
            )

        # Pool vision features
        vision_embeds = vision_outputs.last_hidden_state.mean(dim=1)  # (B, hidden_dim)

        # Project to 768 if necessary
        if self.vision_projection is not None:
            vision_embeds = self.vision_projection(vision_embeds)

        return vision_embeds  # (B, 768)

    def extract_qformer_features(self, images, questions):
        """Extract Q-Former aligned features."""
        if isinstance(images[0], str):
            images = [Image.open(img).convert('RGB') for img in images]

        inputs = self.processor(images=images, text=questions, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            # Get Q-Former outputs
            vision_outputs = self.model.vision_model(pixel_values=inputs.pixel_values)
            image_embeds = vision_outputs.last_hidden_state

            # Q-Former processing
            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=torch.ones(image_embeds.shape[:2], dtype=torch.long, device=self.device),
                return_dict=True
            )

            qformer_feats = query_outputs.last_hidden_state.mean(dim=1)  # (B, hidden_dim)

            # Project to 768 if necessary
            if self.qformer_projection is not None:
                qformer_feats = self.qformer_projection(qformer_feats)

        return qformer_feats  # (B, 768)

    def forward(self, images, questions):
        """Full forward pass."""
        vision_feats = self.extract_vision_features(images)
        qformer_feats = self.extract_qformer_features(images, questions)

        # Use Q-Former features as text features (already vision-language aligned)
        # This is better than raw embeddings since Q-Former does the alignment
        text_feats = qformer_feats  # (B, 768)

        return {
            'vision_features': vision_feats,      # (B, 768)
            'qformer_features': qformer_feats,    # (B, 768)
            'text_features': text_feats           # (B, 768)
        }
