# Dimension and Connection Verification

## Complete Data Flow Dimension Check

### Input Stage
```
Input:
- images: List[PIL.Image] or List[str] (paths)
- questions: List[str]
- batch_size: B
```

---

### 1. BLIP-2 Vision-Language Encoder

**File**: `vision_language/blip2_encoder.py`

```python
# Input
images: List[PIL.Image]  # B images
questions: List[str]      # B questions

# Processing
processor(images, text=questions) → {
    pixel_values: (B, 3, 224, 224),
    input_ids: (B, seq_len)
}

# Vision Model Output
vision_model(pixel_values) → last_hidden_state: (B, num_patches, 1408)
# ViT-L/14 typically: num_patches = 257, hidden = 1408

# Q-Former Output  
qformer_outputs → last_hidden_state: (B, num_queries, 768)
# Default num_queries = 32

# Final Outputs
{
    'vision_features': (B, 768),      # Mean pooled
    'qformer_features': (B, 768),     # Mean pooled over queries
    'text_features': (B, 768)         # Mean pooled over tokens
}
```

**✅ Check**: All outputs are (B, 768) - CORRECT

---

### 2. Entity Extraction + RAG Retrieval

**File**: `retrieval/entity_extractor.py`, `retrieval/rag_retriever.py`

```python
# Entity Extraction
extract_from_text(question: str) → List[str]

# RAG Retrieval
retrieve_for_question(question, entities, top_k=10) → List[Dict]
# Each dict: {'text', 'subject', 'relation', 'object', 'score'}

# Batch retrieval for B questions
batch_knowledge: List[List[Dict]]  # B x top_k knowledge triples
```

**✅ Check**: Returns list structure - CORRECT

---

### 3. Knowledge Summarization

**File**: `retrieval/knowledge_summarizer.py`

```python
# Input
question: str
retrieved_triples: List[Dict]

# Summarization
summarize_knowledge() → str (summary text)

# Feature Generation
generate_knowledge_features(summary: str) → torch.Tensor
# Output: (768,) for single summary

# Batch Generation
batch_generate_features(summaries: List[str]) → torch.Tensor
# Output: (B, 768)
```

**✅ Check**: Output is (B, 768) - CORRECT

---

### 4. Multi-Modal Fusion Layer

**File**: `reasoning/fusion_layer.py`

```python
class MultiModalFusionLayer:
    def __init__(self, vision_dim=768, text_dim=768, kg_dim=768, hidden_dim=1024):
        # Projection layers
        self.vision_proj: Linear(768 → 1024)
        self.text_proj: Linear(768 → 1024)
        self.kg_proj: Linear(768 → 1024)
        
        # Cross-attention
        self.vision_text_attn: MultiheadAttention(1024, num_heads=8)
        self.vision_kg_attn: MultiheadAttention(1024, num_heads=8)
        self.text_kg_attn: MultiheadAttention(1024, num_heads=8)
        
        # Fusion gate
        self.fusion_gate: Sequential(
            Linear(1024*3 → 1024),
            Linear(1024 → 3)
        )
    
    def forward(self, vision_feats, text_feats, kg_feats):
        # Input: (B, 768) each
        v = self.vision_proj(vision_feats).unsqueeze(1)  # (B, 1, 1024)
        t = self.text_proj(text_feats).unsqueeze(1)      # (B, 1, 1024)
        k = self.kg_proj(kg_feats).unsqueeze(1)          # (B, 1, 1024)
        
        # Cross-attention: (B, 1, 1024) each
        vt_attn, _ = self.vision_text_attn(v, t, t)  # (B, 1, 1024)
        vk_attn, _ = self.vision_kg_attn(v, k, k)    # (B, 1, 1024)
        tk_attn, _ = self.text_kg_attn(t, k, k)      # (B, 1, 1024)
        
        # Squeeze: (B, 1024) each
        vt_attn = vt_attn.squeeze(1)  # (B, 1024)
        vk_attn = vk_attn.squeeze(1)  # (B, 1024)
        tk_attn = tk_attn.squeeze(1)  # (B, 1024)
        
        # Concatenate: (B, 3072)
        concat_feats = torch.cat([vt_attn, vk_attn, tk_attn], dim=-1)
        
        # Fusion weights: (B, 3)
        weights = self.fusion_gate(concat_feats)
        
        # Weighted sum: (B, 1024)
        fused = (weights[:, 0:1] * vt_attn + 
                weights[:, 1:2] * vk_attn + 
                weights[:, 2:3] * tk_attn)
        
        # Output: (B, 1024)
        return fused
```

**✅ Check**: Output is (B, 1024) - CORRECT

---

### 5. LLM Reasoning Head

**File**: `reasoning/llm_reasoning_head.py`

```python
class LLMReasoningHead:
    def __init__(self, llm_model, num_answers=3129):
        self.feature_projection: Linear(1024 → llm_hidden_size)
        # Llama-3-8B hidden_size = 4096
        
        self.answer_classifier: Linear(4096 → num_answers)
    
    def forward(self, fused_features, questions, mode='classify'):
        # Input: (B, 1024)
        feature_embeds = self.feature_projection(fused_features)
        # Output: (B, 4096)
        
        feature_embeds = feature_embeds.unsqueeze(1)  # (B, 1, 4096)
        
        if mode == 'classify':
            # Tokenize question
            prompt_ids: (B, seq_len)
            prompt_embeds: (B, seq_len, 4096)
            
            # Concatenate
            combined_embeds: (B, 1+seq_len, 4096)
            
            # LLM forward
            outputs = self.llm(inputs_embeds=combined_embeds)
            last_hidden: (B, 1+seq_len, 4096)
            
            # Take last token: (B, 4096)
            last_token = last_hidden[:, -1, :]
            
            # Classifier: (B, num_answers)
            logits = self.answer_classifier(last_token)
            
            return logits  # (B, num_answers)
        
        else:  # generation mode
            # Generate text
            return generated_text: List[str]
```

**✅ Check**: Output is (B, num_answers) for classification - CORRECT

---

### 6. Unified Model Integration

**File**: `models/kg_vqa_model.py`

```python
class KGVQAModel:
    def forward(self, images, questions, mode='classify'):
        batch_size = len(questions)
        
        # 1. Entity Extraction
        entities: List[List[str]]  # B x variable
        
        # 2. RAG Retrieval
        retrieved_knowledge: List[List[Dict]]  # B x top_k
        
        # 3. Knowledge Summarization
        kg_summaries: List[str]  # B summaries
        kg_feats = summarizer.batch_generate_features(kg_summaries)
        # kg_feats: (B, 768) ✅
        
        # 4. Vision-Language Encoding
        vl_outputs = self.vision_encoder(images, questions)
        vision_feats: (B, 768) ✅
        text_feats: (B, 768) ✅
        
        # 5. Multi-Modal Fusion
        fused_feats = self.fusion_layer(vision_feats, text_feats, kg_feats)
        # fused_feats: (B, 1024) ✅
        
        # 6. LLM Reasoning
        outputs = self.reasoning_head(fused_feats, questions, mode=mode)
        # outputs: (B, num_answers) for classification ✅
        # outputs: List[str] for generation ✅
        
        return outputs
```

**✅ Check**: All connections valid - CORRECT

---

## Potential Issues and Fixes

### Issue 1: BLIP-2 Vision Model Dimensions

**Problem**: Different BLIP-2 models have different hidden sizes
- `blip2-opt-2.7b`: ViT-L/14 → 1408 hidden
- `blip2-flan-t5-xl`: Same

**Current Code**: Uses mean pooling to get 768-dim

**Fix Needed**: Check actual output dimension

```python
# In blip2_encoder.py
def extract_vision_features(self, images):
    vision_outputs = self.model.vision_model(pixel_values=inputs.pixel_values)
    # last_hidden_state may be (B, 257, 1408) not (B, 257, 768)
    
    # Need to project if necessary
    if vision_outputs.last_hidden_state.shape[-1] != 768:
        # Add projection layer
        vision_embeds = self.vision_projection(vision_outputs.last_hidden_state)
    else:
        vision_embeds = vision_outputs.last_hidden_state
    
    return vision_embeds.mean(dim=1)  # (B, 768)
```

### Issue 2: Text Features from BLIP-2

**Problem**: BLIP-2 doesn't have a direct text encoder output

**Current Code**:
```python
text_embeds = self.model.language_model.get_input_embeddings()(text_inputs.input_ids).mean(dim=1)
```

**Issue**: This gets raw embeddings before processing

**Fix**:
```python
# Use Q-Former features as text features (already aligned)
def forward(self, images, questions):
    qformer_feats = self.extract_qformer_features(images, questions)
    
    return {
        'vision_features': self.extract_vision_features(images),
        'qformer_features': qformer_feats,
        'text_features': qformer_feats  # Use same as qformer
    }
```

### Issue 3: KG Features Dimension

**Current**: Flan-T5-Large encoder outputs 1024-dim, but we return 768

**Check**:
```python
# In knowledge_summarizer.py
def generate_knowledge_features(self, summary):
    encoder_outputs = self.model.encoder(**inputs)
    hidden_states = encoder_outputs.last_hidden_state  # (1, seq_len, hidden_dim)
    # Flan-T5-Large: hidden_dim = 1024
    
    features = hidden_states.mean(dim=1).squeeze(0)  # (1024,) NOT (768,)!
```

**Fix Needed**:
```python
# Add projection in __init__
self.projection = nn.Linear(1024, 768) if self.model.config.hidden_size == 1024 else None

def generate_knowledge_features(self, summary):
    encoder_outputs = self.model.encoder(**inputs)
    features = encoder_outputs.last_hidden_state.mean(dim=1).squeeze(0)
    
    if self.projection:
        features = self.projection(features)
    
    return features  # (768,)
```

---

## Critical Fixes Required

### Fix 1: Update `vision_language/blip2_encoder.py`

Add dimension handling:
```python
def __init__(self, ...):
    super().__init__()
    # ... load model ...
    
    # Check vision model output dimension
    vision_hidden = self.model.vision_model.config.hidden_size
    if vision_hidden != 768:
        self.vision_projection = nn.Linear(vision_hidden, 768)
    else:
        self.vision_projection = None
```

### Fix 2: Update `retrieval/knowledge_summarizer.py`

Add dimension projection:
```python
def __init__(self, ...):
    # ... load model ...
    
    encoder_hidden = self.model.config.d_model  # Flan-T5: d_model
    if encoder_hidden != 768:
        self.projection = nn.Linear(encoder_hidden, 768)
    else:
        self.projection = None

def generate_knowledge_features(self, summary):
    # ... encode ...
    features = encoder_outputs.last_hidden_state.mean(dim=1).squeeze(0)
    
    if self.projection:
        features = self.projection(features)
    
    return features  # (768,)
```

### Fix 3: Update `models/kg_vqa_model.py`

Add device handling:
```python
def forward(self, images, questions, mode='classify', entities=None):
    # Ensure all tensors on same device
    device = next(self.parameters()).device
    
    # ... rest of code ...
    
    # Move kg_feats to device
    kg_feats = kg_feats.to(device)
    vision_feats = vision_feats.to(device)
    text_feats = text_feats.to(device)
```

---

## Summary of Issues

| Issue | Severity | File | Fix |
|-------|----------|------|-----|
| BLIP-2 vision dimension | 🔴 HIGH | blip2_encoder.py | Add projection layer |
| Flan-T5 dimension | 🔴 HIGH | knowledge_summarizer.py | Add projection layer |
| Text features source | 🟡 MEDIUM | blip2_encoder.py | Use Q-Former output |
| Device mismatch | 🟡 MEDIUM | kg_vqa_model.py | Add .to(device) |

---

## Testing Checklist

```python
# Test script
import torch
from models.kg_vqa_model import KGVQAModel

config = {
    'blip_model': 'Salesforce/blip2-opt-2.7b',
    'summarizer_model': 'google/flan-t5-large',
    'llm_model': 'meta-llama/Llama-3-8B-Instruct',
    'kg_index_path': 'kg_data/',
    'num_answers': 3129,
    'hidden_dim': 1024,
    'retrieval_top_k': 10,
    'use_lora': True,
    'load_in_8bit': True
}

model = KGVQAModel(config)

# Test forward pass
images = ['path/to/image.jpg']
questions = ['What is in the image?']

try:
    outputs = model(images, questions, mode='classify')
    print(f"✅ Output shape: {outputs.shape}")  # Should be (1, 3129)
    assert outputs.shape == (1, 3129), "Dimension mismatch!"
except Exception as e:
    print(f"❌ Error: {e}")
```

---

## Conclusion

**Current Status**: Code structure is correct, but needs dimension fixes

**Required Fixes**:
1. Add projection layers in BLIP-2 encoder
2. Add projection layer in knowledge summarizer  
3. Ensure device consistency

**After fixes**: All dimensions will be compatible ✅

See next section for corrected code.
