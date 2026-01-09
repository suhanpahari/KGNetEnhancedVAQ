# Enhanced Knowledge Graph-Aware Visual Question Answering System

## Executive Summary

This plan details the implementation of an enhanced KG-aware VQA system with three major architectural modifications:

1. **RAG-style retrieval + LLM summarization** (replacing CEL)
2. **LLM reasoning head** in post-fusion for rich semantic inference
3. **LLM-based KG completion** in preprocessing to enrich static KG

**Base Architecture**: BLIP-2/InstructBLIP (replacing VisualBERT)
**LLM Backend**: Hugging Face Transformers (Llama-3-8B, Flan-T5-Large, Mistral-7B)
**Knowledge Sources**: ConceptNet + Wikipedia + Visual Genome + Custom KG
**Target Datasets**: VQA v2.0, GQA, OK-VQA, ReasonVQA (separate dataloaders)
**Implementation Location**: `update_kgnet/` folder

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                            │
│  • Multi-source KG construction (ConceptNet, Wikipedia, VG, Custom) │
│  • LLM-based KG completion for missing relations                    │
│  • Vector database indexing (FAISS)                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    RUNTIME INFERENCE PIPELINE                        │
│  1. BLIP-2 Vision-Language Encoding                                 │
│  2. RAG Retrieval + LLM Summarization (replaces CEL)                │
│  3. Multi-Modal Fusion (vision + text + knowledge)                  │
│  4. LLM Reasoning Head with Chain-of-Thought                        │
│  5. Answer Generation (classification or free-form)                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Preprocessing - LLM-Based KG Completion

**Goal**: Build enriched multi-source knowledge graph with LLM completion

#### Files to Create:

1. **`update_kgnet/preprocessing/kg_builder.py`**
   - `MultiSourceKGBuilder` class
   - Methods:
     - `build_from_conceptnet()` - Query ConceptNet API (existing pattern)
     - `build_from_wikipedia()` - Wikipedia API + entity linking
     - `build_from_visual_genome()` - Scene graph relationships
     - `build_custom_kg_from_training_data()` - Co-occurrence from VQA datasets
     - `merge_sources()` - Unify into single graph structure
   - Output: `unified_kg.pkl`

2. **`update_kgnet/preprocessing/kg_completion.py`**
   - `LLMKGCompletion` class using Llama-3-8B-Instruct
   - Methods:
     - `complete_missing_relations(entity_pair)` - Infer relations between entities
     - `infer_entity_properties(entity)` - Generate entity attributes
     - `expand_entity_context(entity, relations)` - Add contextual knowledge
   - Prompting strategy:
     ```
     "Given entities '{e1}' and '{e2}', what semantic relations exist between them?
      Provide concise factual relations."
     ```
   - Output: `enriched_kg.pkl`

3. **`update_kgnet/preprocessing/kg_indexer.py`**
   - `KGVectorIndexer` class
   - Embedding model: `sentence-transformers/all-mpnet-base-v2`
   - Convert KG triples to text: `"{subject} {relation} {object}"`
   - Create FAISS index for dense retrieval
   - Methods:
     - `index_knowledge_triples(kg_triples)`
     - `create_entity_index(entities)`
     - `save_index(output_path)`
   - Output: `kg_embeddings.index`, `entity2id.json`, `relation2id.json`

4. **`update_kgnet/preprocessing/run_preprocessing.py`**
   - Main script orchestrating the preprocessing pipeline
   - Command-line interface for configuration
   - Progress tracking and logging

#### Dependencies:
```python
transformers  # For Llama-3-8B
sentence-transformers  # For embeddings
faiss-gpu  # For vector indexing
requests  # For API calls
spacy  # For entity extraction
```

---

### Phase 2: RAG Retrieval Module (Replaces CEL)

**Goal**: Dense retrieval + LLM summarization for knowledge injection

#### Files to Create:

1. **`update_kgnet/retrieval/rag_retriever.py`**
   - `RAGKnowledgeRetriever` class
   - Methods:
     - `retrieve_for_question(question, entities, top_k=10)` - Main retrieval
     - `multi_source_retrieval()` - Query all KG sources
     - `rerank_knowledge()` - Cross-encoder reranking
   - Retrieval strategy:
     - Encode question + entities as dense query
     - FAISS similarity search in vector index
     - Return top-k triples with confidence scores
   - Integration: Replaces simple ConceptNet text augmentation in `visualbert/kg/kg.py`

2. **`update_kgnet/retrieval/knowledge_summarizer.py`**
   - `LLMKnowledgeSummarizer` class using Flan-T5-Large
   - Methods:
     - `summarize_knowledge(question, retrieved_triples)` - Generate summary
     - `generate_knowledge_features(summary)` - Extract feature embeddings
   - Prompt template:
     ```
     "Question: {question}
      Retrieved Knowledge: {triples}
      Summarize the most relevant facts for answering this question in 2-3 sentences."
     ```
   - Output: 768-dim knowledge embedding vector

3. **`update_kgnet/retrieval/entity_extractor.py`**
   - `EntityExtractor` class using spaCy
   - Extract entities from questions and image captions
   - Methods:
     - `extract_from_text(text)` - NER extraction
     - `extract_visual_entities(image_objects)` - From detected objects

4. **`update_kgnet/retrieval/retrieval_controller.py`**
   - `AdaptiveRetrievalController` class
   - Question-type aware retrieval (counting, reasoning, knowledge-seeking, visual)
   - Adjusts `top_k` and sources based on question complexity

---

### Phase 3: Vision-Language Backbone (BLIP-2)

**Goal**: Replace VisualBERT with modern BLIP-2 architecture

#### Files to Create:

1. **`update_kgnet/vision_language/blip2_encoder.py`**
   - `BLIP2VisionLanguageEncoder` class
   - Model: `Salesforce/blip2-opt-2.7b` or `Salesforce/blip2-flan-t5-xl`
   - Methods:
     - `extract_vision_features(images)` - Vision encoder output
     - `extract_qformer_features(images, questions)` - Q-Former aligned features
     - `forward(images, questions)` - Full encoding
   - Configuration:
     - Freeze vision encoder (faster training)
     - Fine-tune Q-Former and language projection
   - Output dimensions: 768-dim for each modality

2. **`update_kgnet/vision_language/instructblip_encoder.py`** (Alternative)
   - `InstructBLIPEncoder` class
   - Model: `Salesforce/instructblip-vicuna-7b`
   - Instruction-aware encoding for better VQA performance

3. **`update_kgnet/vision_language/feature_extractors.py`**
   - Utility functions for image preprocessing
   - Bounding box utilities (reuse from `dataloaders/box_utils.py`)
   - Feature pooling and projection layers

---

### Phase 4: Multi-Modal Fusion + LLM Reasoning Head

**Goal**: Advanced fusion and LLM-based reasoning for answer generation

#### Files to Create:

1. **`update_kgnet/reasoning/fusion_layer.py`**
   - `MultiModalFusionLayer` (PyTorch nn.Module)
   - Architecture:
     - Projection layers for vision, text, and KG features → 1024-dim
     - Cross-attention between modalities (8 heads)
     - Adaptive fusion gates (learned weights per modality)
   - Forward pass:
     ```python
     v, t, k = project(vision, text, knowledge)
     vt_attn = cross_attention(v, t, t)
     vk_attn = cross_attention(v, k, k)
     tk_attn = cross_attention(t, k, k)
     weights = fusion_gate([vt_attn, vk_attn, tk_attn])
     fused = weighted_sum([vt_attn, vk_attn, tk_attn], weights)
     ```

2. **`update_kgnet/reasoning/llm_reasoning_head.py`**
   - `LLMReasoningHead` class using Llama-3-8B-Instruct
   - LoRA configuration for efficient fine-tuning:
     - `r=16, lora_alpha=32`
     - Target modules: `["q_proj", "v_proj"]`
     - 8-bit quantization for memory efficiency
   - Dual mode support:
     - **Classification mode** (VQA v2.0, GQA): Linear classifier on LLM hidden states
     - **Generation mode** (OK-VQA, ReasonVQA): Autoregressive generation
   - Methods:
     - `forward(fused_features, question_text, mode='classify')`
     - Feature projection: 1024-dim → LLM hidden size (4096)
     - Answer decoding with beam search (k=5)

3. **`update_kgnet/reasoning/cot_reasoning.py`**
   - `ChainOfThoughtReasoner` class
   - For complex ReasonVQA questions
   - Prompt template:
     ```
     "Question: {question}
      Visual Context: {visual_summary}
      Knowledge: {kg_summary}

      Let's think step by step:
      1."
     ```
   - Generate intermediate reasoning steps
   - Extract final answer from last step

4. **`update_kgnet/reasoning/answer_decoder.py`**
   - Post-processing utilities
   - Beam search implementation
   - Confidence score computation
   - Answer normalization (lowercase, punctuation removal)

---

### Phase 5: Dataset-Specific Dataloaders

**Goal**: Separate dataloaders for each target dataset

#### Files to Create:

1. **`update_kgnet/dataloaders/base_vqa_dataloader.py`**
   - `BaseVQADataloader` (PyTorch Dataset)
   - Common functionality:
     - Load images (PIL)
     - BLIP-2 preprocessing
     - Entity extraction from questions
     - RAG knowledge retrieval
     - Collation function
   - Abstract methods for dataset-specific logic

2. **`update_kgnet/dataloaders/vqa_v2_dataloader.py`**
   - `VQAv2Dataloader` extending `BaseVQADataloader`
   - Reuses existing IMDB format: `imdb_mirror_train2014.npy`
   - Answer vocabulary: `data/answers_vqa.txt` (3129 classes)
   - VQA scoring: `min(count * 0.3, 1.0)` for soft labels
   - Returns:
     ```python
     {
       'image': PIL.Image,
       'question': str,
       'entities': List[str],
       'retrieved_knowledge': List[Triple],
       'answer_idx': int,
       'question_id': int,
       'mode': 'classify'
     }
     ```

3. **`update_kgnet/dataloaders/gqa_dataloader.py`**
   - `GQADataloader` for GQA dataset
   - Loads GQA scene graphs from Visual Genome
   - Augments retrieval with scene graph relations
   - Classification mode (1878 answer classes)
   - Data format: JSON files with scene graph annotations

4. **`update_kgnet/dataloaders/okvqa_dataloader.py`**
   - `OKVQADataloader` for knowledge-intensive questions
   - Higher `top_k=20` for retrieval (more knowledge needed)
   - Generation mode (free-form answers)
   - Multiple acceptable answers per question
   - Soft accuracy evaluation

5. **`update_kgnet/dataloaders/reasonvqa_dataloader.py`**
   - `ReasonVQADataloader` for multi-step reasoning
   - Includes reasoning rationale annotations
   - Generation mode with chain-of-thought
   - BLEU/ROUGE evaluation metrics
   - Returns reasoning steps alongside answers

#### Dataloader Integration:
- All inherit from `torch.utils.data.Dataset`
- Use BLIP-2 processor for image/text preprocessing
- Integrate RAG retrieval in `__getitem__`
- Custom collate functions for batching

---

### Phase 6: Unified Model Architecture

**Goal**: Integrate all components into single model

#### Files to Create:

1. **`update_kgnet/models/kg_vqa_model.py`**
   - `KGVQAModel` class (main model)
   - Architecture:
     ```python
     class KGVQAModel(nn.Module):
         def __init__(self, config):
             self.vision_encoder = BLIP2VisionLanguageEncoder(...)
             self.rag_retriever = RAGKnowledgeRetriever(...)
             self.knowledge_summarizer = LLMKnowledgeSummarizer(...)
             self.fusion_layer = MultiModalFusionLayer(...)
             self.reasoning_head = LLMReasoningHead(...)

         def forward(self, images, questions, entities, mode='classify'):
             # 1. Vision-language encoding
             vision_feats, text_feats = self.vision_encoder(images, questions)

             # 2. Knowledge retrieval and summarization
             retrieved_kg = self.rag_retriever.retrieve_for_question(questions, entities)
             kg_summary = self.knowledge_summarizer.summarize_knowledge(questions, retrieved_kg)
             kg_feats = self.knowledge_summarizer.generate_knowledge_features(kg_summary)

             # 3. Multi-modal fusion
             fused_feats = self.fusion_layer(vision_feats, text_feats, kg_feats)

             # 4. LLM reasoning
             outputs = self.reasoning_head(fused_feats, questions, mode=mode)

             return outputs
     ```

2. **`update_kgnet/models/model_config.py`**
   - Model configuration dataclass
   - Hyperparameters for each component
   - Easy configuration management

---

### Phase 7: Training Pipeline

**Goal**: Multi-dataset training with unified pipeline

#### Files to Create:

1. **`update_kgnet/training/train_pipeline.py`**
   - `MultiDatasetVQATrainer` class
   - Features:
     - Multi-GPU support with DistributedDataParallel
     - Gradient accumulation (every 4 steps)
     - Mixed precision training (FP16)
     - Checkpoint saving/loading
     - Multi-dataset sampling strategies:
       - Proportional (by dataset size)
       - Uniform (equal samples from each)
       - Temperature-based sampling
   - Training loop:
     ```python
     for epoch in range(num_epochs):
         for dataset_name, batch in multi_dataset_loader:
             outputs = model(batch['images'], batch['questions'],
                           batch['entities'], mode=batch['mode'])

             if batch['mode'] == 'classify':
                 loss = cross_entropy(outputs, batch['answer_idx'])
             else:
                 loss = generation_loss(outputs, batch['answer_text'])

             loss.backward()
             optimizer.step()
     ```

2. **`update_kgnet/training/eval_pipeline.py`**
   - `VQAEvaluator` class
   - Dataset-specific metrics:
     - VQA v2.0: VQA accuracy (soft matching)
     - GQA: Exact match accuracy
     - OK-VQA: Soft accuracy with multiple references
     - ReasonVQA: BLEU-4, ROUGE-L, reasoning evaluation
   - Inference pipeline with batching
   - Result visualization and error analysis

3. **`update_kgnet/training/optimizers.py`**
   - Optimizer configurations
   - AdamW with weight decay: 0.01
   - Layer-wise learning rate decay
   - Separate LRs for different components:
     - Vision encoder: 1e-5 (if not frozen)
     - Fusion layer: 5e-5
     - LLM reasoning head: 1e-4 (LoRA parameters)

4. **`update_kgnet/training/schedulers.py`**
   - Learning rate schedulers
   - Cosine annealing with warmup
   - Warmup steps: 10% of total steps
   - Gradual warmup for stable training

---

### Phase 8: Configuration & Utilities

#### Files to Create:

1. **`update_kgnet/configs/vqa_v2_config.yaml`**
   - Complete configuration for VQA v2.0 training
   - Model architecture parameters
   - Training hyperparameters
   - Data paths

2. **`update_kgnet/configs/gqa_config.yaml`**
   - GQA-specific configuration

3. **`update_kgnet/configs/okvqa_config.yaml`**
   - OK-VQA configuration with generation settings

4. **`update_kgnet/configs/reasonvqa_config.yaml`**
   - ReasonVQA with chain-of-thought settings

5. **`update_kgnet/configs/preprocessing_config.yaml`**
   - KG construction and indexing parameters

6. **`update_kgnet/utils/logger.py`**
   - Logging utilities
   - TensorBoard integration
   - WandB support (optional)
   - Metrics tracking

7. **`update_kgnet/utils/metrics.py`**
   - Evaluation metrics implementation
   - VQA accuracy, BLEU, ROUGE
   - Confidence calibration metrics

8. **`update_kgnet/utils/visualization.py`**
   - Attention visualization
   - Retrieved knowledge display
   - Error analysis tools

---

### Phase 9: Execution Scripts

#### Files to Create:

1. **`update_kgnet/scripts/run_preprocessing.sh`**
   ```bash
   #!/bin/bash
   cd preprocessing/
   python run_preprocessing.py \
     --sources conceptnet wikipedia visual_genome custom \
     --llm_model meta-llama/Llama-3-8B-Instruct \
     --output_dir ../kg_data/
   ```

2. **`update_kgnet/scripts/run_training.sh`**
   ```bash
   #!/bin/bash
   python training/train_pipeline.py \
     --datasets vqa_v2 gqa okvqa reasonvqa \
     --config configs/vqa_v2_config.yaml \
     --num_gpus 2 \
     --batch_size 32 \
     --epochs 30
   ```

3. **`update_kgnet/scripts/run_evaluation.sh`**
   ```bash
   #!/bin/bash
   python training/eval_pipeline.py \
     --checkpoint checkpoints/best_model.pth \
     --datasets vqa_v2 gqa okvqa reasonvqa \
     --output_dir results/
   ```

4. **`update_kgnet/scripts/run_inference.sh`**
   - Single image inference script
   - Demo for testing on custom images

---

## Critical Files Summary

### Top Priority (Core Functionality):

1. **`update_kgnet/models/kg_vqa_model.py`** - Unified model integrating all components
2. **`update_kgnet/retrieval/rag_retriever.py`** - RAG retrieval replacing CEL
3. **`update_kgnet/reasoning/llm_reasoning_head.py`** - LLM reasoning module
4. **`update_kgnet/preprocessing/kg_completion.py`** - LLM-based KG enrichment
5. **`update_kgnet/training/train_pipeline.py`** - Main training loop

### Supporting Infrastructure:

6. **`update_kgnet/vision_language/blip2_encoder.py`** - BLIP-2 backbone
7. **`update_kgnet/dataloaders/vqa_v2_dataloader.py`** - Reference dataloader
8. **`update_kgnet/reasoning/fusion_layer.py`** - Multi-modal fusion
9. **`update_kgnet/preprocessing/kg_indexer.py`** - Vector indexing
10. **`update_kgnet/retrieval/knowledge_summarizer.py`** - LLM summarization

---

## Dependencies

```txt
# Core Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# Vision-Language Models
transformers>=4.35.0
pillow>=10.0.0
timm>=0.9.0

# LLM Optimization
peft>=0.6.0  # LoRA
bitsandbytes>=0.41.0  # 8-bit quantization
accelerate>=0.24.0

# Knowledge Retrieval
sentence-transformers>=2.2.0
faiss-gpu>=1.7.4

# NLP Tools
spacy>=3.7.0
nltk>=3.8.0
rouge-score>=0.1.2

# APIs and Data
requests>=2.31.0
wikipedia-api>=0.6.0
h5py>=3.9.0

# Utilities
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0.0
tqdm>=4.66.0
tensorboard>=2.14.0
wandb>=0.15.0  # Optional
```

---

## Hardware Requirements

**Minimum**: 1x NVIDIA A100 (40GB) or 2x RTX 3090 (24GB)
**Recommended**: 2x NVIDIA A100 (80GB)

**Memory Breakdown**:
- BLIP-2 (ViT-L + Q-Former): ~3GB
- Flan-T5-Large: ~3GB
- Llama-3-8B (8-bit quantized): ~16GB
- Training overhead: ~10GB
- **Total**: ~32GB per GPU

---

## Integration with Existing Codebase

### Reusable Components:

1. **From `dataloaders/vqa_dataset.py`**:
   - IMDB file format and loading logic
   - Answer vocabulary management (`VocabDict`)
   - VQA scoring formula

2. **From `dataloaders/bert_data_utils.py`**:
   - `InputFeatures` structure (adapt for BLIP-2)
   - Tokenization utilities

3. **From `visualbert/kg/kg.py`**:
   - ConceptNet API query patterns
   - Entity extraction from captions

4. **From `visualbert/train.py`**:
   - Checkpoint saving/loading infrastructure
   - Command-line argument parsing
   - Metrics logging

### What Gets Replaced:

1. **VisualBERT architecture** → BLIP-2/InstructBLIP
2. **Simple KG text augmentation** → RAG retrieval + LLM summarization
3. **MLP classifier** → LLM reasoning head
4. **Faster R-CNN features** → BLIP-2 vision encoder

---

## Verification & Testing

### End-to-End Testing:

1. **Preprocessing Verification**:
   - Check KG size (target: 1M+ triples)
   - Validate FAISS index (retrieval speed < 50ms)
   - Inspect LLM-completed relations for quality

2. **Model Testing**:
   - Single-sample forward pass (check shapes)
   - Gradient flow verification (all parameters updating)
   - Memory profiling (ensure < 40GB per GPU)

3. **Training Validation**:
   - Loss convergence (should decrease in first epoch)
   - Overfit single batch (sanity check)
   - Multi-GPU synchronization

4. **Evaluation**:
   - VQA v2.0 test-dev accuracy (target: >72%)
   - GQA test accuracy (target: >60%)
   - OK-VQA val accuracy (target: >50%)
   - ReasonVQA BLEU-4 (target: >20)

---

## Implementation Timeline

**Phase 1 (Preprocessing)**: 2-3 days
- KG construction and LLM completion
- Vector indexing

**Phase 2 (Core Modules)**: 5-7 days
- RAG retrieval
- BLIP-2 integration
- Fusion layer
- LLM reasoning head

**Phase 3 (Dataloaders)**: 3-4 days
- Base dataloader
- 4 dataset-specific loaders
- Testing and validation

**Phase 4 (Training Pipeline)**: 3-4 days
- Training loop
- Evaluation pipeline
- Multi-dataset support

**Phase 5 (Testing & Refinement)**: 3-5 days
- End-to-end testing
- Bug fixes
- Performance optimization

**Total**: ~2-3 weeks for complete implementation

---

## Directory Structure (Complete)

```
update_kgnet/
├── preprocessing/
│   ├── __init__.py
│   ├── kg_builder.py
│   ├── kg_completion.py
│   ├── kg_indexer.py
│   └── run_preprocessing.py
│
├── retrieval/
│   ├── __init__.py
│   ├── rag_retriever.py
│   ├── knowledge_summarizer.py
│   ├── retrieval_controller.py
│   └── entity_extractor.py
│
├── reasoning/
│   ├── __init__.py
│   ├── fusion_layer.py
│   ├── llm_reasoning_head.py
│   ├── cot_reasoning.py
│   └── answer_decoder.py
│
├── vision_language/
│   ├── __init__.py
│   ├── blip2_encoder.py
│   ├── instructblip_encoder.py
│   └── feature_extractors.py
│
├── dataloaders/
│   ├── __init__.py
│   ├── base_vqa_dataloader.py
│   ├── vqa_v2_dataloader.py
│   ├── gqa_dataloader.py
│   ├── okvqa_dataloader.py
│   └── reasonvqa_dataloader.py
│
├── training/
│   ├── __init__.py
│   ├── train_pipeline.py
│   ├── eval_pipeline.py
│   ├── optimizers.py
│   └── schedulers.py
│
├── models/
│   ├── __init__.py
│   ├── kg_vqa_model.py
│   └── model_config.py
│
├── configs/
│   ├── vqa_v2_config.yaml
│   ├── gqa_config.yaml
│   ├── okvqa_config.yaml
│   ├── reasonvqa_config.yaml
│   └── preprocessing_config.yaml
│
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── metrics.py
│   └── visualization.py
│
├── scripts/
│   ├── run_preprocessing.sh
│   ├── run_training.sh
│   ├── run_evaluation.sh
│   └── run_inference.sh
│
├── requirements.txt
└── README.md
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GPU OOM | 8-bit quantization, gradient checkpointing, batch size reduction |
| Slow retrieval | FAISS GPU, approximate NN, query caching |
| LLM hallucination | Confidence thresholding, fallback to classification mode |
| Dataset imbalance | Temperature sampling, balanced batching |
| Poor KG quality | LLM validation step, manual filtering |

---

## Success Criteria

✅ **Preprocessing**: KG with 1M+ triples, retrieval latency < 50ms
✅ **Training**: Stable loss convergence, no OOM errors
✅ **VQA v2.0**: Accuracy > 72% (competitive with VisualBERT baseline)
✅ **Multi-dataset**: All 4 datasets train and evaluate successfully
✅ **Code Quality**: Clean, modular, well-documented code in `update_kgnet/`

