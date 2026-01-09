#!/bin/bash

# Comprehensive evaluation script for all datasets using dataset-specific evaluators

echo "Running evaluation on all VQA datasets with dataset-specific evaluators..."

# Configuration
CHECKPOINT_DIR="../checkpoints/"
CONFIG_DIR="../configs/"
DATA_ROOT="../../data/"
OUTPUT_DIR="../results/"

# Create output directory
mkdir -p $OUTPUT_DIR

# VQA v2.0 Evaluation
echo "=========================================="
echo "Evaluating VQA v2.0..."
echo "=========================================="
cd ../evaluation/
python vqa_v2_eval.py \
    --checkpoint ${CHECKPOINT_DIR}/vqa_v2_best.pth \
    --config ${CONFIG_DIR}/vqa_v2_config.yaml \
    --data_root ${DATA_ROOT}/coco/images/ \
    --imdb_val ${DATA_ROOT}/imdb/imdb_minival2014.npy \
    --answer_vocab ${DATA_ROOT}/answers_vqa.txt \
    --batch_size 16 \
    --output ${OUTPUT_DIR}/vqa_v2_results.json

# GQA Evaluation
echo "=========================================="
echo "Evaluating GQA..."
echo "=========================================="
python gqa_eval.py \
    --checkpoint ${CHECKPOINT_DIR}/gqa_best.pth \
    --config ${CONFIG_DIR}/gqa_config.yaml \
    --data_root ${DATA_ROOT}/gqa/images/ \
    --questions ${DATA_ROOT}/gqa/val_balanced_questions.json \
    --scene_graphs ${DATA_ROOT}/gqa/val_sceneGraphs.json \
    --batch_size 16 \
    --output ${OUTPUT_DIR}/gqa_results.json

# OK-VQA Evaluation
echo "=========================================="
echo "Evaluating OK-VQA..."
echo "=========================================="
python okvqa_eval.py \
    --checkpoint ${CHECKPOINT_DIR}/okvqa_best.pth \
    --config ${CONFIG_DIR}/okvqa_config.yaml \
    --data_root ${DATA_ROOT}/coco/images/ \
    --questions ${DATA_ROOT}/okvqa/OpenEnded_mscoco_val2014_questions.json \
    --annotations ${DATA_ROOT}/okvqa/mscoco_val2014_annotations.json \
    --batch_size 8 \
    --output ${OUTPUT_DIR}/okvqa_results.json

# ReasonVQA Evaluation
echo "=========================================="
echo "Evaluating ReasonVQA..."
echo "=========================================="
python reasonvqa_eval.py \
    --checkpoint ${CHECKPOINT_DIR}/reasonvqa_best.pth \
    --config ${CONFIG_DIR}/reasonvqa_config.yaml \
    --data_root ${DATA_ROOT}/reasonvqa/images/ \
    --questions ${DATA_ROOT}/reasonvqa/val_questions.json \
    --annotations ${DATA_ROOT}/reasonvqa/val_annotations.json \
    --batch_size 8 \
    --output ${OUTPUT_DIR}/reasonvqa_results.json

echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved in: $OUTPUT_DIR"
echo "=========================================="

# Print summary
echo ""
echo "Summary of Results:"
echo "-------------------"
if [ -f "${OUTPUT_DIR}/vqa_v2_results.json" ]; then
    echo "VQA v2.0: $(grep 'overall_accuracy' ${OUTPUT_DIR}/vqa_v2_results.json | head -1)"
fi
if [ -f "${OUTPUT_DIR}/gqa_results.json" ]; then
    echo "GQA: $(grep 'overall_accuracy' ${OUTPUT_DIR}/gqa_results.json | head -1)"
fi
if [ -f "${OUTPUT_DIR}/okvqa_results.json" ]; then
    echo "OK-VQA: $(grep 'overall_accuracy' ${OUTPUT_DIR}/okvqa_results.json | head -1)"
fi
if [ -f "${OUTPUT_DIR}/reasonvqa_results.json" ]; then
    echo "ReasonVQA: $(grep 'exact_match_accuracy' ${OUTPUT_DIR}/reasonvqa_results.json | head -1)"
fi
