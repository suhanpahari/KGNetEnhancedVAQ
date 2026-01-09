#!/bin/bash

# Train KG-VQA model on VQA v2.0

cd ../training/

python train_pipeline.py \
    --config ../configs/vqa_v2_config.yaml \
    --data_root ../../data/coco/images/ \
    --imdb_train ../../data/imdb/imdb_mirror_train2014.npy \
    --imdb_val ../../data/imdb/imdb_minival2014.npy \
    --answer_vocab ../../data/answers_vqa.txt \
    --kg_index_path ../kg_data/ \
    --batch_size 8 \
    --epochs 20 \
    --lr 1e-4 \
    --checkpoint_dir ../checkpoints/

echo "Training complete! Best model saved to ../checkpoints/best_model.pth"
