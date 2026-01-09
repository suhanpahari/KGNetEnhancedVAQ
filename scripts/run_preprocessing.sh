#!/bin/bash

# Run preprocessing to build knowledge graph

cd ../preprocessing/

python run_preprocessing.py \
    --sources conceptnet wikipedia visual_genome custom \
    --entities_file ../../visualbert/kg/entities.json \
    --vqa_train_path ../../data/imdb/imdb_mirror_train2014.npy \
    --scene_graph_path ../../data/visual_genome/scene_graphs.json \
    --output_dir ../kg_data/ \
    --use_llm_completion \
    --llm_model meta-llama/Llama-3-8B-Instruct \
    --max_entities_llm 1000 \
    --completion_types properties

echo "Preprocessing complete! KG saved to ../kg_data/"
