#!/bin/bash

learning_rates=(0.00001 0.0001 0.001 0.01 0.1)
batch_sizes=(1 8 16 32 64)
epochs=(5 10 20 30)
dataset_name="1-tool"

for lr in "${learning_rates[@]}"; do
    for bs in "${batch_sizes[@]}"; do
        for ep in "${epochs[@]}"; do
            model_name="model_${dataset_name}_E${ep}_B${bs}_LR${lr}.pth"

            echo "----------------------------------------"
            echo "MODEL NAME: $model_name"

            if [ -f "./models/$model_name" ]; then
                echo "Model $model_name already exists. Skipping..."
                continue
            fi
            
            echo "Training with learning rate: $lr, batch size: $bs, and epochs: $ep"
            python train.py --amp --batch-size $bs --epochs $ep --learning-rate $lr

            echo "Testing with learning rate: $lr, batch size: $bs, and epochs: $ep"
            python evaluate_dataset.py --model "./models/$model_name"

            echo "----------------------------------------"
        done
    done
done
