#!/bin/bash

# GraphCodeBERT Clone Detection Training with Wandb Logging
# This script uses your exact previous training command with wandb integration added


echo "Starting training with wandb logging..."

python run.py \
    --output_dir=saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=dataset/train_small.txt \
    --eval_data_file=dataset/valid.txt \
    --test_data_file=dataset/test.txt \
    --epochs 1 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 6 \
    --eval_batch_size 6 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 \
    --use_wandb \
    --wandb_project "graphcodebert-clonedetection" \
    --wandb_run_name "train-1epoch-small-dataset" \
    2>&1 | tee saved_models/train_with_wandb.log

echo "Training completed!"
echo "Check your wandb dashboard for training metrics and visualizations."
