#!/usr/bin/env bash
# ==============================================================================
# Multi-GPU ControlNet Launch Script (Dual NVIDIA RTX 3090, 2x 24GB VRAM)
# ==============================================================================
set -e

echo "=== [1/2] Auditing Environment & Libraries ==="
python check_env.py

echo "=== [2/2] Launching Distributed ControlNet Training ==="
accelerate launch \
    --multi_gpu \
    --mixed_precision=fp16 \
    --num_processes=2 \
    train_controlnet.py \
    --pretrained_model_name_or_path="SG161222/Realistic_Vision_V5.1_noVAE" \
    --manifest_path="dataset_manifest.jsonl" \
    --output_dir="./checkpoints/controlnet_indian_faces_512" \
    --train_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-5 \
    --max_train_steps=3500 \
    --checkpointing_steps=500
