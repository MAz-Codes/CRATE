#!/bin/bash
# Resume Phase 3 training

# Activate virtual environment if needed (assuming standard location)
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python train.py \
  --epochs 100 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --word_dropout 0.3 \
  --seq_len 512 \
  --d_model 384 \
  --nhead 8 \
  --num_layers 6 \
  --latent_dim 256 \
  --num_memory_tokens 4 \
  --lr 1e-4 \
  --kl_weight 0.02 \
  --kl_warmup_epochs 20 \
  --weight_decay 0.01 \
  --label_smoothing 0.1 \
  --lr_warmup_epochs 5 \
  --free_bits 0.1 \
  --length_loss_weight 0.1 \
  --grad_clip 1.0 \
  --output_dir checkpoints_phase3 \
  --resume checkpoints_phase3/best_model.pt
