python train.py \
  --epochs 200 \
  --batch_size 6 \
  --seq_len 512 \
  --d_model 320 \
  --nhead 8 \
  --num_layers 5 \
  --latent_dim 160 \
  --num_memory_tokens 4 \
  --lr 1e-4 \
  --kl_weight 0.02 \
  --kl_warmup_epochs 0 \
  --weight_decay 0.0 \
  --label_smoothing 0.0 \
  --lr_warmup_epochs 0 \
  --free_bits 0.2 \
  --length_loss_weight 0.1 \
  --grad_clip 1.0 \
  --output_dir checkpoints \
  --save_every 10

echo ""
echo "Training complete with CLEAN settings:"
echo "✓ Bigger model: d_model 320, 5 layers, latent 160"
echo "✓ Original training: gentle KL warmup, Adam, low LR"
echo "✓ NO label smoothing, NO weight decay, NO aggressive schedules"
