"""
Training script for DrumVAE v2 with auxiliary losses for musical structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GrooveMidiDataset, DataCollator, STYLES, UNKNOWN_STYLE_IDX
from model_v2 import DrumVAE, generate_square_subsequent_mask, get_token_types, TOKEN_TYPE_MAP
import argparse
import os
import math
from tqdm import tqdm
import json
import time
import matplotlib.pyplot as plt


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_token_type_targets(token_ids, tokenizer, device):
    """
    Extract token type targets for auxiliary loss - GPU accelerated.
    Returns tensor of shape [seq_len, batch] with type IDs.
    """
    # Build lookup table once (vocab_size,) - cached per device
    cache_key = f'_lookup_cache_{device.type}_{device.index if device.index else 0}'
    if not hasattr(get_token_type_targets, cache_key):
        vocab_size = len(tokenizer.vocab)
        lookup = torch.zeros(vocab_size, dtype=torch.long, device=device)
        id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        
        for token_id, token_str in id_to_token.items():
            token_str = str(token_str)
            type_id = 0  # Default PAD
            for prefix, type_val in TOKEN_TYPE_MAP.items():
                if token_str.startswith(prefix):
                    type_id = type_val
                    break
            lookup[token_id] = type_id
        
        setattr(get_token_type_targets, cache_key, lookup)
    
    # GPU lookup - single operation instead of nested loops
    lookup = getattr(get_token_type_targets, cache_key)
    type_ids = lookup[token_ids]
    
    return type_ids


def get_structure_targets(token_ids, tokenizer, device):
    """
    Extract bar position targets (0-95) and bar number targets (0-31) - GPU accelerated.
    Returns:
        positions: [seq_len, batch]
        bar_nums: [seq_len, batch]
    """
    # Build lookup tables once - cached per device
    cache_key = f'_cache_{device.type}_{device.index if device.index else 0}'
    if not hasattr(get_structure_targets, cache_key):
        vocab_size = len(tokenizer.vocab)
        id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        
        # Precompute which tokens are Bar or Position tokens
        is_bar_token = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        position_values = torch.zeros(vocab_size, dtype=torch.long, device=device)
        
        for token_id, token_str in id_to_token.items():
            token_str = str(token_str)
            if token_str.startswith('Bar'):
                is_bar_token[token_id] = True
            elif token_str.startswith('Position_'):
                try:
                    pos = int(token_str.split('_')[1])
                    position_values[token_id] = min(pos, 95)
                except:
                    pass
        
        setattr(get_structure_targets, cache_key, (is_bar_token, position_values))
    
    is_bar_token, position_values = getattr(get_structure_targets, cache_key)
    
    seq_len, batch_size = token_ids.shape
    
    # GPU VECTORIZED operations
    # Get bar markers and position values for all tokens at once
    bar_markers = is_bar_token[token_ids]  # [seq_len, batch]
    pos_vals = position_values[token_ids]  # [seq_len, batch]
    
    # Compute bar numbers using cumsum on GPU
    bar_markers_shifted = torch.cat([torch.zeros(1, batch_size, dtype=torch.long, device=device), 
                                     bar_markers[:-1].long()], dim=0)
    bar_nums = torch.cumsum(bar_markers_shifted, dim=0).clamp(max=31)
    
    # Forward-fill positions using cummax trick on GPU
    # We create indices where position tokens occur and use cummax to forward-fill
    has_pos = pos_vals > 0
    
    # Create a sequence index tensor [seq_len, batch]
    seq_idx = torch.arange(seq_len, device=device).unsqueeze(1).expand(-1, batch_size)
    
    # Where we have a position, store the index, otherwise -1
    pos_indices = torch.where(has_pos, seq_idx, torch.tensor(-1, device=device))
    
    # Cummax gives us the index of the last seen position token
    last_pos_idx, _ = torch.cummax(pos_indices, dim=0)
    
    # Gather the position values using the indices
    # Handle -1 indices (before first position) by clamping to 0
    gather_idx = last_pos_idx.clamp(min=0)
    
    # Flatten for gather operation then reshape
    batch_offset = torch.arange(batch_size, device=device) * seq_len
    flat_idx = (gather_idx + batch_offset.unsqueeze(0)).flatten()
    positions = pos_vals.T.flatten()[flat_idx].reshape(seq_len, batch_size)
    
    # Zero out positions before the first position token
    positions = torch.where(last_pos_idx >= 0, positions, torch.tensor(0, device=device, dtype=positions.dtype))
    
    return positions, bar_nums


def plot_training_history(history, output_dir):
    """Plot training metrics."""
    try:
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))

        # Total Loss
        axs[0, 0].plot(history['train_loss'], label='Train')
        axs[0, 0].plot(history['val_loss'], label='Val')
        axs[0, 0].set_title('Total Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Reconstruction Loss
        axs[0, 1].plot(history['train_recon_loss'], label='Train')
        axs[0, 1].plot(history['val_recon_loss'], label='Val')
        axs[0, 1].set_title('Reconstruction Loss')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # KL Loss
        axs[0, 2].plot(history['train_kl_loss'], label='Train')
        axs[0, 2].plot(history['val_kl_loss'], label='Val')
        axs[0, 2].set_title('KL Divergence')
        axs[0, 2].legend()
        axs[0, 2].grid(True)

        # Active Units
        axs[1, 0].plot(history['active_units'], label='Active Units', color='purple')
        axs[1, 0].set_title('Active Latent Units')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Token Type Accuracy (auxiliary loss)
        if 'train_type_acc' in history and history['train_type_acc']:
            axs[1, 1].plot(history['train_type_acc'], label='Train')
            axs[1, 1].plot(history.get('val_type_acc', []), label='Val')
            axs[1, 1].set_title('Token Type Accuracy')
            axs[1, 1].legend()
            axs[1, 1].grid(True)
        
        # Beta (KL weight)
        axs[1, 2].plot(history['beta'], label='Beta', color='orange')
        axs[1, 2].set_title('KL Annealing (Beta)')
        axs[1, 2].legend()
        axs[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_plot.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot training history: {e}")


def train(args):
    # Device selection - prioritize GPU
    device = get_device()
    print(f"Using device: {device}")
    
    # Set optimal settings for device
    if device.type == 'cuda':
        # Enable TF32 for faster computation on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Load datasets
    train_dataset = GrooveMidiDataset(
        split='train', max_seq_len=args.seq_len, max_examples=args.file_limit, augment=True, chunk_bars=args.chunk_bars)
    
    try:
        pad_token_id = train_dataset.tokenizer["PAD_None"]
    except:
        pad_token_id = 0
        
    data_collator = DataCollator(pad_token_id)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=data_collator, num_workers=0)
    
    val_dataset = GrooveMidiDataset(
        split='validation', max_seq_len=args.seq_len, max_examples=args.file_limit, chunk_bars=args.chunk_bars)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=data_collator, num_workers=0)

    vocab_size = len(train_dataset.tokenizer)
    num_styles = len(STYLES) + 1
    print(f"Vocabulary Size: {vocab_size}, Num Styles: {num_styles}")

    # Initialize model
    model = DrumVAE(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        num_conductor_layers=args.num_conductor_layers,
        latent_dim=args.latent_dim,
        max_seq_len=args.seq_len,
        num_styles=num_styles,
        max_bars=args.max_bars,
        num_token_types=10
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, 
        weight_decay=args.weight_decay, betas=(0.9, 0.98))
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # Loss functions
    recon_criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=args.label_smoothing)
    type_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD type
    position_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore non-position tokens

    # Training state
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer state to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint['epoch'] + 1
        
        # CRITICAL CHANGE: Reset best_val_loss to infinity when resuming.
        # This ensures that if we change the metric (e.g. to a stricter one),
        # we don't get stuck comparing against an old, incomparable score.
        print("Resetting best_val_loss to ensure new metric is adopted.")
        best_val_loss = float('inf') 

    # Directories
    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # History
    history = {
        'train_loss': [], 'val_loss': [], 'train_recon_loss': [], 'val_recon_loss': [],
        'train_kl_loss': [], 'val_kl_loss': [], 'train_type_acc': [], 'val_type_acc': [],
        'active_units': [], 'beta': [], 'val_perplexity': []
    }

    # Load history if resuming to prevent overwriting
    if args.resume:
        history_path = os.path.join(logs_dir, 'training_history.json')
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    loaded_history = json.load(f)
                
                # Update history with loaded data
                for k in history.keys():
                    if k in loaded_history:
                        # Truncate to start_epoch to keep sync with model state
                        # This handles cases where we resume from an earlier checkpoint (best_model)
                        # than the latest log entry
                        history[k] = loaded_history[k][:start_epoch]
                
                print(f"Loaded and synced training history (epochs 0-{start_epoch-1})")
            except Exception as e:
                print(f"Warning: Could not load training history: {e}")

    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        
        # KL annealing - CYCLICAL schedule to prevent collapse
        # Cycles every cycle_length epochs, with linear ramp-up
        cycle_length = args.kl_cycle_length
        cycle_position = epoch % cycle_length
        
        # Linear ramp within cycle, capped at beta_max
        beta = min(args.beta_max, cycle_position / max(1, cycle_length // 2))
        
        # Minimum beta floor to always have some KL pressure
        beta = max(args.beta_min, beta)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Metrics
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_type_correct = 0
        total_type_count = 0
        
        epoch_start = time.time()
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (src, src_mask, style_batch, bar_batch) in enumerate(progress):
            if src is None:
                continue
                
            src = src.to(device)
            src_mask = src_mask.to(device)
            style_batch = style_batch.to(device)
            bar_batch = bar_batch.to(device)
            
            # Get token types for auxiliary loss
            token_types = get_token_type_targets(src, train_dataset.tokenizer, device)
            
            # Get structure targets
            positions, bar_nums = get_structure_targets(src, train_dataset.tokenizer, device)
            
            # Word dropout
            tgt = src.clone()
            if args.word_dropout > 0:
                try:
                    mask_token_id = train_dataset.tokenizer["MASK_None"]
                    if isinstance(mask_token_id, list):
                        mask_token_id = mask_token_id[0]
                except:
                    mask_token_id = pad_token_id
                
                prob = torch.rand(tgt.shape, device=device)
                word_dropout_mask = (prob < args.word_dropout) & (tgt != pad_token_id)
                word_dropout_mask[0, :] = False  # Keep first token
                tgt[word_dropout_mask] = mask_token_id
            
            dec_input = tgt[:-1]
            dec_target = tgt[1:]
            dec_mask = src_mask[:, :-1]
            target_types = token_types[1:]  # Shifted for next-token prediction
            target_positions = positions[1:] # Shifted for next-token prediction
            
            tgt_len = dec_input.shape[0]
            tgt_mask = generate_square_subsequent_mask(tgt_len, device)
            
            # Forward pass - encoder uses full src and full token_types
            # decoder uses dec_input and token_types[:-1]
            outputs = model(
                src, dec_input,
                src_token_types=token_types,  # Full sequence for encoder
                tgt_token_types=token_types[:-1],  # Shifted for decoder
                style_id=style_batch,
                bar_id=bar_batch,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=dec_mask,
                tgt_mask=tgt_mask,
                src_bar_pos=positions,
                src_bar_num=bar_nums,
                tgt_bar_pos=positions[:-1],
                tgt_bar_num=bar_nums[:-1]
            )
            
            # Reconstruction loss
            logits = outputs['logits']
            recon_loss = recon_criterion(
                logits.reshape(-1, vocab_size), dec_target.reshape(-1))
            
            # KL loss calculation
            mu, logvar = outputs['mu'], outputs['logvar']
            kl_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            
            # Raw KL: standard VAE KL divergence (for loss and monitoring)
            kl_loss = torch.mean(torch.sum(kl_element, dim=1))
            
            # Free bits penalty: only penalize dimensions that collapse below threshold
            # This is ADDED to loss, not replacing KL
            free_bits = args.free_bits
            kl_per_dim = torch.mean(kl_element, dim=0)  # [latent_dim]
            # Penalty for dims below free_bits threshold
            free_bits_penalty = torch.sum(torch.relu(free_bits - kl_per_dim))
            
            # Token type auxiliary loss
            type_logits = outputs['type_logits']
            type_loss = type_criterion(
                type_logits.reshape(-1, 10), target_types.reshape(-1))
            
            # Position auxiliary loss
            position_logits = outputs['position_logits']
            position_loss = position_criterion(
                position_logits.reshape(-1, 96), target_positions.reshape(-1))
            
            # Calculate type accuracy
            type_preds = type_logits.argmax(dim=-1)
            valid_mask = target_types != 0  # Not PAD
            type_correct = (type_preds == target_types) & valid_mask
            total_type_correct += type_correct.sum().item()
            total_type_count += valid_mask.sum().item()
            
            # Active units (dimensions with KL > threshold)
            active_units = torch.sum(kl_per_dim > 0.1).item()
            
            # Total loss:
            # - recon_loss: reconstruction (cross-entropy)
            # - kl_loss: KL divergence scaled by beta and kl_weight  
            # - free_bits_penalty: prevents individual dimension collapse
            # - auxiliary losses: token type and position prediction
            loss = (recon_loss + 
                    beta * kl_loss * args.kl_weight + 
                    args.free_bits_weight * free_bits_penalty +
                    args.type_loss_weight * type_loss +
                    args.type_loss_weight * position_loss)
            
            loss = loss / args.grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            
            # Accumulate metrics
            total_loss += loss.item() * args.grad_accum_steps
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
            progress.set_postfix({
                'Loss': f'{loss.item() * args.grad_accum_steps:.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'KL': f'{kl_loss.item():.2f}',
                'AU': f'{active_units}',
                'TypeAcc': f'{total_type_correct/max(1,total_type_count):.2%}'
            })
        
        # Epoch metrics
        n_batches = len(train_loader)
        avg_train_loss = total_loss / n_batches
        avg_train_recon = total_recon / n_batches
        avg_train_kl = total_kl / n_batches
        train_type_acc = total_type_correct / max(1, total_type_count)
        
        # Validation
        model.eval()
        val_loss = 0
        val_recon = 0
        val_kl = 0
        val_type_loss = 0
        val_position_loss = 0
        val_type_correct = 0
        val_type_count = 0
        
        with torch.no_grad():
            for src, src_mask, style_batch, bar_batch in val_loader:
                if src is None:
                    continue
                    
                src = src.to(device)
                src_mask = src_mask.to(device)
                style_batch = style_batch.to(device)
                bar_batch = bar_batch.to(device)
                
                token_types = get_token_type_targets(src, train_dataset.tokenizer, device)
                positions, bar_nums = get_structure_targets(src, train_dataset.tokenizer, device)
                
                dec_input = src[:-1]
                dec_target = src[1:]
                dec_mask = src_mask[:, :-1]
                target_types = token_types[1:]
                target_positions = positions[1:]
                
                tgt_len = dec_input.shape[0]
                tgt_mask = generate_square_subsequent_mask(tgt_len, device)
                
                outputs = model(
                    src, dec_input,
                    src_token_types=token_types,  # Full sequence for encoder
                    tgt_token_types=token_types[:-1],  # Shifted for decoder
                    style_id=style_batch,
                    bar_id=bar_batch,
                    src_key_padding_mask=src_mask,
                    tgt_key_padding_mask=dec_mask,
                    tgt_mask=tgt_mask,
                    src_bar_pos=positions,
                    src_bar_num=bar_nums,
                    tgt_bar_pos=positions[:-1],
                    tgt_bar_num=bar_nums[:-1]
                )
                
                recon_loss = recon_criterion(
                    outputs['logits'].reshape(-1, vocab_size), dec_target.reshape(-1))
                
                kl_element = -0.5 * (1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())
                kl_loss = torch.mean(torch.sum(kl_element, dim=1))
                
                type_logits = outputs['type_logits']
                type_loss = type_criterion(
                    type_logits.reshape(-1, 10), target_types.reshape(-1))
                
                position_logits = outputs['position_logits']
                position_loss = position_criterion(
                    position_logits.reshape(-1, 96), target_positions.reshape(-1))
                
                type_preds = type_logits.argmax(dim=-1)
                valid_mask = target_types != 0
                type_correct = (type_preds == target_types) & valid_mask
                val_type_correct += type_correct.sum().item()
                val_type_count += valid_mask.sum().item()
                
                val_recon += recon_loss.item()
                val_kl += kl_loss.item()
                val_type_loss += type_loss.item()
                val_position_loss += position_loss.item()
        
        n_val = len(val_loader)
        avg_val_recon = val_recon / n_val
        avg_val_kl = val_kl / n_val
        avg_val_type_loss = val_type_loss / n_val
        avg_val_position_loss = val_position_loss / n_val
        # Include auxiliary losses in validation for consistency with training loss
        avg_val_loss = (avg_val_recon + 
                        beta * avg_val_kl * args.kl_weight +
                        args.type_loss_weight * avg_val_type_loss +
                        args.type_loss_weight * avg_val_position_loss)
        
        # Metric for model selection: Use fixed beta_max to ensure we select a model 
        # that works well under full KL constraint (crucial for generation)
        val_selection_metric = (avg_val_recon + 
                                args.beta_max * avg_val_kl * args.kl_weight +
                                args.type_loss_weight * avg_val_type_loss +
                                args.type_loss_weight * avg_val_position_loss)

        val_type_acc = val_type_correct / max(1, val_type_count)
        val_perplexity = math.exp(avg_val_recon)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1}/{args.epochs} - Time: {epoch_time:.1f}s")
        print(f"  Train: Loss={avg_train_loss:.4f}, Recon={avg_train_recon:.4f}, KL={avg_train_kl:.2f}, TypeAcc={train_type_acc:.2%}")
        print(f"  Val:   Loss={avg_val_loss:.4f}, Metric={val_selection_metric:.4f}, Recon={avg_val_recon:.4f}, KL={avg_val_kl:.2f}, TypeAcc={val_type_acc:.2%}, PPL={val_perplexity:.2f}")
        print(f"  Beta={beta:.4f}, ActiveUnits={active_units}")
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_recon_loss'].append(avg_train_recon)
        history['val_recon_loss'].append(avg_val_recon)
        history['train_kl_loss'].append(avg_train_kl)
        history['val_kl_loss'].append(avg_val_kl)
        history['train_type_acc'].append(train_type_acc)
        history['val_type_acc'].append(val_type_acc)
        history['active_units'].append(active_units)
        history['beta'].append(beta)
        history['val_perplexity'].append(val_perplexity)
        
        # Scheduler step (use selection metric to avoid fluctuating with beta)
        scheduler.step(val_selection_metric)
        
        # Save best model
        if val_selection_metric < best_val_loss:
            best_val_loss = val_selection_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'vocab_size': vocab_size,
                'num_styles': num_styles,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pt'))
            # Save tokenizer alongside best model for generation consistency
            train_dataset.tokenizer.save_pretrained(args.output_dir)
            print(f"  ✓ Saved best model and tokenizer (Metric: {best_val_loss:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'vocab_size': vocab_size,
                'num_styles': num_styles,
                'pad_token_id': pad_token_id,
                'args': vars(args)
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        # Save history and plot
        with open(os.path.join(logs_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        plot_training_history(history, logs_dir)
        print(f"  ✓ Updated training plot: {os.path.join(logs_dir, 'training_plot.png')}")
        
        # Early stopping check
        if args.early_stopping > 0:
            if len(history['val_loss']) > args.early_stopping:
                recent = history['val_loss'][-args.early_stopping:]
                if all(recent[i] >= recent[i-1] for i in range(1, len(recent))):
                    print(f"\nEarly stopping triggered after {args.early_stopping} epochs without improvement.")
                    break

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'vocab_size': vocab_size,
        'num_styles': num_styles,
        'pad_token_id': pad_token_id,
        'args': vars(args)
    }, os.path.join(args.output_dir, 'final_model.pt'))
    
    train_dataset.tokenizer.save_pretrained(args.output_dir)
    print(f"\nTraining complete! Models saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DrumVAE v2")
    
    # Data
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--file_limit", type=int, default=None)
    parser.add_argument("--chunk_bars", type=int, default=8, help="Number of bars per sequence")
    
    # Model architecture
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_conductor_layers", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--max_bars", type=int, default=32)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    
    # VAE specific - tuned to prevent posterior collapse
    parser.add_argument("--kl_weight", type=float, default=0.1,
                        help="Weight for KL divergence loss")
    parser.add_argument("--kl_cycle_length", type=int, default=10,
                        help="Length of cyclical KL annealing cycle")
    parser.add_argument("--beta_min", type=float, default=0.0,
                        help="Minimum beta value (floor)")
    parser.add_argument("--beta_max", type=float, default=0.5,
                        help="Maximum beta value (cap)")
    parser.add_argument("--free_bits", type=float, default=0.1,
                        help="Minimum KL per latent dimension (prevents collapse)")
    parser.add_argument("--free_bits_weight", type=float, default=0.5,
                        help="Weight for free bits penalty")
    parser.add_argument("--word_dropout", type=float, default=0.25,
                        help="Word dropout rate (lower = decoder relies more on latent)")
    
    # Auxiliary losses
    parser.add_argument("--type_loss_weight", type=float, default=0.1,
                        help="Weight for token type prediction auxiliary loss")
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="checkpoints_v2")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--early_stopping", type=int, default=10,
                        help="Stop after N epochs without improvement (0 to disable)")
    
    args = parser.parse_args()
    train(args)
