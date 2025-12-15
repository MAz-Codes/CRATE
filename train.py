import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GrooveMidiDataset, DataCollator, STYLES, UNKNOWN_STYLE_IDX
from model import MusicTransformerVAE, generate_square_subsequent_mask
import argparse
import os
from tqdm import tqdm
import json
import time
from symusic import Score


def generate_samples(model, tokenizer, device, epoch, output_dir, num_samples=2):
    """
    Generate sample MIDI files during training to monitor progress.
    """
    model.eval()
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    test_cases = [
        {'style': 'funk', 'bars': 4, 'name': 'funk_4bars'},
        {'style': 'rock', 'bars': 8, 'name': 'rock_8bars'}
    ]
    
    print(f"\nGenerating {len(test_cases)} validation samples...")
    
    with torch.no_grad():
        for i, case in enumerate(test_cases):
            # Prepare Style
            style_name = case['style']
            if style_name in STYLES:
                style_id = STYLES.index(style_name)
            else:
                style_id = UNKNOWN_STYLE_IDX
            
            style_tensor = torch.tensor([style_id], dtype=torch.long, device=device)
            
            # Prepare Bar Count
            bar_count = case['bars']
            bar_tensor = torch.tensor([bar_count], dtype=torch.long, device=device)
            
            # Sample Latent Vector
            z = torch.randn(1, model.latent_dim).to(device)
            
            # Predict Length (optional check)
            bar_emb = model.bar_embedding(bar_tensor)
            length_input = torch.cat([z, bar_emb], dim=1)
            predicted_length_normalized = model.length_predictor(length_input).item()
            target_seq_len = int(predicted_length_normalized * model.max_seq_len)
            # Clamp target length to reasonable bounds (e.g. 32 tokens per bar approx)
            target_seq_len = max(32, min(model.max_seq_len, target_seq_len))
            
            # Start Token
            try:
                bos_token = tokenizer["BOS_None"]
                if isinstance(bos_token, list):
                    start_token = bos_token[0]
                else:
                    start_token = int(bos_token)
            except:
                start_token = 1 # Fallback
                
            generated = torch.tensor([[start_token]], dtype=torch.long).to(device)
            
            # Decode Loop
            for _ in range(target_seq_len):
                seq_len = generated.size(0)
                tgt_mask = generate_square_subsequent_mask(seq_len).to(device)
                
                logits = model.decode(z, generated, style_id=style_tensor, bar_id=bar_tensor, tgt_mask=tgt_mask)
                last_logits = logits[-1, :]
                
                # Greedy sampling for stability during training checks
                next_token = torch.argmax(last_logits, dim=-1).unsqueeze(0).unsqueeze(1)
                generated = torch.cat([generated, next_token], dim=0)
                
                # Stop if EOS
                try:
                    eos_token = tokenizer["EOS_None"]
                    eos_id = eos_token[0] if isinstance(eos_token, list) else int(eos_token)
                    if next_token.item() == eos_id:
                        break
                except:
                    pass
            
            # Save MIDI
            gen_seq = generated.squeeze().cpu().numpy().tolist()
            try:
                # Decode tokens to Score
                score = tokenizer.decode(gen_seq)
                filename = f"epoch_{epoch+1}_{case['name']}.mid"
                score.dump_midi(os.path.join(samples_dir, filename))
            except Exception as e:
                print(f"Failed to save sample {case['name']}: {e}")

    model.train()


def validate(model, dataloader, criterion, device, vocab_size):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for src, src_key_padding_mask, style_batch, bar_batch in dataloader:
            if src is None:
                continue

            src = src.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            style_batch = style_batch.to(device)
            bar_batch = bar_batch.to(device)

            tgt = src
            dec_input = tgt[:-1]
            dec_target = tgt[1:]
            dec_padding_mask = src_key_padding_mask[:, :-1]

            tgt_seq_len = dec_input.shape[0]
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)

            logits, mu, logvar, predicted_length = model(src, dec_input, style_id=style_batch, bar_id=bar_batch,
                                       src_key_padding_mask=src_key_padding_mask,
                                       tgt_key_padding_mask=dec_padding_mask,
                                       tgt_mask=tgt_mask)

            recon_loss = criterion(
                logits.reshape(-1, vocab_size), dec_target.reshape(-1))

            kl_loss = -0.5 * \
                torch.mean(
                    torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            
            # Calculate actual sequence lengths (non-padding tokens)
            actual_lengths = (src != 0).sum(dim=0).float()  # [batch_size]
            
            # Normalize lengths to [0, 1] range for stable training
            max_len = float(src.size(0))  # Sequence length dimension
            normalized_actual = actual_lengths / max_len
            
            # Length prediction loss (on normalized values)
            if predicted_length is not None:
                # Predicted length is already in [0, 1] range from sigmoid
                length_loss = F.mse_loss(predicted_length, normalized_actual)
            else:
                length_loss = torch.tensor(0.0, device=device)

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_tokens += (dec_target != 0).sum().item()

    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    import numpy as np
    perplexity = np.exp(avg_recon_loss)

    return avg_recon_loss, avg_kl_loss, perplexity


def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")



    # Load training dataset
    # Bar counts are optional - model will use default of 8 bars if unavailable
    train_dataset = GrooveMidiDataset(
        split='train', max_seq_len=args.seq_len, max_examples=args.file_limit)

    try:
        pad_token = train_dataset.tokenizer["PAD"]
        if isinstance(pad_token, list):
            pad_token_id = pad_token[0]
        else:
            pad_token_id = int(pad_token)
    except:
        pad_token_id = 0
        print("Note: Using token ID 0 as PAD (miditok 3.x default)")

    data_collator = DataCollator(pad_token_id)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=data_collator, num_workers=0) # TFDS is already loaded in memory or iterable, workers=0 safer for now

    val_dataset = GrooveMidiDataset(
        split='validation', max_seq_len=args.seq_len, max_examples=args.file_limit)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=data_collator, num_workers=0)

    print(f"Pad token ID: {pad_token_id}")
    print(f"Sample tokens from first batch to verify...")


    if len(train_dataset.tokenizer) == 0:
        raise ValueError(
            "Tokenizer has empty vocabulary! Check miditok configuration.")


    vocab_size = len(train_dataset.tokenizer)
    num_styles = len(STYLES) + 1
    print(f"Vocabulary Size: {vocab_size}, Num Styles: {num_styles}")

    model = MusicTransformerVAE(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        latent_dim=args.latent_dim,
        max_seq_len=args.seq_len,
        num_styles=num_styles,
        num_bar_classes=33,  # Support 0-32 bars
        num_memory_tokens=args.num_memory_tokens
    ).to(device)

    # Use Adam (not AdamW) if weight_decay is 0, otherwise use AdamW
    if args.weight_decay > 0:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # ReduceLROnPlateau scheduler (original working approach)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    print(f"Using PAD token ID: {pad_token_id}")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=args.label_smoothing)


    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(
                f"Warning: Checkpoint {args.resume} not found. Starting from scratch.")


    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize or load training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    if args.resume and os.path.exists(history_path):
        print(f"Loading training history from {history_path}")
        with open(history_path, 'r') as f:
            history = json.load(f)
        print(f"Loaded {len(history['train_loss'])} previous epochs of history")
    else:
        history = {'train_loss': [], 'val_loss': [], 'val_perplexity': [], 
                   'train_recon_loss': [], 'train_kl_loss': [], 'train_length_loss': [],
                   'val_recon_loss': [], 'val_kl_loss': [], 'beta': [], 'active_units': []}


    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        model.train()

        # Cyclic Annealing Schedule
        # Cycle every 10 epochs (or args.kl_warmup_epochs if provided)
        cycle_length = args.kl_warmup_epochs if args.kl_warmup_epochs > 0 else 10
        cycle = epoch // cycle_length
        epoch_in_cycle = epoch % cycle_length
        
        # Linear warmup for first 50% of cycle, then stay at 1.0
        # beta goes from 0 to 1 over cycle_length/2 epochs
        beta = min(1.0, epoch_in_cycle / (cycle_length * 0.5))
        
        # Linear warmup for learning rate in first few epochs (only if lr_warmup_epochs > 0)
        if args.lr_warmup_epochs > 0 and epoch < args.lr_warmup_epochs:
            lr_scale = (epoch + 1) / args.lr_warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * lr_scale

        epoch_start_time = time.time()
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, (src, src_key_padding_mask, style_batch, bar_batch) in enumerate(progress_bar):
            if src is None:
                continue

            src = src.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            style_batch = style_batch.to(device)
            bar_batch = bar_batch.to(device)


            tgt = src

            # Word Dropout: Randomly mask tokens in decoder input
            # This forces the decoder to rely on the latent variable z
            if args.word_dropout > 0:
                # Create a mask for word dropout (1 = keep, 0 = mask)
                # We don't mask padding tokens or the first token (BOS)
                prob = torch.rand(tgt.shape, device=device)
                # Mask tokens where prob < word_dropout
                # But keep padding (src_key_padding_mask is True for padding in some conventions, check dataset)
                # In PyTorch Transformer, key_padding_mask is True for padded elements.
                # Here src_key_padding_mask is likely True for padding.
                
                # Let's assume we mask with a special token or just 0 (if 0 is PAD, maybe use MASK token if available)
                # If no MASK token, we can replace with PAD or random token.
                # Ideally we should use a MASK token.
                
                # Check if MASK token exists in tokenizer
                try:
                    mask_token_id = train_dataset.tokenizer["MASK_None"]
                    if isinstance(mask_token_id, list): mask_token_id = mask_token_id[0]
                except:
                    mask_token_id = pad_token_id # Fallback to PAD if no MASK
                
                word_dropout_mask = (prob < args.word_dropout) & (tgt != pad_token_id)
                # Don't mask the first token (BOS usually)
                word_dropout_mask[0, :] = False 
                
                dec_input_raw = tgt[:-1].clone()
                dec_input_raw[word_dropout_mask[:-1]] = mask_token_id
                dec_input = dec_input_raw
            else:
                dec_input = tgt[:-1]
            
            dec_target = tgt[1:]
            dec_padding_mask = src_key_padding_mask[:, :-1]

            tgt_seq_len = dec_input.shape[0]
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)

            logits, mu, logvar, predicted_length = model(src, dec_input, style_id=style_batch, bar_id=bar_batch, src_key_padding_mask=src_key_padding_mask,
                                       tgt_key_padding_mask=dec_padding_mask, tgt_mask=tgt_mask)


            recon_loss = criterion(
                logits.reshape(-1, vocab_size), dec_target.reshape(-1))


            # KL divergence with Free Bits
            # Calculate KL for each element in the batch and each latent dimension
            kl_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            
            # Free bits: ensure each dimension contributes at least `free_bits` nats to the KL
            # This prevents the model from completely ignoring dimensions (posterior collapse)
            free_bits = torch.tensor(args.free_bits, device=device)
            kl_loss = torch.mean(torch.sum(torch.max(kl_element, free_bits), dim=1))
            
            # Calculate active units (KL > 0.1)
            # We use the mean KL per dimension across the batch
            mean_kl_per_dim = torch.mean(kl_element, dim=0) # [latent_dim]
            active_units = torch.sum(mean_kl_per_dim > 0.1).item()
            
            # Calculate actual sequence lengths (non-padding tokens)
            actual_lengths = (src != 0).sum(dim=0).float()  # [batch_size]
            
            # Normalize lengths to [0, 1] range for stable training
            max_len = float(src.size(0))  # Sequence length dimension
            normalized_actual = actual_lengths / max_len
            
            # Length prediction loss (on normalized values)
            if predicted_length is not None:
                length_loss = F.mse_loss(predicted_length, normalized_actual)
            else:
                length_loss = torch.tensor(0.0, device=device)

            # Total loss with length prediction
            loss = recon_loss + beta * kl_loss * args.kl_weight + args.length_loss_weight * length_loss
            
            # Scale loss by gradient accumulation steps
            loss = loss / args.grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            # Multiply loss back for logging
            loss_val = loss.item() * args.grad_accum_steps
            total_loss += loss_val
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

            progress_bar.set_postfix({
                'Loss': f'{loss_val:.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'KL': f'{kl_loss.item():.4f}',
                'Length': f'{length_loss.item():.4f}',
                'Beta': f'{beta:.3f}'
            })

        avg_train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time

        print(f"\nRunning validation...")
        val_recon_loss, val_kl_loss, val_perplexity = validate(
            model, val_loader, criterion, device, vocab_size)
        val_total_loss = val_recon_loss + beta * val_kl_loss * args.kl_weight

        print(f"Epoch {epoch+1}/{args.epochs} - Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(
            f"  Val Loss: {val_total_loss:.4f} | Val Recon: {val_recon_loss:.4f} | Val KL: {val_kl_loss:.4f} | Perplexity: {val_perplexity:.2f}")

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_total_loss)
        history['val_perplexity'].append(val_perplexity)
        
        # Log detailed metrics
        history['train_recon_loss'].append(total_recon_loss / len(train_loader))
        history['train_kl_loss'].append(total_kl_loss / len(train_loader))
        history['val_recon_loss'].append(val_recon_loss)
        history['val_kl_loss'].append(val_kl_loss)
        history['beta'].append(beta)
        
        # Log active units (from last batch of epoch - approximation)
        history['active_units'].append(active_units)

        # Step scheduler (ReduceLROnPlateau steps on validation loss)
        scheduler.step(val_total_loss)

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'vocab_size': vocab_size,
                'num_styles': num_styles,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ✓ Saved best model to {checkpoint_path}")


        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'vocab_size': vocab_size,
                'num_styles': num_styles,
                'pad_token_id': pad_token_id,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
            
            # Generate samples for validation
            try:
                generate_samples(model, train_dataset.tokenizer, device, epoch, args.output_dir)
            except Exception as e:
                print(f"Warning: Sample generation failed: {e}")


        history_path = os.path.join(args.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)


    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'vocab_size': vocab_size,
        'num_styles': num_styles,
        'pad_token_id': pad_token_id,
        'args': vars(args)
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")


    train_dataset.tokenizer.save_pretrained(args.output_dir)
    print(f"Tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="Final_GigaMIDI_V1.1_Final", help="Path to MIDI dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_min", type=float, default=1e-5, help="Minimum learning rate for cosine annealing")
    parser.add_argument("--lr_warmup_epochs", type=int, default=0, help="Number of epochs for LR warmup (0 to disable)")
    parser.add_argument("--kl_weight", type=float, default=0.02)
    parser.add_argument("--kl_warmup_epochs", type=int, default=0, help="Number of epochs for KL annealing (0 for full training duration)")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for AdamW")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--scheduler_t0", type=int, default=20, help="T_0 for cosine annealing with warm restarts")
    parser.add_argument("--free_bits", type=float, default=0.1, 
                        help="Minimum KL divergence per latent dimension (nats)")
    parser.add_argument("--file_limit", type=int, default=None,
                        help="Limit number of files for testing")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping threshold")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of data loading workers")
    parser.add_argument("--num_memory_tokens", type=int, default=8,
                        help="Number of memory tokens for latent expansion")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--length_loss_weight", type=float, default=0.1,
                        help="Weight for length prediction loss")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--word_dropout", type=float, default=0.0,
                        help="Probability of masking input tokens to decoder")

    args = parser.parse_args()
    train(args)
