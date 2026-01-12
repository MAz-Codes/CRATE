import torch
from torch.utils.data import DataLoader
from dataset import GrooveMidiDataset, DataCollator, STYLES, STYLE_TO_IDX
from model import CrateVAE, generate_square_subsequent_mask, TOKEN_TYPE_MAP
import argparse
import os
from tqdm import tqdm
import json
import numpy as np

def get_token_type_targets(token_ids, tokenizer, device):
    """
    Extract token type targets for auxiliary loss - GPU accelerated.
    Returns tensor of shape [seq_len, batch] with type IDs.
    """
    cache_key = f'_lookup_cache_{device.type}_{device.index if device.index else 0}'
    if not hasattr(get_token_type_targets, cache_key):
        vocab_size = len(tokenizer.vocab)
        lookup = torch.zeros(vocab_size, dtype=torch.long, device=device)
        id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        
        for token_id, token_str in id_to_token.items():
            token_str = str(token_str)
            type_id = 0
            for prefix, type_val in TOKEN_TYPE_MAP.items():
                if token_str.startswith(prefix):
                    type_id = type_val
                    break
            lookup[token_id] = type_id
        
        setattr(get_token_type_targets, cache_key, lookup)
    
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
    cache_key = f'_cache_{device.type}_{device.index if device.index else 0}'
    if not hasattr(get_structure_targets, cache_key):
        vocab_size = len(tokenizer.vocab)
        id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        
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
    
    bar_markers = is_bar_token[token_ids]
    pos_vals = position_values[token_ids]
    
    bar_markers_shifted = torch.cat([torch.zeros(1, batch_size, dtype=torch.long, device=device), 
                                     bar_markers[:-1].long()], dim=0)
    bar_nums = torch.cumsum(bar_markers_shifted, dim=0).clamp(max=31)
    
    has_pos = pos_vals > 0
    
    seq_idx = torch.arange(seq_len, device=device).unsqueeze(1).expand(-1, batch_size)
    
    pos_indices = torch.where(has_pos, seq_idx, torch.tensor(-1, device=device))
    
    last_pos_idx, _ = torch.cummax(pos_indices, dim=0)
    
    gather_idx = last_pos_idx.clamp(min=0)
    
    batch_offset = torch.arange(batch_size, device=device) * seq_len
    flat_idx = (gather_idx + batch_offset.unsqueeze(0)).flatten()
    positions = pos_vals.T.flatten()[flat_idx].reshape(seq_len, batch_size)
    
    positions = torch.where(last_pos_idx >= 0, positions, torch.tensor(0, device=device, dtype=positions.dtype))
    
    return positions, bar_nums

def evaluate(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    test_dataset = GrooveMidiDataset(
        split='test', max_seq_len=args.seq_len)

    try:
        pad_token_id = test_dataset.tokenizer["PAD_None"]
    except:
        pad_token_id = 0
        print("Note: Using token ID 0 as PAD (miditok 3.x default)")

    data_collator = DataCollator(pad_token_id)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=data_collator, num_workers=4)

    print(f"Loaded {len(test_dataset)} test samples")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location=device)
    vocab_size = checkpoint.get('vocab_size')
    num_styles = checkpoint.get('num_styles', len(STYLES) + 1)
    model_args = checkpoint['args']

    if vocab_size is None:
        raise ValueError("Checkpoint missing 'vocab_size'. Cannot load model.")

    dataset_vocab_size = len(test_dataset.tokenizer)
    if dataset_vocab_size != vocab_size:
        print(
            f"WARNING: Dataset tokenizer vocab size ({dataset_vocab_size}) differs from checkpoint ({vocab_size})")
        print(f"Using checkpoint vocab_size: {vocab_size}")

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Vocab size: {vocab_size}, Num styles: {num_styles}")

    model = CrateVAE(
        vocab_size=vocab_size,
        d_model=model_args['d_model'],
        nhead=model_args['nhead'],
        num_encoder_layers=model_args['num_layers'],
        num_decoder_layers=model_args['num_layers'],
        num_conductor_layers=model_args.get('num_conductor_layers', 4),
        latent_dim=model_args['latent_dim'],
        max_seq_len=model_args.get('seq_len', args.seq_len),
        num_styles=num_styles,
        max_bars=model_args.get('max_bars', 32),
        num_token_types=10
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    pad_token_id = checkpoint.get('pad_token_id', 0)
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=pad_token_id, reduction='none')

    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_tokens = 0
    correct_tokens = 0

    style_losses = {style: [] for style in STYLES}
    style_losses['unknown'] = []

    print("\nEvaluating on test set...")

    with torch.no_grad():
        for src, src_key_padding_mask, style_batch, bar_batch in tqdm(test_loader, desc="Evaluating"):
            if src is None:
                continue

            src = src.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            style_batch = style_batch.to(device)
            bar_batch = bar_batch.to(device)

            token_types = get_token_type_targets(src, test_dataset.tokenizer, device)
            positions, bar_nums = get_structure_targets(src, test_dataset.tokenizer, device)

            tgt = src
            dec_input = tgt[:-1]
            dec_target = tgt[1:]
            dec_padding_mask = src_key_padding_mask[:, :-1]

            tgt_seq_len = dec_input.shape[0]
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)

            outputs = model(
                src, dec_input,
                src_token_types=token_types,
                tgt_token_types=token_types[:-1],
                style_id=style_batch,
                bar_id=bar_batch,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=dec_padding_mask,
                tgt_mask=tgt_mask,
                src_bar_pos=positions,
                src_bar_num=bar_nums,
                tgt_bar_pos=positions[:-1],
                tgt_bar_num=bar_nums[:-1]
            )
            
            logits = outputs['logits']
            mu = outputs['mu']
            logvar = outputs['logvar']

            token_losses = criterion(
                logits.reshape(-1, vocab_size), dec_target.reshape(-1))
            recon_loss = token_losses.mean()

            kl_loss = -0.5 * \
                torch.mean(
                    torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

            predictions = logits.argmax(dim=-1)
            non_pad_mask = (dec_target != pad_token_id)
            correct_tokens += ((predictions == dec_target)
                               & non_pad_mask).sum().item()
            total_tokens += non_pad_mask.sum().item()

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

            for i, style_id in enumerate(style_batch.cpu().numpy()):
                if style_id < len(STYLES):
                    style_name = STYLES[style_id]
                else:
                    style_name = 'unknown'
                style_losses[style_name].append(recon_loss.item())

    num_batches = len(test_loader)
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    perplexity = np.exp(avg_recon_loss)
    token_accuracy = (correct_tokens / total_tokens) * \
        100 if total_tokens > 0 else 0

    style_stats = {}
    for style, losses in style_losses.items():
        if len(losses) > 0:
            style_stats[style] = {
                'avg_loss': float(np.mean(losses)),
                'std_loss': float(np.std(losses)),
                'num_samples': len(losses)
            }

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Reconstruction Loss: {avg_recon_loss:.4f}")
    print(f"  KL Divergence:       {avg_kl_loss:.4f}")
    print(f"  Perplexity:          {perplexity:.2f}")
    print(f"  Token Accuracy:      {token_accuracy:.2f}%")
    print(f"  Total Tokens:        {total_tokens:,}")

    print(f"\nPer-Style Performance:")
    for style in STYLES:
        if style in style_stats:
            stats = style_stats[style]
            print(
                f"  {style:12s}: Loss={stats['avg_loss']:.4f} ±{stats['std_loss']:.4f} (n={stats['num_samples']})")

    if 'unknown' in style_stats:
        stats = style_stats['unknown']
        print(
            f"  {'unknown':12s}: Loss={stats['avg_loss']:.4f} ±{stats['std_loss']:.4f} (n={stats['num_samples']})")

    results = {
        'reconstruction_loss': avg_recon_loss,
        'kl_divergence': avg_kl_loss,
        'perplexity': perplexity,
        'token_accuracy': token_accuracy,
        'total_tokens': total_tokens,
        'style_statistics': style_stats,
        'model_path': args.model_path,
        'epoch': checkpoint['epoch']
    }

    output_path = os.path.join(os.path.dirname(
        args.model_path), 'evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("="*60)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Music VAE on test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str,
                        default="Final_GigaMIDI_V1.1_Final", help="Path to MIDI dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer.json",
                        help="Path to tokenizer config")

    args = parser.parse_args()
    evaluate(args)
