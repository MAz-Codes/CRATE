"""
Generation script for CrateVAE with constrained decoding for musical structure.
"""

import torch
import torch.nn.functional as F
import argparse
import os
from miditok import REMI, TokenizerConfig
from model import CrateVAE, generate_square_subsequent_mask, TOKEN_TYPE_MAP


# REMI token type patterns - enforces grammar
VALID_NEXT_TYPES = {
    'Bar': ['Position', 'TimeSig', 'Tempo'],
    'Position': ['Pitch', 'PitchDrum', 'Rest'],
    'Pitch': ['Velocity'],
    'PitchDrum': ['Velocity'],
    'Velocity': ['Duration'],
    'Duration': ['Position', 'Pitch', 'PitchDrum', 'Rest', 'Bar'],
    'Tempo': ['Position', 'Pitch', 'PitchDrum'],
    'TimeSig': ['Position', 'Tempo'],
    'Rest': ['Position', 'Bar'],
    'BOS': ['Bar', 'TimeSig', 'Tempo'],
    'EOS': [],
    'PAD': ['Bar', 'Position'],
}


def get_token_type(token_str):
    """Extract token type from token string."""
    token_str = str(token_str)
    # Check PitchDrum before Pitch to avoid false match
    if token_str.startswith('PitchDrum'):
        return 'PitchDrum'
    for prefix in TOKEN_TYPE_MAP.keys():
        if token_str.startswith(prefix):
            return prefix
    return 'PAD'


def create_type_mask(prev_type, vocab, id_to_token, device):
    """
    Create a mask that only allows valid next tokens based on REMI grammar.
    Returns a tensor of shape [vocab_size] with -inf for invalid tokens.
    """
    valid_types = VALID_NEXT_TYPES.get(prev_type, ['Bar', 'Position', 'Pitch', 'PitchDrum'])
    
    mask = torch.full((len(vocab),), float('-inf'), device=device)
    
    for token_id in range(len(vocab)):
        token_str = str(id_to_token.get(token_id, ''))
        token_type = get_token_type(token_str)
        
        if token_type in valid_types:
            mask[token_id] = 0.0
    
    return mask


def get_structure_info(token_ids, id_to_token, device):
    """
    Extract bar position and bar number for the sequence.
    """
    seq_len = token_ids.size(0)
    positions = torch.zeros_like(token_ids)
    bar_nums = torch.zeros_like(token_ids)
    
    current_pos = 0
    current_bar = 0
    
    for s in range(seq_len):
        tid = token_ids[s, 0].item()
        token_str = str(id_to_token.get(tid, ''))
        
        if token_str.startswith('Bar'):
            if s > 0:
                current_bar += 1
            current_pos = 0
        elif token_str.startswith('Position_'):
            try:
                pos = int(token_str.split('_')[1])
                current_pos = min(pos, 95)
            except:
                pass
        
        positions[s, 0] = current_pos
        bar_nums[s, 0] = min(current_bar, 31)
        
    return positions, bar_nums


def get_token_types_for_seq(token_ids, id_to_token, device):
    """
    Compute token type IDs for the generated sequence.
    This ensures train/test consistency for the decoder's token type embeddings.
    """
    seq_len = token_ids.size(0)
    type_ids = torch.zeros_like(token_ids)
    
    for s in range(seq_len):
        tid = token_ids[s, 0].item()
        token_str = str(id_to_token.get(tid, ''))
        type_id = 0  # Default: PAD/unknown
        
        # Check PitchDrum before Pitch to avoid false match
        if token_str.startswith('PitchDrum'):
            type_id = TOKEN_TYPE_MAP.get('PitchDrum', 3)
        else:
            for prefix, tval in TOKEN_TYPE_MAP.items():
                if token_str.startswith(prefix):
                    type_id = tval
                    break
        
        type_ids[s, 0] = type_id
    
    return type_ids


def generate_constrained(model, tokenizer, z, style_id, bar_id, device, 
                         max_len=512, temperature=0.9, top_k=50, top_p=0.95):
    """
    Generate a sequence with constrained decoding to enforce REMI grammar.
    """
    model.eval()
    
    vocab = tokenizer.vocab
    id_to_token = {v: k for k, v in vocab.items()}
    
    # Start with BOS or first valid token
    try:
        start_token = tokenizer["BOS_None"]
        if isinstance(start_token, list):
            start_token = start_token[0]
    except:
        start_token = 1
    
    generated = torch.tensor([[start_token]], dtype=torch.long, device=device)
    prev_type = 'BOS'
    
    # Track bar structure
    current_bar = 0
    current_position = 0
    notes_in_bar = 0
    target_bars = bar_id.item() if bar_id is not None else 8
    
    with torch.inference_mode():
        for step in range(max_len):
            seq_len = generated.size(0)
            tgt_mask = generate_square_subsequent_mask(seq_len, device)
            
            # Get structure info
            positions, bar_nums = get_structure_info(generated, id_to_token, device)
            
            # Get token types for decoder (ensures train/test consistency)
            token_types = get_token_types_for_seq(generated, id_to_token, device)
            
            # Decode
            logits, type_logits, _ = model.decode(
                z, generated, 
                token_types=token_types,
                style_id=style_id, 
                bar_id=bar_id,
                tgt_mask=tgt_mask,
                bar_pos=positions,
                bar_num=bar_nums
            )
            
            # Get last token logits
            last_logits = logits[-1, 0, :]  # [vocab_size]
            last_type_logits = type_logits[-1, 0, :]  # [num_types]
            
            # Apply temperature
            last_logits = last_logits / temperature
            
            # Create grammar mask
            grammar_mask = create_type_mask(prev_type, vocab, id_to_token, device)
            
            # Apply grammar constraints
            last_logits = last_logits + grammar_mask
            
            # Force Bar token if we've gone too long without one (prevent infinite bars)
            # A typical dense drum bar is ~30-60 tokens. 128 is a safe upper limit.
            tokens_since_last_bar = step - last_bar_step if 'last_bar_step' in locals() else step
            if tokens_since_last_bar > 128 and current_bar < target_bars:
                 # Boost Bar token probability significantly to force a bar change
                bar_token_ids = [tid for tid, tok in id_to_token.items() if 'Bar' in str(tok)]
                for tid in bar_token_ids:
                    last_logits[tid] += 10.0
            
            # Force position progression within bar
            if prev_type == 'Duration' and notes_in_bar > 0:
                # Prefer positions that are >= current_position
                for tid, tok in id_to_token.items():
                    tok_str = str(tok)
                    if tok_str.startswith('Position_'):
                        try:
                            pos = int(tok_str.split('_')[1])
                            if pos < current_position:
                                last_logits[tid] -= 3.0  # Penalize backward positions
                            elif pos == current_position:
                                last_logits[tid] += 1.0  # Slight boost for same position (chord)
                        except:
                            pass
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = last_logits < torch.topk(last_logits, top_k)[0][..., -1, None]
                last_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    0, sorted_indices, sorted_indices_to_remove)
                last_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(last_logits, dim=-1)
            
            # Handle all -inf case
            if torch.all(probs == 0) or torch.any(torch.isnan(probs)):
                # Fallback: sample from valid tokens only
                valid_mask = grammar_mask == 0
                if valid_mask.any():
                    uniform_probs = torch.zeros_like(probs)
                    uniform_probs[valid_mask] = 1.0 / valid_mask.sum()
                    probs = uniform_probs
                else:
                    break
            
            next_token_id = torch.multinomial(probs, 1).item()
            next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            generated = torch.cat([generated, next_token], dim=0)
            
            # Update state
            token_str = str(id_to_token.get(next_token_id, ''))
            prev_type = get_token_type(token_str)
            
            # Track musical structure
            if prev_type == 'Bar':
                current_bar += 1
                current_position = 0
                notes_in_bar = 0
                last_bar_step = step
            elif token_str.startswith('Position_'):
                try:
                    current_position = int(token_str.split('_')[1])
                except:
                    pass
            elif prev_type in ['Pitch', 'PitchDrum']:
                notes_in_bar += 1
            
            # Stop conditions
            # We want to complete the Nth bar, so we stop when we see the start of the (N+1)th bar
            if current_bar > target_bars:
                # Remove the last token (the extra Bar token) and break
                generated = generated[:-1]
                break
            
            try:
                eos_token = tokenizer["EOS_None"]
                eos_id = eos_token[0] if isinstance(eos_token, list) else int(eos_token)
                if next_token_id == eos_id:
                    break
            except:
                pass
    
    return generated.squeeze().cpu().tolist()


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(args):
    # Device - prioritize GPU (CUDA > MPS > CPU)
    device = get_device()
    print(f"Using device: {device}")
    
    # Optimize for GPU inference
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load tokenizer from checkpoint directory (ensures vocabulary consistency)
    checkpoint_dir = os.path.dirname(args.model_path)
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.json')
    
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {checkpoint_dir}")
        try:
            # Try the new miditok API first
            tokenizer = REMI(params=tokenizer_path)
        except (TypeError, Exception):
            # Fallback: create fresh tokenizer with same config
            print("Note: Using fresh tokenizer configuration")
            config = TokenizerConfig(
                num_velocities=16, use_chords=False, use_programs=False,
                use_rests=True, use_tempos=True, use_time_signatures=True,
                use_sustain_pedals=False, use_pitch_bends=False
            )
            tokenizer = REMI(config)
    else:
        print("Warning: No saved tokenizer found, creating fresh tokenizer (may cause vocab mismatch)")
        config = TokenizerConfig(
            num_velocities=16, use_chords=False, use_programs=False,
            use_rests=True, use_tempos=True, use_time_signatures=True,
            use_sustain_pedals=False, use_pitch_bends=False
        )
        tokenizer = REMI(config)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    train_args = checkpoint.get('args', {})
    
    model = CrateVAE(
        vocab_size=checkpoint['vocab_size'],
        d_model=train_args.get('d_model', 512),
        nhead=train_args.get('nhead', 8),
        num_encoder_layers=train_args.get('num_layers', 6),
        num_decoder_layers=train_args.get('num_layers', 6),
        num_conductor_layers=train_args.get('num_conductor_layers', 4),
        latent_dim=train_args.get('latent_dim', 256),
        max_seq_len=train_args.get('seq_len', 512),
        num_styles=checkpoint.get('num_styles', 20),
        max_bars=train_args.get('max_bars', 32),
        num_token_types=10
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")
    
    # Style mapping
    from dataset import STYLES, STYLE_TO_IDX, UNKNOWN_STYLE_IDX
    
    style_name = args.style.lower()
    if style_name in STYLE_TO_IDX:
        style_id = STYLE_TO_IDX[style_name]
    else:
        print(f"Warning: Unknown style '{args.style}', using default")
        style_id = UNKNOWN_STYLE_IDX
    
    style_tensor = torch.tensor([style_id], dtype=torch.long, device=device)
    bar_tensor = torch.tensor([args.num_bars], dtype=torch.long, device=device)
    
    print(f"Generating {args.num_bars}-bar {args.style} beat...")
    
    # Sample latent
    z = torch.randn(1, train_args.get('latent_dim', 256), device=device)
    
    # Generate with constraints
    tokens = generate_constrained(
        model, tokenizer, z, style_tensor, bar_tensor, device,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    print(f"Generated {len(tokens)} tokens")
    
    # Check if we actually reached the target number of bars
    # Count 'Bar' tokens in the output
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    bar_count = sum(1 for t in tokens if 'Bar' in str(id_to_token.get(t, '')))
    print(f"Generated {bar_count} bars (Target: {args.num_bars})")
    
    if bar_count < args.num_bars:
        print(f"Warning: Sequence truncated at {len(tokens)} tokens (max_len={args.max_len}).")
        print("Try increasing --max_len if you want more bars.")

    # Decode to MIDI
    try:
        score = tokenizer.decode([tokens])
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        
        score.dump_midi(args.output)
        print(f"✓ Saved to {args.output}")
        
        # Print summary
        total_notes = sum(len(t.notes) for t in score.tracks)
        duration = score.end() / score.ticks_per_quarter if score.ticks_per_quarter > 0 else 0
        print(f"  Notes: {total_notes}, Duration: {duration:.1f} beats")
        
    except Exception as e:
        print(f"Error decoding MIDI: {e}")
        # Save tokens for debugging
        import json
        with open(args.output + '.tokens.json', 'w') as f:
            json.dump(tokens, f)
        print(f"Saved raw tokens to {args.output}.tokens.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate drums with CrateVAE")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output", type=str, default="samples/generated.mid")
    parser.add_argument("--style", type=str, default="rock")
    parser.add_argument("--num_bars", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_len", type=int, default=1024)
    
    args = parser.parse_args()
    main(args)
