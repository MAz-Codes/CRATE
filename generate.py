import torch
import torch.nn.functional as F
import os
import json
from model import MusicTransformerVAE, generate_square_subsequent_mask
from miditok import REMI, TokenizerConfig
import argparse
from symusic import Score
from dataset import STYLES, UNKNOWN_STYLE_IDX
from pathlib import Path
from miditok import MusicTokenizer

# Monkey patch MusicTokenizer.__getitem__ to handle tuple indices
# This fixes a KeyError: (track, token) issue in some miditok versions
original_getitem = MusicTokenizer.__getitem__
def patched_getitem(self, item):
    if isinstance(item, tuple):
        return original_getitem(self, item[1])
    return original_getitem(self, item)
MusicTokenizer.__getitem__ = patched_getitem


def generate(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load Tokenizer (miditok 3.x API)
    try:
        config = TokenizerConfig(
            num_velocities=16,
            use_chords=False,
            use_programs=False, # Match dataset config
            use_rests=True,
            use_tempos=True,
            use_time_signatures=True
        )
        tokenizer = REMI(config)

        # Try to load saved tokenizer config if it exists
        if os.path.exists(args.tokenizer_path):
            print(f"Loading tokenizer from {args.tokenizer_path}")
            try:
                tokenizer = REMI.from_pretrained(args.tokenizer_path)
            except Exception as e:
                print(f"Could not use from_pretrained: {e}")
                print("Attempting manual config load...")
                try:
                    with open(args.tokenizer_path, 'r') as f:
                        data = json.load(f)
                    
                    config_dict = data['config']
                    
                    # Fix beat_res keys
                    if 'beat_res' in config_dict:
                        new_beat_res = {}
                        for k, v in config_dict['beat_res'].items():
                            if isinstance(k, str) and '_' in k:
                                parts = k.split('_')
                                try:
                                    key = (int(parts[0]), int(parts[1]))
                                    new_beat_res[key] = v
                                except:
                                    new_beat_res[k] = v
                            else:
                                new_beat_res[k] = v
                        config_dict['beat_res'] = new_beat_res

                    # Fix beat_res_rest keys
                    if 'beat_res_rest' in config_dict:
                        new_beat_res_rest = {}
                        for k, v in config_dict['beat_res_rest'].items():
                            if isinstance(k, str) and '_' in k:
                                parts = k.split('_')
                                try:
                                    key = (int(parts[0]), int(parts[1]))
                                    new_beat_res_rest[key] = v
                                except:
                                    new_beat_res_rest[k] = v
                            else:
                                new_beat_res_rest[k] = v
                        config_dict['beat_res_rest'] = new_beat_res_rest

                    # Fix time_signature_range keys
                    if 'time_signature_range' in config_dict:
                        new_ts = {}
                        for k, v in config_dict['time_signature_range'].items():
                            try:
                                key = int(k)
                                new_ts[key] = v
                            except:
                                new_ts[k] = v
                        config_dict['time_signature_range'] = new_ts

                    config = TokenizerConfig(**config_dict)
                    tokenizer = REMI(config)
                    print("✓ Loaded tokenizer from manual config parse")
                except Exception as e2:
                    print(f"Manual load failed: {e2}")
                    raise e2
        else:
            print(
                f"Tokenizer file not found at {args.tokenizer_path}, using default config")
    except Exception as e:
        print(f"Could not load tokenizer params, using default: {e}")
        config = TokenizerConfig(
            num_velocities=16,
            use_chords=False,
            use_programs=False,
            use_rests=True,
            use_tempos=True,
            use_time_signatures=True
        )
        tokenizer = REMI(config)

    # Load Checkpoint first to get model args and vocab info
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Use vocab_size and num_styles from checkpoint to ensure consistency
    vocab_size = checkpoint.get('vocab_size', len(tokenizer))
    num_styles = checkpoint.get('num_styles', len(STYLES) + 1)

    # Verify tokenizer matches checkpoint
    current_vocab_size = len(tokenizer)
    if current_vocab_size != vocab_size:
        print(
            f"WARNING: Tokenizer vocab size ({current_vocab_size}) differs from checkpoint ({vocab_size})")
        print(f"Using checkpoint vocab_size: {vocab_size}")

    # Determine model parameters
    if isinstance(checkpoint, dict) and 'args' in checkpoint:
        print(
            "Found training arguments in checkpoint, using them for model initialization.")
        train_args = checkpoint['args']
        d_model = train_args.get('d_model', args.d_model)
        nhead = train_args.get('nhead', args.nhead)
        num_layers = train_args.get('num_layers', args.num_layers)
        latent_dim = train_args.get('latent_dim', args.latent_dim)
        max_seq_len = train_args.get('seq_len', args.seq_len)
        num_memory_tokens = train_args.get('num_memory_tokens', 4)  # Default to 4 (new architecture)
    else:
        print("No training arguments found in checkpoint, using command line arguments.")
        d_model = args.d_model
        nhead = args.nhead
        num_layers = args.num_layers
        latent_dim = args.latent_dim
        max_seq_len = args.seq_len
        num_memory_tokens = 4  # Default

    # Load Model
    model = MusicTransformerVAE(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        latent_dim=latent_dim,
        max_seq_len=max_seq_len,
        num_styles=num_styles,
        num_memory_tokens=num_memory_tokens
    ).to(device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print("Generating music...")

    # Determine Latent Vector z
    if args.input_midi and os.path.exists(args.input_midi):
        print(f"Loading input MIDI for reconstruction: {args.input_midi}")
        try:
            score = Score.from_midi(args.input_midi)
            tokens = tokenizer.encode(score)
            
            # Extract IDs
            token_ids = []
            if isinstance(tokens, list) and len(tokens) > 0:
                if hasattr(tokens[0], 'ids'):
                    token_ids = tokens[0].ids
                elif isinstance(tokens[0], int):
                    token_ids = tokens
            elif hasattr(tokens, 'ids'):
                token_ids = tokens.ids
                
            if len(token_ids) > 0:
                # Truncate to max_seq_len
                token_ids = token_ids[:max_seq_len]
                src = torch.tensor(token_ids, dtype=torch.long).unsqueeze(1).to(device) # [seq_len, 1]
                
                with torch.no_grad():
                    # Encode
                    # We don't provide style_id for encoding in this simple VAE setup usually, 
                    # or we provide the style of the input if known. 
                    # For now, let's assume unconditioned encoding or use UNKNOWN.
                    # The model.encode method signature: encode(self, src, style_id=None, ...)
                    # If the model expects style during encoding (it might if it concatenates style to input),
                    # we should check model.py. 
                    # Checking model.py from context: encode takes style_id but might not use it if not implemented.
                    # Let's pass UNKNOWN_STYLE_IDX just in case.
                    dummy_style = torch.tensor([UNKNOWN_STYLE_IDX], dtype=torch.long, device=device)
                    mu, logvar = model.encode(src, style_id=dummy_style)
                    
                    if args.reconstruct:
                        # Use mean for best reconstruction
                        z = mu
                        print("Encoded input MIDI to latent space (using mean).")
                    else:
                        # Sample with noise (variation)
                        z = model.reparameterize(mu, logvar)
                        print("Encoded input MIDI and sampled nearby (variation).")
            else:
                print("Error: No tokens found in input MIDI. Falling back to random sampling.")
                z = torch.randn(1, latent_dim).to(device)
        except Exception as e:
            print(f"Error processing input MIDI: {e}. Falling back to random sampling.")
            z = torch.randn(1, latent_dim).to(device)
    else:
        # Random sampling
        z = torch.randn(1, latent_dim).to(device)

    # Prepare Style
    if args.style:
        if args.style in STYLES:
            style_id = STYLES.index(args.style)
            print(f"Conditioning on style: {args.style}")
        else:
            print(
                f"Style '{args.style}' not found. Available: {STYLES}. Using unknown.")
            style_id = UNKNOWN_STYLE_IDX
    else:
        style_id = UNKNOWN_STYLE_IDX

    style_tensor = torch.tensor([style_id], dtype=torch.long, device=device)
    
    # Prepare bar count and calculate target length
    bar_count = max(2, min(32, args.num_bars))  # Clamp to 2-32
    bar_tensor = torch.tensor([bar_count], dtype=torch.long, device=device)
    print(f"Conditioning on {bar_count} bars")
    
    # Get predicted length from model (if available)
    with torch.no_grad():
        bar_emb = model.bar_embedding(bar_tensor)  # [1, d_model]
        length_input = torch.cat([z, bar_emb], dim=1)  # [1, latent_dim + d_model]
        predicted_length_normalized = model.length_predictor(length_input).item()  # Scalar in [0, 1]
        
        # Convert normalized prediction to actual token count
        # Predicted length is fraction of max_seq_len
        predicted_tokens = int(predicted_length_normalized * max_seq_len)
        print(f"Model predicted length: {predicted_tokens} tokens ({predicted_length_normalized:.3f} of max_seq_len {max_seq_len})")
    
    # Load tokens-per-bar statistics for fallback
    try:
        with open('tokens_per_bar_stats.json', 'r') as f:
            tpb_stats = json.load(f)
        mean_tokens_per_bar = tpb_stats['mean_tokens_per_bar']
        statistical_estimate = int(bar_count * mean_tokens_per_bar)
        print(f"Statistical estimate: {statistical_estimate} tokens ({mean_tokens_per_bar:.1f} tokens/bar)")
        
        # Use model prediction if available, otherwise use statistical estimate
        target_seq_len = predicted_tokens if predicted_tokens > 0 else statistical_estimate
    except FileNotFoundError:
        print("Warning: tokens_per_bar_stats.json not found, using model prediction only")
        target_seq_len = predicted_tokens if predicted_tokens > 0 else args.seq_len
    
    print(f"Target sequence length: {target_seq_len} tokens")

    # Get BOS (Beginning of Sequence) token from tokenizer
    # miditok 3.x may not have explicit BOS, so we'll use a safe default
    try:
        bos_token = tokenizer["BOS"]
        if isinstance(bos_token, list):
            start_token = bos_token[0]
        else:
            start_token = int(bos_token)
        print(f"Using BOS token: {start_token}")
    except:
        # miditok 3.x doesn't have BOS token, use first valid token (1)
        # Token 0 is typically reserved/padding
        start_token = 1
        print(
            f"Note: Using token ID {start_token} as start token (miditok 3.x default)")

    generated = torch.tensor([[start_token]], dtype=torch.long).to(device)

    # Condition on Tempo if requested
    if args.tempo:
        try:
            # Find all Tempo tokens in vocab
            tempo_tokens = []
            for token_str, token_id in tokenizer.vocab.items():
                if str(token_str).startswith("Tempo_"):
                    try:
                        bpm = float(str(token_str).split("_")[1])
                        tempo_tokens.append((bpm, token_id))
                    except ValueError:
                        pass
            
            if tempo_tokens:
                # Find closest tempo
                target_bpm = float(args.tempo)
                closest_tempo = min(tempo_tokens, key=lambda x: abs(x[0] - target_bpm))
                closest_bpm, closest_token_id = closest_tempo
                
                print(f"Conditioning on tempo: {target_bpm} BPM (closest token: {closest_bpm} BPM)")
                
                # Append tempo token to generated sequence
                tempo_tensor = torch.tensor([[closest_token_id]], dtype=torch.long).to(device)
                generated = torch.cat([generated, tempo_tensor], dim=0)
            else:
                print("WARNING: No Tempo tokens found in vocabulary. Ignoring --tempo.")
        except Exception as e:
            print(f"WARNING: Failed to set tempo: {e}")

    with torch.no_grad():
        for _ in range(target_seq_len):  # Use calculated target length
            seq_len = generated.size(0)
            tgt_mask = generate_square_subsequent_mask(seq_len).to(device)

            # Decode
            logits = model.decode(
                z, generated, style_id=style_tensor, bar_id=bar_tensor, tgt_mask=tgt_mask)

            # Get last token logits
            last_logits = logits[-1, :]

            # Temperature sampling
            probs = torch.softmax(last_logits / args.temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)

            # Ensure next_token is [1, batch]
            next_token = next_token.transpose(0, 1)

            generated = torch.cat([generated, next_token], dim=0)

            # Stop if EOS token generated (if available)
            next_token_id = next_token.item()
            try:
                eos_token = tokenizer["EOS"]
                if isinstance(eos_token, list):
                    eos_id = eos_token[0]
                else:
                    eos_id = int(eos_token)

                if next_token_id == eos_id:
                    print(
                        f"EOS token generated at position {generated.size(0)}")
                    break
            except:
                # No EOS token in miditok 3.x, just continue
                pass

    # Convert tokens to MIDI
    tokens = generated.squeeze().cpu().tolist()
    print(f"Generated {len(tokens)} tokens")

    try:
        # Decode tokens to MIDI using miditok 3.x
        # miditok 3.x decode expects tokens wrapped in proper format
        from miditok import TokSequence
        
        # Create TokSequence object
        # If one_token_stream is True, we need to wrap tokens appropriately
        if hasattr(tokenizer, 'one_token_stream') and tokenizer.one_token_stream:
            # For single stream tokenizers
            tok_seq = TokSequence(ids=tokens)
            midi = tokenizer.decode([tok_seq])
        else:
            # Try direct decode first (fallback)
            try:
                tok_seq = TokSequence(ids=tokens)
                midi = tokenizer.decode([tok_seq])
            except:
                # Last resort: direct list decode
                midi = tokenizer.decode([tokens])

        # Post-processing: Merge all tracks into a single Piano track
        # This addresses the "16 different midi channels" issue
        if args.single_track and len(midi.tracks) > 0:
            print(
                f"Merging {len(midi.tracks)} tracks into a single Drum track...")
            all_notes = []
            for track in midi.tracks:
                all_notes.extend(track.notes)

            # Sort notes by time
            all_notes.sort(key=lambda x: x.time)

            # Use the first track as the container
            main_track = midi.tracks[0]
            main_track.program = 0 
            main_track.is_drum = True # DRUMS
            main_track.name = "Generated Drums"
            main_track.notes = all_notes

            # Replace tracks list with just the main track
            midi.tracks = [main_track]

        # Save MIDI file (symusic Score object)
        if hasattr(midi, 'dump_midi'):
            midi.dump_midi(args.output_path)
            print(f"✓ Saved generated MIDI to {args.output_path}")
        elif hasattr(midi, 'dump'):
            midi.dump(args.output_path)
            print(f"✓ Saved generated MIDI to {args.output_path}")
        else:
            raise ValueError(
                f"Unknown MIDI object type: {type(midi)}. Expected symusic Score with dump_midi() method.")

    except Exception as e:
        print(f"ERROR decoding/saving MIDI: {type(e).__name__}: {e}")
        print(f"Token sample (first 20): {tokens[:20]}")
        raise


def get_unique_path(base_path):
    """
    Given a path, returns a unique path by appending a number if the file already exists.
    Example: generated.mid -> generated_1.mid -> generated_2.mid
    """
    if not os.path.exists(base_path):
        return base_path

    path_obj = Path(base_path)
    parent = path_obj.parent
    stem = path_obj.stem
    suffix = path_obj.suffix

    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return str(new_path)
        counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    # Changed default to save in a folder
    parser.add_argument("--output_path", type=str, default="generated_samples/generated.mid")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--style", type=str, default=None,
                        help="Style to generate")
    parser.add_argument("--num_bars", type=int, default=8,
                        help="Number of bars to generate (2-32)")
    parser.add_argument("--single_track", action="store_true",
                        help="Merge all tracks into a single piano track (default: False, keep original tracks)")
    parser.add_argument("--tempo", type=int, default=None,
                        help="Target tempo in BPM")
    parser.add_argument("--input_midi", type=str, default=None,
                        help="Path to input MIDI for reconstruction")
    parser.add_argument("--reconstruct", action="store_true",
                        help="Reconstruct the input MIDI instead of random sampling")

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get unique path
    args.output_path = get_unique_path(args.output_path)
    print(f"Output will be saved to: {args.output_path}")

    generate(args)
