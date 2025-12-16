"""
Batch generation script for DrumVAE v2.
Generates 40 diverse drum samples across all available genres.
"""

import torch
import os
import argparse
from miditok import REMI, TokenizerConfig
from model_v2 import DrumVAE
from generate_v2 import generate_constrained, get_device
from dataset import STYLES
import random
from tqdm import tqdm


def generate_batch_samples(model_path, output_dir, num_samples=40, max_len=1024):
    """
    Generate multiple samples across different genres with varied parameters.
    
    Args:
        model_path: Path to the trained model checkpoint
        output_dir: Directory to save generated samples
        num_samples: Total number of samples to generate (default: 40)
        max_len: Maximum sequence length for generation
    """
    # Device setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Optimize for GPU inference
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load tokenizer
    checkpoint_dir = os.path.dirname(model_path)
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.json')
    
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {checkpoint_dir}")
        try:
            tokenizer = REMI(params=tokenizer_path)
        except (TypeError, Exception):
            print("Note: Using fresh tokenizer configuration")
            config = TokenizerConfig(
                num_velocities=16, use_chords=False, use_programs=False,
                use_rests=True, use_tempos=True, use_time_signatures=True,
                use_sustain_pedals=False, use_pitch_bends=False
            )
            tokenizer = REMI(config)
    else:
        print("Warning: No saved tokenizer found, creating fresh tokenizer")
        config = TokenizerConfig(
            num_velocities=16, use_chords=False, use_programs=False,
            use_rests=True, use_tempos=True, use_time_signatures=True,
            use_sustain_pedals=False, use_pitch_bends=False
        )
        tokenizer = REMI(config)
    
    # Load model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    train_args = checkpoint.get('args', {})
    
    model = DrumVAE(
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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generation parameters - create diverse configurations
    latent_dim = train_args.get('latent_dim', 256)
    
    # Calculate samples per genre (distribute 40 across 18 genres)
    # Some genres get 2 samples, some get 3 for total of 40
    num_genres = len(STYLES)
    base_samples_per_genre = num_samples // num_genres  # 2
    extra_samples = num_samples % num_genres  # 4
    
    generation_configs = []
    sample_idx = 1
    
    for genre_idx, genre in enumerate(STYLES):
        # Some genres get an extra sample
        samples_for_this_genre = base_samples_per_genre + (1 if genre_idx < extra_samples else 0)
        
        for i in range(samples_for_this_genre):
            # Vary parameters for diversity
            config = {
                'sample_id': sample_idx,
                'genre': genre,
                'genre_idx': genre_idx,
                'num_bars': random.choice([4, 8, 16]),
                'temperature': round(random.uniform(0.7, 1.1), 2),
                'top_k': random.randint(30, 70),
                'top_p': round(random.uniform(0.90, 0.98), 2),
                'seed': random.randint(1, 1000000)
            }
            generation_configs.append(config)
            sample_idx += 1
    
    # Shuffle configs for variety in generation order
    random.shuffle(generation_configs)
    
    print(f"\n{'='*70}")
    print(f"Generating {num_samples} samples across {num_genres} genres")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    successful = 0
    failed = 0
    
    # Generate samples with progress bar
    for config in tqdm(generation_configs, desc="Generating samples"):
        try:
            # Set random seed for reproducibility of this sample
            torch.manual_seed(config['seed'])
            random.seed(config['seed'])
            
            # Prepare tensors
            style_tensor = torch.tensor([config['genre_idx']], dtype=torch.long, device=device)
            bar_tensor = torch.tensor([config['num_bars']], dtype=torch.long, device=device)
            
            # Sample from latent space
            z = torch.randn(1, latent_dim, device=device)
            
            # Generate tokens
            tokens = generate_constrained(
                model, tokenizer, z, style_tensor, bar_tensor, device,
                max_len=max_len,
                temperature=config['temperature'],
                top_k=config['top_k'],
                top_p=config['top_p']
            )
            
            # Decode to MIDI
            score = tokenizer.decode([tokens])
            
            # Create filename
            filename = f"{config['genre']}_{config['sample_id']:02d}_bars{config['num_bars']}.mid"
            output_path = os.path.join(output_dir, filename)
            
            # Save MIDI
            score.dump_midi(output_path)
            
            successful += 1
            
            # Log details for first few and any interesting ones
            if config['sample_id'] <= 5 or config['num_bars'] >= 16:
                total_notes = sum(len(t.notes) for t in score.tracks)
                tqdm.write(f"  ✓ {filename}: {total_notes} notes, temp={config['temperature']}, "
                          f"top_k={config['top_k']}, top_p={config['top_p']}")
            
        except Exception as e:
            failed += 1
            tqdm.write(f"  ✗ Failed {config['genre']}_{config['sample_id']:02d}: {str(e)[:50]}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Generation Complete!")
    print(f"{'='*70}")
    print(f"Successful: {successful}/{num_samples}")
    print(f"Failed: {failed}/{num_samples}")
    print(f"Output directory: {output_dir}")
    print(f"\nGenre distribution:")
    
    # Count samples per genre
    genre_counts = {}
    for config in generation_configs:
        genre = config['genre']
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    for genre, count in sorted(genre_counts.items()):
        print(f"  {genre:15s}: {count} samples")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Batch generate drum samples with DrumVAE v2")
    parser.add_argument("--model_path", type=str, default="checkpoints_v2/best_model.pt",
                       help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="samples/batch_generation",
                       help="Directory to save generated samples")
    parser.add_argument("--num_samples", type=int, default=40,
                       help="Total number of samples to generate")
    parser.add_argument("--max_len", type=int, default=1024,
                       help="Maximum sequence length for generation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set global seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    generate_batch_samples(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_len=args.max_len
    )


if __name__ == "__main__":
    main()
