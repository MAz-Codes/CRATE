import torch
from torch.utils.data import DataLoader
from dataset import GrooveMidiDataset, DataCollator, STYLES, STYLE_TO_IDX
from model import MusicTransformerVAE, generate_square_subsequent_mask
import argparse
import os
from tqdm import tqdm
import json
import numpy as np


def evaluate(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load test dataset
    test_dataset = GrooveMidiDataset(
        split='test', max_seq_len=args.seq_len)

    # Get PAD token ID - use 0 as default for miditok 3.x
    try:
        pad_token = test_dataset.tokenizer["PAD"]
        if isinstance(pad_token, list):
            pad_token_id = pad_token[0]
        else:
            pad_token_id = int(pad_token)
    except:
        pad_token_id = 0
        print("Note: Using token ID 0 as PAD (miditok 3.x default)")

    data_collator = DataCollator(pad_token_id)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=data_collator, num_workers=4)

    print(f"Loaded {len(test_dataset)} test samples")

    # Load checkpoint
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location=device)
    vocab_size = checkpoint.get('vocab_size')
    num_styles = checkpoint.get('num_styles', len(STYLES) + 1)
    model_args = checkpoint['args']

    if vocab_size is None:
        raise ValueError("Checkpoint missing 'vocab_size'. Cannot load model.")

    # Verify dataset tokenizer matches checkpoint
    dataset_vocab_size = len(test_dataset.tokenizer)
    if dataset_vocab_size != vocab_size:
        print(
            f"WARNING: Dataset tokenizer vocab size ({dataset_vocab_size}) differs from checkpoint ({vocab_size})")
        print(f"Using checkpoint vocab_size: {vocab_size}")

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Vocab size: {vocab_size}, Num styles: {num_styles}")

    # Initialize model
    model = MusicTransformerVAE(
        vocab_size=vocab_size,
        d_model=model_args['d_model'],
        nhead=model_args['nhead'],
        num_encoder_layers=model_args['num_layers'],
        num_decoder_layers=model_args['num_layers'],
        latent_dim=model_args['latent_dim'],
        max_seq_len=model_args.get('seq_len', args.seq_len),
        num_styles=num_styles
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Use PAD token ID from checkpoint if available
    pad_token_id = checkpoint.get('pad_token_id', 0)
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=pad_token_id, reduction='none')

    # Evaluation metrics
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_tokens = 0
    correct_tokens = 0

    # Style-specific metrics
    style_losses = {style: [] for style in STYLES}
    style_losses['unknown'] = []

    print("\nEvaluating on test set...")

    with torch.no_grad():
        for src, src_key_padding_mask, style_batch in tqdm(test_loader, desc="Evaluating"):
            if src is None:
                continue

            src = src.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            style_batch = style_batch.to(device)

            tgt = src
            dec_input = tgt[:-1]
            dec_target = tgt[1:]
            dec_padding_mask = src_key_padding_mask[:, :-1]

            # Create mask for decoder input length
            tgt_seq_len = dec_input.shape[0]
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)

            logits, mu, logvar = model(src, dec_input, style_id=style_batch,
                                       src_key_padding_mask=src_key_padding_mask,
                                       tgt_key_padding_mask=dec_padding_mask,
                                       tgt_mask=tgt_mask)

            # Reconstruction loss per token
            token_losses = criterion(
                logits.reshape(-1, vocab_size), dec_target.reshape(-1))
            recon_loss = token_losses.mean()

            # KL Divergence: average over batch and latent dimensions
            kl_loss = -0.5 * \
                torch.mean(
                    torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

            # Token accuracy (non-padded tokens)
            predictions = logits.argmax(dim=-1)
            non_pad_mask = (dec_target != pad_token_id)
            correct_tokens += ((predictions == dec_target)
                               & non_pad_mask).sum().item()
            total_tokens += non_pad_mask.sum().item()

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

            # Track per-style losses
            for i, style_id in enumerate(style_batch.cpu().numpy()):
                if style_id < len(STYLES):
                    style_name = STYLES[style_id]
                else:
                    style_name = 'unknown'
                style_losses[style_name].append(recon_loss.item())

    # Compute final metrics
    num_batches = len(test_loader)
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    perplexity = np.exp(avg_recon_loss)
    token_accuracy = (correct_tokens / total_tokens) * \
        100 if total_tokens > 0 else 0

    # Compute per-style statistics
    style_stats = {}
    for style, losses in style_losses.items():
        if len(losses) > 0:
            style_stats[style] = {
                'avg_loss': float(np.mean(losses)),
                'std_loss': float(np.std(losses)),
                'num_samples': len(losses)
            }

    # Print results
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

    # Save results to JSON
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
