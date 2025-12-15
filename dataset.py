import os
import torch
from torch.utils.data import Dataset
from miditok import REMI, TokenizerConfig
from symusic import Score
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import tempfile
import json

# Define supported styles for mapping
# Define supported styles from Groove MIDI Dataset
STYLES = ['afrobeat', 'afrocuban', 'blues', 'country', 'dance', 'funk', 'gospel', 
          'highlife', 'hiphop', 'jazz', 'latin', 'middleeastern', 'neworleans', 
          'pop', 'punk', 'reggae', 'rock', 'soul']
STYLE_TO_IDX = {style: i for i, style in enumerate(STYLES)}
UNKNOWN_STYLE_IDX = len(STYLES)

class GrooveMidiDataset(Dataset):
    def __init__(self, split='train', max_seq_len=1024, max_examples=None, use_bar_filter=False):
        self.max_seq_len = max_seq_len
        self.split = split
        
        # Load bar counts (optional)
        # Bar counts are not critical for model functionality
        bar_counts_file = 'bar_counts_filtered.json' if use_bar_filter else 'bar_counts.json'
        bar_counts_path = os.path.join(os.path.dirname(__file__), bar_counts_file)
        
        self.bar_counts = {}
        if os.path.exists(bar_counts_path):
            try:
                with open(bar_counts_path, 'r') as f:
                    self.bar_counts = json.load(f)
                print(f"Loaded bar counts from {bar_counts_file}")
            except Exception as e:
                print(f"Note: Could not load bar counts ({e}). Using default value of 8 bars.")

        # Initialize Tokenizer (miditok 3.x)
        # Optimized for Drums: No chords, no programs (single instrument implied)
        config = TokenizerConfig(
            num_velocities=16,
            use_chords=False,
            use_programs=False, # Drums don't need program changes for this VAE
            use_rests=True,
            use_tempos=True,
            use_time_signatures=True
        )
        self.tokenizer = REMI(config)
        
        print(f"Loading Groove MIDI Dataset (split: {split})...")
        data_dir = os.path.join(os.path.dirname(__file__), 'dataset')
        
        # Load the dataset
        ds, info = tfds.load(
            name="groove/full-midionly",
            split=split,
            with_info=True,
            try_gcs=True,
            data_dir=data_dir
        )
        
        self.style_primary_feature = info.features['style']['primary']
        self.examples = []
        
        # Pre-process loop to tokenize and chunk
        print(f"Processing and chunking {split} data...")
        
        # Get Bar token ID for counting bars in chunks
        try:
            bar_token_id = self.tokenizer["Bar_None"]
        except:
            bar_token_id = -1 # Fallback

        for example in tqdm(ds, desc=f"Loading GMD {split}"):
            try:
                example_id = example['id'].numpy().decode('utf-8')
                
                if use_bar_filter and example_id not in self.bar_counts:
                    continue
                
                # Get Style
                style_int = example['style']['primary'].numpy()
                style_name = self.style_primary_feature.int2str(style_int)
                style_id = STYLE_TO_IDX.get(style_name.lower(), UNKNOWN_STYLE_IDX)
                
                # Get Metadata
                bpm = int(example['bpm'].numpy())
                # Default to 8 bars if not in bar_counts (safe fallback)
                total_bars = self.bar_counts.get(example_id, 8)

                # Load MIDI directly from bytes
                midi_bytes = example['midi'].numpy()
                score = Score.from_midi(midi_bytes)

                # Tokenize
                tokens = self.tokenizer.encode(score)
                
                # Extract IDs
                token_ids = []
                if isinstance(tokens, list) and len(tokens) > 0:
                    if hasattr(tokens[0], 'ids'):
                        token_ids = tokens[0].ids
                    elif isinstance(tokens[0], int):
                        token_ids = tokens
                elif hasattr(tokens, 'ids'):
                    token_ids = tokens.ids
                
                if len(token_ids) == 0:
                    continue

                # CHUNKING STRATEGY
                # Split long sequences into chunks of max_seq_len with some overlap or just simple tiling
                # For VAEs, discrete chunks are usually fine.
                
                chunk_size = self.max_seq_len
                
                # Calculate average tokens per bar for this file
                avg_tokens_per_bar = len(token_ids) / max(1, total_bars)

                # If sequence is short enough, just use it
                if len(token_ids) <= chunk_size:
                    # Recalculate bars for short sequences too (sometimes metadata is wrong)
                    actual_bars = total_bars
                    if bar_token_id != -1:
                        count = token_ids.count(bar_token_id)
                        if count > 0:
                            actual_bars = count
                        else:
                            # Estimate
                            actual_bars = max(1, int(len(token_ids) / avg_tokens_per_bar))

                    self.examples.append({
                        'input_ids': torch.tensor(token_ids, dtype=torch.long),
                        'style_id': style_id,
                        'num_bars': actual_bars,
                        'id': example_id
                    })
                else:
                    # Split into chunks
                    # We create multiple examples from one file
                    # We stride by chunk_size (no overlap) to maximize unique data
                    for i in range(0, len(token_ids), chunk_size):
                        chunk = token_ids[i : i + chunk_size]
                        
                        # Discard very short last chunks (e.g. < 50 tokens) to avoid noise
                        if len(chunk) < 50:
                            continue
                        
                        # Calculate actual bars in this chunk
                        chunk_bars = 0
                        if bar_token_id != -1:
                            chunk_bars = chunk.count(bar_token_id)
                        
                        # If no bar tokens found or count is suspicious, estimate using file average
                        if chunk_bars == 0:
                            chunk_bars = max(1, int(len(chunk) / avg_tokens_per_bar))

                        self.examples.append({
                            'input_ids': torch.tensor(chunk, dtype=torch.long),
                            'style_id': style_id,
                            'num_bars': chunk_bars,
                            'id': f"{example_id}_chunk_{i//chunk_size}"
                        })

            except Exception as e:
                # print(f"Error processing {example_id}: {e}")
                continue
            
        if max_examples:
            self.examples = self.examples[:max_examples]
            
        print(f"Processed {len(self.examples)} training sequences.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return (
            example['input_ids'], 
            torch.tensor(example['style_id'], dtype=torch.long), 
            torch.tensor(example['num_bars'], dtype=torch.long)
        )

# Keep the old class for valid reference but prevent its usage errors if dependencies are missing
class MidiDataset(Dataset):
    # ... (rest of old code essentially commented out or kept as legacy)
    # For now we just keep it as is, but specific functionality is replaced by GrooveMidiDataset
     # Class-level set to track corrupt files we've already warned about
    _warned_corrupt_files = set()
    
    def __init__(self, data_dir, split='train', max_seq_len=1024, file_limit=None):
        # Allow fallback instantiation
        self.data_dir = data_dir
        self.files = []
        self.tokenizer = None
        pass
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        raise NotImplementedError("MidiDataset is deprecated. Use GrooveMidiDataset.")



class DataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # Filter out empty tensors (first element of tuple is empty)
        batch = [item for item in batch if item[0].numel() > 0]
        if len(batch) == 0:
            return None, None, None, None


        # Unzip
        seqs, styles, num_bars = zip(*batch)

        # Pad sequences
        src_batch = torch.nn.utils.rnn.pad_sequence(
            list(seqs), batch_first=False, padding_value=self.pad_token_id)

        # Create masks
        src_key_padding_mask = (src_batch == self.pad_token_id).transpose(0, 1)

        # Stack styles and bar counts
        style_batch = torch.stack(styles)
        bar_batch = torch.stack(num_bars)

        return src_batch, src_key_padding_mask, style_batch, bar_batch
