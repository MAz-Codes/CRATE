import os
import torch
from torch.utils.data import Dataset
from miditok import REMI, TokenizerConfig
from symusic import Score, Note
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
# Prevent TensorFlow from grabbing all GPU memory
tf.config.set_visible_devices([], 'GPU')

import tempfile
import json
import random
import copy

# Define supported styles for mapping
STYLES = ['afrobeat', 'afrocuban', 'blues', 'country', 'dance', 'funk', 'gospel', 
          'highlife', 'hiphop', 'jazz', 'latin', 'middleeastern', 'neworleans', 
          'pop', 'punk', 'reggae', 'rock', 'soul']
STYLE_TO_IDX = {style: i for i, style in enumerate(STYLES)}
UNKNOWN_STYLE_IDX = len(STYLES)

class GrooveMidiDataset(Dataset):
    def __init__(self, split='train', max_seq_len=1024, max_examples=None, use_bar_filter=False, augment=False, chunk_bars=8):
        self.max_seq_len = max_seq_len
        self.split = split
        self.augment = augment
        self.chunk_bars = chunk_bars
        
        # Load bar counts (optional)
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

        # Initialize Tokenizer (REMI - Revamped MIDI)
        # Switched back to REMI for Hierarchical Model: Explicit 'Bar' tokens are crucial 
        # for the Decoder to align with the Conductor's bar-level embeddings.
        config = TokenizerConfig(
            num_velocities=16,
            use_chords=False,
            use_programs=False,
            use_rests=True,
            use_tempos=True,
            use_time_signatures=True,
            use_sustain_pedals=False,
            use_pitch_bends=False
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
        
        print(f"Indexing {split} data...")
        
        for example in tqdm(ds, desc=f"Indexing GMD {split}"):
            try:
                example_id = example['id'].numpy().decode('utf-8')
                
                if use_bar_filter and example_id not in self.bar_counts:
                    continue
                
                # Get Style
                style_int = example['style']['primary'].numpy()
                style_name = self.style_primary_feature.int2str(style_int)
                style_id = STYLE_TO_IDX.get(style_name.lower(), UNKNOWN_STYLE_IDX)
                
                # Load MIDI bytes
                midi_bytes = example['midi'].numpy()
                
                # Parse Score to determine chunks
                score = Score.from_midi(midi_bytes)
                
                # Calculate bars
                end_tick = score.end()
                if end_tick == 0:
                    continue
                    
                tpq = score.ticks_per_quarter
                ticks_per_bar = tpq * 4 
                
                if len(score.time_signatures) > 0:
                    ts = score.time_signatures[0]
                    ticks_per_bar = int(tpq * 4 * ts.numerator / ts.denominator)
                
                chunk_ticks = ticks_per_bar * self.chunk_bars
                
                # Create chunks
                for i in range(0, end_tick, chunk_ticks):
                    start = i
                    end = min(i + chunk_ticks, end_tick)
                    
                    # Skip short last chunks
                    if (end - start) < ticks_per_bar:
                        continue
                        
                    self.examples.append({
                        'midi_bytes': midi_bytes, 
                        'start_tick': start,
                        'end_tick': end,
                        'style_id': style_id,
                        'num_bars': self.chunk_bars,
                        'id': f"{example_id}_chunk_{i//chunk_ticks}"
                    })

            except Exception as e:
                continue
            
        if max_examples:
            self.examples = self.examples[:max_examples]
            
        print(f"Indexed {len(self.examples)} sequences.")

    def __len__(self):
        return len(self.examples)

    def apply_augmentation(self, score):
        # 1. Velocity Noise
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            for track in score.tracks:
                for note in track.notes:
                    new_vel = int(note.velocity * factor)
                    note.velocity = max(1, min(127, new_vel))
        
        # 2. Timing Humanization (Micro-timing)
        if random.random() < 0.3:
            jitter = int(score.ticks_per_quarter * 0.05)
            for track in score.tracks:
                for note in track.notes:
                    shift = random.randint(-jitter, jitter)
                    note.time = max(0, note.time + shift)
                    
        # 3. Mirroring (Drum Kit Swaps)
        if random.random() < 0.3:
            swap_map = {
                50: 45, 45: 50, 
                48: 43, 43: 48,
                49: 57, 57: 49,
            }
            for track in score.tracks:
                for note in track.notes:
                    if note.pitch in swap_map:
                        note.pitch = swap_map[note.pitch]

    def __getitem__(self, idx):
        meta = self.examples[idx]
        
        # Load Score
        score = Score.from_midi(meta['midi_bytes'])
        
        # Clip to chunk
        clipped_score = score.clip(meta['start_tick'], meta['end_tick'])
        
        # Shift to start at 0
        shifted_score = clipped_score.shift_time(-meta['start_tick'])
        
        # Augment
        if self.augment:
            self.apply_augmentation(shifted_score)
            
        # Tokenize
        tokens = self.tokenizer.encode(shifted_score)
        
        # Extract IDs
        token_ids = []
        if isinstance(tokens, list) and len(tokens) > 0:
            if hasattr(tokens[0], 'ids'):
                token_ids = tokens[0].ids
            elif isinstance(tokens[0], int):
                token_ids = tokens
        elif hasattr(tokens, 'ids'):
            token_ids = tokens.ids
            
        # Pad or Truncate
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            # Pad
            try:
                pad_id = self.tokenizer["PAD_None"]
            except:
                pad_id = 0 # Fallback
                
            token_ids += [pad_id] * (self.max_seq_len - len(token_ids))
            
        return (
            torch.tensor(token_ids, dtype=torch.long), 
            torch.tensor(meta['style_id'], dtype=torch.long), 
            torch.tensor(meta['num_bars'], dtype=torch.long)
        )

class DataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        batch = [item for item in batch if item[0] is not None]
        if len(batch) == 0:
            return None, None, None, None

        seqs, styles, num_bars = zip(*batch)

        src_batch = torch.stack(seqs).transpose(0, 1) # [seq_len, batch_size]
        src_key_padding_mask = (src_batch == self.pad_token_id).transpose(0, 1) # [batch_size, seq_len]

        style_batch = torch.stack(styles)
        bar_batch = torch.stack(num_bars)

        return src_batch, src_key_padding_mask, style_batch, bar_batch
