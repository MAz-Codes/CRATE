import os
import torch
from torch.utils.data import Dataset
from miditok import REMI, TokenizerConfig
from symusic import Score
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import json

STYLES = ['afrobeat', 'afrocuban', 'blues', 'country', 'dance', 'funk', 'gospel', 
          'highlife', 'hiphop', 'jazz', 'latin', 'middleeastern', 'neworleans', 
          'pop', 'punk', 'reggae', 'rock', 'soul']
STYLE_TO_IDX = {style: i for i, style in enumerate(STYLES)}
UNKNOWN_STYLE_IDX = len(STYLES)

class GrooveMidiDataset(Dataset):
    def __init__(self, split='train', max_seq_len=1024, max_examples=None, use_bar_filter=False, augment=False, chunk_bars=8, beat_only=True):
        self.max_seq_len = max_seq_len
        self.split = split
        self.augment = augment
        self.chunk_bars = chunk_bars
        
        config = TokenizerConfig(
            num_velocities=16,
            use_chords=False,
            use_programs=False,
            use_rests=True,
            use_tempos=True,
            use_time_signatures=True,
            use_sustain_pedals=False,
            use_pitch_bends=False,
            special_tokens=["BOS", "EOS"]
        )
        self.tokenizer = REMI(config)
        
        print(f"Loading Groove MIDI Dataset from CSV (split: {split})...")
        
        csv_path = os.path.join(os.path.dirname(__file__), 'dataset', 'e-gmd-v1.0.0.csv')
        df = pd.read_csv(csv_path)
        
        if split == 'validation':
            df = df[df['split'] == 'validation']
        elif split == 'test':
            df = df[df['split'] == 'test']
        else:
            df = df[df['split'] == 'train']
        
        if beat_only:
            df = df[df['beat_type'] == 'beat']
            print(f"Filtering for beats only: {len(df)} samples")
            
        self.examples = []
        dataset_root = os.path.join(os.path.dirname(__file__), 'dataset')
        
        print(f"Indexing {split} data...")
        
        count = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Indexing GMD {split}"):
            if max_examples and count >= max_examples:
                break
                
            try:
                style_raw = str(row['style'])
                style_name = style_raw.split('/')[0].lower()
                style_id = STYLE_TO_IDX.get(style_name, UNKNOWN_STYLE_IDX)
                
                midi_rel_path = row['midi_filename']
                midi_path = os.path.join(dataset_root, midi_rel_path)
                
                if not os.path.exists(midi_path):
                    continue
                    
                score = Score(midi_path)
                
                
                end_tick = score.end()
                if end_tick == 0:
                    continue
                    
                tpq = score.ticks_per_quarter
                ticks_per_bar = tpq * 4
                
                if len(score.time_signatures) > 0:
                    ts = score.time_signatures[0]
                    ticks_per_bar = int(tpq * 4 * ts.numerator / ts.denominator)
                
                chunk_ticks = ticks_per_bar * self.chunk_bars
                
                for i in range(0, end_tick, chunk_ticks):
                    start = i
                    end = min(i + chunk_ticks, end_tick)
                    
                    if (end - start) < ticks_per_bar:
                        continue
                    
                    
                    self.examples.append({
                        'score': score,
                        'start_tick': start,
                        'end_tick': end,
                        'style_id': style_id,
                        'num_bars': self.chunk_bars,
                        'id': f"{row['id']}_chunk_{i//chunk_ticks}"
                    })
                    
                count += 1

            except Exception as e:
                continue
            
        if max_examples:
             self.examples = self.examples[:max_examples]
            
        print(f"Indexed {len(self.examples)} sequences from {count} files.")

    def __len__(self):
        return len(self.examples)

    def apply_augmentation(self, score):
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            for track in score.tracks:
                for note in track.notes:
                    new_vel = int(note.velocity * factor)
                    note.velocity = max(1, min(127, new_vel))
        
        if random.random() < 0.3:
            jitter = int(score.ticks_per_quarter * 0.05)
            for track in score.tracks:
                for note in track.notes:
                    shift = random.randint(-jitter, jitter)
                    note.time = max(0, note.time + shift)
                    
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
        
        clipped_score = meta['score'].clip(meta['start_tick'], meta['end_tick'])
        
        shifted_score = clipped_score.shift_time(-meta['start_tick'])
        
        if self.augment:
            self.apply_augmentation(shifted_score)
            
        tokens = self.tokenizer.encode(shifted_score)
        
        token_ids = []
        if isinstance(tokens, list) and len(tokens) > 0:
            if hasattr(tokens[0], 'ids'):
                token_ids = tokens[0].ids
            elif isinstance(tokens[0], int):
                token_ids = tokens
        elif hasattr(tokens, 'ids'):
            token_ids = tokens.ids
            
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            try:
                pad_id = self.tokenizer["PAD_None"]
            except:
                pad_id = 0
                
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

        src_batch = torch.stack(seqs).transpose(0, 1)
        src_key_padding_mask = (src_batch == self.pad_token_id).transpose(0, 1)

        style_batch = torch.stack(styles)
        bar_batch = torch.stack(num_bars)

        return src_batch, src_key_padding_mask, style_batch, bar_batch
