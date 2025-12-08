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

# Define supported styles for mapping
# Define supported styles from Groove MIDI Dataset
STYLES = ['afrobeat', 'afrocuban', 'blues', 'country', 'dance', 'funk', 'gospel', 
          'highlife', 'hiphop', 'jazz', 'latin', 'middleeastern', 'neworleans', 
          'pop', 'punk', 'reggae', 'rock', 'soul']
STYLE_TO_IDX = {style: i for i, style in enumerate(STYLES)}
UNKNOWN_STYLE_IDX = len(STYLES)

class GrooveMidiDataset(Dataset):
    def __init__(self, split='train', max_seq_len=1024, max_examples=None):
        self.max_seq_len = max_seq_len
        self.split = split
        
        print(f"Loading Groove MIDI Dataset (split: {split})...")
        # Load the full dataset (MIDI only)
        # tfds.load returns a tf.data.Dataset
        data_dir = os.path.join(os.path.dirname(__file__), 'dataset')
        ds, info = tfds.load(
            name="groove/full-midionly",
            split=split,
            with_info=True,
            try_gcs=True,
            data_dir=data_dir
        )
        
        # Get style mapping from info
        self.style_primary_feature = info.features['style']['primary']
        
        # Convert to list for random access (GMD is small ~1150 samples total)
        # This allows us to use PyTorch's generic DataLoader shuffling easily
        self.examples = []
        for example in tqdm(ds, desc=f"Loading GMD {split}"):
            # Style is returned as int (ClassLabel)
            style_int = example['style']['primary'].numpy()
            style_name = self.style_primary_feature.int2str(style_int)
            
            self.examples.append({
                'midi': example['midi'].numpy(),
                'style': style_name,
                'bpm': int(example['bpm'].numpy()),
                'id': example['id'].numpy().decode('utf-8')
            })
            
        if max_examples:
            self.examples = self.examples[:max_examples]
            
        print(f"Loaded {len(self.examples)} examples.")

        # Initialize Tokenizer (miditok 3.x)
        # Note: Groove MIDI acts as a drum track (Channel 10 usually)
        config = TokenizerConfig(
            num_velocities=16,
            use_chords=False, # Drums usually don't need chord tokens in the same way, but REMI uses them. 
                              # We'll keep settings close to original but disable unnecessary ones if possible.
                              # Actually, keeping it consistent with the model's expectation is safer for now.
            use_programs=True, 
            use_rests=True, # Drums have lots of silence
            use_tempos=True,
            use_time_signatures=True
        )
        self.tokenizer = REMI(config)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        midi_bytes = example['midi']
        style_name = example['style']
        
        # Map style
        # GMD styles might differ, simple mapping for now
        # GMD styles: 'rock', 'funk', 'jazz', etc.
        style_id = STYLE_TO_IDX.get(style_name.lower(), UNKNOWN_STYLE_IDX)
        
        try:
            # miditok/symusic usually requires a file path. 
            # We'll write to a temp file.
            with tempfile.NamedTemporaryFile(suffix='.mid',  delete=True) as tmp:
                tmp.write(midi_bytes)
                tmp.flush()
                
                # Load MIDI
                midi = Score(tmp.name)

            # Tokenize
            tokens = self.tokenizer.encode(midi)
            
            token_ids = []
            if isinstance(tokens, list): # Handle legacy/complex returns
                 if len(tokens) > 0:
                    if hasattr(tokens[0], 'ids'):
                         token_ids = tokens[0].ids
                    elif isinstance(tokens[0], int):
                        token_ids = tokens
            elif hasattr(tokens, 'ids'):
                 token_ids = tokens.ids

            if len(token_ids) == 0:
                # print(f"WARNING: Zero tokens for {example['id']}")
                return torch.tensor([], dtype=torch.long), torch.tensor(style_id, dtype=torch.long)

            # Truncate
            if len(token_ids) > self.max_seq_len:
                token_ids = token_ids[:self.max_seq_len]

            return torch.tensor(token_ids, dtype=torch.long), torch.tensor(style_id, dtype=torch.long)

        except Exception as e:
            print(f"ERROR processing {example['id']}: {e}")
            return torch.tensor([], dtype=torch.long), torch.tensor(style_id, dtype=torch.long)

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
            return None, None, None


        # Unzip
        seqs, styles = zip(*batch)

        # Pad sequences
        src_batch = torch.nn.utils.rnn.pad_sequence(
            list(seqs), batch_first=False, padding_value=self.pad_token_id)

        # Create masks
        src_key_padding_mask = (src_batch == self.pad_token_id).transpose(0, 1)

        # Stack styles
        style_batch = torch.stack(styles)

        return src_batch, src_key_padding_mask, style_batch
