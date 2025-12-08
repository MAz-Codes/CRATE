# Dataset: Groove MIDI Dataset (GMD)

The project now utilizes the **Groove MIDI Dataset**, a collection of 13.6 hours of aligned MIDI and audio drum performances captured on a Roland TD-11 electronic drum kit.

## Source
- **Name**: `groove/full-midionly` (via TensorFlow Datasets)
- **Creators**: Magenta (Google Brain)
- **Content**: Human-performed drum grooves.

## Structure & Usage
We use the `full-midionly` split, which contains:
- **MIDI**: The beat patterns.
- **Metadata**: Style, tempo, and drummer ID.

### Integration
The `GrooveMidiDataset` class in `dataset.py` handles loading:
1.  **Download**: Automatically handled by TFDS into the `dataset/` directory.
2.  **Processing**:
    - Raw MIDI bytes are extracted.
    - Converted to `symusic.Score` objects.
    - Tokenized using `miditok.REMI` with drum-optimized settings.

### Styles
The dataset maps to 18 specific genres, ensuring precise conditioning:
- afrobeat, afrocuban, blues, country, dance, funk, gospel, highlife, hiphop, jazz, latin, middleeastern, neworleans, pop, punk, reggae, rock, soul.

## Tokenization
We use the **REMI** tokenization strategy with:
- `use_chords=False` (Drums are polyphonic but treated as single instrument tracks here).
- `use_rests=True` (Crucial for rhythm).
- `use_tempos=True` (Preserves performance tempo).
- `num_velocities=16` (Quantized dynamics).
