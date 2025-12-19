# CRATE (Customizeable Rhythm Autoencoder with Transformer Encoding)

A transformer-based Variational Autoencoder that generates multi-bar drum performances with explicit control over musical style, bar lengths and structure. It combines a hierarchical Conductor network with grammar-constrained decoding to produce MIDI drum patterns.

**This is a work-in-progress and will be updated regularly**

## Key Features

*   **Style & Structure Control**: Generate beats in specific styles (Rock, Funk, Jazz, etc.) with precise bar lengths.
*   **Advanced Architecture**: Uses Relative Positional Encoding and Bar-Position Aware Decoding for coherent musical structure.
*   **Robust Training**: Implements auxiliary losses for bar and position prediction to ensure rhythmic accuracy.

## Quick Start

The project needs PyTorch. Install the appropriate version (inside the venv) depending on your CUDA version / MPS availablity.

### 1. Setup
```bash
# Create environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train
Train the model on the Groove MIDI Dataset:
```bash
python train.py --epochs 100 --batch_size 16
```

### 3. Generate
Generate new drum samples:
```bash
# Generate a 4-bar Funk beat
python generate.py --style funk --num_bars 4

# Generate a batch of diverse samples
python generate_batch.py
```

## Project Structure

*   `model.py`: The CrateVAE model architecture.
*   `train.py`: Training script with auxiliary losses.
*   `generate.py`: Generation script with constrained decoding.
*   `dataset.py`: Data loading and processing using `miditok`.
*   `checkpoints/`: Saved model weights and logs.
*   `samples/`: Generated MIDI files.


