# Training and Architecture

## Model Architecture
The core model is a **MusicTransformerVAE**, combining:
1.  **Transformer Encoder**: Compresses the input drum sequence into a latent representation.
2.  **Variational Bottleneck (VAE)**: Maps the encoded sequence to a probabilistic latent space ($z$), enabling generation of diverse variations.
3.  **Transformer Decoder**: Reconstructs the sequence from the latent vector $z$ and conditioning information (Style, Tempo).

### Conditioning
The model uses **Style Embeddings** and **Bar Count Embeddings** to guide generation:
- **Style ID** (e.g., "Funk"): Embedded and added to decoder input, forcing the model to adopt genre-specific rhythm and velocity characteristics
- **Bar Count** (2-32 bars): Embedded and added to decoder input, teaching the model to generate sequences of specific lengths

## Training Process
- **Objective**: Optimize the Evidence Lower Bound (ELBO), consisting of:
    - **Reconstruction Loss**: Cross-Entropy loss between generated and target tokens.
    - **KL Divergence**: Regularizes the latent space to be close to a standard normal distribution.
- **Optimizer**: AdamW.
- **Scheduler**: Cosine Annealing.

### Hyperparameters
- `d_model`: 512 (Embedding dimension).
- `nhead`: 8 (Attention heads).
- `num_layers`: 6 (Encoder/Decoder layers).
- `seq_len`: 512 (Max sequence length in tokens).
- `latent_dim`: 256 (Size of latent vector).

## Generating
During inference (`generate.py`), we:
1.  Sample a random latent vector $z \sim \mathcal{N}(0, I)$.
2.  Provide a Style ID and Bar Count.
3.  Decode auto-regressively to produce a new MIDI file of the specified length.
