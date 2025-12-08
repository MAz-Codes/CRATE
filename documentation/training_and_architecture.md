# Training and Architecture

## Model Architecture
The core model is a **MusicTransformerVAE**, combining:
1.  **Transformer Encoder**: Compresses the input drum sequence into a latent representation.
2.  **Variational Bottleneck (VAE)**: Maps the encoded sequence to a probabilistic latent space ($z$), enabling generation of diverse variations.
3.  **Transformer Decoder**: Reconstructs the sequence from the latent vector $z$ and conditioning information (Style, Tempo).

### Conditioning
The model uses **Style Embeddings** to guide generation. A style ID (e.g., for "Funk") is embedded and added to the decoder's input, forcing the model to adopt the rhythm and velocity characteristics of that genre.

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
2.  Provide a Style ID.
3.  Decode auto-regressively to produce a new MIDI file.
