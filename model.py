import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  # type: ignore
        return self.dropout(x)


class MusicTransformerVAE(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, latent_dim=256, max_seq_len=1024, num_styles=12):
        super(MusicTransformerVAE, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.style_embedding = nn.Embedding(num_styles, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model, dropout, max_len=max_seq_len)

        # Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_encoder_layers)

        # Special [CLS] token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # VAE Bottleneck
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_var = nn.Linear(d_model, latent_dim)

        # Project latent back to d_model for decoder memory
        self.fc_z_to_memory = nn.Linear(latent_dim, d_model)

        # Decoder
        decoder_layers = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers, num_decoder_layers)

        # Output projection
        self.output_layer = nn.Linear(d_model, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, src, style_id=None, src_mask=None, src_key_padding_mask=None):
        # src: [seq_len, batch_size]
        batch_size = src.size(1)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(1, batch_size, -1)
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        # Concatenate CLS + Embeddings
        # Note: We need to adjust masks if we use them, but for simplicity in VAE,
        # we often just let it attend to everything.
        # If src_key_padding_mask is provided, we need to pad it for CLS token (False = not masked)
        # Clone mask to avoid in-place modification side effects
        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(
                (batch_size, 1), dtype=torch.bool, device=src.device)
            src_key_padding_mask = torch.cat(
                [cls_mask, src_key_padding_mask.clone()], dim=1)

        encoder_input = torch.cat([cls_tokens, src_emb], dim=0)

        # Add Style Embedding
        if style_id is not None:
            style_emb = self.style_embedding(
                style_id).unsqueeze(0)  # [1, batch, d_model]
            encoder_input = encoder_input + style_emb

        output = self.transformer_encoder(
            encoder_input, src_key_padding_mask=src_key_padding_mask)

        # Take the state of the [CLS] token (first token)
        cls_output = output[0]  # [batch_size, d_model]

        mu = self.fc_mu(cls_output)
        logvar = self.fc_var(cls_output)

        return mu, logvar

    def decode(self, tgt, z, style_id=None, tgt_mask=None, tgt_key_padding_mask=None):
        # tgt: [seq_len, batch_size]
        # z: [batch_size, latent_dim]

        # Project z to be the "memory" for the decoder
        # memory shape: [1, batch_size, d_model] (sequence length of 1)
        memory = self.fc_z_to_memory(z).unsqueeze(0)

        # Add Style Embedding to memory
        if style_id is not None:
            style_emb = self.style_embedding(style_id).unsqueeze(0)
            memory = memory + style_emb

        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        output = self.transformer_decoder(
            tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self.output_layer(output)
        return logits

    def forward(self, src, tgt, style_id=None, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        # src: [src_len, batch_size]
        # tgt: [tgt_len, batch_size] (usually src shifted by 1 for teacher forcing)

        mu, logvar = self.encode(
            src, style_id=style_id, src_key_padding_mask=src_key_padding_mask)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(tgt, z, style_id=style_id, tgt_mask=tgt_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask)

        return logits, mu, logvar


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask
