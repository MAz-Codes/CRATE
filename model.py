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
                 num_decoder_layers=6, num_conductor_layers=4, dim_feedforward=2048, 
                 dropout=0.1, latent_dim=256, max_seq_len=1024, num_styles=12, 
                 max_bars=16):
        super(MusicTransformerVAE, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.max_bars = max_bars

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.style_embedding = nn.Embedding(num_styles, d_model)
        self.bar_embedding = nn.Embedding(33, d_model)  # Bar count conditioning (keep 33 for compatibility)
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

        # --- HIERARCHICAL CONDUCTOR ---
        # Transforms Z into Bar Embeddings
        self.fc_z_to_conductor = nn.Linear(latent_dim, d_model)
        
        # Learned queries for each bar position [max_bars, 1, d_model]
        self.bar_queries = nn.Parameter(torch.randn(max_bars, 1, d_model))
        
        conductor_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.conductor = nn.TransformerEncoder(conductor_layers, num_conductor_layers)

        # Decoder
        decoder_layers = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers, num_decoder_layers)

        # Output projection
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Length prediction head
        self.length_predictor = nn.Sequential(
            nn.Linear(latent_dim + d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

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

        # Take the state of the [CLS] token
        cls_output = output[0]  # [batch_size, d_model]

        mu = self.fc_mu(cls_output)
        logvar = self.fc_var(cls_output)

        return mu, logvar

    def decode(self, z, tgt, style_id=None, bar_id=None, tgt_key_padding_mask=None, tgt_mask=None, use_latent_dropout=False):
        """
        Decode from latent z to output sequence using Hierarchical Conductor.
        """
        # Latent dropout
        if use_latent_dropout and self.training:
            dropout_mask = torch.bernoulli(torch.full((z.size(0), 1), 0.9, device=z.device))
            z = z * dropout_mask
        
        batch_size = z.size(0)
        
        # --- CONDUCTOR PASS ---
        # Project Z to d_model
        z_emb = self.fc_z_to_conductor(z).unsqueeze(0) # [1, batch, d_model]
        
        # Expand queries
        queries = self.bar_queries.expand(-1, batch_size, -1) # [max_bars, batch, d_model]
        
        # Add Z to queries (Conditioning)
        conductor_input = queries + z_emb
        
        # Add Style to Conductor
        if style_id is not None:
            style_emb = self.style_embedding(style_id).unsqueeze(0)
            conductor_input = conductor_input + style_emb
            
        # Run Conductor
        # Output: [max_bars, batch, d_model]
        bar_embeddings = self.conductor(conductor_input)
        
        # Create Mask for Conductor Output (if bar_id/num_bars provided)
        memory_key_padding_mask = None
        if bar_id is not None:
            # bar_id is num_bars [batch]
            # mask: [batch, max_bars] (True = masked)
            # We mask bars > num_bars
            # bar_id is 1-based count.
            # indices are 0 to max_bars-1
            # if num_bars=4, indices 0,1,2,3 are valid. 4+ masked.
            mask = torch.arange(self.max_bars, device=z.device).expand(batch_size, -1) >= bar_id.unsqueeze(1)
            memory_key_padding_mask = mask

        # --- DECODER PASS ---
        # Embed target tokens
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Add style embedding to decoder too (optional, but helps)
        if style_id is not None:
            style_emb = self.style_embedding(style_id)
            tgt_emb = tgt_emb + style_emb.unsqueeze(0)
        
        # Add bar count embedding (global context)
        if bar_id is not None:
            bar_count_emb = self.bar_embedding(bar_id)
            tgt_emb = tgt_emb + bar_count_emb.unsqueeze(0)
        
        tgt_emb = self.pos_encoder(tgt_emb)

        # Decode
        # Memory is bar_embeddings
        output = self.transformer_decoder(
            tgt_emb, bar_embeddings,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # Project to vocabulary
        logits = self.output_layer(output)
        return logits

    def forward(self, src, tgt, style_id=None, bar_id=None, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        # Encode
        mu, logvar = self.encode(
            src, src_key_padding_mask=src_key_padding_mask)

        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Predict length
        if bar_id is not None:
            bar_emb = self.bar_embedding(bar_id)
            length_input = torch.cat([z, bar_emb], dim=1)
            predicted_length = self.length_predictor(length_input).squeeze(-1)
        else:
            predicted_length = None

        # Decode
        logits = self.decode(
            z, tgt, style_id=style_id, bar_id=bar_id,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
            use_latent_dropout=True
        )

        return logits, mu, logvar, predicted_length


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask
