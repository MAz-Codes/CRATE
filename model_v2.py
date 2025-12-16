"""
Improved Drum VAE Model (v2)
Key improvements for musical structure:
1. Token Type Embeddings - separate embeddings for Bar, Position, Pitch, Velocity, Duration
2. Relative Position Encoding - better for repeating patterns
3. Bar-Position Aware Decoder - knows where it is in the musical structure
4. Auxiliary Losses - bar prediction, position prediction for structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding that captures distances between tokens.
    Better for music as patterns repeat at regular intervals.
    """
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        
        # Standard sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Dynamically extend PE if sequence is longer than buffer
        if x.size(0) > self.pe.size(0):
            self._extend_pe(x.size(0), x.device)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    def _extend_pe(self, new_len, device):
        """Extend positional encoding buffer if needed."""
        pe = torch.zeros(new_len, self.d_model, device=device)
        position = torch.arange(0, new_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)


class TokenTypeEmbedding(nn.Module):
    """
    Learns separate embeddings for different token types in REMI:
    Bar, Position, Pitch/PitchDrum, Velocity, Duration, Tempo, TimeSig, Rest
    """
    def __init__(self, d_model, num_types=8):
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, d_model)
        # Token type IDs: 0=PAD, 1=Bar, 2=Position, 3=Pitch, 4=Velocity, 5=Duration, 6=Tempo, 7=TimeSig, 8=Rest
        
    def forward(self, token_types):
        return self.type_embedding(token_types)


class MusicalPositionEncoding(nn.Module):
    """
    Encodes position within bar (0-95 for 96 ticks per bar) separately from absolute position.
    This helps the model learn recurring patterns within bars.
    """
    def __init__(self, d_model, max_bar_position=96, max_bars=32):
        super().__init__()
        self.bar_position_emb = nn.Embedding(max_bar_position, d_model // 2)
        self.bar_number_emb = nn.Embedding(max_bars, d_model // 2)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, bar_positions, bar_numbers):
        """
        bar_positions: [seq_len, batch] - position within bar (0-95)
        bar_numbers: [seq_len, batch] - which bar (0-31)
        """
        pos_emb = self.bar_position_emb(bar_positions)
        bar_emb = self.bar_number_emb(bar_numbers)
        combined = torch.cat([pos_emb, bar_emb], dim=-1)
        return self.proj(combined)


class DrumVAE(nn.Module):
    """
    Improved VAE for drum pattern generation with better musical structure.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, num_conductor_layers=4, dim_feedforward=2048, 
                 dropout=0.1, latent_dim=256, max_seq_len=1024, num_styles=20, 
                 max_bars=32, num_token_types=10):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.max_bars = max_bars

        # === EMBEDDINGS ===
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.token_type_embedding = nn.Embedding(num_token_types, d_model)
        self.style_embedding = nn.Embedding(num_styles, d_model)
        self.bar_count_embedding = nn.Embedding(33, d_model)
        
        # Position encodings - use max_seq_len, will dynamically extend if needed during generation
        self.pos_encoder = RelativePositionalEncoding(d_model, dropout, max_len=max_seq_len)
        self.musical_pos_encoder = MusicalPositionEncoding(d_model, max_bar_position=96, max_bars=max_bars)

        # === ENCODER ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # [CLS] token for sequence-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # === VAE BOTTLENECK ===
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        
        # === HIERARCHICAL CONDUCTOR ===
        self.z_to_conductor = nn.Linear(latent_dim, d_model)
        self.bar_queries = nn.Parameter(torch.randn(max_bars, 1, d_model))
        self.conductor_proj = nn.Linear(d_model * 2, d_model)
        
        conductor_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.conductor = nn.TransformerEncoder(conductor_layer, num_conductor_layers)

        # === DECODER ===
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # === OUTPUT HEADS ===
        # Main token prediction
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Auxiliary heads for structure (multi-task learning)
        self.token_type_head = nn.Linear(d_model, num_token_types)  # Predict next token type
        self.bar_position_head = nn.Linear(d_model, 96)  # Predict position within bar
        
        # Length prediction
        self.length_predictor = nn.Sequential(
            nn.Linear(latent_dim + d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot for better training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, src, token_types=None, style_id=None, src_key_padding_mask=None, 
               bar_pos=None, bar_num=None):
        """
        Encode input sequence to latent representation.
        
        Args:
            src: [seq_len, batch] - token IDs
            token_types: [seq_len, batch] - token type IDs (optional)
            style_id: [batch] - style conditioning
            src_key_padding_mask: [batch, seq_len] - padding mask
            bar_pos: [seq_len, batch] - position within bar (0-95)
            bar_num: [seq_len, batch] - bar index (0-31)
        """
        batch_size = src.size(1)
        
        # Token embeddings
        x = self.token_embedding(src) * math.sqrt(self.d_model)
        
        # Add token type embeddings if provided
        if token_types is not None:
            x = x + self.token_type_embedding(token_types)
            
        # Add musical position encoding if provided
        if bar_pos is not None and bar_num is not None:
            x = x + self.musical_pos_encoder(bar_pos, bar_num)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(1, batch_size, -1)
        x = torch.cat([cls_tokens, x], dim=0)
        
        # Update padding mask for [CLS]
        if src_key_padding_mask is not None:
            cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=src.device)
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
        
        # Add style embedding
        if style_id is not None:
            style_emb = self.style_embedding(style_id).unsqueeze(0)
            x = x + style_emb
        
        # Encode
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Extract [CLS] representation
        cls_output = encoded[0]  # [batch, d_model]
        
        mu = self.fc_mu(cls_output)
        logvar = self.fc_logvar(cls_output)
        
        return mu, logvar

    def decode(self, z, tgt, token_types=None, style_id=None, bar_id=None, 
               tgt_key_padding_mask=None, tgt_mask=None, use_latent_dropout=False,
               bar_pos=None, bar_num=None):
        """
        Decode from latent z to output sequence.
        """
        batch_size = z.size(0)
        
        # Latent dropout (for training)
        if use_latent_dropout and self.training:
            dropout_mask = torch.bernoulli(torch.full_like(z, 0.5))
            z = z * dropout_mask * 2.0
        
        # === CONDUCTOR ===
        z_emb = self.z_to_conductor(z).unsqueeze(0)  # [1, batch, d_model]
        queries = self.bar_queries.expand(-1, batch_size, -1)
        z_expanded = z_emb.expand(self.max_bars, -1, -1)
        
        conductor_input = torch.cat([queries, z_expanded], dim=-1)
        conductor_input = self.conductor_proj(conductor_input)
        
        if style_id is not None:
            style_emb = self.style_embedding(style_id).unsqueeze(0)
            conductor_input = conductor_input + style_emb
            
        bar_embeddings = self.conductor(conductor_input)
        
        # Memory mask for bar count
        memory_mask = None
        if bar_id is not None:
            memory_mask = torch.arange(self.max_bars, device=z.device).expand(batch_size, -1) >= bar_id.unsqueeze(1)
        
        # === DECODER ===
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
        
        # Add token type embeddings
        if token_types is not None:
            tgt_emb = tgt_emb + self.token_type_embedding(token_types)
            
        # Add musical position encoding if provided
        if bar_pos is not None and bar_num is not None:
            tgt_emb = tgt_emb + self.musical_pos_encoder(bar_pos, bar_num)
        
        # Add latent to all positions (strong conditioning)
        tgt_emb = tgt_emb + z_emb
        
        # Add style
        if style_id is not None:
            tgt_emb = tgt_emb + self.style_embedding(style_id).unsqueeze(0)
        
        # Add bar count
        if bar_id is not None:
            tgt_emb = tgt_emb + self.bar_count_embedding(bar_id).unsqueeze(0)
        
        # Positional encoding
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Decode
        decoded = self.decoder(
            tgt_emb, bar_embeddings,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_mask
        )
        
        # Output projections
        logits = self.output_proj(decoded)
        type_logits = self.token_type_head(decoded)
        position_logits = self.bar_position_head(decoded)
        
        return logits, type_logits, position_logits

    def forward(self, src, tgt, src_token_types=None, tgt_token_types=None, 
                style_id=None, bar_id=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None,
                src_bar_pos=None, src_bar_num=None, tgt_bar_pos=None, tgt_bar_num=None):
        """
        Full forward pass.
        
        Args:
            src: [seq_len, batch] - encoder input tokens
            tgt: [seq_len-1, batch] - decoder input tokens
            src_token_types: [seq_len, batch] - token types for encoder
            tgt_token_types: [seq_len-1, batch] - token types for decoder
        """
        # Encode (uses src and src_token_types)
        mu, logvar = self.encode(src, src_token_types, style_id, src_key_padding_mask,
                                 bar_pos=src_bar_pos, bar_num=src_bar_num)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Length prediction
        if bar_id is not None:
            bar_emb = self.bar_count_embedding(bar_id)
            length_input = torch.cat([z, bar_emb], dim=1)
            predicted_length = self.length_predictor(length_input).squeeze(-1)
        else:
            predicted_length = None
        
        # Decode (uses tgt and tgt_token_types)
        logits, type_logits, position_logits = self.decode(
            z, tgt, tgt_token_types, style_id, bar_id,
            tgt_key_padding_mask, tgt_mask,
            use_latent_dropout=True,
            bar_pos=tgt_bar_pos, bar_num=tgt_bar_num
        )
        
        return {
            'logits': logits,
            'type_logits': type_logits,
            'position_logits': position_logits,
            'mu': mu,
            'logvar': logvar,
            'predicted_length': predicted_length,
            'z': z
        }


def generate_square_subsequent_mask(sz, device=None):
    """Generate causal mask for autoregressive decoding."""
    if device is None:
        device = torch.device('cpu')
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


# Token type mapping for REMI tokenizer
TOKEN_TYPE_MAP = {
    'PAD': 0,
    'Bar': 1,
    'Position': 2,
    'Pitch': 3,
    'PitchDrum': 3,  # Same as Pitch
    'Velocity': 4,
    'Duration': 5,
    'Tempo': 6,
    'TimeSig': 7,
    'Rest': 8,
    'BOS': 9,
    'EOS': 9,
}


def get_token_types(token_ids, tokenizer):
    """
    Convert token IDs to token type IDs for the model.
    
    Args:
        token_ids: [seq_len, batch] tensor of token IDs
        tokenizer: miditok tokenizer with vocab
        
    Returns:
        [seq_len, batch] tensor of token type IDs
    """
    # Build reverse vocab
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    
    device = token_ids.device
    seq_len, batch_size = token_ids.shape
    type_ids = torch.zeros_like(token_ids)
    
    for b in range(batch_size):
        for s in range(seq_len):
            tid = token_ids[s, b].item()
            token_str = str(id_to_token.get(tid, 'PAD'))
            
            # Determine type from token string
            type_id = 0  # Default to PAD
            for prefix, tid_val in TOKEN_TYPE_MAP.items():
                if token_str.startswith(prefix):
                    type_id = tid_val
                    break
            
            type_ids[s, b] = type_id
    
    return type_ids


# Keep backward compatibility with old model name
MusicTransformerVAE = DrumVAE
