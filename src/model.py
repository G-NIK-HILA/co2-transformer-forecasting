"""
Custom Transformer components for CO2 time-series forecasting.

Everything here is written directly in PyTorch. No nn.TransformerEncoder,
nn.MultiheadAttention, or externally implemented Transformer models are used.
"""

import math

import torch
import torch.nn as nn


class InputProjection(nn.Module):
    """
    Project the raw feature vector at each time step into the
    Transformer embedding dimension.

    Input:
        (batch_size, sequence_length, n_features)

    Output:
        (batch_size, sequence_length, d_model)
    """

    def __init__(self, n_features, d_model):
        super().__init__()

        self.projection = nn.Linear(
            n_features,
            d_model,
        )

    def forward(self, x):
        return self.projection(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.

    Self-attention alone does not know the order of observations.
    This adds a deterministic representation of each step's
    position in the input window.

    The positional encoding has no trainable parameters.
    """

    def __init__(self, d_model, max_length=512):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(
                "d_model must be even for this positional encoding."
            )

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_length,
            d_model,
            dtype=torch.float32,
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # Shape: (1, max_length, d_model).
        # register_buffer keeps the positional encoding as part of
        # the model state without making it trainable.
        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(self, x):
        steps = x.size(1)

        if steps > self.pe.size(1):
            raise ValueError(
                "Input sequence is longer than max_length."
            )

        return x + self.pe[:, :steps, :]


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head scaled dot-product self-attention implemented directly
    with PyTorch tensor operations.

    Input:
        (batch, steps, d_model)

    Output:
        (batch, steps, d_model)

    No causal mask is required because every observation in the input
    window occurs before the forecast target.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.0,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Produce query, key, and value representations together.
        self.qkv = nn.Linear(
            d_model,
            3 * d_model,
        )

        # Combine the attention heads.
        self.out = nn.Linear(
            d_model,
            d_model,
        )

        self.attn_dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        return_attention=False,
    ):
        batch, steps, _ = x.shape

        # (B, L, d_model)
        # -> (B, L, 3*d_model)
        qkv = self.qkv(x)

        # -> (B, L, 3, heads, d_head)
        qkv = qkv.reshape(
            batch,
            steps,
            3,
            self.n_heads,
            self.d_head,
        )

        # -> (3, B, heads, L, d_head)
        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4,
        )

        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        # Scaled dot-product attention:
        #
        # Q K^T / sqrt(d_head)
        #
        # Shape:
        # (B, heads, L, L)
        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.d_head)

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        weights = self.attn_dropout(
            weights
        )

        # Weighted sum of values:
        # (B, heads, L, d_head)
        context = weights @ v

        # Merge the heads:
        # (B, heads, L, d_head)
        # -> (B, L, d_model)
        context = (
            context
            .transpose(1, 2)
            .contiguous()
            .reshape(
                batch,
                steps,
                self.d_model,
            )
        )

        output = self.out(
            context
        )

        if return_attention:
            return output, weights

        return output


class EncoderBlock(nn.Module):
    """
    One pre-norm Transformer encoder block.

    Architecture:

        x
        -> LayerNorm
        -> Multi-Head Self-Attention
        -> Dropout
        -> Residual Addition
        -> LayerNorm
        -> Feed-Forward Network
        -> Dropout
        -> Residual Addition

    Input:
        (batch, steps, d_model)

    Output:
        (batch, steps, d_model)
    """

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout=0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                d_model,
                d_ff,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                d_ff,
                d_model,
            ),
        )

        self.dropout2 = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        return_attention=False,
    ):
        # ---------------------------------------------------------
        # Self-attention sublayer
        # ---------------------------------------------------------

        normalized = self.norm1(
            x
        )

        if return_attention:
            attention_output, attention_weights = self.attention(
                normalized,
                return_attention=True,
            )
        else:
            attention_output = self.attention(
                normalized
            )
            attention_weights = None

        # Residual connection around self-attention.
        x = x + self.dropout1(
            attention_output
        )

        # ---------------------------------------------------------
        # Feed-forward sublayer
        # ---------------------------------------------------------

        normalized = self.norm2(
            x
        )

        feed_forward_output = self.feed_forward(
            normalized
        )

        # Residual connection around feed-forward network.
        x = x + self.dropout2(
            feed_forward_output
        )

        if return_attention:
            return x, attention_weights

        return x


class CO2Transformer(nn.Module):
    """
    Transformer model used to predict a correction to the causal
    persistence forecast for the six-point CO2 profile.

    Input:
        (batch, context_steps, n_features)

    Output:
        (batch, n_targets)

    The returned output is the residual/correction. The final physical
    forecast is constructed outside the network as:

        forecast = persistence + predicted_correction

    Architecture:

        raw input
        -> input projection
        -> sinusoidal positional encoding
        -> input dropout
        -> pre-norm Transformer encoder blocks
        -> final LayerNorm
        -> final time-step representation
        -> residual prediction head
    """

    def __init__(
        self,
        n_features=110,
        n_targets=6,
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_layers=2,
        dropout=0.1,
        max_length=18,
    ):
        super().__init__()

        if n_layers < 1:
            raise ValueError(
                "n_layers must be at least 1."
            )

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads."
            )

        self.n_features = n_features
        self.n_targets = n_targets
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.max_length = max_length

        # ---------------------------------------------------------
        # Raw features -> Transformer embedding
        # ---------------------------------------------------------

        self.input_projection = InputProjection(
            n_features=n_features,
            d_model=d_model,
        )

        # ---------------------------------------------------------
        # Temporal position information
        # ---------------------------------------------------------

        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_length=max_length,
        )

        self.input_dropout = nn.Dropout(
            dropout
        )

        # ---------------------------------------------------------
        # Transformer encoder stack
        # ---------------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # A pre-norm Transformer stack uses a final normalization
        # after the last encoder block.
        self.final_norm = nn.LayerNorm(
            d_model
        )

        # ---------------------------------------------------------
        # Residual forecasting head
        # ---------------------------------------------------------

        self.prediction_head = nn.Linear(
            d_model,
            n_targets,
        )

        # The network predicts a correction to persistence.
        #
        # Starting this head at zero means that before training:
        #
        #     predicted correction = 0
        #
        # and therefore:
        #
        #     final forecast = persistence
        #
        # Training must learn a useful correction beyond that
        # strong causal baseline.
        nn.init.zeros_(
            self.prediction_head.weight
        )

        nn.init.zeros_(
            self.prediction_head.bias
        )

    def forward(
        self,
        x,
        return_attention=False,
    ):
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(batch, context_steps, n_features)."
            )

        if x.size(-1) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} input features, "
                f"but received {x.size(-1)}."
            )

        if x.size(1) > self.max_length:
            raise ValueError(
                f"Input context length {x.size(1)} exceeds "
                f"max_length={self.max_length}."
            )

        # ---------------------------------------------------------
        # Input representation
        # ---------------------------------------------------------

        x = self.input_projection(
            x
        )

        x = self.positional_encoding(
            x
        )

        x = self.input_dropout(
            x
        )

        # ---------------------------------------------------------
        # Transformer encoder blocks
        # ---------------------------------------------------------

        attention_maps = []

        for block in self.blocks:
            if return_attention:
                x, attention = block(
                    x,
                    return_attention=True,
                )

                attention_maps.append(
                    attention
                )

            else:
                x = block(
                    x
                )

        # Final normalization for the pre-norm stack.
        x = self.final_norm(
            x
        )

        # ---------------------------------------------------------
        # Sequence representation
        # ---------------------------------------------------------

        # The final historical position has access to the complete
        # input window through self-attention.
        final_state = x[:, -1, :]

        # ---------------------------------------------------------
        # Predict correction to persistence
        # ---------------------------------------------------------

        correction = self.prediction_head(
            final_state
        )

        if return_attention:
            return correction, attention_maps

        return correction