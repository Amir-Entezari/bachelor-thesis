import math
import torch
from momentfm import MOMENTPipeline
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MomentTransformer(nn.Module):
    def __init__(
        self,
        emb_size=512,
        n_channels = 16,
        patch_len = 512,
        stride = 512,

        heads = 8, # number of heads
        depth = 1, # number of transformer layers,

        max_sequence_length = 2560,
        **kwargs
      ):
        super().__init__()


        self.transformer = LinearAttentionTransformer(
                dim=emb_size,
                heads=heads,
                depth=depth,
                max_seq_len=1024,
                attn_layer_dropout=0.2,  # dropout right after self-attention layer
                attn_dropout=0.2,  # dropout post-attention
            )

        self.patch_len = patch_len
        self.stride = stride
        self.n_channels = n_channels
        self.positional_encoding = PositionalEncoding(emb_size)

        # channel token, N_channels >= your actual channels
        self.channel_tokens = nn.Embedding(n_channels, emb_size)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )
        self.max_sequence_length = max_sequence_length


    def forward(self, x, perturb=False, saved_embeddings=None):
        """
        x: [batch_size, channel, num_patch, ts]
        saved_embeddings: Pre-computed embeddings that bypass the `MomentEEG` forward pass.
        """

        # Apply channel embedding and positional encoding here
        batch_size, ts, _ = x.shape
        channel_emb = []

        for i in range(self.n_channels):
            # Channel token embedding (repeat across time steps)
            channel_token_emb = (
                self.channel_tokens(self.index[i])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )

            # Add positional encoding to the embeddings
            emb_with_channel_pos = self.positional_encoding(x + channel_token_emb)
            channel_emb.append(emb_with_channel_pos)

        # Stack embeddings from all channels and average them
        emb = torch.cat(channel_emb, dim=1)  # (batch_size, 16 * ts, emb)

        emb = self.transformer(emb)
        # (batch_size, emb)
        emb = emb.mean(dim=1)

        return emb




# # supervised classifier module using pre-embedded samples
# class MomentClassifier(nn.Module):
#     def __init__(self, emb_size=512, n_channels=16, n_classes=6, **kwargs):
#         super().__init__()
#         self.n_channels = n_channels
#         self.emb_size = emb_size

#         # Channel embedding and positional encoding will now be handled here
#         self.positional_encoding = PositionalEncoding(emb_size)

#         # Initialize learnable channel embeddings
#         self.channel_tokens = nn.Embedding(n_channels, emb_size)
#         self.index = nn.Parameter(torch.LongTensor(range(n_channels)), requires_grad=False)

#         self.classifier = nn.Linear(emb_size, n_classes)

#     def forward(self, x, perturb=False, saved_embeddings=None):
#         """
#         x: [batch_size, channel, num_patch, ts]
#         saved_embeddings: Pre-computed embeddings that bypass the `MomentEEG` forward pass.
#         """

#         # Apply channel embedding and positional encoding here
#         batch_size, ts, _ = x.shape
#         channel_emb = []

#         for i in range(self.n_channels):
#             # Channel token embedding (repeat across time steps)
#             channel_token_emb = (
#                 self.channel_tokens(self.index[i])
#                 .unsqueeze(0)
#                 .unsqueeze(0)
#                 .repeat(batch_size, ts, 1)
#             )

#             # Add positional encoding to the embeddings
#             emb_with_channel_pos = self.positional_encoding(x + channel_token_emb)
#             channel_emb.append(emb_with_channel_pos)

#         # Stack embeddings from all channels and average them
#         emb = torch.cat(channel_emb, dim=1)  # (batch_size, 16 * ts, emb)

#         emb = self.transformer(emb)
#         # (batch_size, emb)
#         emb = emb.mean(dim=1)

#         emb = self.classifier(emb)
#         return emb


# supervised classifier module
class MomentClassifier(nn.Module):
    def __init__(self, emb_size=512, n_classes=6, n_channels=19, **kwargs):
        super().__init__()
        self.transformer = MomentTransformer(emb_size, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x, perturb = False):
        x = self.transformer(x, perturb = perturb)
        x = self.classifier(x)
        return x