import math
import torch
from momentfm import MOMENTPipeline
import torch.nn as nn


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
        max_sequence_length = 2560,
        **kwargs
      ):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.n_channels = n_channels
        self.positional_encoding = PositionalEncoding(emb_size)

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

        batch_size, ts, _ = x.shape
        channel_emb = []

        for i in range(self.n_channels):
            channel_token_emb = (
                self.channel_tokens(self.index[i])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )

            emb_with_channel_pos = self.positional_encoding(x + channel_token_emb)
            channel_emb.append(emb_with_channel_pos)

        emb = torch.cat(channel_emb, dim=1)  # (batch_size, 16 * ts, emb)

        # (batch_size, emb)
        emb = emb.mean(dim=1)

        return emb



# supervised classifier module
class MomentClassifier(nn.Module):
    def __init__(self, emb_size=512, n_channels=16, n_classes=6, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.emb_size = emb_size

        self.positional_encoding = PositionalEncoding(emb_size)

        self.channel_tokens = nn.Embedding(n_channels, emb_size)
        self.index = nn.Parameter(torch.LongTensor(range(n_channels)), requires_grad=False)

        self.classifier = nn.Linear(emb_size, n_classes)

    def forward(self, x, perturb=False, saved_embeddings=None):
        """
        x: [batch_size, channel, num_patch, ts]
        saved_embeddings: Pre-computed embeddings that bypass the `MomentEEG` forward pass.
        """

        batch_size, ts, _ = x.shape
        channel_emb = []

        for i in range(self.n_channels):
            channel_token_emb = (
                self.channel_tokens(self.index[i])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )

            emb_with_channel_pos = self.positional_encoding(x + channel_token_emb)
            channel_emb.append(emb_with_channel_pos)

        emb = torch.cat(channel_emb, dim=1)  # (batch_size, 16 * ts, emb)


        # (batch_size, emb)
        emb = emb.mean(dim=1)
        
        emb = self.classifier(emb)
        return emb
