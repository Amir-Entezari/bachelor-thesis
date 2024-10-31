import math
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

def create_patch(xb, patch_len, stride, max_sequence_length):

  """
  xb: [bs x n_vars x seq_len ]
  """

  seq_len = max_sequence_length if max_sequence_length is not None else xb.shape[2]
  mask = torch.ones(xb.shape)
  num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
  tgt_len = patch_len  + stride * num_patch
  pd = tgt_len - seq_len
  pad1 = (0, pd)
  xb = F.pad(xb, pad1, "constant", 0)
  mask = F.pad(mask, pad1, "constant", 0)
  xb = xb.unfold(dimension=-1, size=patch_len, step=stride)                 # xb: [bs x n_vars x num_patch x patch_len]
  mask = mask.unfold(dimension=-1, size=patch_len, step=stride)                 # xb: [bs x n_vars x num_patch x patch_len]
  return xb, mask

class MomentEEG(nn.Module):
    def __init__(
        self,
        emb_size=512,
        n_channels = 64,
        patch_len = 512,
        stride = 512,
        max_sequence_length = 2560,
        **kwargs
      ):
      super().__init__()

      self.tokenizer = MOMENTPipeline.from_pretrained(
          "AutonLab/MOMENT-1-small",
          model_kwargs={'task_name': 'embedding',
                        'reduction': 'mean'},
        )
      self.tokenizer.init()

      self.patch_len = patch_len
      self.stride = stride

      self.positional_encoding = PositionalEncoding(emb_size)

      # channel token, N_channels >= your actual channels
      self.channel_tokens = nn.Embedding(n_channels, emb_size)
      self.index = nn.Parameter(
          torch.LongTensor(range(n_channels)), requires_grad=False
      )
      self.max_sequence_length = max_sequence_length


    def forward(self, x, perturb = False):
        """
        x: [batch_size, channel, num_patch, ts]
        output: [batch_size, emb_size]
        """
        x, m = create_patch(x, patch_len = self.patch_len, stride = self.stride, max_sequence_length = self.max_sequence_length)
        emb_seq = []
        for i in range(x.shape[1]):
            xb = x[:, i : i + 1, :, :].squeeze(1)
            mb=  m[:, i : i + 1, :, :].squeeze(1)
            bs, num_patch, patch_len = xb.shape
            xb = torch.reshape(xb, (bs * num_patch, 1, patch_len))
            mb = torch.reshape(mb, (bs * num_patch, 1, patch_len))
            xb = self.tokenizer(x_enc = xb.detach()).embeddings

            xb = torch.reshape(xb, (bs, num_patch, -1))

            batch_size, ts, _ = xb.shape
            # (batch_size, ts, emb)
            channel_token_emb = (
                self.channel_tokens(self.index[i])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            # (batch_size, ts, emb)
            channel_emb = self.positional_encoding(xb + channel_token_emb)

            # perturb
            if self.training and perturb:
              ts = channel_emb.shape[1]
              step = np.random.randint(2, ts)
              if step < ts -1: # step = ts -1 means we do not perturb the sequence
                  start_point = np.random.randint(1, ts - 1) # 1, ts -1  because we do not want to drop fisrt and last token
                  drop_list = list(range(start_point, ts - 1, step))
                  selected_ts = [i for i in range(ts) if i not in drop_list]
                  channel_emb = channel_emb[:, selected_ts]
            emb_seq.append(channel_emb)

        # (batch_size, 16 * ts, emb)
        emb = torch.cat(emb_seq, dim=1)
        # (batch_size, emb)
        emb = emb.mean(dim=1)
        print('*'*20)
        print(emb.shape)
        return emb



# supervised classifier module
class MomentClassifier(nn.Module):
    def __init__(self, emb_size=512, n_classes=6, **kwargs):
        super().__init__()
        self.transformer = MomentEEG(emb_size, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x, perturb = False):
        x = self.transformer(x, perturb = perturb)
        x = self.classifier(x)
        return x