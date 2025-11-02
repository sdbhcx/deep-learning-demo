import torch as th
import torch.nn as nn
import torch.nn.functional as F

class PositionWiseFFN(nn.Module):
  def __init__(self, model_dim, dropOut = 0.0):
    super().__init__()
    ffn_dim = model_dim * 4
    self.linear1 = nn.Linear(model_dim, ffn_dim)
    self.linear2 = nn.Linear(ffn_dim, model_dim)
    self.dropout = nn.Dropout(dropOut)
    self.layer_norm = nn.LayerNorm(model_dim)

  def forward(self, x):
    output = F.relu(self.linear1(x))
    output = self.linear2(output)
    output = self.dropout(output)
    output = self.layer_norm(x + output)
    return output
