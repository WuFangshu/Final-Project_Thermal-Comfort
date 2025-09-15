import torch
import torch.nn as nn
from xLSTM.xLSTM import xLSTM as xlstm

batch_size = 4
seq_lenght = 8
input_size = 32
x_example = torch.zeros(batch_size, seq_lenght, input_size)
factor = 2 # how much input_size will be multiply to give hidden_size
depth = 4 # number of blocks for q, k and v
layers = 'ms' # m for mLSTMblock and s for sLSTMblock

model = xlstm(layers, x_example, factor=factor, depth=depth)

x = torch.randn(batch_size, seq_lenght, input_size)
out = model(x)
print(out.shape)