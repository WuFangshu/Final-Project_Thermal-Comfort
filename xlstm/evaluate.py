# evaluate.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig

# Load test data
x = np.load("data/x_seq_small.npy")
y = np.load("data/y_label_small.npy")
y = y.astype(np.int64) + 3  # Shift label to [0,6]

# Auto pad embedding dim
seq_len, feature_dim = x.shape[1], x.shape[2]
if feature_dim % 4 != 0:
    new_dim = ((feature_dim + 3) // 4) * 4
    padded = np.zeros((x.shape[0], seq_len, new_dim), dtype=np.float32)
    padded[:, :, :feature_dim] = x
    x = padded

# Construct DataLoader
X = torch.tensor(x)
Y = torch.tensor(y)
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=64)

# Load model
mlstm_config = mLSTMBlockConfig()
slstm_config = sLSTMBlockConfig()

model_config = xLSTMBlockStackConfig(
    embedding_dim=X.shape[2],
    num_blocks=2,
    slstm_at=[1],
    dropout=0.1,
    bias=False,
    mlstm_block=mlstm_config,
    slstm_block=slstm_config,
    context_length=X.shape[1]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = xLSTMBlockStack(model_config).to(device)

# Load trained weights if saved, e.g., model.load_state_dict(torch.load("xlstm_model.pt"))

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.numpy())

# Evaluation
print(classification_report(all_labels, all_preds, digits=4))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(7), yticklabels=range(7))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
