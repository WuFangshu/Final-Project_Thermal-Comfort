#!/usr/bin/env python3
"""
autotherm_gan_full.py
Supports GAN synthesis scripts with discrete fields (gender, emotion, key points, etc.) and timestamps.

python autotherm_gan_full.py 
    --csv_path /path/autotherm.csv 
    --rows 10000 
    --out autotherm_gan_10k.csv
"""

import argparse, torch, torch.nn as nn
from torch.autograd import grad
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder

# ---------- hyperparameters ----------
Z_DIM = 32
HIDDEN = 128
EPOCHS = 100
BATCH = 256
LAMBDA_GP = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- Network ----------
class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, out_dim):
        super().__init__()
        self.embed = nn.Embedding(num_classes, num_classes)  # one-hot
        self.net = nn.Sequential(
            nn.Linear(z_dim + num_classes, HIDDEN), nn.LeakyReLU(0.2, True),
            nn.Linear(HIDDEN, HIDDEN), nn.LeakyReLU(0.2, True),
            nn.Linear(HIDDEN, out_dim))

    def forward(self, z, label):
        label_emb = self.embed(label)
        x = torch.cat([z, label_emb], dim=1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(in_dim + num_classes, HIDDEN), nn.LeakyReLU(0.2, True),
            nn.Linear(HIDDEN, HIDDEN), nn.LeakyReLU(0.2, True),
            nn.Linear(HIDDEN, 1))

    def forward(self, x, label):
        label_emb = self.embed(label)
        x = torch.cat([x, label_emb], dim=1)
        return self.net(x)

# ---------- Gradient penalty ----------
def gradient_penalty(D, real, fake, label):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=DEVICE)
    interp = alpha * real + (1 - alpha) * fake
    interp = torch.clamp(interp, -5, 5)
    interp.requires_grad_(True)
    d_interp = D(interp, label)
    grad_outputs = torch.ones_like(d_interp)

    grad_interp = grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True)[0]

    eps = 1e-12
    grad_norm = grad_interp.norm(2, dim=1)
    grad_norm = torch.clamp(grad_norm, min=eps)
    return ((grad_norm - 1) ** 2).mean()

# ---------- main process ----------
def main(args):
    # 1. read CSV and dropna
    df = pd.read_csv(args.csv_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # 2. Column definition
    num_cols = ['Age', 'Weight', 'Height', 'Bodyfat', 'Bodytemp',
                'Sport-Last-Hour', 'Time-Since-Meal', 'Tiredness',
                'Clothing-Level', 'Radiation-Temp',
                'PCE-Ambient-Temp', 'Air-Velocity', 'Metabolic-Rate',
                'Wrist_Skin_Temperature', 'Heart_Rate', 'GSR',
                'Ambient_Temperature', 'Ambient_Humidity', 'Solar_Radiation']
    label_col = 'Label'
    cat_cols = ['Gender', 'Emotion-Self', 'Emotion-ML',
                'Nose', 'Neck', 'RShoulder', 'RElbow',
                'LShoulder', 'LElbow', 'REye', 'LEye', 'REar', 'LEar']
    timestamp_col = 'Timestamp'
    file_col = 'file_name'          # template filling

    # 3. Numerical column scaling
    scaler = RobustScaler(quantile_range=(1.0, 99.0))
    X_num = scaler.fit_transform(df[num_cols]).clip(-5, 5)

    # 4. Timestamp scaling
    df['ts_unix'] = pd.to_datetime(df[timestamp_col]).astype(np.int64) // 1e9
    ts_scaler = RobustScaler()
    X_ts = ts_scaler.fit_transform(df[['ts_unix']]).clip(-5, 5)

    # 5. discrete series One-Hot
    cat_encoders = {}
    cat_onehots = []
    cat_dims = []          # Number of categories per column, used for decoding
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        oh = pd.get_dummies(df[col], prefix=col, dtype='float32')
        cat_onehots.append(oh.values)
        cat_encoders[col] = le
        cat_dims.append(len(le.classes_))
    X_cat = np.hstack(cat_onehots) if cat_onehots else np.empty((len(df), 0))

    # 6. Assemble complete features
    X_all = np.hstack([X_num, X_ts, X_cat]).astype(np.float32)

    # 7. Title
    le_label = LabelEncoder()
    y = le_label.fit_transform(df[label_col])

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_all), torch.tensor(y, dtype=torch.long))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, shuffle=True, drop_last=True)

    # 8. Network
    num_features = X_all.shape[1]
    num_classes = len(le_label.classes_)
    G = Generator(Z_DIM, num_classes, num_features).to(DEVICE)
    D = Discriminator(num_features, num_classes).to(DEVICE)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
            nn.init.zeros_(m.bias)
    G.apply(weights_init)
    D.apply(weights_init)

    opt_G = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.5, 0.999))

    # 9. Train
    for epoch in range(EPOCHS):
        for real_x, real_y in loader:
            real_x, real_y = real_x.to(DEVICE), real_y.to(DEVICE)
            z = torch.randn(real_x.size(0), Z_DIM, device=DEVICE)
            fake_x = G(z, real_y)

            # ---- Training D ----
            opt_D.zero_grad()
            loss_real = D(real_x, real_y).mean()
            loss_fake = D(fake_x.detach(), real_y).mean()
            gp = gradient_penalty(D, real_x, fake_x, real_y)
            loss_D = loss_fake - loss_real + LAMBDA_GP * gp
            loss_D.backward(retain_graph=True)
            opt_D.step()

            # ---- Training G ----
            opt_G.zero_grad()
            loss_G = -D(fake_x, real_y).mean()
            loss_G.backward()
            opt_G.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | D {loss_D.item():.4f} | G {loss_G.item():.4f}")

    # 10. Generation
    G.eval()
    with torch.no_grad():
        z = torch.randn(args.rows, Z_DIM, device=DEVICE)
        labels = torch.randint(0, num_classes, (args.rows,), device=DEVICE)
        fake = G(z, labels).cpu().numpy()

        # split
        len_num = len(num_cols)
        fake_num = fake[:, :len_num]
        fake_ts_norm = fake[:, len_num]
        fake_cat = fake[:, len_num + 1:] if cat_dims else np.empty((args.rows, 0))

        # inverse transformation
        fake_num = scaler.inverse_transform(fake_num)
        fake_ts = ts_scaler.inverse_transform(fake_ts_norm.reshape(-1, 1))
        fake_time = pd.to_datetime(fake_ts.flatten(), unit='s')

        # set DataFrame
        syn_df = pd.DataFrame(fake_num, columns=num_cols)
        syn_df[timestamp_col] = fake_time
        syn_df[label_col] = le_label.inverse_transform(labels.cpu().numpy())

        # Decoding discrete columns
        start = 0
        for col, dim in zip(cat_cols, cat_dims):
            probs = fake_cat[:, start:start + dim]
            cat_id = probs.argmax(axis=1)
            syn_df[col] = cat_encoders[col].inverse_transform(cat_id)
            start += dim

        # File name template
        syn_df[file_col] = df[file_col].iloc[0]

    # 11. save
    df = df.drop(columns=['ts_unix'])
    syn_df = syn_df[df.columns]
    syn_df.to_csv(args.out, index=False)
    print(f" {len(syn_df)} synthetic data is generated → {args.out}")

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path",default='/mnt/nvme2n1/xc/teach/autotherm-3CFF/parquetdata/original/test-00000-of-00001.csv' , help="原始 csv 路径")
    parser.add_argument("--rows", type=int, default=5000,
                        help="synthetic rows, default 5000")
    parser.add_argument("--out", default="/mnt/nvme2n1/xc/teach/autotherm-3CFF/parquetdata/gan/test-00000-of-00001_gan.csv",
                        help="output file, default autotherm_gan.csv")
    args = parser.parse_args()
    main(args)