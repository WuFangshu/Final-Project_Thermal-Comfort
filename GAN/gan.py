#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gan_2.py
Stable release fixes All-NaN, timestamp view warnings, initial D/G NaN issues, etc.
- Numeric columns: to_numeric → Remove entire NaN columns → Median fill → RobustScaler
- Timestamps: utc + astype(“int64”) for safe conversion
- Categorical columns: Top-K + OTHER restriction for one-hot dimensions
- Pre-training: Strict NaN/Inf assertions + np.nan_to_num fallback
- Model: Conditionally one-hot concatenated WGAN-GP with more stable hyperparameters

Running example:
python Gan_2.py --csv_path "E:/thermal_project/Gan/data/test.csv" --rows 5000 --out "E:/thermal_project/Gan/data/gan.csv"
"""

import argparse
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler, LabelEncoder
from torch.autograd import grad

# parameter
Z_DIM = 32 #The dimension of the input noise vector z
HIDDEN = 128
EPOCHS = 50
BATCH = 256
LAMBDA_GP = 1.0            
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOPK_EMOTION_ML = 100      # Top K retention for high-frequency categories
TOPK_GENERAL_CAT = 50

# Random seeds
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#Retain only the top k most frequent classes in the Pandas Series
def topk_map_series(s: pd.Series, k=50, other_token="OTHER"):
    s = s.astype(str) #Forced conversion to string to ensure consistency
    cnt = Counter(s) #Count the frequency of occurrence for each category
    keep = set([v for v, _ in cnt.most_common(k)]) #Take the k most common category names and place them into a set.
    return s.where(s.isin(keep), other_token)

#Integrity check to ensure all values are finite numbers
def finite_assert(name: str, arr: np.ndarray):
    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr)) #Obtaining the position (index) of non-finite values
        raise ValueError(f"[{name}] Containing non-finite values (NaN/Inf). Row{bad[0][:5]}, Column{bad[1][:5]}")

# model
class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, out_dim):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(z_dim + num_classes, HIDDEN), nn.LeakyReLU(0.2, inplace=True), #Input layer
            nn.Linear(HIDDEN, HIDDEN), nn.LeakyReLU(0.2, inplace=True), #medium layer
            nn.Linear(HIDDEN, out_dim) #output layer
        )

    def forward(self, z, label):
        onehot = F.one_hot(label, num_classes=self.num_classes).float()
        x = torch.cat([z, onehot], dim=1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim + num_classes, HIDDEN), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(HIDDEN, HIDDEN), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(HIDDEN, 1)
        )

    def forward(self, x, label):
        onehot = F.one_hot(label, num_classes=self.num_classes).float()
        x = torch.cat([x, onehot], dim=1)
        return self.net(x)

def gradient_penalty(D, real, fake, label):
    batch = real.size(0) #Retrieve the size of the current batch
    alpha = torch.rand(batch, 1, device=DEVICE).expand_as(real) #Generate a set of random weights for linear interpolation between real and fake values.
    interp = alpha * real + (1 - alpha) * fake #Generate interpolated samples
    interp.requires_grad_(True)
    d_interp = D(interp, label) #The output of the classifier for interpolated samples
    grad_outputs = torch.ones_like(d_interp)
    grad_interp = grad(
        outputs=d_interp, inputs=interp, grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_norm = grad_interp.view(batch, -1).norm(2, dim=1) #Flatten the gradient tensor into two dimensions, then compute the L2 norm for each sample.
    return ((grad_norm - 1) ** 2).mean()

# Main process
def main(args):
    set_seed(42)

    # ---------- Read CSV ----------
    # Using the default C engine with `low_memory=False` reduces misclassification of mixed data types
    df = pd.read_csv(args.csv_path, low_memory=False)
    # Replace Inf with NaN, initially refrain from dropping rows, and subsequently process according to the column strategy
    df = df.replace([np.inf, -np.inf], np.nan)
    df1 = df.copy() 

    # ---------- Column definition ----------
    base_num_cols = [
        'Age', 'Weight', 'Height', 'Bodyfat', 'Bodytemp',
        'Sport-Last-Hour', 'Time-Since-Meal', 'Tiredness',
        'Clothing-Level', 'Radiation-Temp',
        'PCE-Ambient-Temp', 'Air-Velocity', 'Metabolic-Rate',
        'Wrist_Skin_Temperature', 'Heart_Rate', 'GSR',
        'Ambient_Temperature', 'Ambient_Humidity', 'Solar_Radiation'
    ]
    keypoint_cols = ['Nose', 'Neck', 'RShoulder', 'RElbow',
                     'LShoulder', 'LElbow', 'REye', 'LEye', 'REar', 'LEar']
    # Merge the key points of existence into a numerical column
    num_cols = base_num_cols + [c for c in keypoint_cols if c in df.columns]
    #Merge key points into a numerical column
    #Categorical variable column
    cat_cols = [c for c in ['Gender', 'Emotion-Self', 'Emotion-ML'] if c in df.columns]

    label_col = 'Label'
    timestamp_col = 'Timestamp'
    file_col = 'file_name'
    for must in [label_col, timestamp_col, file_col]:
        if must not in df.columns:
            raise ValueError(f"Missing required columns: {must}")

    # ---------- Numeric column cleaning + scaling ----------
    # Convert all values to numeric type (unparsable values → NaN)
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Remove the numerical column where the entire row consists of NaN values (print it out to verify).
    num_cols = [c for c in num_cols if c in df.columns and df[c].notna().any()]
    if not num_cols:
        raise ValueError("The numeric column is empty after cleaning. Please check the num_cols configuration or data quality.")

    # For columns still containing sporadic NaN values, employ median imputation
    for c in num_cols:
        med = df[c].median(skipna=True)
        df[c] = df[c].fillna(med)

    # Only now can safely discard NaN values from other non-critical information (to prevent irrelevant columns from influencing the outcome)
    df = df.dropna(subset=[label_col, timestamp_col, file_col])

    # Scaling
    scaler = RobustScaler(quantile_range=(1.0, 99.0))
    X_num = scaler.fit_transform(df[num_cols]).clip(-5, 5).astype(np.float32)

    # ---------- Timestamp (UTC + astype) ----------
    ts = pd.to_datetime(df[timestamp_col], utc=True, errors='coerce')
    mask_ts = ts.notna()
    df = df[mask_ts].copy()
    ts = ts[mask_ts]
    # second-level Unix time
    df['ts_unix'] = (ts.astype('int64') // 1_000_000_000).astype(np.int64)

    ts_scaler = RobustScaler()
    X_ts = ts_scaler.fit_transform(df[['ts_unix']]).clip(-5, 5).astype(np.float32)

    # Synchronous trimming (if the previous step removed rows with empty timestamps)
    if len(X_num) != len(df):
        X_num = X_num[(-np.inf < df['ts_unix']).values]  # Simple alignment, typically unnecessary

    # ---------- Category column：Top-K + OTHER → LabelEncoder → one-hot ----------
    cat_onehots, cat_dims, cat_encoders = [], [], {}
    for col in cat_cols:
        s = df[col].astype(str)
        if col == 'Emotion-ML':
            s = topk_map_series(s, k=TOPK_EMOTION_ML, other_token="OTHER")
        else:
            s = topk_map_series(s, k=TOPK_GENERAL_CAT, other_token="OTHER")

        le = LabelEncoder()
        int_ids = le.fit_transform(s)                # include OTHER
        oh = pd.get_dummies(int_ids, prefix=col, dtype='float32') #Convert integer labels to one-hot encoded vectors
        cat_onehots.append(oh.values.astype(np.float32))
        cat_encoders[col] = le
        cat_dims.append(len(le.classes_))
    
    #Concatenate the one-hot encoded results of all category columns
    if cat_onehots:
        X_cat = np.hstack(cat_onehots).astype(np.float32)
    else:
        X_cat = np.empty((len(df), 0), dtype=np.float32)

    # ---------- Assembly characteristics + Training labels ----------
    X_all = np.hstack([X_num, X_ts, X_cat]).astype(np.float32)
    # Fallback: Clear any residual NaN/Inf values to prevent initialisation errors
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)
    finite_assert("X_all", X_all)

    le_label = LabelEncoder() #Encode textual labels as integers
    y = le_label.fit_transform(df[label_col].astype(str))

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_all), torch.tensor(y, dtype=torch.long)
    )

    # ---------- DataLoader ----------
    drop_last_flag = len(dataset) >= BATCH
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        drop_last=drop_last_flag,
        num_workers=0,        
        pin_memory=False
    )

    # ---------- Network ----------
    num_features = X_all.shape[1]
    num_classes = len(le_label.classes_)
    G = Generator(Z_DIM, num_classes, num_features).to(DEVICE)
    D = Discriminator(num_features, num_classes).to(DEVICE)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
            nn.init.zeros_(m.bias) #Initialise the bias term to 0
    G.apply(weights_init); D.apply(weights_init)
    
    #Optimizer
    opt_G = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.0, 0.9))
    n_critic = 5 #Before each update to the generator, train the discriminator five times

    # ---------- Train ----------
    for epoch in range(EPOCHS):
        for real_x, real_y in loader:
            real_x = real_x.to(DEVICE)
            real_y = real_y.to(DEVICE)

            # Critic multiple steps
            for _ in range(n_critic):
                z = torch.randn(real_x.size(0), Z_DIM, device=DEVICE)
                fake_x = G(z, real_y) #Randomly sample a vector z and input it alongside the label real_y into the generator G to produce a synthetic sample fake_x.

                opt_D.zero_grad(set_to_none=True)
                d_real = D(real_x, real_y).mean()
                d_fake = D(fake_x.detach(), real_y).mean()
                gp = gradient_penalty(D, real_x, fake_x.detach(), real_y)
                loss_D = d_fake - d_real + LAMBDA_GP * gp
                if not torch.isfinite(loss_D):
                    raise RuntimeError("loss_D is not finite. Please check whether the input data contains NaN/Inf")
                loss_D.backward() #Backward updates the gradient
                opt_D.step() #Optimiser updates discriminator weights

            # Generator one step
            z = torch.randn(real_x.size(0), Z_DIM, device=DEVICE)
            fake_x = G(z, real_y)
            opt_G.zero_grad(set_to_none=True) #clear the gradient cache
            loss_G = -D(fake_x, real_y).mean()
            if not torch.isfinite(loss_G):
                raise RuntimeError("loss_G is not finite. Please check whether the input data contains NaN/Inf")
            loss_G.backward()
            opt_G.step()

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | D {loss_D.item():.4f} | G {loss_G.item():.4f}", flush=True)

    # ---------- Generation ----------
    G.eval() #Set generator G to evaluation mode
    with torch.no_grad():
        z = torch.randn(args.rows, Z_DIM, device=DEVICE) #Sampling latent vectors z and random labels
        labels = torch.randint(0, num_classes, (args.rows,), device=DEVICE) #Use the trained G to synthesise args.rows row data
        fake = G(z, labels).cpu().numpy() #Output converted to a NumPy array

        # Split
        len_num = len(num_cols)
        fake_num = fake[:, :len_num]
        fake_ts_norm = fake[:, len_num]
        fake_cat = fake[:, len_num + 1:] if cat_dims else np.empty((args.rows, 0), dtype=np.float32)

        # Inverse transform
        fake_num = scaler.inverse_transform(fake_num)
        fake_ts = ts_scaler.inverse_transform(fake_ts_norm.reshape(-1, 1))
        fake_time = pd.to_datetime(fake_ts.flatten(), unit='s', utc=True).tz_convert(None)

        # Assembly DataFrame
        syn_df = pd.DataFrame(fake_num, columns=num_cols)
        syn_df[timestamp_col] = fake_time
        syn_df[label_col] = le_label.inverse_transform(labels.cpu().numpy())

        # Reverse Decoding Category
        start = 0
        for col, dim in zip(cat_cols, cat_dims):
            if dim == 0:
                continue
            probs = fake_cat[:, start:start + dim]
            cat_id = probs.argmax(axis=1).astype(int)
            syn_df[col] = cat_encoders[col].inverse_transform(cat_id)
            start += dim

        # File name template
        syn_df[file_col] = df[file_col].iloc[0]

    if 'ts_unix' in df.columns:
        df = df.drop(columns=['ts_unix'])
    final_cols = [c for c in df.columns if c in syn_df.columns]
    if final_cols:
        syn_df = syn_df[final_cols]

    # ---------- Preserve metadata ----------
    import pickle
    meta = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "label_col": label_col,
        "timestamp_col": timestamp_col,
        "file_col": file_col,
        "scaler": scaler,
        "ts_scaler": ts_scaler,
        "label_encoder": le_label,
        "cat_encoders": cat_encoders,
        "cat_dims": cat_dims,
        "topk": {"Emotion-ML": TOPK_EMOTION_ML, "_general": TOPK_GENERAL_CAT}
    }
    meta_path = os.path.splitext(args.out)[0] + "_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"Metadata has been saved → {meta_path}")
    
    agent = len(syn_df)
    df1_sample = df1.sample(n=agent, random_state=42).reset_index(drop=True)
    syn_df = syn_df.reset_index(drop=True)
    re_idx = [1, 16, 17]
    max_idx = 27
    for idx in re_idx:
      syn_df.iloc[:, idx] = df1_sample.iloc[:, idx].values
    insert_block = df1_sample.iloc[:, 18:28]
    r_idx = 17
    syn_df = pd.concat([syn_df.iloc[:, :r_idx+1], insert_block, syn_df.iloc[:, r_idx+1:]], axis=1)
    
    # Save the CSV file
    syn_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"{len(syn_df)} synthetic data points have been generated → {args.out}")

# ===================== CLI =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="E:/thermal_project/Gan/data/test.csv", help="original CSV path")
    parser.add_argument("--rows", type=int, default=5000, help="number of sythetic data rows（default 5000）")
    parser.add_argument("--out", default="E:/thermal_project/Gan/data/gan.csv", help="output CSV path")
    args = parser.parse_args()
    main(args)
