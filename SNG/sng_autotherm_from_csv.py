# -*- coding: utf-8 -*-
"""
AutoTherm CSV -> SNG 
    conda activate sng_env
    python sng_autotherm_from_csv.py
"""
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ========= Path =========
TRAIN_CSV = r"E:/thermal_project/Synthetic-Data-Generation-by-Supervised-Neural-Gas-Network/data/train-00001-of-00005.csv"
TEST_CSV  = r"E:/thermal_project/Synthetic-Data-Generation-by-Supervised-Neural-Gas-Network/data/train-00001-of-00005.csv"
OUT_CSV   = "synthetic_from_train.csv"
META_JSON = OUT_CSV.replace(".csv", "_meta.json")

# ========= SNG Parameter =========
PROTOS_PER_CLASS = 64    # Number of prototypes per class for SNG
EPOCHS = 30              # Number of training epochs for SNG
LR_BMU0, LR_NBR0 = 0.5, 0.05  # Initial learning rates for BMU and neighbors
SYNTH_MULTIPLIER = 1.0   # Multiplier: how many synthetic samples to generate per real sample

# ========= Column definition =========
KEYPOINT_COLS = [
    "Nose","Neck","RShoulder","RElbow","LShoulder","LElbow",
    "REye","LEye","REar","LEar"
]
CATEGORICAL_COLS = ["Gender","Emotion-Self","Emotion-ML"]  # categorical columns that will be one-hot encoded
LIKELY_LABEL_NAMES = ["Label","label","tsv","TSV","thermal_sensation","vote"]

def read_csv(path):
    """Read CSV file with automatic delimiter handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)  # default comma delimiter
    except Exception:
        # if comma fails, try automatic delimiter inference
        return pd.read_csv(path, sep=None, engine="python")

def split_keypoint_series(s: pd.Series):
    """
    Split 'x~y~c' string into three numeric columns.
    Each represents: x coordinate, y coordinate, confidence.
    If parsing fails, fill with NaN.
    """
    x = []; y = []; c = []
    for v in s.astype(str).values:
        parts = str(v).split("~")
        if len(parts) >= 3:
            try:
                xv = float(parts[0]); yv = float(parts[1]); cv = float(parts[2])
            except Exception:
                xv = yv = cv = np.nan
        else:
            xv = yv = cv = np.nan
        x.append(xv); y.append(yv); c.append(cv)
    return np.array(x), np.array(y), np.array(c)

def expand_keypoints(df: pd.DataFrame):
    """
    Expand keypoint columns into separate numeric columns (x, y, confidence).
    Replace original string columns with expanded numeric ones.
    """
    df = df.copy()
    for kp in KEYPOINT_COLS:
        if kp in df.columns:
            x, y, c = split_keypoint_series(df[kp])
            df[f"{kp}_x"] = x
            df[f"{kp}_y"] = y
            df[f"{kp}_c"] = c
            df.drop(columns=[kp], inplace=True)
    return df

def detect_label_col(df: pd.DataFrame):
    """
    Detect which column is the label column.
    Priority:
      1. Common known names.
      2. Last column if numeric with limited categories.
      3. Numeric column with fewest unique values.
      4. Fallback: last column.
    """
    for name in LIKELY_LABEL_NAMES:
        if name in df.columns:
            return name
    last = df.columns[-1]
    if pd.api.types.is_numeric_dtype(df[last]) and df[last].nunique() <= max(20, int(0.02*len(df))):
        return last
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric:
        uniq = sorted([(c, df[c].nunique()) for c in numeric], key=lambda t: t[1])
        return uniq[0][0]
    return last

def one_hot_encode(df: pd.DataFrame, cols):
    """
    Apply one-hot encoding to categorical columns.
    """
    df = df.copy()
    exists = [c for c in cols if c in df.columns]
    if not exists:
        return df
    return pd.get_dummies(df, columns=exists, dummy_na=True)

# ----------------- SNG / Neural Gas -----------------
def train_sng(X, K=64, epochs=30, lr_bmu0=0.5, lr_nbr0=0.05):
    """
    Train a Self-Organizing Neural Gas (SNG) prototype set on data X.
    X: input data (N,D)
    K: number of prototypes
    """
    rng = np.random.default_rng(0)
    K = min(max(4, K), len(X))
    idx = rng.choice(len(X), size=K, replace=False)
    W = X[idx].copy()  # initial prototypes
    lam0 = max(2.0, K/2)
    for ep in range(epochs):
        order = rng.permutation(len(X))  # shuffle samples
        lr_bmu = lr_bmu0 * (1 - ep/epochs)
        lr_nbr = lr_nbr0 * (1 - ep/epochs)
        lam = max(1.0, lam0 * (1 - ep/epochs))
        for i in order:
            x = X[i]
            d = np.linalg.norm(W - x, axis=1)
            rank = np.argsort(np.argsort(d))
            h = np.exp(-rank / lam)  # neighborhood function
            eta = lr_nbr * h
            bmu = np.argmin(d)  # best matching unit
            eta[bmu] = lr_bmu
            W += (eta[:, None] * (x - W))  # update prototypes
    return W

def estimate_local_stats(X, W, batch_size=10000):
    """
    Estimate local statistics around prototypes:
    - pis: prototype weights (relative frequency)
    - vari: per-dimension variance (diagonal covariance)
    Uses batch assignment to avoid large memory.
    """
    N, D = X.shape
    K = W.shape[0]
    w_norm2 = np.sum(W * W, axis=1)  # squared norms of prototypes

    assign_counts = np.zeros(K, dtype=np.int64)
    sum_per_k = np.zeros((K, D), dtype=np.float64)
    sumsq_per_k = np.zeros((K, D), dtype=np.float64)

    for start in range(0, N, batch_size):
        xb = X[start:start + batch_size]           
        x_norm2 = np.sum(xb * xb, axis=1, keepdims=True)      
        dot = xb.astype(np.float64) @ W.astype(np.float64).T  
        d2 = x_norm2 + w_norm2[None, :] - 2.0 * dot           
        bmu = np.argmin(d2, axis=1)                           

        for k in range(K):
            mask = (bmu == k)
            if not np.any(mask):
                continue
            Xk = xb[mask].astype(np.float64)
            assign_counts[k] += Xk.shape[0]
            sum_per_k[k]   += Xk.sum(axis=0)
            sumsq_per_k[k] += (Xk * Xk).sum(axis=0)

    pis = assign_counts.astype(np.float64)
    pis = np.maximum(pis, 1e-12)
    pis /= pis.sum()

    vari = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        n = assign_counts[k]
        if n <= 1:
            # fallback: use global variance
            vari[k] = np.var(X, axis=0).astype(np.float32) + 1e-6
        else:
            mean_k = (sum_per_k[k] / n)
            ex2_k  = (sumsq_per_k[k] / n)
            var_k  = np.maximum(ex2_k - mean_k * mean_k, 1e-6)
            vari[k] = var_k.astype(np.float32)

    return pis.astype(np.float32), vari

def sample_from_sng(M, W, pis, vari, rng=None):
    """
    Generate synthetic samples from SNG prototypes.
    For each sample:
      1. Randomly choose a prototype (according to pis).
      2. Add Gaussian noise with variance vari.
    """
    rng = np.random.default_rng() if rng is None else rng
    ks = rng.choice(len(W), size=M, p=pis)
    eps = rng.normal(size=(M, W.shape[1]))
    return W[ks] + eps * np.sqrt(vari[ks])

# ----------------- Main process -----------------
def main():
    print(">> Loading CSV ...")
    df = read_csv(TRAIN_CSV)

    # Expand pose keypoints into numeric columns
    df = expand_keypoints(df)
    # One-hot encode categorical columns
    df = one_hot_encode(df, CATEGORICAL_COLS)

    # Detect label column automatically
    label_col = detect_label_col(df)
    print(f"Detected label column: {label_col}")

    # Select numeric feature columns (exclude label)
    feature_cols = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns available. Please check input.")

    # Clean: remove columns with only one unique value
    tmp = df[feature_cols].copy()
    nunique = tmp.nunique()
    keep = nunique[nunique > 1].index.tolist()
    feature_cols = keep

    # Encode label as integer if not numeric
    if not pd.api.types.is_numeric_dtype(df[label_col]):
        df[label_col] = df[label_col].astype("category").cat.codes

    # Drop rows with missing values
    df = df[feature_cols + [label_col]].dropna()
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy()

    # Standardize features
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X).astype(np.float32)

    # Train SNG per class and generate synthetic samples
    classes = np.unique(y)
    rng = np.random.default_rng(42)
    syn_rows = []
    syn_labels = []
    for c in classes:
        Xc = Xz[y == c]
        if len(Xc) == 0:
            continue
        Kc = min(PROTOS_PER_CLASS, max(4, len(Xc)))
        print(f">> Train SNG: class={c} N={len(Xc)} K={Kc}")
        W = train_sng(Xc, K=Kc, epochs=EPOCHS, lr_bmu0=LR_BMU0, lr_nbr0=LR_NBR0)
        pis, vari = estimate_local_stats(Xc, W)
        M = int(len(Xc) * SYNTH_MULTIPLIER)
        Xsyn = sample_from_sng(M, W, pis, vari, rng=rng)
        Xsyn = scaler.inverse_transform(Xsyn)
        syn_rows.append(Xsyn)
        syn_labels.append(np.full(M, c))

    if not syn_rows:
        raise RuntimeError("No synthetic rows produced.")
    Xsyn = np.vstack(syn_rows)
    ysyn = np.concatenate(syn_labels)

    # Save synthetic dataset to CSV
    out = pd.DataFrame(Xsyn, columns=feature_cols)
    out[label_col] = ysyn
    out.to_csv(OUT_CSV, index=False)
    print(f">> Saved: {OUT_CSV} shape={out.shape}")

    # Save metadata (feature columns, label, etc.)
    meta = {
        "train_csv": TRAIN_CSV,
        "label_col": label_col,
        "feature_cols": feature_cols,
        "keypoints_expanded": KEYPOINT_COLS,
        "categorical_encoded": CATEGORICAL_COLS,
        "protos_per_class": PROTOS_PER_CLASS,
        "epochs": EPOCHS,
        "synth_multiplier": SYNTH_MULTIPLIER,
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(">> Wrote meta:", META_JSON)

if __name__ == "__main__":
    main()
