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

# ========= 路径（按你的实际路径）=========
TRAIN_CSV = r"E:\thermal_project\autotherm_data\train.csv"
TEST_CSV  = r"E:\thermal_project\autotherm_data\test.csv"   # 可选，用不到也没关系
OUT_CSV   = "synthetic_from_train.csv"
META_JSON = OUT_CSV.replace(".csv", "_meta.json")

# ========= SNG 参数 =========
PROTOS_PER_CLASS = 64    # 每个类的原型数
EPOCHS = 30
LR_BMU0, LR_NBR0 = 0.5, 0.05
SYNTH_MULTIPLIER = 1.0   # 每类合成等量样本

# ========= 列定义（关键点名列表；会把 "Nose" 拆成 Nose_x/Nose_y/Nose_c 等）=========
KEYPOINT_COLS = [
    "Nose","Neck","RShoulder","RElbow","LShoulder","LElbow",
    "REye","LEye","REar","LEar"
]
CATEGORICAL_COLS = ["Gender","Emotion-Self","Emotion-ML"]  # 会做 one-hot
LIKELY_LABEL_NAMES = ["Label","label","tsv","TSV","thermal_sensation","vote"]

def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # 防止逗号/分隔符问题：默认逗号，若失败再尝试分隔符推断
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")

def split_keypoint_series(s: pd.Series):
    """把 'x~y~c' 拆成三列浮点。缺失/异常填 NaN。"""
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
    # 优先按常见名字
    for name in LIKELY_LABEL_NAMES:
        if name in df.columns:
            return name
    # 否则以最后一列为候选（若是数值且类别数较小）
    last = df.columns[-1]
    if pd.api.types.is_numeric_dtype(df[last]) and df[last].nunique() <= max(20, int(0.02*len(df))):
        return last
    # 兜底：找类别数最小的数值列
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric:
        uniq = sorted([(c, df[c].nunique()) for c in numeric], key=lambda t: t[1])
        return uniq[0][0]
    return last

def one_hot_encode(df: pd.DataFrame, cols):
    df = df.copy()
    exists = [c for c in cols if c in df.columns]
    if not exists:
        return df
    return pd.get_dummies(df, columns=exists, dummy_na=True)

# ----------------- SNG / Neural Gas -----------------
def train_sng(X, K=64, epochs=30, lr_bmu0=0.5, lr_nbr0=0.05):
    rng = np.random.default_rng(0)
    K = min(max(4, K), len(X))
    idx = rng.choice(len(X), size=K, replace=False)
    W = X[idx].copy()
    lam0 = max(2.0, K/2)
    for ep in range(epochs):
        order = rng.permutation(len(X))
        lr_bmu = lr_bmu0 * (1 - ep/epochs)
        lr_nbr = lr_nbr0 * (1 - ep/epochs)
        lam = max(1.0, lam0 * (1 - ep/epochs))
        for i in order:
            x = X[i]
            d = np.linalg.norm(W - x, axis=1)
            rank = np.argsort(np.argsort(d))
            h = np.exp(-rank / lam)
            eta = lr_nbr * h
            bmu = np.argmin(d)
            eta[bmu] = lr_bmu
            W += (eta[:, None] * (x - W))
    return W

def estimate_local_stats(X, W):
    d = np.linalg.norm(X[:, None, :] - W[None, :, :], axis=2)
    assign = d.argmin(axis=1)
    K = W.shape[0]
    pis = np.bincount(assign, minlength=K).astype(float)
    pis = np.maximum(pis, 1e-8); pis /= pis.sum()
    vari = np.zeros_like(W)
    for k in range(K):
        Xk = X[assign == k]
        if len(Xk) >= 2:
            vari[k] = Xk.var(axis=0) + 1e-6
        else:
            vari[k] = X.var(axis=0) + 1e-6
    return pis, vari

def sample_from_sng(M, W, pis, vari, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    ks = rng.choice(len(W), size=M, p=pis)
    eps = rng.normal(size=(M, W.shape[1]))
    return W[ks] + eps * np.sqrt(vari[ks])

# ----------------- 主流程 -----------------
def main():
    print(">> Loading CSV ...")
    df = read_csv(TRAIN_CSV)

    # 关键点拆分 & 类别编码
    df = expand_keypoints(df)
    df = one_hot_encode(df, CATEGORICAL_COLS)

    # 识别标签列
    label_col = detect_label_col(df)
    print(f"Detected label column: {label_col}")

    # 选特征列：数值 & 非标签
    feature_cols = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
    if len(feature_cols) == 0:
        raise ValueError("没有可用的数值特征列，请检查输入。")

    # 清洗：去除全空/全常数列
    tmp = df[feature_cols].copy()
    nunique = tmp.nunique()
    keep = nunique[nunique > 1].index.tolist()
    feature_cols = keep

    # 处理标签为整数编码
    if not pd.api.types.is_numeric_dtype(df[label_col]):
        df[label_col] = df[label_col].astype("category").cat.codes

    # 丢掉缺失样本
    df = df[feature_cols + [label_col]].dropna()
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy()

    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)

    # 分类训练 SNG & 采样
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

    out = pd.DataFrame(Xsyn, columns=feature_cols)
    out[label_col] = ysyn
    out.to_csv(OUT_CSV, index=False)
    print(f">> Saved: {OUT_CSV} shape={out.shape}")

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
