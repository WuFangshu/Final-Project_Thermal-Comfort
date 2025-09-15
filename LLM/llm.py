"""
synthesize_person_numeric.py
LLM synthesis pipeline for pure numerical values + timestamps + labels
"""
import pandas as pd
import numpy as np
from scipy.stats import truncnorm
import requests
import json
import datetime
import os
from tqdm import tqdm
import re
# ========== Configuration ==========
INPUT_CSV  = "E:/thermal_project/LLMs/test.csv"
OUTPUT_CSV = "E:/thermal_project/LLMs/llm.csv"
SYN_ROWS   = 200          # number of rows does each synthesis produce
TIME_STEP  = 0.5         # second，Time interval

# ========== Read and extract key points==========
df = pd.read_csv(INPUT_CSV)
# Break the key points down into three floats
kpt_cols = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'LShoulder',
            'LElbow', 'REye', 'LEye', 'REar', 'LEar']
for col in kpt_cols:
    xyz = df[col].str.split('~', expand=True).astype(float)
    for i, suffix in enumerate(['x', 'y', 'z']):
        df[f"{col}_{suffix}"] = xyz[i]

# ========== Select pure numeric columns ==========
PERSON_KEYS = ['Age', 'Gender', 'Weight', 'Height', 'Bodyfat']
non_person = [c for c in df.columns if c not in PERSON_KEYS + ['Timestamp', 'file_name', 'Label',
                                                               'Emotion-Self', 'Emotion-ML','Sport-Last-Hour','Tiredness','Metabolic-Rate','Solar_Radiation']]
numeric_cols = [c for c in non_person if pd.api.types.is_numeric_dtype(df[c])]

# ========== LLM Call function ==========
from openai import OpenAI

# ========== Initialize Qwen3 Client ==========
client = OpenAI(
    api_key="sk-XvhaqKBt4xjeMlk6D0Ab257816B54c0dB86594F127726333",  # 🔑 你的 AiHubMix 秘钥
    base_url="https://aihubmix.com/v1",
)

# ========== LLM Call function ==========
def llm_label(row_dict: dict) -> str:
    # conversion to primitive type
    clean = {k: (v.item() if hasattr(v, 'item') else v) for k, v in row_dict.items()}
    prompt = (
        "Thermal comfort refers to the degree to which an individual perceives satisfaction with their environment "
        "based on thermal influences. It is estimated through a seven-point thermal sensation scale: "
        "“cold, cool, slightly cool, comfortable, slightly warm, warm, hot”. "
        "Based on the following human and environmental data, output the label most closely matching the perceived thermal comfort:\n"
        f"{json.dumps(clean, ensure_ascii=False)}\n"
        "Optional labels: cold, cool, slightly cool, comfortable, slightly warm, warm, hot\n"
        "Returns only one tag, without explanation"
    )

    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen3-30B-A3B",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0.3
        )
        resp = completion.choices[0].message.content.strip()
        return resp
    except Exception as e:
        print("LLM Request failed:", e)
        return "Neutral"


# ========== Sampling ==========
def sample_trunc(mean, std, low=None, high=None):
    if low is not None and high is not None and low == high:
            return mean
    std = max(std, 1e-6)  # avoid std <= 0
    a = (low - mean) / std if low is not None else None
    b = (high - mean) / std if high is not None else None
    return truncnorm.rvs(a, b, loc=mean, scale=std)

# ========== Generation ==========
all_rows = []
groups = df.groupby(PERSON_KEYS)

for person_vals, group in tqdm(groups, desc="Processing"):
    person_dict = dict(zip(PERSON_KEYS, person_vals))
    stats = {c: {"mean": group[c].mean(), "std": max(group[c].std(), 1e-3)} for c in numeric_cols}

    base_time = datetime.datetime.strptime(group.iloc[-1]['Timestamp'][:19], "%Y-%m-%d %H:%M:%S")
    for i in range(SYN_ROWS):
        ts = base_time + datetime.timedelta(seconds=i * TIME_STEP)
        row = person_dict.copy()
        row["Timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row["file_name"] = f"synth_{'_'.join(map(str, person_vals))}_{i:04d}.jpg"

        # Sampling value column
        for col in numeric_cols:
            mn, st = stats[col]["mean"], stats[col]["std"]
            lo, hi = group[col].min(), group[col].max()
            if lo == hi:        # Single-value interval → Directly assign the mean value
                row[col] = mn
            else:
                row[col] = sample_trunc(mn, st, lo, hi)

        # Placeholder category
        cat_range = {
            'Emotion-Self': df['Emotion-Self'].dropna().unique().tolist(),
            'Emotion-ML'  : df['Emotion-ML'].dropna().unique().tolist(),
            'Sport-Last-Hour'  : df['Sport-Last-Hour'].dropna().unique().tolist(),
            'Tiredness'  : df['Tiredness'].dropna().unique().tolist(),
            'Metabolic-Rate'  : df['Metabolic-Rate'].dropna().unique().tolist(),
            'Solar_Radiation'  : df['Solar_Radiation'].dropna().unique().tolist()
        }
        # Random Sampling
        for c in ['Emotion-Self', 'Emotion-ML','Sport-Last-Hour','Tiredness','Metabolic-Rate','Solar_Radiation']:
            row[c] = np.random.choice(cat_range[c])


        # Mapping
        thermo_map = {
            "cold": -3,
            "cool": -2,
            "slightly cool": -1,
            "comfortable": 0,
            "slightly warm": 1,
            "warm": 2,
            "hot": 3,
        }
        label_text = str(llm_label(row)).strip()
        # Replace the existing LLM labelling line
        row["Label"] = thermo_map.get(llm_label(row), None) 
        if row["Label"]!= None:
           all_rows.append(row)

# ========== Output ==========
out_df = pd.DataFrame(all_rows)
print(out_df.columns)
# Ensure that the column order remains consistent with the original
# Reassemble the key points x/y/z into the ~ form.
kpt_cols = ['Nose', 'Neck', 'RShoulder', 'RElbow',
            'LShoulder', 'LElbow', 'REye', 'LEye',
            'REar', 'LEar']          

for k in kpt_cols:
    # Assemble into a string array
    out_df[k] = (out_df[f"{k}_x"].astype(str) + "~" +
                 out_df[f"{k}_y"].astype(str) + "~" +
                 out_df[f"{k}_z"].astype(str))
    # Remove the three columns that have been split off
    out_df.drop(columns=[f"{k}_x", f"{k}_y", f"{k}_z"], inplace=True)

# Reorder according to the original sequence
df = pd.read_csv(INPUT_CSV)
out_df = out_df[df.columns]
out_df.to_csv(OUTPUT_CSV, index=False)
print(f" Generation complete！ {len(out_df)} rows → {OUTPUT_CSV}")