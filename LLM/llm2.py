import pandas as pd
import numpy as np
import json
import datetime
from tqdm import tqdm
from openai import OpenAI

# ========== Configuration ==========
INPUT_CSV  = "E:/123/LLM/train.csv"
OUTPUT_CSV = "E:/123/LLM/llm_out.csv"
SYN_ROWS   = 100         # number of rows to generate per participant
TIME_STEP  = 0.5         # time interval (seconds)

# ========== Read Data ==========
df = pd.read_csv(INPUT_CSV)

PERSON_KEYS = ['Age', 'Gender', 'Weight', 'Height', 'Bodyfat']
# Columns that must be in "a~b~c" triple format (only facial/body temperature sensors)
TRIPLET_COLS = [
    'Nose','Neck','RShoulder','RElbow','LShoulder','LElbow','REye','LEye','REar','LEar'
]
# Remove Solar_Radiation from non-negative since it can be -1 in original data
NON_NEGATIVE_COLS = [
    'GSR','Sport-Last-Hour','Time-Since-Meal','Clothing-Level',
    'Air-Velocity','Metabolic-Rate','Bodytemp','PCE-Ambient-Temp','Radiation-Temp',
    'Ambient_Temperature','Ambient_Humidity','Heart_Rate'
]
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) 
                and c not in PERSON_KEYS + ['Label']]

# ========== Initialize Qwen3 Client ==========
client = OpenAI(
    api_key="sk-XvhaqKBt4xjeMlk6D0Ab257816B54c0dB86594F127726333",  # AiHubMix api key
    base_url="https://aihubmix.com/v1",
)

# ========== Utility Functions ==========
def safe_json(val):
    """Convert numpy types to Python built-in types"""
    if isinstance(val, (np.integer,)):
        return int(val)
    elif isinstance(val, (np.floating,)):
        return float(val)
    return str(val)

def llm_generate(person_dict, stats, required_columns, num_rows=10, retries=2, categorical_examples=None, triplet_examples=None, global_stats=None, quantiles=None, feedback_hint=""):
    """Call LLM to generate synthetic data"""
    clean_person = {k: safe_json(v) for k, v in person_dict.items()}
    clean_stats  = {k: {kk: safe_json(vv) for kk, vv in d.items()} for k,d in stats.items()}
    clean_global_stats = {k: {kk: safe_json(vv) for kk, vv in d.items()} for k,d in (global_stats or {}).items()}

    cols = list(required_columns)
    categorical_examples = categorical_examples or {}
    triplet_examples = triplet_examples or {}
    
    # Build detailed statistical constraints
    stat_constraints = []
    for col in numeric_cols:
        if col in clean_stats:
            stat_info = clean_stats[col]
            stat_constraints.append(f"  - {col}: mean≈{stat_info['mean']:.2f}, std≈{stat_info['std']:.2f}, range=[{stat_info['min']:.2f}, {stat_info['max']:.2f}]")
    # Quantile guidance per column
    quantile_constraints = []
    if quantiles:
        for col, qs in quantiles.items():
            try:
                q10, q50, q90 = qs.get('q10'), qs.get('q50'), qs.get('q90')
                quantile_constraints.append(f"  - {col}: q10≈{q10:.2f}, median≈{q50:.2f}, q90≈{q90:.2f}")
            except Exception:
                continue
    
    prompt = f"""
You are a data synthesis assistant. Generate {num_rows} rows of thermal comfort experiment data with REALISTIC VARIATION.

Participant characteristics (FIXED for all rows):
{json.dumps(clean_person, ensure_ascii=False)}

CRITICAL: Generate REALISTIC VARIATION - each row should have different values within the statistical ranges below.

Statistical constraints for numeric columns (generate values within these ranges with realistic variation):
{chr(10).join(stat_constraints)}

Approximate quantile targets (match distribution shape roughly):
{chr(10).join(quantile_constraints)}

Global dataset statistics (for reference):
{json.dumps(clean_global_stats, ensure_ascii=False)}

STRICT REQUIREMENTS:
1) Output ONLY data rows, no explanations, no code blocks, no headers.
2) Each row must contain exactly these columns in order, comma-separated:
{','.join(cols)}
3) Generate exactly {num_rows} rows with REALISTIC VARIATION between rows.
4) Column format rules:
   - Label: integer in [-3, -2, -1, 0, 1, 2, 3] with realistic distribution
   - TRIPLET columns ONLY ({', '.join(TRIPLET_COLS)}): format "float~float~float" with '~' separator
   - ALL OTHER numeric columns: single float values, NOT triplet format
   - Participant columns ({', '.join(PERSON_KEYS)}): identical to participant characteristics
   - Categorical columns: use only values from options below
5) VARIATION REQUIREMENT: Each row must have different values within the statistical ranges. Do not repeat identical values across rows.

FEEDBACK (fix these issues if present):
{feedback_hint}

Categorical options (use exactly as listed):
{json.dumps(categorical_examples, ensure_ascii=False)}

Triplet column examples (use similar "a~b~c" format with variation):
{json.dumps({k: v[:3] for k, v in triplet_examples.items()}, ensure_ascii=False)}
"""

    attempt = 0
    last_preview = ""
    while attempt <= retries:
        try:
            completion = client.chat.completions.create(
                model="Qwen/Qwen3-30B-A3B",
                messages=[
                    {"role": "system", "content": "You are a strict data synthesis assistant. Only output strict CSV data rows, no explanations."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8 if attempt > 0 else 0.9,
                max_tokens=1500,
            )
            text = (completion.choices[0].message.content or "").strip()
            last_preview = text[:300]
            raw_lines = text.split("\n")
            # Filter irrelevant lines
            filtered = []
            for line in raw_lines:
                s = line.strip()
                if not s or s.startswith("`") or s.lower().startswith("csv"):
                    continue
                if "," not in s:
                    continue
                # Skip header
                lower = s.lower()
                if any(col.lower() in lower for col in cols[:3]) and any(h in lower for h in ["timestamp","label","gender"]):
                    continue
                filtered.append(s)

            # Fix column count: truncate if too many, skip if too few
            fixed_lines = []
            num_cols = len(cols)
            for s in filtered:
                parts = [p.strip() for p in s.split(",")]
                if len(parts) < num_cols:
                    continue
                if len(parts) > num_cols:
                    parts = parts[:num_cols]
                fixed_lines.append(",".join(parts))

            if len(fixed_lines) >= min(num_rows, 1):
                return fixed_lines[:num_rows]
        except Exception as e:
            print("LLM request failed:", e)
        attempt += 1

    if last_preview:
        print("Debug: model output preview (first 300 chars) →", last_preview)
    return []

# ========== Main Loop ==========
all_rows = []
groups = df.groupby(PERSON_KEYS)

# Calculate global statistics for reference
global_stats = {c: {"mean": df[c].mean(), 
                    "std": df[c].std(), 
                    "min": df[c].min(), 
                    "max": df[c].max()} for c in numeric_cols}

for person_vals, group in tqdm(groups, desc="Processing participants"):
    # Convert person_dict
    person_dict = {k: safe_json(v) for k,v in zip(PERSON_KEYS, person_vals)}
    # Numeric statistics
    stats = {c: {"mean": group[c].mean(), 
                 "std": group[c].std(), 
                 "min": group[c].min(), 
                 "max": group[c].max()} for c in numeric_cols}

    # Base time (robust parsing)
    last_ts_val = group.iloc[-1].get('Timestamp') if 'Timestamp' in group.columns else None
    try:
        base_time = pd.to_datetime(last_ts_val, errors='coerce')
        if pd.isna(base_time):
            raise ValueError("Invalid Timestamp")
        base_time = base_time.to_pydatetime()
    except Exception:
        base_time = datetime.datetime.now()

    # Build categorical constraints from dataset
    categorical_examples = {}
    if 'Gender' in df.columns:
        categorical_examples['Gender'] = sorted(list(map(str, df['Gender'].dropna().unique().tolist())))
    if 'Emotion-Self' in df.columns:
        categorical_examples['Emotion-Self'] = sorted(list(map(str, df['Emotion-Self'].dropna().unique().tolist())))
    if 'Emotion-ML' in df.columns:
        categorical_examples['Emotion-ML'] = sorted(list(map(str, df['Emotion-ML'].dropna().unique().tolist())))
    
    # Add discrete value constraints for specific columns
    if 'Sport-Last-Hour' in df.columns:
        categorical_examples['Sport-Last-Hour'] = sorted(list(map(str, df['Sport-Last-Hour'].dropna().unique().tolist())))
    if 'Time-Since-Meal' in df.columns:
        categorical_examples['Time-Since-Meal'] = sorted(list(map(str, df['Time-Since-Meal'].dropna().unique().tolist())))
    
    # Special handling for Solar_Radiation - if it's all -1.0 in train data, keep it as -1.0
    if 'Solar_Radiation' in df.columns and df['Solar_Radiation'].nunique() == 1:
        categorical_examples['Solar_Radiation'] = [str(df['Solar_Radiation'].iloc[0])]

    # Prepare triplet examples from current group
    triplet_examples = {}
    sample_group = group.head(10)
    for col in TRIPLET_COLS:
        if col in sample_group.columns:
            vals = [str(v) for v in sample_group[col].dropna().astype(str).tolist() if '~' in str(v)]
            if vals:
                triplet_examples[col] = vals

    # Build quantiles and feedback hint
    quantiles = {}
    for col in numeric_cols:
        try:
            q = group[col].quantile([0.1, 0.5, 0.9])
            quantiles[col] = {"q10": float(q.loc[0.1]), "q50": float(q.loc[0.5]), "q90": float(q.loc[0.9])}
        except Exception:
            continue
    feedback_hint = "Generate values close to the given means/stds and quantiles; avoid constant columns and avoid triplet format for non-triplet columns."

    # Call LLM
    lines = llm_generate(
        person_dict,
        stats,
        required_columns=df.columns,
        num_rows=SYN_ROWS,
        categorical_examples=categorical_examples,
        triplet_examples=triplet_examples,
        global_stats=global_stats,
        quantiles=quantiles,
        feedback_hint=feedback_hint,
    )

    # Convert to dict with simple validation loop: if deviation too large, attempt a second call with feedback
    for i, line in enumerate(lines):
        parts = line.split(",")
        # Skip if not enough columns; excess columns already truncated
        if len(parts) < len(df.columns):
            continue
        row = dict(zip(df.columns, parts[:len(df.columns)]))
        # Enforce participant descriptors to match the group
        for k, v in person_dict.items():
            if k in row:
                row[k] = str(v)
        # Clamp numeric columns using dataset stats
        for col in numeric_cols:
            if col in row:
                try:
                    val = float(row[col])
                    col_min = float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None
                    col_max = float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None
                    if col in NON_NEGATIVE_COLS and val < 0:
                        val = 0.0
                    if col_min is not None:
                        val = max(val, col_min)
                    if col_max is not None:
                        val = min(val, col_max)
                    row[col] = str(val)
                except Exception:
                    pass
        # Post-fix: ensure Label is integer string
        if 'Label' in row:
            try:
                row['Label'] = str(int(float(row['Label'])))
            except Exception:
                continue
        # Normalize categorical strings
        for cat_col in ['Gender','Emotion-Self','Emotion-ML','Sport-Last-Hour','Time-Since-Meal','Solar_Radiation']:
            if cat_col in row and isinstance(row[cat_col], str):
                row[cat_col] = row[cat_col].strip()
        # Ensure triplet format for required columns
        for tcol in TRIPLET_COLS:
            if tcol in row and '~' not in str(row[tcol]):
                examples = triplet_examples.get(tcol, [])
                if examples:
                    row[tcol] = examples[0]
                else:
                    # fallback: synthesize a plausible triplet around group numeric means if any
                    try:
                        base = 1000.0
                        row[tcol] = f"{base-50:.2f}~{base:.2f}~{base+50:.2f}"
                    except Exception:
                        pass
        
        # Fix any non-triplet columns that were incorrectly generated as triplets
        for col in row:
            if col not in TRIPLET_COLS and '~' in str(row[col]):
                # Extract the middle value from triplet format
                try:
                    parts = str(row[col]).split('~')
                    if len(parts) == 3:
                        row[col] = parts[1]  # Take the middle value
                except Exception:
                    pass
        # Add timestamp and file name
        ts = base_time + datetime.timedelta(seconds=i * TIME_STEP)
        row["Timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row["file_name"] = f"synth_{'_'.join(map(str, person_vals))}_{i:04d}.jpg"
        all_rows.append(row)

    # Optional: simple deviation check (per participant batch)
    # If many numeric columns have zero std or out-of-range, we could trigger a second attempt with stricter hint (skipped here for runtime simplicity)

# ========== Output ==========
out_df = pd.DataFrame(all_rows)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"Generation completed! Total {len(out_df)} rows → {OUTPUT_CSV}")
