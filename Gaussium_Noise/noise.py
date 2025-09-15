# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import gc

# src = 'E:/thermal_project/tsaug/test.csv'
# dst = 'E:/thermal_project/tsaug/noise1.csv'

# num_cols = [
#     'Weight','Height','Bodyfat','Bodytemp','Clothing-Level',
#     'Radiation-Temp','PCE-Ambient-Temp','Air-Velocity',
#     'Wrist_Skin_Temperature','Heart_Rate','GSR',
#     'Ambient_Temperature','Ambient_Humidity','Solar_Radiation'
# ]


# scaler = StandardScaler()
# for chunk in pd.read_csv(src, usecols=num_cols, chunksize=10_000, dtype=float):
#     scaler.partial_fit(chunk.values)


# first = True
# with open(dst, 'w', newline='', encoding='utf-8') as f_out:   # 关键：newline=''
#     for chunk in pd.read_csv(src, chunksize=10_000):
#         x = chunk[num_cols].astype(float).values
#         x_norm  = scaler.transform(x)
#         x_noise = x_norm + np.random.normal(0, 0.02, x.shape)
#         x_clip  = np.clip(
#             scaler.inverse_transform(x_noise),
#             chunk[num_cols].min(),
#             chunk[num_cols].max()
#         )
#         chunk[num_cols] = x_clip
#         # Key: lineterminator='\n'
#         chunk.to_csv(f_out, header=first, index=False, lineterminator='\n')
#         first = False

#         del chunk, x, x_norm, x_noise, x_clip
#         gc.collect()

# print("Done — noise.csv generated.")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import gc

src = 'E:/thermal_project/tsaug/test.csv'
dst = 'E:/thermal_project/tsaug/noise.csv'

num_cols = ['Age', 'Weight', 'Height', 'Bodyfat', 'Bodytemp', 'Clothing-Level',
            'Radiation-Temp', 'PCE-Ambient-Temp', 'Air-Velocity',
            'Wrist_Skin_Temperature', 'Heart_Rate', 'GSR',
            'Ambient_Temperature', 'Ambient_Humidity', 'Solar_Radiation']

# global mean variance
scaler = StandardScaler()
for chunk in pd.read_csv(src, usecols=num_cols, chunksize=10_000, dtype=float): 
#Read large files in chunks (processing 10,000 rows at a time) to conserve memory.Select only the specified num_cols numerical columns, forcing them to be read as float type.
    scaler.partial_fit(chunk.values) #StandardScaler incrementally update the mean and variance across each chunk.

# Enhance block by block, retaining only ‘changed’ rows
header = pd.read_csv(src, nrows=0).columns   # Complete listing
first_out = True #When writing CSV files in chunks, the header is only output during the initial write operation, avoiding duplicate column names.

with open(dst, 'w') as f_out:
    for chunk in pd.read_csv(src, chunksize=10_000):
        # Original values
        x_orig = chunk[num_cols].astype(float).values #Extract the specified num_cols numeric columns from the current data block. Cast to float type.

        # Add noise
        x_norm  = scaler.transform(x_orig) #Standardisation
        x_noise = x_norm + np.random.normal(0, 0.02, x_norm.shape) # Add Gaussian white noise with a standard deviation of 0.02 to the standardised data.
        x_clip  = np.clip(scaler.inverse_transform(x_noise), #Reverse standardisation to restore original dimensions.      
                          chunk[num_cols].min(), chunk[num_cols].max()) #Crop the results to fall within the range of the block's original minimum and maximum values, preventing noise from generating unreasonable extremes.

        # Find rows that differ from the original
        mask_changed = ~np.all(np.isclose(x_clip, x_orig, rtol=1e-5, atol=1e-8), axis=1)

        # This block has changed rows
        if mask_changed.any():                       
            out_df = chunk.loc[mask_changed].copy()
            out_df[num_cols] = x_clip[mask_changed]
            out_df.to_csv(f_out, header=first_out, index=False)
            first_out = False

        #Clear variables to release memory
        del chunk, x_orig, x_norm, x_noise, x_clip, mask_changed
        gc.collect()

print("Done — Retained only changed rows, original rows discarded.")