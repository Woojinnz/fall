from pathlib import Path
import re
import pandas as pd
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

#1st column is the acceleration data in the X axis measured by the sensor ADXL345.
#2nd column is the acceleration data in the Y axis measured by the sensor ADXL345.
#3rd column is the acceleration data in the Z axis measured by the sensor ADXL345.

#4th column is the rotation data in the X axis measured by the sensor ITG3200.
#5th column is the rotation data in the Y axis measured by the sensor ITG3200.
#6th column is the rotation data in the Z axis measured by the sensor ITG3200.

#7th column is the acceleration data in the X axis measured by the sensor MMA8451Q.
#8th column is the acceleration data in the Y axis measured by the sensor MMA8451Q.
#9th column is the acceleration data in the Z axis measured by the sensor MMA8451Q.

#Data are in bits with the following characteristics:

#ADXL345:
#Resolution: 13 bits
#Range: +-16g

#ITG3200
#Resolution: 16 bits
#Range: +-2000?s

#MMA8451Q:
#Resolution: 14 bits
#Range: +-8g


#In order to convert the acceleration data (AD) given in bits into gravity, use this equation:

#Acceleration [g]: [(2*Range)/(2^Resolution)]*AD

DATA_DIR = Path("sisfall")

# still gotta add calculation of acceleration
# length =  time / frequency
# 60s / 1hz
max_seconds = 60
buffer = deque(maxlen=max_seconds)


def update_buffer(new_point):
    buffer.append(new_point)


def get_buffer_as_dataframe():
    return pd.DataFrame(buffer)


def load_sisfall_file(path: Path) -> pd.DataFrame:
    raw = path.read_text().replace(';','\n')
    return pd.read_csv(
        StringIO(raw),
        header=None,
        names=['AccX','AccY','AccZ','GyroX','GyroY','GyroZ','OriX','OriY','OriZ'],
        skip_blank_lines=True
    )


def downsample_to_1hz(df, original_freq=200):
    # Group every `original_freq` rows (since 200Hz → 1Hz = 200:1 downsampling)
    downsampled = (
        df.groupby(df.index // original_freq)
        .mean(numeric_only=True)  # Only average numeric columns
    )
    return downsampled

pat_fall = re.compile(r'^F\d{2}_(SE\d{2})_R\d{2}\.txt$')
pat_adl  = re.compile(r'^D\d{2}_(SE\d{2})_R\d{2}\.txt$')

dfs_fall = []
for subj_dir in DATA_DIR.iterdir():
    if not subj_dir.is_dir(): continue
    for f in subj_dir.glob("*.txt"):
        if pat_fall.match(f.name):
            df = load_sisfall_file(f)
            df = downsample_to_1hz(df)
            df['file'] = f.name
            dfs_fall.append(df)
falls = pd.concat(dfs_fall, ignore_index=True)

dfs_adl = []
for subj_dir in DATA_DIR.iterdir():
    if not subj_dir.is_dir(): continue
    for f in subj_dir.glob("*.txt"):
        if pat_adl.match(f.name):
            df = load_sisfall_file(f)
            df = downsample_to_1hz(df)
            df['file'] = f.name
            dfs_adl.append(df)
adls = pd.concat(dfs_adl, ignore_index=True)

range_g, res_bits = 16.0, 13
scale = (2*range_g)/(2**res_bits)
for df in (falls, adls):
    for ax in ('X','Y','Z'):
        df[f'Acc{ax}_g'] = df[f'Acc{ax}'] * scale
    df['res_a'] = np.sqrt(df['AccX_g']**2 +
                         df['AccY_g']**2 +
                         df['AccZ_g']**2)
    df['dyn_a'] = (df['res_a'] - 1.0).abs()

fall_peaks = falls .groupby('file')['dyn_a'].max()
adl_peaks  = adls  .groupby('file')['dyn_a'].max()

print("Fall peaks (g):")
print(fall_peaks.describe())

print("\nADL peaks (g):")
print(adl_peaks.describe())


peak_idxs = falls.groupby('file')['dyn_a'].idxmax()
impact_rows = falls.loc[peak_idxs, ['file','AccX_g','AccY_g','AccZ_g','dyn_a']]
print(impact_rows.head())

trial = falls[falls.file=='F05_SE06_R04.txt']
plt.plot(trial['dyn_a'])
plt.xlabel('Sample Index')
plt.ylabel('dyn_a (g)')
plt.title('Fall Trial dyn_a Trace')
plt.show()
