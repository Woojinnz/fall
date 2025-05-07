from pathlib import Path
import re
import pandas as pd
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = Path("sisfall")

def load_sisfall_file(path: Path) -> pd.DataFrame:
    raw = path.read_text().replace(';','\n')
    return pd.read_csv(
        StringIO(raw),
        header=None,
        names=['AccX','AccY','AccZ','GyroX','GyroY','GyroZ','OriX','OriY','OriZ'],
        skip_blank_lines=True
    )

pat_fall = re.compile(r'^F\d{2}_(SE\d{2})_R\d{2}\.txt$')
pat_adl  = re.compile(r'^D\d{2}_(SE\d{2})_R\d{2}\.txt$')

dfs_fall = []
for subj_dir in DATA_DIR.iterdir():
    if not subj_dir.is_dir(): continue
    for f in subj_dir.glob("*.txt"):
        if pat_fall.match(f.name):
            df = load_sisfall_file(f)
            df['file'] = f.name
            dfs_fall.append(df)
falls = pd.concat(dfs_fall, ignore_index=True)

dfs_adl = []
for subj_dir in DATA_DIR.iterdir():
    if not subj_dir.is_dir(): continue
    for f in subj_dir.glob("*.txt"):
        if pat_adl.match(f.name):
            df = load_sisfall_file(f)
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
