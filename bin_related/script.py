import numpy as np
from pathlib import Path

raw = Path("data.bin").read_bytes()          # raw bytes
arr = np.frombuffer(raw, dtype=np.uint8)     # one big vector of bytes
print(arr[:64])                              # first 64 bytes
arr16 = np.frombuffer(raw, dtype='<i2')      # little‑endian int16
arr16 = arr16[16:]                           # skip 32‑byte header you saw
records = arr16.reshape(-1, 8)               # each row = 8 int16s
print(records[:5])


import pandas as pd
pd.DataFrame(records).to_csv("raw_imu_data.csv", index=False)
