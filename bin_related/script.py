import numpy as np
from pathlib import Path

raw = Path("data.bin").read_bytes()        
arr = np.frombuffer(raw, dtype=np.uint8)     
print(arr[:64])                              
arr16 = np.frombuffer(raw, dtype='<i2')     
arr16 = arr16[16:]                          
records = arr16.reshape(-1, 8)              
print(records[:5])


import pandas as pd
pd.DataFrame(records).to_csv("raw_imu_data.csv", index=False)
