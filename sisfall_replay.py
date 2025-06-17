
import json, time, redis, numpy as np, pandas as pd
from pathlib import Path
from io import StringIO

SISFALL_FILE = Path("sisfall/SA01/F03_SA01_R01.txt")  
REDIS_DB     = 2
KEY          = "accel:stream"
PLAYBACK_HZ  = 10                                    
MAX_LEN      = 60                                   

r = redis.Redis(host="localhost", port=6379, db=REDIS_DB)
r.delete(KEY)

raw_txt = SISFALL_FILE.read_text().replace(";", "\n")
df = pd.read_csv(StringIO(raw_txt), header=None,
                 names=["AccX","AccY","AccZ",
                        "GyroX","GyroY","GyroZ",
                        "OriX","OriY","OriZ"])

factor  = int(200 / PLAYBACK_HZ)            
df      = df.groupby(df.index // factor).mean(numeric_only=True).reset_index(drop=True)

scale = (2*16.0) / (2**13)                 
for ax in "XYZ":
    df[f"Acc{ax}_g"] = df[f"Acc{ax}"] * scale

df["res_a"] = np.linalg.norm(df[[f"Acc{a}_g" for a in "XYZ"]], axis=1)
df["dyn_a"] = (df["res_a"] - 1.0).abs()    

for i, row in df.iterrows():
    payload = {
        "tag_id": 1,
        "t"     : round(i / PLAYBACK_HZ, 3),
        "AccX_g": row.AccX_g,
        "AccY_g": row.AccY_g,
        "AccZ_g": row.AccZ_g,
        "dyn_a" : row.dyn_a
    }
    r.rpush(KEY, json.dumps(payload))
    r.ltrim(KEY, -MAX_LEN, -1)
    print(f"[push] t={payload['t']:5.2f}s  dyn_a={payload['dyn_a']:.2f}")
    time.sleep(1 / PLAYBACK_HZ)
