import time, json, pandas as pd, redis
from convert_acceleration import pos_to_accel

CSV_FILE   = "mock_positions.csv"
SLEEP_S    = 0.10
CLEAR_LIST = True

REDIS_DB   = 2
KEY        = "accel:stream"
MAX_LEN    = 60

r = redis.Redis(host="localhost", port=6379, db=REDIS_DB)
if CLEAR_LIST:
    r.delete(KEY)

def push_accel_sample(sample: dict):
    r.rpush(KEY, json.dumps(sample))
    r.ltrim(KEY, -MAX_LEN, -1)
    print(f"[push] tag={sample['tag_id']}  t={sample['t']:.3f} "
          f"dyn_a={sample['dyn_a']:.3f}")

df = pd.read_csv(CSV_FILE).sort_values(["pid", "t"])

for row in df.itertuples(index=False):
    acc = pos_to_accel(
        tag_id=int(row.pid),
        timestamp=row.t,
        x=row.x, y=row.y, z=row.z
    )
    if acc is not None:
        sample = {"tag_id": int(row.pid), "t": row.t, **acc}
        push_accel_sample(sample)        
    time.sleep(SLEEP_S)

print("Replay finished ✔")
