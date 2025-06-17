import json, time, redis, pandas as pd, joblib
import numpy as np

def classify_buffer(buffer_df: pd.DataFrame, model, scaler):
    peak_idx = buffer_df['dyn_a'].idxmax()
    post = buffer_df.loc[peak_idx + 1:, 'dyn_a']
    pre  = buffer_df.loc[:peak_idx - 1, 'dyn_a']

    feat = {
        'max_dyn_a'     : buffer_df['dyn_a'].max(),
        'pre_impact_mean': pre.mean()  if not pre.empty  else 0,
        'pre_impact_var' : pre.var()   if not pre.empty  else 0,
        'post_imp_mean'  : post.mean() if not post.empty else 0,
        'post_imp_var'   : post.var()  if not post.empty else 0,
    }

    
    X = scaler.transform(pd.DataFrame([feat]))
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return pred, prob


model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

REDIS_DB  = 2
KEY       = "accel:stream"
r = redis.Redis(host="localhost", port=6379, db=REDIS_DB)

def buffer_df() -> pd.DataFrame:
    raw = r.lrange(KEY, 0, -1)
    return pd.DataFrame([json.loads(x) for x in raw]) if raw else pd.DataFrame()

MIN_SAMPLES = 20
SLEEP       = 0.5
seen_ready  = False

while True:
    df = buffer_df()
    n  = len(df)

    if n:
        last = df.iloc[-1]
        print(f"[stream] n={n:<3} tag={last['tag_id']} "
              f"t={last['t']:.3f} dyn_a={last['dyn_a']:.3f}")

    if not seen_ready and n >= MIN_SAMPLES:
        print(f"🟢 buffer now has ≥{MIN_SAMPLES} samples – classifier active.")
        seen_ready = True

    if n >= MIN_SAMPLES:
        pred, prob = classify_buffer(df, model, scaler)
        print(f"🧮  {'FALL DETECTED!' if pred else 'No fall.'}  prob={prob:.3f}")

    time.sleep(SLEEP)
