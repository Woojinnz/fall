# train_fall_model.py
from pathlib import Path
import re, pandas as pd, numpy as np, joblib
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

def load_file(p: Path):
    raw = p.read_text().replace(';', '\n')
    return pd.read_csv(StringIO(raw), header=None, names=[
        'AccX','AccY','AccZ','GyroX','GyroY','GyroZ','OriX','OriY','OriZ'
    ])

def downsample(df, orig_f=200):
    return df.groupby(df.index // orig_f).mean(numeric_only=True)

DATA_DIR = Path("sisfall")
pat_fall = re.compile(r'^F\d{2}_(S[A-Z]\d{2})_R\d{2}\.txt$')
pat_adl  = re.compile(r'^D\d{2}_(S[A-Z]\d{2})_R\d{2}\.txt$')

def gather(regex, label):
    out = []
    for sd in DATA_DIR.iterdir():
        if not sd.is_dir(): continue
        for f in sd.glob("*.txt"):
            if regex.match(f.name):
                df = downsample(load_file(f))
                df['file']  = f.name
                df['label'] = label
                out.append(df)
    return pd.concat(out, ignore_index=True)

falls = gather(pat_fall, 1)
adls  = gather(pat_adl,  0)

scale = (2*16.0)/(2**13)
for df in (falls, adls):
    for ax in 'XYZ':
        df[f'Acc{ax}_g'] = df[f'Acc{ax}']*scale
    df['res_a'] = np.linalg.norm(df[[f'Acc{a}_g' for a in 'XYZ']], axis=1)
    df['dyn_a'] = (df['res_a'] - 1).abs()

def extract(group):
    peak = group['dyn_a'].idxmax()
    pre  = group.loc[:peak-1,'dyn_a'];    post = group.loc[peak+1:,'dyn_a']
    return pd.Series({
        'max_dyn_a'      : group['dyn_a'].max(),
        'pre_impact_mean': pre.mean(),   'pre_impact_var': pre.var(),
        'post_imp_mean'  : post.mean(),  'post_imp_var'  : post.var()
    })

df_all   = pd.concat([falls, adls])
Xy       = df_all.groupby('file').apply(extract)
y        = df_all.groupby('file')['label'].first()
Xy['label'] = y

X_train, X_test, y_train, y_test = train_test_split(
    Xy.drop(columns='label'), Xy['label'], test_size=0.2, random_state=42)

X_train.describe()

scaler = StandardScaler().fit(X_train)
X_train_sc, X_test_sc = scaler.transform(X_train), scaler.transform(X_test)

model = XGBClassifier(scale_pos_weight=1.5, eval_metric='aucpr').fit(X_train_sc, y_train)

print(classification_report(y_test, model.predict(X_test_sc)))

joblib.dump(model,  "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Saved model.pkl and scaler.pkl")
