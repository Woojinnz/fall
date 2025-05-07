import struct
import pandas as pd

PACKET_SIZE = 40
FMT_HEADER  = "<I4sI4s"
FMT_PAYLOAD = "<ddd"

ts, xs, ys, zs, pids = [], [], [], [], []
with open("data.bin","rb") as f:
    f.seek(32)
    while (chunk := f.read(PACKET_SIZE)):
        if len(chunk)!=PACKET_SIZE: break
        t,_,pid,_ = struct.unpack(FMT_HEADER, chunk[:16])
        ax,ay,az = struct.unpack(FMT_PAYLOAD, chunk[16:])
        ts.append(t); xs.append(ax); ys.append(ay); zs.append(az); pids.append(pid)

df = pd.DataFrame({
    "t": ts,
    "ax": xs,
    "ay": ys,
    "az": zs,
    "pid": pids
})

print(df)

# df["dt_ms"] = df["t"].diff().fillna(0)

# print("Packet-ID counts:", Counter(df["pid"]))

# plt.figure(figsize=(6,3))
# plt.hist(df["dt_ms"], bins=100, log=True)
# plt.xlabel("Δt (ms)")
# plt.ylabel("Count")
# plt.title("Inter‐sample time distribution")
# plt.show()

# fast = df[df["dt_ms"] <= 100].copy()
# print(f"\nHigh-rate samples (dt_ms ≤100): {len(fast)} rows")

# idx = fast.index.to_numpy()
# breaks = np.where(np.diff(idx) > 1)[0] + 1
# segs = np.split(idx, breaks)
# print(f"\nFound {len(segs)} high‐rate bursts; lengths:")
# for seg in segs:
#     print(f"  • Burst of {len(seg)} samples  (dt_ms range {fast['dt_ms'].iloc[seg].min():.0f}–{fast['dt_ms'].iloc[seg].max():.0f} ms)")
