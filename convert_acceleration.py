from collections import defaultdict, deque
import numpy as np

G = 9.81                           
HISTORY = defaultdict(lambda: deque(maxlen=3))

def pos_to_accel(tag_id: int, timestamp: float, x: float, y: float, z: float):

    H = HISTORY[tag_id]
    H.append((timestamp, np.array([x, y, z], dtype=float)))

    if len(H) < 3:                       
        return None

    (t0, p0), (t1, p1), (t2, p2) = H
    dt1, dt2 = t1 - t0, t2 - t1
    if dt1 == 0 or dt2 == 0:           
        return None

    v0 = (p1 - p0) / dt1
    v1 = (p2 - p1) / dt2
    a  = (v1 - v0) / dt2                

    ax_g, ay_g, az_g = a / G           
    res_a = np.linalg.norm([ax_g, ay_g, az_g])
    dyn_a = abs(res_a - 1.0) 

    return {
        "AccX_g": ax_g,
        "AccY_g": ay_g,
        "AccZ_g": az_g,
        "dyn_a":  dyn_a,
    }
