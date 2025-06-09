import struct
import time

def read_from_binary(file_path):
    with open(file_path, 'rb') as f:
        while chunk := f.read(struct.calcsize('i3dQ')):
            id, x, y, z, timestamp = struct.unpack('i3dQ', chunk)
            # readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
            print(f"ID: {id}, X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}, Timestamp: {timestamp}")


read_from_binary('./data.bin')
