import json
import socket
import threading
import ctypes
from flask import Flask, jsonify
import queue
import redis
import datetime
from positioning import TagPositioning
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from convert_acceleration import pos_to_accel

app = Flask(__name__)

redis_client = redis.Redis(host='localhost', port=6379, db=0)
redis_client.flushdb()
redis_client1 = redis.Redis(host='localhost', port=6379, db=1)
redis_client1.flushdb()
redis_queue = queue.Queue()

class UWBMsg(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double),
                ("y", ctypes.c_double),
                ("z", ctypes.c_double)]

anchorArray = (UWBMsg * 8)()
active_tags = {}

# Set the UDP_IP to 0.0.0.0 when connecting to the actual UWB devices
UDP_IP = "0.0.0.0"
UDP_PORT = 8887

DATA_PATH = "data.bin"
BATCH_SIZE = 10
MAX_QUEUE_SIZE = 1000

buffer = []

# Add a lock for thread-safe access to active_tags
active_tags_lock = Lock()

def set_anchor(anchor_id, x, y, z=2.45):
    anchorArray[anchor_id].x = x
    anchorArray[anchor_id].y = y
    anchorArray[anchor_id].z = z

def sync_anchor(data):
    anchor_data = data.get("anchor_info")
    map_size = data.get("map_size")

    for anchor in anchor_data:
        anchor_id = int(anchor.get("name")[1:]) - 1  # name is in the format "A1", "A2", etc. Starting from 1, need to subtract 1 to get the index
        x = float(anchor.get("x") / map_size.get("image_width") * map_size.get("room_width"))
        y = float(anchor.get("y") / map_size.get("image_height") * map_size.get("room_height"))
        set_anchor(anchor_id, x, y)
    return jsonify({"message": "Anchors successfully synced"}), 200

def sync_anchor_starter():
    with app.app_context():
        while True:
            try:
                mapIds = redis_client1.keys()

                for mapId in mapIds:
                    map_id_str = mapId.decode('utf-8')
                    data = redis_client1.get(mapId)

                    if data:
                        data = json.loads(data.decode('utf-8'))
                        sync_anchor(data)
                time.sleep(1)
            except Exception as e:
                print(f"Error syncing anchors: {e}")


def save_to_binary(file_path, data):
    with open(file_path, 'ab') as f:
        f.write(data)

# ***DEPRECATED***
@app.route('/get_latest_position/', methods=['GET'])
def get_latest_position():
    global latest_position, active_tags
    tag_pos = {}
    c_time = datetime.datetime.now()
    for tag_id, tag in active_tags.items():
        tag_pos[tag_id] = [tag.current_location["x"], tag.current_location["y"], tag.current_location["z"], c_time]
    return jsonify(tag_pos), 200


def data_writer():
    last_saved_time = {}
    global buffer
    
    while True:
        try:
            if redis_queue.qsize() > MAX_QUEUE_SIZE:
                with redis_queue.mutex:
                    redis_queue.queue.clear()
                continue

            tag_id, position = redis_queue.get()
            redis_client.set(tag_id, json.dumps(position))
            
            current_time = time.time()
            
            if tag_id not in last_saved_time or (current_time - last_saved_time[tag_id] >= 60):
                timestamp = int(current_time)
                packed_data = struct.pack('i3dQ', int(tag_id), position["x"], position["y"], position["z"], timestamp)
                buffer.append(packed_data)
                last_saved_time[tag_id] = current_time

                if len(buffer) >= BATCH_SIZE:
                    save_to_binary(DATA_PATH, b''.join(buffer))
                    buffer = []

            redis_queue.task_done()
            
        except Exception as e:
            print(f"Error writing to Redis or saving data: {e}")

def process_tag_data(tag_data, anchorArray):
    try:
        tag_id = tag_data.get("TagID")
        with active_tags_lock:
            if tag_id not in active_tags:
                current_tag = TagPositioning(tag_id)
                latest_position = current_tag.positioning_service(tag_data, anchorArray)
                active_tags[tag_id] = current_tag
            else:
                current_tag = active_tags[tag_id]
                latest_position = current_tag.positioning_service(tag_data, anchorArray)

        redis_queue.put((tag_id, current_tag.current_location))
        print(f"Latest position for {tag_id}: {current_tag.current_location}")
    except Exception as e:
        print(f"Error processing tag data: {e}")


def udp_server():
    global anchorArray, active_tags
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    # Create a thread pool for processing tag data
    with ThreadPoolExecutor(max_workers=10) as executor:
        while True:
            try:
                data, _ = sock.recvfrom(1024)
                raw_data = data.decode('utf-8')
                tag_data = json.loads(raw_data)
                
                if tag_data["Command"] != "UpLink":
                    continue
                
                # Submit the tag processing task to the thread pool
                executor.submit(process_tag_data, tag_data, anchorArray)
                
            except Exception as e:
                print(f"Error in UDP server: {e}")

def broadcast_time_sync():
    broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    
    # Use the standard broadcast address
    broadcast_address = '192.168.1.255'
    
    while True:
        try:
            # Get current timestamp in microseconds
            current_time = int(time.time() * 1000000)
            
            # Create time sync message
            sync_message = {
                "Command": "TimeSync",
                "TimeStamp": current_time
            }
            
            # Convert to JSON and send broadcast
            message = json.dumps(sync_message).encode('utf-8')
            broadcast_sock.sendto(message, (broadcast_address, 54321))
            
            print(f"Broadcasting time sync message: {sync_message}")
            
            # Sleep for 5 seconds before next sync
            time.sleep(5)
        except Exception as e:
            print(f"Error in time sync broadcast: {e}")
            time.sleep(5)  # Still sleep on error to prevent rapid retries

threading.Thread(target=data_writer, daemon=True).start()

threading.Thread(target=udp_server, daemon=True).start()

threading.Thread(target=sync_anchor_starter, daemon=True).start()

# Add time sync broadcast thread
threading.Thread(target=broadcast_time_sync, daemon=True).start()

if __name__ == '__main__':
    app.run(port=8886)
