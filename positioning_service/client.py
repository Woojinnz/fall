import socket
import time

UDP_IP = "0.0.0.0"
UDP_PORT = 8887

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

messages = [
    '{"Command":"UpLink","AnchorID":0,"TagID":0,"Distance":0}',
    '{"Command":"UpLink","AnchorID":1,"TagID":0,"Distance":6}',
    '{"Command":"UpLink","AnchorID":2,"TagID":0,"Distance":7.1421}',
    '{"Command":"UpLink","AnchorID":3,"TagID":0,"Distance":10}',

    '{"Command":"UpLink","AnchorID":0,"TagID":0,"Distance":5}',
    '{"Command":"UpLink","AnchorID":1,"TagID":0,"Distance":5}',
    '{"Command":"UpLink","AnchorID":2,"TagID":0,"Distance":9.1803}',
    '{"Command":"UpLink","AnchorID":3,"TagID":0,"Distance":2.1803}',

    '{"Command":"UpLink","AnchorID":0,"TagID":0,"Distance":7.07107}',
    '{"Command":"UpLink","AnchorID":1,"TagID":0,"Distance":7.07107}',
    '{"Command":"UpLink","AnchorID":2,"TagID":0,"Distance":7.07107}',
    '{"Command":"UpLink","AnchorID":3,"TagID":0,"Distance":7.07107}',

    '{"Command":"UpLink","AnchorID":0,"TagID":0,"Distance":7.07107}',
    '{"Command":"UpLink","AnchorID":1,"TagID":0,"Distance":7.07107}',
    '{"Command":"UpLink","AnchorID":2,"TagID":0,"Distance":7.07107}',
    '{"Command":"UpLink","AnchorID":3,"TagID":0,"Distance":7.07107}',

    '{"Command":"UpLink","AnchorID":0,"TagID":1,"Distance":7.18}',
    '{"Command":"UpLink","AnchorID":1,"TagID":1,"Distance":9.18}',
    '{"Command":"UpLink","AnchorID":2,"TagID":1,"Distance":5.00}',
    '{"Command":"UpLink","AnchorID":3,"TagID":1,"Distance":5.00}',

    '{"Command":"UpLink","AnchorID":0,"TagID":1,"Distance":7.07107}',
    '{"Command":"UpLink","AnchorID":1,"TagID":1,"Distance":7.07107}',
    '{"Command":"UpLink","AnchorID":2,"TagID":1,"Distance":7.07107}',
    '{"Command":"UpLink","AnchorID":3,"TagID":1,"Distance":7.07107}',

    '{"Command":"UpLink","AnchorID":0,"TagID":1,"Distance":7.07107}',
    '{"Command":"UpLink","AnchorID":1,"TagID":1,"Distance":7.07107}',
    '{"Command":"UpLink","AnchorID":2,"TagID":1,"Distance":7.07107}',
    '{"Command":"UpLink","AnchorID":3,"TagID":1,"Distance":7.07107}',

    '{"Command":"UpLink","AnchorID":0,"TagID":1,"Distance":2.18}',
    '{"Command":"UpLink","AnchorID":1,"TagID":1,"Distance":10.18}',
    '{"Command":"UpLink","AnchorID":2,"TagID":1,"Distance":12.0}',
    '{"Command":"UpLink","AnchorID":3,"TagID":1,"Distance":3.97}',
]

try:
    while True:
        for message in messages:
            sock.sendto(message.encode('utf-8'), (UDP_IP, UDP_PORT))
            print(f"Sent: {message}")
            time.sleep(0.5)
except KeyboardInterrupt:
    print("\nClient stopped.")
    sock.close()
