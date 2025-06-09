# Positioning Service

This service provides real-time positioning functionality for UWB (Ultra-Wideband) tags using trilateration. It processes distance measurements from anchors to calculate tag positions and stores the position data for later analysis.

## Prerequisites

- Python 3.x
- Redis server
- Required Python packages:
  - flask
  - redis
  - ctypes (built-in)
  - struct (built-in)
  - socket (built-in)
  - threading (built-in)

## Installation

1. Install Redis server:
   ```bash
   # For macOS
   brew install redis
   
   # For Ubuntu/Debian
   sudo apt-get install redis-server
   ```

2. Install Python dependencies:
   ```bash
   pip install flask redis
   ```

3. Make sure the `trilateration.so` library is in the same directory as the Python files.

## Components

- `main.py`: The main positioning service that:
  - Listens for UDP messages from tags
  - Processes tag positions using trilateration
  - Stores position data in Redis and binary file
  - Broadcasts time sync messages to anchors
  - Provides a web API for position queries

- `client.py`: A test client that simulates tag messages
  - Sends sample distance measurements to the positioning service
  - Useful for testing the system without actual UWB hardware

- `positioning.py`: Contains the trilateration logic
  - Uses C library (`trilateration.so`) for position calculations
  - Maintains tag state and processes distance measurements

- `visualize_bin.py`: Tool to read and display stored position data
  - Reads the binary data file
  - Displays tag positions with timestamps

## Usage

1. Start Redis server:
   ```bash
   redis-server
   ```

2. Start the positioning service:
   ```bash
   python main.py
   ```
   The service will:
   - Start listening on UDP port 8887 for tag messages
   - Begin broadcasting time sync messages on port 54321
   - Start the web server on port 8886

3. (Optional) Run the test client to simulate tag messages:
   ```bash
   python client.py
   ```
   This will send sample distance measurements to test the positioning service.

4. View stored position data:
   ```bash
   python visualize_bin.py
   ```
   This will display the stored tag positions with their timestamps.

## Data Format

### Tag Messages (UDP)
```json
{
    "Command": "UpLink",
    "AnchorID": 0,
    "TagID": 0,
    "Distance": 7.07107
}
```

### Time Sync Messages (UDP Broadcast)
```json
{
    "Command": "TimeSync",
    "TimeStamp": 1234567891234000
}
```

### Position Data (Binary File)
The binary file (`data.bin`) stores position data in the following format:
- Tag ID (integer)
- X coordinate (double)
- Y coordinate (double)
- Z coordinate (double)
- Timestamp (unsigned long long)

## API Endpoints

- `GET /get_latest_position/`: Returns the latest position of all active tags
  - Response format: `{tag_id: [x, y, z, timestamp]}`

## Configuration

Key parameters in `main.py`:
- `UDP_PORT`: Port for receiving tag messages (default: 8887)
- `BATCH_SIZE`: Number of positions to buffer before writing to file (default: 10)
- `MAX_QUEUE_SIZE`: Maximum size of the Redis queue (default: 1000)

## Troubleshooting

1. If the positioning service can't connect to Redis:
   - Ensure Redis server is running
   - Check Redis connection settings in `main.py`

2. If no position data is being saved:
   - Check if the `data.bin` file is being created
   - Verify that tags are sending messages to the correct UDP port
   - Check the console output for any error messages

3. If positions are inaccurate:
   - Verify anchor positions are correctly set
   - Check if all required anchors are sending distance measurements
   - Ensure the `trilateration.so` library is properly compiled for your system

## Notes

- The service uses a thread pool to handle multiple tags concurrently
- Position data is stored both in Redis (for real-time access) and binary file (for historical analysis)
- Time sync messages are broadcast every 5 seconds to keep anchors synchronized
- The system supports up to 8 anchors and multiple tags 