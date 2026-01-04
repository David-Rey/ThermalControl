import serial
import csv
import time

PORT = 'COM5'
BAUD = 115200
FILE_NAME = "peltier_data.csv"
MAX_LINES = 100000

# Use 'with' to ensure the port closes even if the script crashes
try:
    with serial.Serial(PORT, BAUD, timeout=1) as ser, \
            open(FILE_NAME, 'w', newline='') as f:

        writer = csv.writer(f)
        # Write Header
        writer.writerow(["Timestamp", "Type", "Setpoint", "Current", "PWM"])

        print(f"Connected to {PORT}. Logging {MAX_LINES} lines...")

        count = 0
        while count < MAX_LINES:
            if ser.in_waiting > 0:
                try:
                    # Read and decode
                    line = ser.readline().decode('utf-8', errors='replace').strip()
                    print(line)

                    # Split CSV data (assuming "%.2f,%.2f,%d" from Teensy)
                    parts = line.split(',')

                    if len(parts) in [3, 4]:
                        # Add a timestamp from the PC side for analysis
                        row = [time.time()] + parts
                        writer.writerow(row)

                        count += 1
                        if count % 100 == 0:
                            print(f"Captured {count}/{MAX_LINES} lines...")

                except Exception as e:
                    print(f"Error parsing line: {e}")
                    continue

    print(f"Done! Data saved to {FILE_NAME}")

except serial.SerialException as e:
    print(f"Could not open {PORT}. Is the Teensy plugged in or Serial Monitor open?")