import time
import serial

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
ser.flush() 