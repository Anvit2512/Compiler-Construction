import serial
import matplotlib.pyplot as plt
from collections import deque

# adjust COM port and baud rate
ser = serial.Serial('COM7', 9600)  
window = 200  # number of samples to show
data = deque([0]*window, maxlen=window)

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(data)
ax.set_ylim(0, 1023)   # EMG range from Arduino 10-bit ADC
ax.set_title("MyoWare EMG Signal")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")

while True:
    try:
        value = int(ser.readline().decode().strip())
        data.append(value)
        line.set_ydata(data)
        ax.set_xlim(0, window)
        plt.pause(0.01)
    except:
        pass
