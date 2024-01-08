import serial
import time

Obj = serial.Serial("/dev/cu.usbmodem1101")
Obj.Baudrate = 9600
Obj.Bytesize = 8
Obj.parity = 'N'
Obj.stopbits = 1

time.sleep(3)

Obj.write(b"3")

