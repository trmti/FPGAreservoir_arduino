import os
import csv
import serial
import time

file = './test.csv'
ser = serial.Serial('/dev/cu.usbmodem11401', 1000000) #ポートの情報を記入
not_used = ser.readline()

if (os.path.exists(file)):
    os.remove(file)

with open(file, 'w') as f:
    writer = csv.writer(f)
    count = 0
    try:
        time.sleep(5)
        while count < 10000:
            val_arduino = str(ser.readline(), "ascii").strip().split(',",", ,')[0].split(",")
            print(val_arduino)
            writer.writerow([''.join(val_arduino[0].split(",")), ''.join(val_arduino[1].split(","))])
            count += 1
                
    except KeyboardInterrupt:
        f.close()
        ser.close()

    finally:
        f.close()
        ser.close()