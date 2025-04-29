import time
import serial
import threading


class MIC_array:
    def __init__(self,COM):
        self.ser_ = serial.Serial(port=COM, baudrate=115200)#, parity=serial.PARITY_ODD,stopbits=serial.STOPBITS_TWO,bytesize=serial.SEVENBITS)
        thread = threading.Thread(target=self.reader)
        thread.start()
        self.now_pos = -1
        print("voice direction detector:", self.ser_.isOpen)

    def reader(self):
        if self.ser_.in_waiting > 0:
            data  = self.ser_.readline()
            datastr = data.decode('utf-8')
            self.now_pos = float(datastr.strip())
            print(self.now_pos)
            time.sleep(0.01)

    def get_pos(self):
        return self.now_pos




if __name__ == "__main__":
    mic = MIC_array("COM5")
    while True:
        mic.reader()
        print(mic.get_pos())
        time.sleep(1)