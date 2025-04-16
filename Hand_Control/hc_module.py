import time
import serial

"""
GLOBAL VARIABLES
"""

PORT = 'COM4'
HAND_ID = 2
BAUDRATE = 115200

class Hand:

    def __init__(self, hand_id=HAND_ID, port=PORT, baudrate=BAUDRATE):
        self.hand_id = hand_id
        self.ser=serial.Serial(port, baudrate)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Must add this magic method to prevent the port not closing and obstract the next initialization.
        """
        self.ser.close()
        

    def __delattr__(self):
        self.ser.close()

    """
    Byte-level functions. Most are private functions.
    """
    def _data2bytes(self, data):
        """
        把数据分成高字节和低字节
        """
        rdata = [0xff]*2
        if data == -1:
            rdata[0] = 0xff
            rdata[1] = 0xff
        else:
            rdata[0] = data&0xff
            rdata[1] = (data>>8)&(0xff)
        return rdata

    def _num2str(self, num):
        """
        把十六进制或十进制的数转成bytes
        """
        str = hex(num)
        str = str[2:4]
        if(len(str) == 1):
            str = '0'+ str
        str = bytes.fromhex(str)     
        #print(str)
        return str

    def _checknum(self, data, leng):
        """
        求校验和
        """
        result = 0
        for i in range(2,leng):
            result += data[i]
        result = result&0xff
        #print(result)
        return result
    
    def _interface(self, writein, receive_len):
        """
        向串口发送数据
        """
        putdata = b''
        
        for byte in writein:
            putdata = putdata + self._num2str(byte)
        self.ser.write(putdata)
        print('发送的数据：')
        for byte in putdata:
            print(hex(putdata[i-1]))
        
        getdata= self.ser.read(receive_len)
        print('返回的数据：')
        for i in range(1,10):
            print(hex(getdata[i-1]))

        return getdata
    
    def _pack_data(self, header, data):

        datanum = (len(header) + len(data)).to_bytes(1)

        const = [0xEB, 0x90, self.hand_id, datanum]
        data_pkg = const + header + data

        summed_pkg = data_pkg + self._checknum(data_pkg, len(data_pkg))
        return summed_pkg

    """
    Direct command to the hand.
    """

    def setpos(self, pos1, pos2, pos3, pos4, pos5, pos6):
        """
        设置六个驱动器位置------参数pos范围-1-2000 
        """
        positions = [pos1,pos2,pos3,pos4,pos5,pos6]

        for pos in positions:
            if (pos < -1 or pos > 2000):
                print('数据超出正确范围：-1-2000')
                return
        
         # Header (1 for R/W, 2 for address)
        header = [0x12, 0xC2, 0x05]

        # Data
        data = [self._data2bytes(pos) for pos in positions]
        flattened_data = [item for sublist in data for item in sublist]

        # Pack data
        packed_data = self._pack_data(header, flattened_data)

        # Send data
        _ = self._interface(packed_data, 9)
        
        return
    
    def setangle(self, angle1, angle2, angle3, angle4, angle5, angle6):
        """
        设置灵巧手角度------参数angle范围-1-1000
        """
        angles = [angle1, angle2, angle3, angle4, angle5, angle6]
        for angle in angles:
            if (angle < -1 or angle > 1000):
                print('数据超出正确范围：-1-1000')
                return
        
         # Header (1 for R/W, 2 for address)
        header = [0x12, 0xCE, 0x05]

        # Data
        data = [self._data2bytes(angle) for angle in angles]
        flattened_data = [item for sublist in data for item in sublist]

        # Pack data
        packed_data = self._pack_data(header, flattened_data)
        
        # Send data
        _ = self._interface(packed_data, 9)
        return
    
    def setpower(self, power1, power2, power3, power4, power5, power6):
        """
        设置力控阈值------参数power范围0-1000
        """
        powers = [power1, power2, power3, power4, power5, power6]

        for power in powers:
            if (power < 0 or power > 1000):
                print('数据超出正确范围：0-1000')
                return
            
         # Header (1 for R/W, 2 for address)
        header = [0x12, 0xDA, 0x05]

        # Data
        data = [self._data2bytes(power) for power in powers]
        flattened_data = [item for sublist in data for item in sublist]

        # Pack data
        packed_data = self._pack_data(header, flattened_data)

        # Send data
        _ = self._interface(packed_data, 9)
        return
    
    def setspeed(self, speed1, speed2, speed3, speed4, speed5, speed6):
        """
        设置速度------参数speed范围0-1000
        """
        speeds = [speed1, speed2, speed3, speed4, speed5, speed6]
        for speed in speeds:
            if (speed < 0 or speed > 1000):
                print('数据超出正确范围：0-1000')
                return
            
         # Header (1 for R/W, 2 for address)
        header = [0x12, 0xF2, 0x05]

        # Data
        data = [self._data2bytes(speed) for speed in speeds]
        flattened_data = [item for sublist in data for item in sublist]

        # Pack data
        packed_data = self._pack_data(header, flattened_data)

        # Send data
        _ = self._interface(packed_data, 9)

    def get_actpos(self):
        """
        读取驱动器实际的位置值
        """
         # Header (1 for R/W, 2 for address)
        header = [0x11, 0xFE, 0x05]

        # Data
        data = [0x0C]

        # Pack data
        packed_data = self._pack_data(header, data)

        # Send data
        getdata = self._interface(packed_data, 20)
        
        # Unpack data
        actpos = [0]*6
        for i in range(1,7):
            if getdata[i*2+5]== 0xff and getdata[i*2+6]== 0xff:
                actpos[i-1] = -1
            else:
                actpos[i-1] = getdata[i*2+5] + (getdata[i*2+6]<<8)
        
        return actpos
    
    def get_actangle(self):
        """
        读取实际的角度值
        """
         # Header (1 for R/W, 2 for address)
        header = [0x11, 0x0A, 0x06]

        # Data
        data = [0x0C]

        # Pack data
        packed_data = self._pack_data(header, data)

        # Send data
        getdata = self._interface(packed_data, 20)

        # Unpack data
        actangle = [0]*6
        for i in range(1,7):
            if getdata[i*2+5]== 0xff and getdata[i*2+6]== 0xff:
                actangle[i-1] = -1
            else:
                actangle[i-1] = getdata[i*2+5] + (getdata[i*2+6]<<8)
        return actangle
    
    def get_actforce(self):
        """
        读取实际的受力
        """
         # Header (1 for R/W, 2 for address)
        header = [0x11, 0x2E, 0x06]

        # Data
        data = [0x0C]

        # Pack data
        packed_data = self._pack_data(header, data)

        # Send data
        getdata = self._interface(packed_data, 20)

        # Unpack data
        actforce = [0]*6
        for i in range(1,7):
            if getdata[i*2+5]== 0xff and getdata[i*2+6]== 0xff:
                actforce[i-1] = -1
            else:
                actforce[i-1] = getdata[i*2+5] + (getdata[i*2+6]<<8)
        return actforce
    
    def get_setpos(self):
        """
        取驱动器设置的位置值
        """
         # Header (1 for R/W, 2 for address)
        header = [0x11, 0xC2, 0x05]

        # Data
        data = [0x0C]

        # Pack data
        packed_data = self._pack_data(header, data)

        # Send data
        getdata = self._interface(packed_data, 20)

        # Unpack data
        setpos = [0]*6
        for i in range(1,7):
            if getdata[i*2+5]== 0xff and getdata[i*2+6]== 0xff:
                setpos[i-1] = -1
            else:
                setpos[i-1] = getdata[i*2+5] + (getdata[i*2+6]<<8)
        return setpos
    
    def get_setangle(self):
        """
        取设置的角度值
        """
         # Header (1 for R/W, 2 for address)
        header = [0x11, 0xCE, 0x05]

        # Data
        data = [0x0C]

        # Pack data
        packed_data = self._pack_data(header, data)

        # Send data
        getdata = self._interface(packed_data, 20)

        # Unpack data
        setangle = [0]*6
        for i in range(1,7):
            if getdata[i*2+5]== 0xff and getdata[i*2+6]== 0xff:
                setangle[i-1] = -1
            else:
                setangle[i-1] = getdata[i*2+5] + (getdata[i*2+6]<<8)
        return setangle
    
    def get_setpower(self):
        """
        取设置的力控阈值
        """
         # Header (1 for R/W, 2 for address)
        header = [0x11, 0xDA, 0x05]

        # Data
        data = [0x0C]

        # Pack data
        packed_data = self._pack_data(header, data)

        # Send data
        getdata = self._interface(packed_data, 20)

        # Unpack data
        setpower = [0]*6
        for i in range(1,7):
            if getdata[i*2+5]== 0xff and getdata[i*2+6]== 0xff:
                setpower[i-1] = -1
            else:
                setpower[i-1] = getdata[i*2+5] + (getdata[i*2+6]<<8)
        return setpower
    
    def get_error(self):
        """
        读取故障信息
        """
        # Header (1 for R/W, 2 for address)
        header = [0x11, 0x46, 0x06]

        # Data
        data = [0x06]

        # Pack data
        packed_data = self._pack_data(header, data)

        # Send data
        getdata = self._interface(packed_data, 14)

        # Unpack data
        error = [0]*6
        for i in range(1,7):
            error[i-1] = getdata[i+6]
        return error
    
    def get_status(self):
        """
        读取状态信息
        """
        # Header (1 for R/W, 2 for address)
        header = [0x11, 0x4C, 0x06]

        # Data
        data = [0x06]

        # Pack data
        packed_data = self._pack_data(header, data)

        # Send data
        getdata = self._interface(packed_data, 14)

        # Unpack data
        status = [0]*6
        for i in range(1,7):
            status[i-1] = getdata[i+6]
        return status
    
    def get_temp(self):
        """
        读取温度信息
        """
        # Header (1 for R/W, 2 for address)
        header = [0x11, 0x52, 0x06]

        # Data
        data = [0x06]
        
        # Pack data
        packed_data = self._pack_data(header, data)

        # Send data
        getdata = self._interface(packed_data, 14)

        # Unpack data
        temp = [0]*6
        for i in range(1,7):
            temp[i-1] = getdata[i+6]
        
        return temp
    
    def get_current(self):
        """
        读取电流信息
        """
        # Header (1 for R/W, 2 for address)
        header = [0x11, 0x3A, 0x06]

        # Data
        data = [0x0C]

        # Pack data
        packed_data = self._pack_data(header, data)

        # Send data
        getdata = self._interface(packed_data, 20)

        # Unpack data
        current = [0]*6
        for i in range(1,7):
            if getdata[i*2+5]== 0xff and getdata[i*2+6]== 0xff:
                current[i-1] = -1
            else:
                current[i-1] = getdata[i*2+5] + (getdata[i*2+6]<<8)
        return current
    
    def set_clear_error(self):
        """
        清除错误
        """
        # Header (1 for R/W, 2 for address)
        header = [0x12, 0xEC, 0x03]

        # Data
        data = [0x01]

        # Pack data
        packed_data = self._pack_data(header, data)

        # Send data
        _ = self._interface(packed_data, 9)

        return
    
    def setdefaultspeed(self, speed1, speed2, speed3, speed4, speed5, speed6):
        """
        设置默认速度------参数speed范围0-1000
        """
        speeds = [speed1, speed2, speed3, speed4, speed5, speed6]
        for speed in speeds:
            if (speed < 0 or speed > 1000):
                print('数据超出正确范围：0-1000')
                return
            
         # Header (1 for R/W, 2 for address)
        header = [0x12, 0x08, 0x04]

        # Data
        data = [self._data2bytes(speed) for speed in speeds]
        flattened_data = [item for sublist in data for item in sublist]

        # Pack data
        packed_data = self._pack_data(header, flattened_data)

        # Send data
        _ = self._interface(packed_data, 9)

    def setdefaultpower(self, power1, power2, power3, power4, power5, power6):
        """
        设置默认力控阈值------参数power范围0-1000
        """
        powers = [power1, power2, power3, power4, power5, power6]

        for power in powers:
            if (power < 0 or power > 1000):
                print('数据超出正确范围：0-1000')
                return
            
         # Header (1 for R/W, 2 for address)
        header = [0x12, 0x14, 0x04]

        # Data
        data = [self._data2bytes(power) for power in powers]
        flattened_data = [item for sublist in data for item in sublist]

        # Pack data
        packed_data = self._pack_data(header, flattened_data)

        # Send data
        _ = self._interface(packed_data, 9)
        return
    
    def set_save_flash(self):
        """
        保存数据到flash
        """
         # Header (1 for R/W, 2 for address)
        header = [0x12, 0xED, 0x03]

        # Data
        data = [0x01]

        # Pack data
        packed_data = self._pack_data(header, data)

        # Send data
        _ = self._interface(packed_data, 18)
        return
    
    """
    Operation-level functions. Can be used in Main Functions.
    """

    def c1(self):
        """
        Heavy Prismatic Wrap (Code: 1): Designed for strong, firm gripping of large tools or objects with long prismatic shapes, such as hammer handles or thick pipes.
        """
        self.setangle(999, 999, 999, 999, 999, 999)
        time.sleep(2)
        self.setangle(999, 999, 999, 999, 999, 400)
        time.sleep(2)
        self.setangle(0,0,0,0,0,400)
        return
    
    def c4(self):
        """
        Adducted Thumb Grasp (Code: 4): Used for stable manipulation of knobs or handles, such as door handles or mechanical
        """
        self.setangle(999, 999, 999, 999, 999, 999)
        time.sleep(2)
        self.setangle(999, 999, 999, 999, 999, 400)
        time.sleep(2)
        self.setangle(0,0,0,0,999,999)
        time.sleep(0.5)
        self.setangle(0, 0, 0, 0, 9, 999)
        return
    
    def c5(self):
        """
        Light Tool Grasp (Code: 5): Suitable for lightly holding small tools like screwdrivers or pens.
        """
        self.setangle(999, 999, 999, 999, 999, 999)
        time.sleep(2)
        self.setangle(999, 999, 999, 999, 999, 200)
        time.sleep(2)
        self.setangle(0,0,0,0,999,200)
        time.sleep(0.5)
        self.setangle(0, 0, 0, 0, 800, 999)
        return
    
    def c6(self):
        """
        4-Finger Pinch (Code: 6): Suitable for holding slightly larger cylindrical objects, such as cups or bottles, with a secure grip.
        """
        self.setangle(999, 999, 999, 999, 999, 999)
        time.sleep(1)
        self.setangle(900, 999, 999, 999, 999, 0)
        time.sleep(1)
        self.setangle(0, 50, 50, 50, 0, 0)
        return
    
    def c7(self):
        """
        Used for gripping medium-sized objects, such as tool handles or pens, with moderate control.
        """
        self.setangle(999, 999, 999, 999, 999, 999)
        time.sleep(1)
        self.setangle(0, 999, 999, 999, 999, 0)
        time.sleep(1)
        self.setangle(0, 50, 50, 50, 0, 0)
        return
    
    def c9(self):
        """
        Thumb-Index Finger Pinch (Code: 9): Perfect for handling very small or delicate objects, such as needles or tiny screws, where precise control is required.
        改为拇指-三指
        """
        self.setangle(999, 999, 999, 999, 999, 999)
        time.sleep(1)
        self.setangle(9, 999, 999, 999, 999, 400)
        time.sleep(1)
        self.setangle(9, 0, 0, 200, 100, 650)
        time.sleep(1)
        self.setangle(9, 0, 0, 200, 100, 950)
        return
    
    def c13(self):
        """
        Large Sphere Grasp (Code: 13): Designed for picking up larger spherical objects or irregularly shaped items like cloths, where all five fingers are evenly distributed around the object and apply force toward its center.
        """
        self.setangle(999, 999, 999, 999, 999, 999)
        time.sleep(1)
        self.setangle(999, 999, 999, 999, 999, 99)
        time.sleep(1)
        self.setangle(9, 0, 0, 9, 9, 100)
        return
    
    def c14(self):
        """
        Sphere Grasp (Code: 14): Suitable for grasping spherical objects like balls or oranges, where the fingers are evenly distributed around the object.
        """
        self.setangle(999, 999, 999, 999, 999, 999)
        time.sleep(1)
        self.setangle(9, 9, 999, 999, 999, 99)
        time.sleep(1)
        self.setangle(9, 0, 0, 200, 9, 100)
        self.setangle(9, 0, 0, 9, 9, 100)
        return
    
    def release(self):
        """
        Release the grip.
        """
        self.setangle(999, 999, 999, 999, 999, 999)
        return
    
    def pregrip(self):
        """
        Pre-grip function to prepare the hand for gripping.
        """
        self.setangle(999, 999, 999, 999, 999, 999)
        return


# Main function for the Master program
def control_hand(hand: Hand, c):

    match c:
        case "1":
            Hand.c1()
        case "4":
            Hand.c4()
        case "5":
            Hand.c5()
        case "6":
            Hand.c6()
        case "7":
            Hand.c7()
        case "9":
            Hand.c9()
        case "13":
            Hand.c13()
        case "14":
            Hand.c14()
        case _:
            Hand.release()
    
    return

if __name__ == "__main__":
    
    # Initialize the Hand object (in main.py)
    # Given a command, do something

    hand = Hand()
    given_command = "1"  # Example command

    # Example usage
    control_hand(hand, given_command)
    