from time import sleep
import urx
import sys
import time
import serial

class Hand:

    """
    Custom class to control the Hand. All functions are originally defined in hand_action.py and inspire_hand.py.
    """

    def __init__(self, hand_id=2, port='COM4', baudrate=115200):
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
        print(data)
        result = 0
        for i in range(2,leng):
            result += data[i]
        print(result)
        result = result&0xff
        #print(result)
        return [result]
    
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
            print(hex(byte))
        
        getdata= self.ser.read(receive_len)
        print('返回的数据：')
        for i in range(1,10):
            print(hex(getdata[i-1]))

        return getdata
    
    def _pack_data(self, header, data):

        datanum = (len(header) + len(data))

        const = [0xEB, 0x90, self.hand_id, datanum]
        data_pkg = const + header + data

        summed_pkg = data_pkg + self._checknum(data_pkg, len(data_pkg))
        return summed_pkg

    """
    Direct command to the hand. For information, refer to handReadme.txt
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
    
    def close(self):
        """
        Close the serial port. Should be called whenever the program ends.
        """
        self.ser.close()
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

# Basic Unit: meter
DISPLACEMENT = 0.05    # Displacement
VELOCITY = 0.2    # Velocity
ACCELERATION = 0.1     # Acceleration

# O-point
# [-0.0950121533928039, 0.4809898169216103, 0.06002943736198503, -1.5774495438896732, 4.626599225228091e-06, 0.06718130273957854]

def reset_to_standby(arm: urx.URRobot):

    # J [-4.267364327107565, -1.5765263042845667, -2.3882980346679688, 3.9583703714558105, -1.1640966574298304, 3.1896891593933105]
    # L [-2.7236719140396465e-07, 0.4000310982860804, 0.20001619645724564, -1.5774533355169895, 2.066791994465247e-05, 0.06710359852509011]
    arm.movej((-4.267364327107565, -1.5765263042845667, -2.3882980346679688, 3.9583703714558105, -1.1640966574298304, 3.1896891593933105), acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)

def grab_0(hand: Hand, arm: urx.URRobot, start_coord=(0.17,0.4,0.06,4.7,0,-0.2)):
    #c1(),release(),pregrip()
    arm.movel((0.17,0,0,0,0,0), acc=ACCELERATION, vel=VELOCITY, wait=True, relative=True)
    arm.movel(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)
    hand.pregrip()
    arm.movel((0,0.02,0,0,0,0), acc=ACCELERATION, vel=VELOCITY , wait=True, relative=True)
    hand.c1()
    arm.movel((0,0,0.2,0,0,0), acc=ACCELERATION, vel=VELOCITY , wait=True, relative=True)
    arm.movel((-0.31,0,0,0,0,0), acc=ACCELERATION, vel=VELOCITY, wait=True, relative=True)
    arm.movej((3.5,0,0,0,0,0), acc=ACCELERATION, vel=VELOCITY, wait=True, relative=True)
    # arm.movel((0,0,-0.2,0,0,0) ,acc=ACCELERATION, vel=VELOCITY, wait=True, relative=True)
    hand.release()
    hand.pregrip()
    sleep(0.5)
    arm.movel((0,0,0.15,0,0,0) ,acc=ACCELERATION, vel=VELOCITY, wait=True, relative=True)

def grab_1(hand: Hand, arm: urx.URRobot, start_coord = (0,0,0,0,0,0)):

    arm.movej(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)

def grab_2(hand: Hand, arm: urx.URRobot, start_coord = (0,0,0,0,0,0)):

    arm.movej(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)

def grab_3(hand: Hand, arm: urx.URRobot, start_coord = (0,0,0,0,0,0)):

    arm.movej(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)

# Main function for the Master program. It is intent to control both the arm and the hand.
def grab_medicine(hand: Hand, arm: urx.URRobot, medicine: str):

    print("Arm start moving")

    match medicine:
        case "ACE_Inhibitor":
            grab_0(hand, arm)
        case "Metformin":
            grab_0(hand, arm)
        case "Atorvastatin":
            grab_0(hand, arm)
        case "Amitriptyline":
            grab_0(hand, arm)
        case _:
            print("Error")
    
    # Wait grabbing finish
    while True:
        sleep(0.1)
        if not arm.is_program_running():
            break
    
    # arm.cleanup()
    print("Arm stop moving, continue master program")
    return

def main():

    """
    Avoid adding hand-related lines in grab_0-4 functions to maintain modularity.
    """
    try:

        # These three are given from external modules, don't need to care if configuring the arm motions only.
        # I include the Hand here. Please see grab_0 to know how to use it.
        medicine = "ACE Inhibitor" 
        hand = Hand()
        arm = urx.URRobot("192.168.12.21", useRTInterface=True)

        print("Arm start moving")

        reset_to_standby(arm)
        match medicine:
            case "ACE Inhibitor":
                grab_0(hand, arm)
            case "Metformin":
                grab_1(hand, arm)
            case "Atorvastatin":
                grab_2(hand, arm)
            case "Amitriptyline":
                grab_3(hand, arm)
            case _:
                print("Error")
        reset_to_standby(arm)
        
        # Wait grabbing finish
        while True:
            sleep(0.1)
            if not arm.is_program_running():
                break
        
        #arm.cleanup()

        print("Arm stop moving, exiting the program")

        return
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
        arm.cleanup()
        hand.close()
        sys.exit(0)

if __name__ == "__main__":

    main()