import time
from inspire_hand import setpos,setangle,setspeed,setpower,get_actangle,get_actforce
import serial


def control_arm(x,y,z):
    pass

# ser=serial.Serial('COM4',115200)
#Lateral Pinch (Code: 16): Ideal for gripping thin, flat objects like keys or cards, where the thumb presses against the side of the index finger.
'pinch'
def c16():
    setangle(999, 999, 999, 999, 999, 999)
    time.sleep(1)
    setangle(50,50,50,50, 999, 999)
    time.sleep(1.5)
    setangle(50, 50, 50, 50, 999, 999)
    time.sleep(0.5)
    setangle(50, 50, 50, 50, 0,999)
    return



#Heavy Prismatic Wrap (Code: 1): Designed for strong, firm gripping of large tools or objects with long prismatic shapes, such as hammer handles or thick pipes.
'heavy grab'
def c1():
    setangle(999, 999, 999, 999, 999, 999)
    time.sleep(2)
    setangle(999, 999, 999, 999, 999, 400)
    time.sleep(2)
    setangle(0,0,0,0,0,400)
    return


#Adducted Thumb Grasp (Code: 4): Used for stable manipulation of knobs or handles, such as door handles or mechanical
'adducted thumb'
def c4():
    setangle(999, 999, 999, 999, 999, 999)
    time.sleep(2)
    setangle(999, 999, 999, 999, 999, 400)
    time.sleep(2)
    setangle(0,0,0,0,999,999)
    time.sleep(0.5)
    setangle(0, 0, 0, 0, 9, 999)
    return



#Light Tool Grasp (Code: 5): Suitable for lightly holding small tools like screwdrivers or pens.
def c5():
    setangle(999, 999, 999, 999, 999, 999)
    time.sleep(2)
    setangle(999, 999, 999, 999, 999, 200)
    time.sleep(2)
    setangle(0,0,0,0,999,200)
    time.sleep(0.5)
    setangle(0, 0, 0, 0, 800, 999)
    return

#Used for gripping medium-sized objects, such as tool handles or pens, with moderate control.
def c7():
    setangle(999, 999, 999, 999, 999, 999)
    time.sleep(1)
    setangle(0, 999, 999, 999, 999, 0)
    time.sleep(1)
    setangle(0, 50, 50, 50,0 , 0)
    return

#4-Finger Pinch (Code: 6): Suitable for holding slightly larger cylindrical objects, such as cups or bottles, with a secure grip.
def c6():
    setangle(999, 999, 999, 999, 999, 999)
    time.sleep(1)
    setangle(900, 999, 999, 999, 999, 0)
    time.sleep(1)
    setangle(0, 50, 50, 50,0 , 0)
    return

'改为拇指-三指'
#Thumb-Index Finger Pinch (Code: 9): Perfect for handling very small or delicate objects, such as needles or tiny screws, where precise control is required.
def c9():
    setangle(999, 999, 999, 999, 999, 999)
    time.sleep(1)
    setangle(9, 999, 999, 999, 999, 400)
    time.sleep(1)
    setangle(9, 0, 0, 200, 100, 650)
    time.sleep(1)
    setangle(9, 0, 0, 200, 100, 950)
    return

#Large Sphere Grasp (Code: 13): Designed for picking up larger spherical objects or irregularly shaped items like cloths, where all five fingers are evenly distributed around the object and apply force toward its center.
def c13():
    setangle(999, 999, 999, 999, 999, 999)
    time.sleep(1)
    setangle(999, 999, 999, 999, 999, 99)
    time.sleep(1)
    setangle(9, 0, 0, 9, 9, 100)



#Small Sphere Grasp (Code: 14): Used for controlling small spherical objects, such as marbles or balls, where the thumb, index, and middle fingers are evenly distributed around the object and apply force toward its center.
def c14():
    setangle(999, 999, 999, 999, 999, 999)
    time.sleep(1)
    setangle(9, 9, 999, 999, 999, 99)
    time.sleep(1)
    setangle(9, 0, 0, 200, 9, 100)
    setangle(9, 0, 0, 9, 9, 100)
    #setangle(999, 999, 999, 999, 999, 999)
def release():
    setangle(999, 999, 999, 999, 999, 999)
def pregrab():
    setangle(999, 999, 999, 999, 999, 300)


c1()
#c7()
#没有c6(
#c9()
#c14()
#c14()

#release()
#pregrab()

def take_action(c,x,y,z):
    print("执行手势{c}")
    if c =='16':
        c16()
    elif c == '1':
        c1()
    elif c == '4':
        c4()
    elif c == '5':
        c5()
    elif c == '7':
        c7()
    elif c == '6':
        c6()
    elif c == '9':
        c9()
    elif c == '13':
        c13()
    elif c == '14':
        c14()
    else:
        c1()
    #control_arm()





#ser.close()













