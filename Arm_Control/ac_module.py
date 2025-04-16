from time import sleep
import urx
import sys
from Hand_Control import hand_action
from Hand_Control.hc_module import Hand

# Basic Unit: meter
DISPLACEMENT = 0.05    # Displacement
VELOCITY = 0.2    # Velocity
ACCELERATION = 0.1     # Acceleration


def grab_0(hand: Hand, arm: urx.URRobot, start_coord =(0.17,0.5,0.06,-0.4,-2.35,-2.3)):
    #c1(),release(),pregrab()
    hand.c1()
    arm.movel(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)
    arm.movel((0,0,0.2,0,0,0), acc=ACCELERATION, vel=VELOCITY , wait=True, relative=True)
    arm.movel((-0.3,0,0,0,0,0), acc=ACCELERATION, vel=VELOCITY, wait=True, relative=True)
    arm.movel((0,0,-0.2,0,0,0) ,acc=ACCELERATION, vel=VELOCITY, wait=True, relative=True)
    hand.release()

def grab_1(hand: Hand, arm: urx.URRobot, start_coord = (0,0,0,0,0,0)):

    arm.movel(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)

def grab_2(hand: Hand, arm: urx.URRobot, start_coord = (0,0,0,0,0,0)):

    arm.movel(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)

def grab_3(hand: Hand, arm: urx.URRobot, start_coord = (0,0,0,0,0,0)):

    arm.movel(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)

# Main function for the Master program
def control_arm(hand: Hand, arm: urx.URRobot, medicine: str):

    print("Arm start moving")

    match medicine:
        case "ACE_Inhibitor":
            grab_0(hand, arm)
        case "Metformin":
            grab_1(hand, arm)
        case "Atorvastatin":
            grab_2(hand, arm)
        case "Amitriptyline":
            grab_3(hand, arm)
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

def main(medicine):

    hand = Hand()
    arm = urx.URRobot("192.168.12.21", useRTInterface=True)

    print("Arm start moving")

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
    
    # Wait grabbing finish
    while True:
        sleep(0.1)
        if not arm.is_program_running():
            break
    
    #arm.cleanup()

    print("Arm stop moving, exiting the program")

    return

if __name__ == "__main__":
    
    # Medicine name for testing arm
    med = "ACE Inhibitor"

    main(med)