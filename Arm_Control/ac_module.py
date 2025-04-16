from time import sleep
import urx
import sys
from Hand_Control.hc_module import Hand

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