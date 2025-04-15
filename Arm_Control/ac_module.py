from time import sleep
import urx
import sys
import action
# Basic Unit: meter
DISPLACEMENT = 0.05    # Displacement
VELOCITY = 0.2    # Velocity
ACCELERATION = 0.1     # Acceleration


def grab_0(robot, start_coord =(-0.23,0.48,0.07,-0.25,-2.35,-2.3)):
    #c1(),release(),pregrab()
    action.c1()
    robot.movel(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)
    robot.movel((0,0,0,0,0,0), acc=ACCELERATION, vel=VELOCITY , wait=True, relative=True)
    robot.movel((0,0,0,0,0,0), acc=ACCELERATION, vel=VELOCITY, wait=True, relative=True)
    robot.movel((0,0,0,0,0,0) ,acc=ACCELERATION, vel=VELOCITY, wait=True, relative=True)
    action.release()

def grab_1(robot, start_coord = (0,0,0,0,0,0)):

    robot.movel(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)

def grab_2(robot, start_coord = (0,0,0,0,0,0)):

    robot.movel(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)

def grab_3(robot, start_coord = (0,0,0,0,0,0)):

    robot.movel(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)

# Main function for the Master program
def control_arm(robot: urx.URRobot, medicine: str):

    print("Arm start moving")

    match medicine:
        case "ACE Inhibitor":
            grab_0(robot)
        case "Metformin":
            grab_1(robot)
        case "Atorvastatin":
            grab_2(robot)
        case "Amitriptyline":
            grab_3(robot)
        case _:
            print("Error")
    
    # Wait grabbing finish
    while True:
        sleep(0.1)
        if not robot.is_program_running():
            break
    
    # robot.cleanup()
    print("Arm stop moving, continue master program")
    return

def main(medicine):

    robot = urx.URRobot("192.168.12.21", useRTInterface=True)

    print("Arm start moving")

    match medicine:
        case "ACE Inhibitor":
            grab_0(robot)
        case "Metformin":
            grab_1(robot)
        case "Atorvastatin":
            grab_2(robot)
        case "Amitriptyline":
            grab_3(robot)
        case _:
            print("Error")
    
    # Wait grabbing finish
    while True:
        sleep(0.1)
        if not robot.is_program_running():
            break
    
    robot.cleanup()

    print("Arm stop moving, exiting the program")

    return

if __name__ == "__main__":

    # Medicine name for testing arm
    med = "ACE Inhibitor"

    main(med)