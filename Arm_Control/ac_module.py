from time import sleep
import urx

# Basic Unit: meter
DISPLACEMENT = 0.05    # Displacement
VELOCITY = 0.2    # Velocity
ACCELERATION = 0.1     # Acceleration


def grab_0(robot, start_coord = (1.5, -1.9, -2.5, -2, -1.6, 0)):

    robot.movej(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)
    robot.movej((0, 0.5, 0, -0.3, 0, 0), acc=ACCELERATION, vel=VELOCITY , wait=True, relative=True)
    robot.movej((0.5, 0, 0, 0, 0, 0), acc=ACCELERATION, vel=VELOCITY, wait=True, relative=True)
    robot.movej((0, -0.5, 0, 0.3, 0, 0), acc=ACCELERATION, vel=VELOCITY, wait=True, relative=True)

def grab_1(robot, start_coord = (0,0,0,0,0,0)):

    robot.movej(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)

def grab_2(robot, start_coord = (0,0,0,0,0,0)):

    robot.movej(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)

def grab_3(robot, start_coord = (0,0,0,0,0,0)):

    robot.movej(start_coord, acc=ACCELERATION, vel=VELOCITY, wait=True, relative=False)

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