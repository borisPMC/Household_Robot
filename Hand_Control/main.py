import urx
from urx import RobotException
from time import sleep

rob = urx.Robot("192.168.12.21")
rob.set_tcp((0, 0, 0.1, 0, 0, 0))
rob.set_payload(2, (0, 0, 0.1))
sleep(5)  #leave some time to robot to process the setup commands

# Basic Unit: meter
l = 0.05    # Displacement
v = 0.01    # Velocity
a = 0.1     # Acceleration

# pose = rob.getl() # (x, y, z, rx, ry, rz)
# pose = rob.get_pose()
# print(pose)

try:

    # rob.movej((1, 2, 3, 4, 5, 6), a, v)
    # rob.movel((, y, z, rx, ry, rz), a, v)
    # print("Current tool pose is: ",  rob.getl())
    # rob.movel((-0.1, 0, 0, 0, 0, 0), a, v, relative=True)  # move relative to current pose
    # rob.translate((0.1, 0, 0), a, v)  #move tool and keep orientation
    # rob.stopj(a)

    # rob.movel((x, y, z, rx, ry, rz), wait=False)
    # while True:
    #     sleep(0.1)  #sleep first since the robot may not have processed the command yet
    #     if rob.is_program_running():
    #         break

    # rob.movel((x, y, z, rx, ry, rz), wait=False)
    # while rob.getForce() < 50:
    #     sleep(0.01)
    #     if not rob.is_program_running():
    #         break

    # rob.stopl()
    pose = rob.getj()
    print(pose)
    rob.movej((-1.5,-1.5,-1.5,0,0,0), relative=False)
    # print(rob.getl())

except RobotException as ex:
    rob.stopl()
    print("Robot could not execute move (emergency stop for example), do something", ex)

finally:
    rob.close()
    print("end robot arm")