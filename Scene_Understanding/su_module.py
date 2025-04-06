import time
from Scene_Understanding.scene_understanding import PoseEstimator_ViTPose, live_capture, process_video
   

# Main function for the Master program
# Expected to be run forever
def find_user_thread(pose_class: PoseEstimator_ViTPose, shared_dict, listen_event) -> None:

    while True:

        listen_event.wait()

        # Idle when grabbing medicine
        if shared_dict["user_flag"] and shared_dict["cmd_flag"]:
            # print("SU Thread: Idle")
            time.sleep(5) 
            continue

        # Simulate finding a user
        # time.sleep(5)  # Simulate the 5-second duration for finding a user
        shared_dict["keypoints"], shared_dict["user_flag"] = live_capture(pose_class, timer=shared_dict["THREAD_PROCESS_TIMER"])

        # Wait 1 second before looping again
        time.sleep(2)
