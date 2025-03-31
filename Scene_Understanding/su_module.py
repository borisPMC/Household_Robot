from multiprocessing import Value
import time
from Scene_Understanding.scene_understanding import PoseEstimator_ViTPose, live_capture, process_video

# From scene_understanding.main()
def find_user(pose_estimator: PoseEstimator_ViTPose):
    
    # Initialize the pose estimator (e.g., ViTPose or YOLO)
    pose_estimator = PoseEstimator_ViTPose() 

    # Capture a 5-second footage and return the cache file path
    live_capture()

    # Process the video
    person_detected = process_video(pose_estimator, confidence_threshold=0.5)

    # Output the final result
    # print("Person detected in video:", person_detected)

    return person_detected

# Main function for the Master program
# Expected to be run forever
def find_user_thread(pose_class: PoseEstimator_ViTPose, user_flag, cmd_flag) -> None:

    while True:
        # Idle when grabbing medicine
        if user_flag.value and cmd_flag.value:
            # print("SU Thread: Idle")
            time.sleep(5) 
            continue

        # Simulate finding a user
        # time.sleep(5)  # Simulate the 5-second duration for finding a user
        user_flag.value = find_user(pose_class)

        # Wait 1 second before looping again
        time.sleep(1)
