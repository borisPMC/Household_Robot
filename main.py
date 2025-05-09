# PIP packages
from multiprocessing import Manager
import time, torch, threading, urx
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Custom Modules
from modules import Audio_Listener, Grabber, Object_Finder, User_Recogniser, Depth_Detector

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# This function is specifically to control the timing flags.
def grabbing_process(request_queue: str, model_dict: dict, states_dict: dict) -> None:

    for item in request_queue:

        coord_list = Object_Finder.detect_medicine_redefined(model_dict, item, 3)

        # Multiple bottles may exist. Get one only
        if coord_list:
            # Convertion from depth cam coordination to UR coordination
            tgt_coord = Grabber.convert_to_arm_coord(coord_list)
            # Moving medicine to specific location from fixed starting point
            Grabber.reset_to_standby(model_dict["robot_arm"])
            Grabber.grab_medicine(model_dict["robot_hand"], model_dict["robot_arm"], item, states_dict["user_angle"], tgt_coord)
            Grabber.reset_to_standby(model_dict["robot_arm"])

    print("Grabbing done. Return to listening...")
    return

def handle_request(states_dict, model_dict):
    match states_dict["current_cmd"]:
        case "1": # Retrieve
            grabbing_process(states_dict["queued_objects"], model_dict, states_dict)
        case _:
            print("To be developed in future...")
    return

def main():

    # Shared variables and queues (State Cache)
    manager = Manager()
    states_dict = manager.dict({
        "user_flag": False,         # Bool
        "current_cmd": "",          # Bool
        "queued_objects": [],
        "keypoints": [],            # List[list]: List of 2 scalar (x, y) lists
        "THREAD_PROCESS_TIMER": 5,  # CONSTANT, UNEXPECTED TO ALTER
        "user_angle": 0,            # Between 0-1 radian
    })

    listen_event = threading.Event()  # Event to control the listening thread
    listen_event.set()  # Set the event to allow the thread to run

    # Define & Initialize ALL Models and Hardwares (or related model classes) to prevent repeated loading
    # Loading on first time is slower and require Internet connection. Afterwards, no Internet should be need.
    model_dict = {
        "asr_pipe":             Audio_Listener.load_asr_pipeline(),
        "med_list_pipe":        Audio_Listener.load_med_list_pipeline(),
        "intent_pipe":          Audio_Listener.load_intent_pipeline(),
        "detect_med_model":     YOLO("./modules/siou150.pt"),
        "ocr_model":            PaddleOCR(use_angle_cls=True, lang='en'),  # Set language to English
        "pose_model":           User_Recogniser.PoseEstimator_ViTPose(),
        "robot_arm":            urx.URRobot("192.168.12.21", useRTInterface=True),
        "robot_hand":           Grabber.Hand(2, 'COM4', 115200),
        # Assign the right camera before running
        "user_camera_index":    0,
        "object_cam":           Depth_Detector.init_depth_detector(),
        "audio_dir_detector":   User_Recogniser.MIC_array("COM7", states_dict, listen_event)
    }

    model_dict["asr_pipe"].generation_config.forced_decoder_ids = None
    
    # Configure Robot Arm
    model_dict["robot_hand"].pregrip()
    model_dict["robot_arm"].set_tcp((0, 0, 0.1, 0, 0, 0))
    model_dict["robot_arm"].set_payload(2, (0, 0, 0.1))
    Grabber.reset_to_standby(model_dict["robot_arm"])


    # Note: Torch should use CUDA; Paddle should use CPU to avoid device collision

    # Create threads
    user_thread = threading.Thread(target=User_Recogniser.find_user_thread, args=(model_dict, states_dict, listen_event), daemon=True)
    audio_thread = threading.Thread(target=Audio_Listener.listen_audio_thread, args=(model_dict, states_dict, listen_event), daemon=True)

    user_thread.start()
    audio_thread.start()

    print("\nSetup done, ready to operate.\n")

    try:
        while True:
            
            time.sleep(1)
            print("User Found:", states_dict["user_flag"] ,"|", "Command Heard:", states_dict["current_cmd"])

            # Trigger find_medicine() if conditions are met
            if states_dict["user_flag"] and len(states_dict["current_cmd"]) > 0:

                # Keep pushing queued label until the queue is empty
                listen_event.clear()  # Stop listening for User & Audio
                print("\Pause listening mode\n")

                # Request Handler
                handle_request(states_dict, model_dict)
                
                # Reset after grabbing all objects in 
                states_dict["user_flag"] = False
                states_dict["current_cmd"] = ""
                states_dict["keypoints"] = []
                states_dict["queued_objects"] = []

                listen_event.set()  # Resume listening for User & Audio
                print("\nReturn to listening mode\n")
    
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down the system...")
        pass

    finally:
        """
        Safe exit whenever program ends
        """
        model_dict["robot_arm"].cleanup() # Close hardware
        model_dict["robot_hand"].close()
        model_dict["object_cam"].close()
        model_dict["audio_dir_detector"].close()
        print("Stopping threads...")

if __name__ == "__main__":
    main()