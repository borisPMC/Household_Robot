# PIP packages
from multiprocessing import Manager
import time, torch, threading, urx
from huggingface_hub import login
from transformers import pipeline
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Custom Modules
from Scene_Understanding.su_module import find_user_thread, PoseEstimator_ViTPose
from Intent_Prediction.ip_module import listen_audio_thread, load_asr_pipeline, load_intent_pipeline, load_med_list_pipeline
from Object_Detection.od_module import detect_medicine
from Hand_Control.hc_module import Hand, control_hand
from Arm_Control.ac_module import grab_medicine

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

login("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")

# This function is specifically to control the timing flags.
def grabbing_process(class_label: str, model_dict: dict) -> None:

    # time.sleep(10)  # Simulate the 10-second duration for grabbing a medicine
    coord_list = detect_medicine(model_dict["detect_med_model"], model_dict["ocr_model"], class_label, 3) # a list of dict: [{xmin, ymin, xmax, ymax}]
    print(coord_list)

    # Simulate Grabbing, comment if implemeneted
    # time.sleep(10)

    for item in coord_list:

        # TODO: pass item & img to depth function

        # TODO: Grab the medicine
        
        # Moving medicine to specific location from fixed starting point
        grab_medicine(model_dict["robot_arm"], class_label)

        # TODO: Release the medicine


    print("Grabbing done. Return to listening...")
    return

def main():

    # Define & Initialize ALL Models and Hardwares (or related model classes) to prevent repeated loading
    model_dict = {
        "asr_pipe":             load_asr_pipeline("borisPMC/MedicGrabber_WhisperSmall"),
        "med_list_pipe":        load_med_list_pipeline("borisPMC/MedicGrabber_multitask_BERT_ner"),
        "intent_pipe":          load_intent_pipeline("borisPMC/MedicGrabber_multitask_BERT_intent"),
        "detect_med_model":     YOLO("./Object_Detection/siou150.pt"),
        "ocr_model":            PaddleOCR(use_angle_cls=True, lang='en'),  # Set language to English
        "pose_model":           PoseEstimator_ViTPose(),
        "robot_arm":            urx.URRobot("192.168.12.21", useRTInterface=True),
        "robot_hand":           Hand('COM4', 2, 115200),
        # Assign the right camera before running
        "user_camera_index":    0,
        "medicine_camera_index":0,
    }

    model_dict["asr_pipe"].generation_config.forced_decoder_ids = None
    
    # Configure Robot Arm
    model_dict["robot_arm"].set_tcp((0, 0, 0.1, 0, 0, 0))
    model_dict["robot_arm"].set_payload(2, (0, 0, 0.1))
    model_dict["robot_arm"].set_freedrive(True)


    # Note: Torch should use CUDA; Paddle should use CPU to avoid device collision

    # Shared variables and queues
    manager = Manager()
    states_dict = manager.dict({
        "user_flag": False,         # Bool
        "cmd_flag": False,          # Bool
        "play_sound_flag": False,
        "label_command": "",        # Str
        "queued_commands": [],
        "keypoints": [],            # List[list]: List of 2 scalar (x, y) lists
        "THREAD_PROCESS_TIMER": 5,  # CONSTANT, UNEXPECTED TO ALTER
    })

    listen_event = threading.Event()  # Event to control the listening thread
    listen_event.set()  # Set the event to allow the thread to run

    # Create threads
    user_thread = threading.Thread(target=find_user_thread, args=(model_dict, states_dict, listen_event), daemon=True)
    audio_thread = threading.Thread(target=listen_audio_thread, args=(model_dict, states_dict, listen_event), daemon=True)

    user_thread.start()
    audio_thread.start()

    print("\nSetup done, ready to operate.\n")

    try:
        while True:
            
            time.sleep(1)
            # Trigger find_medicine() if conditions are met
            print("User Found:", states_dict["user_flag"] ,"|", "Command Heard:", states_dict["cmd_flag"])

            if states_dict["user_flag"] and states_dict["cmd_flag"]:

                # Keep pushing queued label until the queue is empty
                listen_event.clear()  # Stop listening to audio commands
                queue = states_dict["queued_commands"]

                for queue_item in queue:

                    states_dict["label_command"] = queue_item
                    grabbing_process(states_dict["label_command"], model_dict=model_dict)
                
                # Reset after grabbing
                states_dict["user_flag"] = False
                states_dict["cmd_flag"] = False
                states_dict["label_command"] = ""
                states_dict["keypoints"] = []
                states_dict["queued_commands"] = []

                listen_event.set()  # Resume listening to audio commands
                print("\nReturn to listening mode\n")
    
    except KeyboardInterrupt:
        model_dict["robot_arm"].cleanup()
        model_dict["robot_hand"].close()
        print("Stopping threads...")

if __name__ == "__main__":
    main()