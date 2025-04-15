from multiprocessing import Manager, Value, Array
import time
from huggingface_hub import login
import torch
from ultralytics import YOLO
import paddle
from paddleocr import PaddleOCR
import threading
import urx
from transformers import pipeline, Pipeline

# Custom modules (The 5 modules)
# from folder.file import function
from Scene_Understanding.scene_understanding import PoseEstimator_ViTPose
# from Intent_Prediction.Models import TableSearcher
from Scene_Understanding.su_module import find_user_thread
from Intent_Prediction.ip_module import listen_audio_thread
from Object_Detection.od_module import detect_medicine
from Hand_Control.hc_module import control_hand
from Hand_Control.ac_module import control_arm

# IP Address to Robot Arm

ARM_ADRESS = "192.168.12.21"

# IP Address to Robot Arm

ARM_ADRESS = "192.168.12.21"

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
        control_arm(model_dict["robot_arm"], class_label)

        # TODO: Release the medicine


    print("Grabbing done. Return to listening...")
    return

def main():

    # Initialize ALL Models (or related model classes) to prevent repeated loading
    model_dict = {
        # ASR_Pipe: Manually removed "forced_decoder_ids": [ [ 1, 50259 ], [ 2, 50359 ], [ 3, 50363 ] ], to prevent exception.
        "asr_pipe": pipeline("automatic-speech-recognition", model="borisPMC/MedicGrabber_WhisperSmall"),
        "med_list_pipe": pipeline("token-classification", "borisPMC/MedicGrabber_multitask_BERT_ner"),
        "intent_pipe": pipeline("text-classification", "borisPMC/MedicGrabber_multitask_BERT_intent"),
        # "nlp_pipe": pipeline("text-classification", model="borisPMC/bert_grab_medicine_intent", tokenizer="bert-base-multilingual-uncased"),
        "detect_med_model": YOLO("./Object_Detection/siou150.pt"),
        "ocr_model": PaddleOCR(use_angle_cls=True, lang='en'),  # Set language to English
        "pose_model": PoseEstimator_ViTPose(),
        # "manual_nlp": TableSearcher(),
        "robot_arm": urx.URRobot(ARM_ADRESS, useRTInterface=True)
    }

    model_dict["asr_pipe"].generation_config.forced_decoder_ids = None # Manually removed "forced_decoder_ids": [ [ 1, 50259 ], [ 2, 50359 ], [ 3, 50363 ] ], to prevent exception.

    # Note: Torch should use CUDA; Paddle should use CPU to avoid device collision

    # Shared variables and queues
    
    # Use multiprocessing.Manager to create shared variables
    manager = Manager()
    shared_dict = manager.dict({
        "user_flag": False,         # Bool
        "cmd_flag": False,          # Bool
        "play_sound_flag": False,
        "label_command": "",        # Str
        "queued_commands": [],
        "keypoints": [],            # List[list]: List of 2 scalar (x, y) lists
        "THREAD_PROCESS_TIMER": 5,  # CONSTANT, UNEXPECTED TO ALTER
    })

    # Configure Robot Arm
    model_dict["robot_arm"].set_tcp((0, 0, 0.1, 0, 0, 0))
    model_dict["robot_arm"].set_payload(2, (0, 0, 0.1))
    model_dict["robot_arm"].set_freedrive(True)

    # Create threads
    user_thread = threading.Thread(target=find_user_thread, args=(model_dict["pose_model"], shared_dict, listen_event), daemon=True)
    audio_thread = threading.Thread(target=listen_audio_thread, args=(model_dict, shared_dict, listen_event), daemon=True)

    # Start threads
    user_thread.start()
    audio_thread.start()
    # cache_class = "Empty"

    print("\nSetup done, ready to operate.\n")

    try:
        while True:
            
            # Wait prior threads to finish detection first
            time.sleep(7.5)

            # Trigger find_medicine() if conditions are met
            print("User Found:", shared_dict["user_flag"] ,"|", "Command Heard:", shared_dict["cmd_flag"])

            if shared_dict["user_flag"] and shared_dict["cmd_flag"]:

                # Push queued label until the queue is empty
                listen_event.clear()  # Stop listening to audio commands
                queue = shared_dict["queued_commands"]

                for queue_item in queue:
                    shared_dict["label_command"] = queue_item

                    grabbing_process(shared_dict["label_command"], model_dict=model_dict)
                
                # Reset after grabbing
                shared_dict["user_flag"] = False
                shared_dict["cmd_flag"] = False
                shared_dict["label_command"] = ""
                shared_dict["keypoints"] = []
                shared_dict["queued_commands"] = []

                listen_event.set()  # Resume listening to audio commands
                print("\nReturn to listening mode\n")
    
    except KeyboardInterrupt:
        print("Stopping threads...")

if __name__ == "__main__":
    main()