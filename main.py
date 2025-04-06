from multiprocessing import Manager, Value, Array
import time
from huggingface_hub import login
import torch
from ultralytics import YOLO
import paddle
from paddleocr import PaddleOCR
import threading
from queue import Queue
from transformers import pipeline

# Custom modules (The 5 modules)
# from folder.file import function
from Scene_Understanding.scene_understanding import PoseEstimator_ViTPose
from Intent_Prediction.Models import TableSearcher
from Scene_Understanding.su_module import find_user_thread
from Intent_Prediction.ip_module import listen_audio_thread
from Object_Detection.od_module import detect_medicine
from Hand_Control.hc_module import control_hand
from Arm_Control.ac_module import control_arm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# u are advise to make a virtual environment (with VS Code), and run the cmd in sys.txt
login("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")

# This function is specifically to control the timing flags.
def grabbing_process(class_label: str, model_dict: dict) -> None:

    # time.sleep(10)  # Simulate the 10-second duration for grabbing a medicine
    coord_list = detect_medicine(model_dict["detect_med_model"], model_dict["ocr_model"], class_label, 3) # a list of dict: [{xmin, ymin, xmax, ymax}]
    print(coord_list)

    # Simulate Grabbing, comment if implemeneted
    time.sleep(10)

    for item in coord_list:
        control_hand(item, class_label)
        control_arm()
    print("Grabbing {} done.".format(class_label))
    return 

def main():

    # Initialize ALL Models (or related model classes) to prevent repeated loading
    model_dict = {
        # ASR_Pipe: Manually removed "forced_decoder_ids": [ [ 1, 50259 ], [ 2, 50359 ], [ 3, 50363 ] ], to prevent exception.
        "asr_pipe": pipeline("automatic-speech-recognition", model="borisPMC/whisper_small_grab_medicine_intent", processor="borisPMC/whisper_small_grab_medicine_intent"),
        "nlp_pipe": pipeline("text-classification", model="borisPMC/bert_grab_medicine_intent", tokenizer="bert-base-multilingual-uncased"),
        "detect_med_model": YOLO("./Object_Detection/siou150.pt"),
        "ocr_model": PaddleOCR(use_angle_cls=True, lang='en'),  # Set language to English
        "pose_model": PoseEstimator_ViTPose(),
        "manual_nlp": TableSearcher(),
    }

    model_dict["asr_pipe"].generation_config.forced_decoder_ids = None

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

    # If it is set (True), the threads will listen. Else, clear (false) until set again. 
    listen_event = threading.Event()
    listen_event.set()

    # Create threads
    user_thread = threading.Thread(target=find_user_thread, args=(model_dict["pose_model"], shared_dict, listen_event), daemon=True)
    audio_thread = threading.Thread(target=listen_audio_thread, args=(model_dict["asr_pipe"], model_dict["manual_nlp"], shared_dict, listen_event), daemon=True)

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