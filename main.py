from multiprocessing import Value
from huggingface_hub import login
import torch
from ultralytics import YOLO
import paddle
from paddleocr import PaddleOCR
import threading
from queue import Queue
from transformers import pipeline, Pipeline

# Custom modules (The 5 modules)
# from folder.file import function
from Scene_Understanding.scene_understanding import PoseEstimator_ViTPose
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
    for item in coord_list:
        control_hand(item, class_label)
        control_arm()
    print("Grabbing done. Return to listening...")
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
    }

    # Note: Torch should use CUDA; Paddle should use CPU to avoid device collision

    # Shared variables and queues
    user_flag = Value('b', False)   # Shared boolean flag for Scene Understanding
    cmd_flag = Value('b', False)    # Shared boolean flag for Intent Prediction
    label_queue = Queue()  # Queue to store labels from the audio process

    stop_receiving_commands = Value('b', False)  # If the robot is grabbing, set to True then the two Threads stop listening until set False

    # Create threads
    user_thread = threading.Thread(target=find_user_thread, args=(model_dict["pose_model"], user_flag, cmd_flag), daemon=True)
    audio_thread = threading.Thread(target=listen_audio_thread, args=(model_dict["asr_pipe"], model_dict["nlp_pipe"], user_flag, cmd_flag, label_queue), daemon=True)

    # Start threads
    user_thread.start()
    audio_thread.start()
    cache_class = "Empty"

    print("\nSetup done, ready to operate.\n")

    try:
        while True:
            # If not grabbing, listen to command
            if not user_flag.value or not cmd_flag.value:
                class_label = label_queue.get()
            
            # Get the label from the queue; Use the newest command (exclude Empty)
            if class_label != "Empty":
                cache_class = class_label
            print(cache_class)
            
            # Trigger find_medicine() if conditions are met
            if user_flag.value and cmd_flag.value:
                grabbing_process(cache_class, model_dict=model_dict)
                # Reset after grabbing
                user_flag.value = False
                cmd_flag.value = False
                cache_class = "Empty"
            else:
                if not user_flag.value:
                    print("User not found!")
                if not cmd_flag.value:
                    print("No executable commands!")
    
    except KeyboardInterrupt:
        print("Stopping threads...")

if __name__ == "__main__":
    main()