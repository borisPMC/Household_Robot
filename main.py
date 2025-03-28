from multiprocessing import Value
from huggingface_hub import login
from ultralytics import YOLO
from paddleocr import PaddleOCR
import threading
# import time
from queue import Queue
# import numpy as np
from transformers import pipeline, Pipeline

# Custom modules (The 5 modules)
# from folder.file import function
from Scene_Understanding.su_module import find_user_thread
from Intent_Prediction.ip_module import listen_audio_thread
from Object_Detection.od_module import detect_medicine
from Hand_Control.hc_module import control_hand
from Arm_Control.ac_module import control_arm


# u are advise to make a virtual environment (with VS Code), and run the cmd in sys.txt
login("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")

# This function is specifically to control the timing flags.
def grabbing_process(model_dict: dict, class_label: str, finding_medicine, label_queue) -> None:

    # time.sleep(10)  # Simulate the 10-second duration for grabbing a medicine
    coord_list = detect_medicine(model_dict["detect_med_model"], model_dict["ocr_model"], class_label, 3) # a list of dict: [{xmin, ymin, xmax, ymax}]
    for item in coord_list:
        control_hand(item, class_label)
        control_arm()
    # Reset flag for future detection
    finding_medicine.value = False
    label_queue.put("Empty")  # Put "Empty" in the queue to reset the label
    print("Grabbing done. Return to listening...")
    return

def main():

    # Initialize ALL Models to prevent repeated loading
    model_dict = {
        "model_dict": pipeline("automatic-speech-recognition", model="borisPMC/whisper_small_grab_medicine_intent"),
        "nlp_pipe": pipeline("text-classification", model="borisPMC/bert_grab_medicine_intent", tokenizer="bert-base-multilingual-uncased"),
        "detect_med_model": YOLO("./Object_Detection/siou150.pt"),
        "ocr_model": PaddleOCR(use_angle_cls=True, lang='en')  # Set language to English
    }

    # Shared variables and queues
    user_flag = Value('b', True)  # Shared boolean flag for user detection
    label_queue = Queue()  # Queue to store labels from the audio process

    stop_receiving_commands = Value('b', False)  # If the robot is grabbing, set to True then the two Threads stop listening until set False

    # Create threads
    user_thread = threading.Thread(target=find_user_thread, args=(user_flag, stop_receiving_commands), daemon=True)
    audio_thread = threading.Thread(target=listen_audio_thread, args=(model_dict["asr_pipe"], model_dict["nlp_pipe"], label_queue, stop_receiving_commands), daemon=True)

    # Start threads
    user_thread.start()
    audio_thread.start()
    class_label = "Empty"

    try:
        while True:
            class_label = label_queue.get()  # Get the label from the queue
            if user_flag.value and class_label != "Empty":
                stop_receiving_commands.value = True
                grabbing_process(class_label, finding_medicine=stop_receiving_commands, label_queue=label_queue)  # Trigger find_medicine() if conditions are met
            else:
                stop_receiving_commands.value = False
                print("No executable commands detected, please request again.\n")
    
    except KeyboardInterrupt:
        print("Stopping threads...")

if __name__ == "__main__":
    main()