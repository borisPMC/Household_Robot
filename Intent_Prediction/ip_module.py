import re
import time
from queue import Queue
import numpy as np
from transformers import Pipeline
import sounddevice as sd

# 20250403: Table Searcher
# Pre-requisite: the patient KNOWS what disease he/she is suffering from
def detect_class(script: str) -> str:

    # Origin table from paper
    # TABLE = {
    #     "ACE Inhibitor": ["高血壓", "血管緊張素轉換酶抑制劑", "高血壓藥", "high blood pressure", "hypertension", "ACE Inhibitor"],
    #     "Metformin": ["糖尿病", "糖尿病藥", "甲福明", "Diabetes", "Metformin"],
    #     "Atorvastatin": ["冠脈病", "冠心病", "心臟病", "Coronary heart disease", "CAD", "阿伐他汀", "心臟病藥", "Atorvastatin"],
    #     "Amitriptyline": ["抑鬱症", "Depression", "Depression disorder", "阿米替林", "抗抑鬱藥", "Amitriptyline"]
    # }

    # Optimised for searching
    TABLE = {
        "ACE Inhibitor": ["高血壓", "血管緊張素轉換酶抑制劑", "high blood pressure", "hypertension", "ACE Inhibitor"],
        "Metformin": ["糖尿病", "甲福明", "Diabetes", "Metformin"],
        "Atorvastatin": ["冠脈病", "冠心病", "心臟病", "Coronary heart disease", "CAD", "阿伐他汀", "Atorvastatin"],
        "Amitriptyline": ["抑鬱", "Depression", "Depression disorder", "阿米替林", "Amitriptyline"]
    }

    # Verb for Intention (Direct & Assertive words): Commands often use imperative verbs (action words) and sound direct.
    VERB_LIST = ["俾", "遞", "交", "攞", "拎", "揸", "執", "抓", "要", "grab",
            "offer", "provide", "present", "hand over", "deliver", "give",
            "clutch", "grasp", "pick up", "take", "capture"]

    # Hedging words (Uncertain and Questioning): Confused statements often contain hedging words (maybe, not sure, which one) or question-like structures.
    CONFUSE_LIST = ["邊個", "邊隻", "邊款", "唔知", "Which", "What", "Unsure", "Should"]

    # Default
    detected_med = []
    is_direct = False
    is_confused = False

    #TODO: If Medicine name is in the script, append detected_med with the corresponding key

    #TODO: If a VERB_LIST item exists in the script, is_direct = True

    #TODO: If a CONFUSE_LIST item exists in the script, is_confused = True

    # If detected_med has only 1 item, and a VERB_LIST item exists in the script, semantic = "certain_one"; ->          execute grabbing
    # If detected_med has multiple items, and a VERB_LIST item exists in the script, semantic = "certain_multi"; ->     ask which one
    # If detected_med is empty, and a VERB_LIST item exists in the script, semantic = "certain_others"; ->              "The requested medicine is unsupported", ask specific medicine
    # If detected_med has only 1 item, and a CONFUSE_LIST item exists in the script, semantic = "confused_one"; ->      "It seems you are confused", confirm the one
    # If detected_med has multiple items, and a CONFUSE_LIST item exists in the script, semantic = "confused_multi"; -> "It seems you are confused", ask which one
    # If detected_med is empty, and a CONFUSE_LIST item exists in the script, semantic = "confused_None"; ->            "It seems you are confused", ask specific medicine
    
    



    return detected_med, is_direct, is_confused

def confirm_command(detected_med, is_direct, is_confused):

    result = []

    if is_direct and not is_confused:
        result = detected_med
    
    return result

def play_audio_thread(shared_dict: dict, semantic: bool):

    # match shared_dict["label_command"]:
    

    pass

# Main function for the Master program
# # Expected to be run forever
# def listen_audio_thread(asr_pipe: Pipeline, nlp_pipe: Pipeline, shared_dict: dict) -> None:

#     while True:
#         # Idle when grabbing medicine
#         if shared_dict["user_flag"] and shared_dict["cmd_flag"]:
#             # print("IP Thread: Idle")
#             time.sleep(shared_dict["THREAD_PROCESS_TIMER"])
#             continue
        
#         # # Not recommended entering this condition.
#         # if not user_flag.value and cmd_flag.value:
#         #     print("Wait existing user, cached command...")
#         #     time.sleep(11)
#         #     continue

#         audio_array = None
#         transcript = None
#         class_label = None

#         # print("\nRecording for 5 seconds...")
#         audio_data = sd.rec(int(shared_dict["THREAD_PROCESS_TIMER"] * 16000), samplerate=16000, channels=1, dtype="float32")
#         sd.wait()  # Wait until recording is finished
#         audio_array = np.squeeze(audio_data)  # Convert to 1D array

#         transcript = asr_pipe(audio_array)
#         class_label = nlp_pipe(transcript)
#         class_label, semantic = detect_class(transcript)

#         # print("Transcript:", transcript["text"])

#         # Put the label in the queue (If received class not Empty, use it; If Empty, use previous one)
#         if bool(re.search(r"[^a-zA-Z0-9\s'\u4e00-\u9fff]", transcript["text"])) or shared_dict["label_command"] != "Empty":
#             shared_dict["label_command"] = "Empty" # Put "Empty" in the queue if the transcript contains special characters, avoid accident inputs.
#         else:
#             shared_dict["label_command"] = class_label["label"]

#         shared_dict["cmd_flag"] = shared_dict["cmd_flag"] or (class_label["label"] != "Empty")
        
#         # Wait 1 second before looping again 
#         time.sleep(2)

# 20250403: table search
# Main function for the Master program
# Expected to be run forever
def listen_audio_thread(asr_pipe: Pipeline, shared_dict: dict) -> None:

    while True:
        # Idle when grabbing medicine
        if (shared_dict["user_flag"] and shared_dict["cmd_flag"]) or shared_dict["play_sound_flag"]:
            # print("IP Thread: Idle")
            time.sleep(shared_dict["THREAD_PROCESS_TIMER"])
            continue
        
        # # Not recommended entering this condition.
        # if not user_flag.value and cmd_flag.value:
        #     print("Wait existing user, cached command...")
        #     time.sleep(11)
        #     continue

        audio_array = None
        transcript = None

        # print("\nRecording for 5 seconds...")
        audio_data = sd.rec(int(shared_dict["THREAD_PROCESS_TIMER"] * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()  # Wait until recording is finished
        audio_array = np.squeeze(audio_data)  # Convert to 1D array

        transcript = asr_pipe(audio_array)
        shared_dict["detected_meds"], semantic = detect_class(transcript)

        shared_dict["confirmed_commands"] = confirm_command()

        elif shared_dict["user_flag"]:

            # play audio thread
            play_audio_thread = ""
            
            shared_dict["label_command"] = class_label["label"]

        shared_dict["cmd_flag"] = shared_dict["cmd_flag"] or (class_label["label"] != "Empty")
        
        # Wait 1 second before looping again 
        time.sleep(2)