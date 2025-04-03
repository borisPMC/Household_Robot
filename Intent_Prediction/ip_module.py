import re
import time
from queue import Queue
import numpy as np
from transformers import Pipeline
import sounddevice as sd

# 20250403: Table Searcher
# Pre-requisite: the patient KNOWS what disease he/she is suffering from
def process_script(shared_dict, script: str) -> int:

    lower_script = script.lower()

    # Origin table from paper
    # TABLE = {
    #     "ACE Inhibitor": ["高血壓", "血管緊張素轉換酶抑制劑", "高血壓藥", "high blood pressure", "hypertension", "ACE Inhibitor"],
    #     "Metformin": ["糖尿病", "糖尿病藥", "甲福明", "Diabetes", "Metformin"],
    #     "Atorvastatin": ["冠脈病", "冠心病", "心臟病", "Coronary heart disease", "CAD", "阿伐他汀", "心臟病藥", "Atorvastatin"],
    #     "Amitriptyline": ["抑鬱症", "Depression", "Depression disorder", "阿米替林", "抗抑鬱藥", "Amitriptyline"]
    # }

    # Optimised for searching
    MEDICINE_TABLE = {
        "ACE Inhibitor": ["高血壓", "血管緊張素轉換酶抑制劑", "high blood pressure", "hypertension", "ace inhibitor"],
        "Metformin": ["糖尿病", "甲福明", "diabetes", "metformin"],
        "Atorvastatin": ["冠脈病", "冠心病", "心臟病", "coronary heart disease", "cad", "阿伐他汀", "atorvastatin"],
        "Amitriptyline": ["抑鬱", "depression", "depression disorder", "阿米替林", "amitriptyline"]
    }

    # Verb for Intention (Direct & Assertive words): Commands often use imperative verbs (action words) and sound direct.
    VERB_LIST = ["俾", "遞", "交", "攞", "拎", "揸", "執", "抓", "要", "grab", "want",
            "offer", "provide", "present", "hand over", "deliver", "give",
            "clutch", "grasp", "pick up", "take", "capture" , "where"]

    # Hedging words (Uncertain and Questioning): Confused statements often contain hedging words (maybe, not sure, which one) or question-like structures.
    CONFUSE_LIST = ["邊個", "邊隻", "邊款", "which", "what", "unsure", "uncertain", "not sure"]

    # Assertive words
    ASSERTIVE_LIST = ["yes", "ok", "係"]

    # Default
    detected_med = []
    is_direct = False
    is_confused = False
    is_request_confirm = len(shared_dict["cache_requests"]) > 0

    #TODO: Medicine Classification: If Medicine name is in the script, append detected_med with the corresponding key
    for key in MEDICINE_TABLE.keys():
        for word in MEDICINE_TABLE[key]:
            if re.search(word, lower_script):
                detected_med.append(key)
                break

    #TODO: Semantic Analysis: if confused, return; else, see if it is response to confirmation; else, see if requesting a medicine
    for word in CONFUSE_LIST:
        if re.search(word, lower_script):
            is_confused = True
            break
    if not is_confused:
        if is_request_confirm:
            searching_list = ASSERTIVE_LIST
            shared_dict["cache_requests"] = []
        else:
            searching_list = VERB_LIST
        for word in searching_list:
            if re.search(word, lower_script):
                is_direct = True
                break    

    response_type = process_request(shared_dict, detected_med, is_direct, is_confused)
    return response_type

def process_request(shared_dict, detected_med, is_direct, is_confused):

    #1 If detected_med has item(s), and a VERB_LIST item exists in the script, semantic = "certain"; ->                  "Understand."
    #2 If detected_med has item(s), and a CONFUSE_LIST item exists in the script, semantic = "confused_multi"; ->        "It seems you are confused", ask for confirmation
    #3 If detected_med is empty, and a CONFUSE_LIST item exists in the script, semantic = "confused_None"; ->            "The request is unsupported", ask specific medicine
    #4 If detected_med is empty, and a VERB_LIST item exists in the script, semantic = "certain_others"; ->              "The requested medicine is unsupported", ask specific medicine

    response_type = -1

    if len(detected_med) > 0 and is_direct:
        shared_dict["queued_commands"] = shared_dict["queued_commands"] + detected_med
        response_type = 0

    if len(detected_med) > 0 and is_confused:
        shared_dict["cache_requests"] = detected_med
        response_type = 1

    if len(detected_med) == 0 and is_direct:
        response_type = 2

    if len(detected_med) == 0 and is_confused:
        response_type = 3

    return response_type

def detect_language(string):
    cantonese_found = any("\u4E00" <= char <= "\u9FFF" for char in string)
    english_found = any(char.isalpha() for char in string)

    if cantonese_found:
        return "Cantonese"
    elif english_found:
        return "English"
    else:
        return "Unknown"

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
        lang = detect_language(transcript)
        response_type = process_script(shared_dict, transcript)

        shared_dict["cmd_flag"] = len(shared_dict["queued_commands"]) > 0

        print(response_type, lang)
        
        # Wait 1 second before looping again 
        time.sleep(2)