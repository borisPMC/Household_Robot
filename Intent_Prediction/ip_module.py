import re
import time
from queue import Queue
import numpy as np
from transformers import Pipeline
import sounddevice as sd

# Main function for the Master program
# Expected to be run forever
def listen_audio_thread(asr_pipe: Pipeline, nlp_pipe: Pipeline, shared_dict) -> None:

    while True:
        # Idle when grabbing medicine
        if shared_dict["user_flag"] and shared_dict["cmd_flag"]:
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
        class_label = None

        # print("\nRecording for 5 seconds...")
        audio_data = sd.rec(int(shared_dict["THREAD_PROCESS_TIMER"] * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()  # Wait until recording is finished
        audio_array = np.squeeze(audio_data)  # Convert to 1D array

        transcript = asr_pipe(audio_array)
        class_label = nlp_pipe(transcript)

        # print("Transcript:", transcript["text"])

        # Put the label in the queue (If received class not Empty, use it; If Empty, use previous one)
        if bool(re.search(r"[^a-zA-Z0-9\s'\u4e00-\u9fff]", transcript["text"])) or shared_dict["label_command"] != "Empty":
            shared_dict["label_command"] = "Empty" # Put "Empty" in the queue if the transcript contains special characters, avoid accident inputs.
        else:
            shared_dict["label_command"] = class_label["label"]

        shared_dict["cmd_flag"] = shared_dict["cmd_flag"] or (class_label["label"] != "Empty")
        
        # Wait 1 second before looping again 
        time.sleep(2)