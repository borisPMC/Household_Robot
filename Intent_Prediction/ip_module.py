import re
import time
from queue import Queue
import numpy as np
from transformers import Pipeline
import sounddevice as sd
from Intent_Prediction.Models import TableSearcher



# Main function for the Master program
# Expected to be run forever
def listen_audio_thread(asr_pipe: Pipeline, classifier: TableSearcher, shared_dict: dict, listen_event) -> None:

    while True:

        listen_event.wait()

        # Idle when grabbing medicine
        if (shared_dict["user_flag"] and shared_dict["cmd_flag"]) or shared_dict["play_sound_flag"]:
            # print("IP Thread: Idle")
            time.sleep(shared_dict["THREAD_PROCESS_TIMER"])
            continue

        audio_array = None
        transcript = None

        # print("\nRecording for 5 seconds...")
        audio_data = sd.rec(int(shared_dict["THREAD_PROCESS_TIMER"] * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()  # Wait until recording is finished
        audio_array = np.squeeze(audio_data)  # Convert to 1D array

        transcript = asr_pipe(audio_array)["text"]
        if len(transcript) > 0 :

            # lang = classifier.(transcript)
            # response_type = process_script(shared_dict, transcript)

            # Change states in classifier once predicted a script
            classifier.predict(transcript)
            response = classifier.generate_response()
            print(response)

            if len(classifier.pop_medicine_list()) > 0:
                shared_dict["queued_commands"] = shared_dict["queued_commands"] + classifier.pop_medicine_list()

            shared_dict["cmd_flag"] = len(shared_dict["queued_commands"]) > 0

            print(shared_dict["queued_commands"])

        else:
            print("No speech detected.")
        
        # Wait 1 second before looping again 
        time.sleep(2)