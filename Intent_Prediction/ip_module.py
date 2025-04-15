import time
import numpy as np
import sounddevice as sd
from Intent_Prediction.Datasets import New_PharmaIntent_Dataset

# INTENT_LABEL = ["other_intents", "retrieve_med", "search_med", "enquire_suitable_med"]
# Main function for the Master program
# Expected to be run forever
def listen_audio_thread(model_dict: dict, shared_dict: dict, listen_event) -> None:
    
    asr_pipe = model_dict["asr_pipe"]
    intent_pipe = model_dict["intent_pipe"]
    med_pipe = model_dict["med_list_pipe"]

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
        # print("Transcript:", transcript)

        if len(transcript) > 0 :

            intent = intent_pipe(transcript)[0]["label"][-1]
            med_list = New_PharmaIntent_Dataset.post_process_med(med_pipe(transcript))
            # print(intent, med_list)

            clean_meds = []
            for med in med_list:
                if med != "Empty":
                    clean_meds.append(med)

            # The second printing line should be audio
            if  intent == "1" and len(clean_meds) > 0:
                # print("Command Heard: Retrieve Medicine")
                print("\nRetrieving medicine. Please wait while I get it for you.")
                shared_dict["cmd_flag"] = True
                shared_dict["queued_commands"] = shared_dict["queued_commands"] + clean_meds
            
            elif intent == "2" and len(clean_meds) > 0:
                # print("Command Heard: Search Medicine")
                print("\nSeems like you are looking for a medicine. Please wait while I search for it.")
                shared_dict["cmd_flag"] = True
                shared_dict["queued_commands"] = shared_dict["queued_commands"] + clean_meds

            elif intent == "3":
                # print("Command Heard: Enquire Suitable Medicine")
                print("\nMy apologies, I am not able to diagnosis medical issues. Please consult to professional to get the best advice.")
            
            elif intent in ["1", "2"] and len(clean_meds) == 0:
                # print("Command Heard: Retrieve/Search Medicine")
                print("\nSorry, I only retrieve designated medicines. Please try again.")

            elif intent in ["0", "3"] and len(clean_meds) > 0:
                # print("Command Heard: Other Intents")
                print("\nHeard that you mentioned about chronic disease medicines. If you search for them, please let me know.")

        else:
            print("No speech detected.")
        
        # Wait 1 second before looping again 
        time.sleep(2)