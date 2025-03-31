import re
import time
from queue import Queue
import numpy as np
from transformers import Pipeline
import sounddevice as sd

# Main function for the Master program
# Expected to be run forever
def listen_audio_thread(asr_pipe: Pipeline, nlp_pipe: Pipeline, user_flag, cmd_flag, label_queue: Queue) -> None:

    while True:
        # Idle when grabbing medicine
        if user_flag.value and cmd_flag.value:
            print("IP Thread: Idle")
            time.sleep(5)
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
        audio_data = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()  # Wait until recording is finished
        audio_array = np.squeeze(audio_data)  # Convert to 1D array

        transcript = asr_pipe(audio_array)
        class_label = nlp_pipe(transcript)

        # print("Transcript:", transcript["text"])

        # Put the label in the queue (If received class not Empty, use it; If Empty, use previous one)
        if bool(re.search(r"[^a-zA-Z0-9\s'\u4e00-\u9fff]", transcript["text"])):
            label_queue.put("Empty") # Put "Empty" in the queue if the transcript contains special characters, avoid accident inputs.
        else:
            label_queue.put(class_label["label"])

        cmd_flag.value = cmd_flag.value or (class_label["label"] != "Empty")
        
        # Wait 1 second before looping again
        time.sleep(1)